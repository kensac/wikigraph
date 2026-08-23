//! HTTP API over the graph. Public, read-only, permissive CORS.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::get;
use axum::Router;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

use crate::algo;
use crate::graph::Graph;

const MAX_SEARCH_SCAN: usize = 20_000;
const MAX_PAGE: usize = 1_000;
const MAX_BFS_VISITED: usize = 20_000_000;

pub struct App {
    g: Graph,
    /// Optional whole-graph layout, served verbatim (file bytes = wire bytes).
    atlas: Option<memmap2::Mmap>,
    top_in: Vec<u32>,
    top_out: Vec<u32>,
    top_rank: Vec<u32>,
    /// Well-known, well-connected articles used to seed game puzzles.
    top_rank_pool: Vec<u32>,
    /// Top edges by min-endpoint rank, built lazily for the skeleton mode.
    skeleton: std::sync::OnceLock<Vec<(u32, u32)>>,
    /// Per-community stats for semantic zoom, built lazily from the atlas.
    communities: std::sync::OnceLock<Vec<CommunityInfo>>,
    /// Bounds concurrent expensive work (BFS, DAGs, meshes) so bursts queue
    /// instead of piling up blocking threads.
    heavy: tokio::sync::Semaphore,
    rng: AtomicU64,
}

const SKELETON_MAX: usize = 5_000_000;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    id: u32,
    title: String,
    rank: f32,
    out_degree: usize,
    in_degree: usize,
    /// Atlas community id, when an atlas is loaded (65535 = "other").
    #[serde(skip_serializing_if = "Option::is_none")]
    community: Option<u16>,
}

impl App {
    pub fn new(g: Graph, atlas: Option<memmap2::Mmap>) -> Arc<Self> {
        let n = g.node_count() as u32;
        let top = |key: &dyn Fn(u32) -> f64| -> Vec<u32> {
            let mut heap: BinaryHeap<(std::cmp::Reverse<u64>, u32)> = BinaryHeap::new();
            for v in 0..n {
                let k = key(v).to_bits();
                if heap.len() < 100 {
                    heap.push((std::cmp::Reverse(k), v));
                } else if heap.peek().unwrap().0 .0 < k {
                    heap.pop();
                    heap.push((std::cmp::Reverse(k), v));
                }
            }
            let mut v = heap.into_vec();
            v.sort_by_key(|&(std::cmp::Reverse(k), _)| std::cmp::Reverse(k));
            v.into_iter().map(|(_, id)| id).collect()
        };
        let top_in = top(&|v| g.in_degree(v) as f64);
        let top_out = top(&|v| g.out_degree(v) as f64);
        let top_rank = top(&|v| g.rank(v) as f64);

        // puzzle pool: top ~150k by rank that are traversable in both directions
        let mut by_rank: Vec<u32> = (0..n).collect();
        by_rank.sort_unstable_by(|&a, &b| g.rank(b).total_cmp(&g.rank(a)));
        let top_rank_pool: Vec<u32> = by_rank
            .into_iter()
            .filter(|&v| g.out_degree(v) >= 20 && g.in_degree(v) >= 20)
            .take(150_000)
            .collect();

        let seed = std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos() as u64 | 1;
        Arc::new(App {
            g,
            atlas,
            top_in,
            top_out,
            top_rank,
            top_rank_pool,
            skeleton: std::sync::OnceLock::new(),
            communities: std::sync::OnceLock::new(),
            heavy: tokio::sync::Semaphore::new(6),
            rng: AtomicU64::new(seed),
        })
    }

    fn node(&self, v: u32) -> Node {
        Node {
            id: v,
            title: self.g.title(v).to_string(),
            rank: self.g.rank(v),
            out_degree: self.g.out_degree(v),
            in_degree: self.g.in_degree(v),
            community: self.community_of(v),
        }
    }

    /// Community section lives at byte 12 + 4n in the atlas (after x and y).
    fn community_of(&self, v: u32) -> Option<u16> {
        let atlas = self.atlas.as_ref()?;
        let n = self.g.node_count();
        let off = 12 + 4 * n + 2 * v as usize;
        let bytes = atlas.get(off..off + 2)?;
        Some(u16::from_le_bytes(bytes.try_into().ok()?))
    }

    /// Atlas position of a node (x section at 12, y section at 12 + 2n).
    fn pos_of(&self, v: u32) -> Option<(u16, u16)> {
        let atlas = self.atlas.as_ref()?;
        let n = self.g.node_count();
        let x = atlas.get(12 + 2 * v as usize..12 + 2 * v as usize + 2)?;
        let yo = 12 + 2 * n + 2 * v as usize;
        let y = atlas.get(yo..yo + 2)?;
        Some((
            u16::from_le_bytes(x.try_into().ok()?),
            u16::from_le_bytes(y.try_into().ok()?),
        ))
    }

    /// xorshift64*, good enough for picking pages
    fn next_rand(&self) -> u64 {
        let mut x = self.rng.load(Ordering::Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng.store(x, Ordering::Relaxed);
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Queue expensive work behind the shared semaphore.
    async fn heavy_permit(&self) -> Result<tokio::sync::SemaphorePermit<'_>, (StatusCode, Json<Value>)> {
        self.heavy
            .acquire()
            .await
            .map_err(|_| err(StatusCode::SERVICE_UNAVAILABLE, "shutting down"))
    }

    fn check(&self, v: u32) -> Result<u32, (StatusCode, Json<Value>)> {
        if (v as usize) < self.g.node_count() {
            Ok(v)
        } else {
            Err(err(StatusCode::NOT_FOUND, "no such node"))
        }
    }
}

/// WIKIGRAPH_ORIGINS: comma-separated origin allowlist; unset = permissive
/// (local dev). Restricting origins gates browsers to the portfolio site.
fn cors_layer() -> CorsLayer {
    match std::env::var("WIKIGRAPH_ORIGINS") {
        Ok(list) if !list.trim().is_empty() => {
            let origins: Vec<axum::http::HeaderValue> = list
                .split(',')
                .filter_map(|o| o.trim().parse().ok())
                .collect();
            CorsLayer::new()
                .allow_origin(origins)
                .allow_methods([axum::http::Method::GET])
        }
        _ => CorsLayer::permissive(),
    }
}

fn err(code: StatusCode, msg: &str) -> (StatusCode, Json<Value>) {
    (code, Json(json!({ "error": msg })))
}

pub fn router(app: Arc<App>, tiles_dir: Option<std::path::PathBuf>) -> Router {
    let (metric_layer, metric_handle) = axum_prometheus::PrometheusMetricLayer::pair();
    let mut router = Router::new();
    router = router.route(
        "/metrics",
        get(move || std::future::ready(metric_handle.render())),
    );
    if let Some(dir) = tiles_dir {
        router = router.nest_service("/tiles", tower_http::services::ServeDir::new(dir));
    }
    router
        .route("/healthz", get(|| async { "ok" }))
        .route("/stats", get(stats))
        .route("/search", get(search))
        .route("/node/{id}", get(node))
        .route("/node/{id}/links", get(links))
        .route("/node/{id}/neighbors.bin", get(neighbors_bin))
        .route("/node/{id}/mesh.bin", get(mesh_bin))
        .route("/node/{id}/links/search", get(links_search))
        .route("/titles", get(titles))
        .route("/atlas.bin", get(atlas_bin))
        .route("/edges.bin", get(edges_bin))
        .route("/edges_region.bin", get(edges_region))
        .route("/communities", get(communities))
        .route("/bfs.bin", get(bfs_bin))
        .route("/edge", get(edge))
        .route("/links_among", get(links_among))
        .route("/neighborhood", get(neighborhood))
        .route("/distance", get(distance))
        .route("/game/new", get(game_new))
        .route("/path", get(path))
        .route("/paths", get(paths))
        .route("/random", get(random))
        .route("/top", get(top))
        .layer(cors_layer())
        .layer(metric_layer)
        .layer(axum::middleware::from_fn(log_request))
        .layer(tower_http::compression::CompressionLayer::new())
        // everything is immutable between data snapshots; let browsers and the
        // CDN cache anything that doesn't set its own policy
        .layer(tower_http::set_header::SetResponseHeaderLayer::if_not_present(
            axum::http::header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("public, max-age=3600"),
        ))
        .with_state(app)
}

/// One structured line per request; docker json logs -> promtail -> loki.
async fn log_request(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let start = std::time::Instant::now();
    let res = next.run(req).await;
    tracing::info!(
        target: "request",
        %method,
        path,
        status = res.status().as_u16(),
        ms = start.elapsed().as_millis() as u64,
    );
    res
}

async fn stats(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({
        "nodes": app.g.node_count(),
        "edges": app.g.edge_count(),
    }))
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

fn default_limit() -> usize {
    10
}

/// Titles are stored in underscore form. Tiers: prefix match on the query and
/// its capitalization variants (first-letter and per-word — "albert einstein"
/// must find Albert_Einstein), then a parallel substring scan as a fallback.
/// Exact title matches are always pinned first.
async fn search(
    State(app): State<Arc<App>>,
    Query(p): Query<SearchParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let q = p.q.trim().replace(' ', "_");
    if q.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "empty query"));
    }
    let limit = p.limit.min(50);
    let app2 = app.clone();
    let candidates = tokio::task::spawn_blocking(move || {
        let g = &app2.g;
        let mut variants = vec![q.clone(), first_upper(&q), title_case(&q)];
        variants.dedup();
        let mut candidates: Vec<u32> = Vec::new();
        for v in &variants {
            let range = g.prefix_range(v.as_bytes());
            let end = range.end.min(range.start + MAX_SEARCH_SCAN as u32);
            candidates.extend(range.start..end);
        }
        // substring supplements thin prefix results ("einstein" must also
        // surface Albert_Einstein); fuzzy is the last tier for typos
        if candidates.len() < limit && q.len() >= 4 {
            candidates.extend(substring_scan(g, &q));
        }
        if candidates.is_empty() && (4..=40).contains(&q.len()) {
            candidates = fuzzy_scan(g, &q);
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates.sort_by(|a, b| g.rank(*b).total_cmp(&g.rank(*a)));
        // exact matches (any variant) pinned first
        let mut pinned: Vec<u32> = variants
            .iter()
            .filter_map(|v| g.find(v.as_bytes()))
            .collect();
        pinned.dedup();
        candidates.retain(|v| !pinned.contains(v));
        pinned.extend(candidates);
        pinned
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "search failed"))?;
    let items: Vec<Node> = candidates
        .into_iter()
        .take(limit)
        .map(|v| app.node(v))
        .collect();
    Ok(Json(json!({ "items": items })))
}

fn first_upper(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Capitalize the first letter of every underscore-separated word.
fn title_case(s: &str) -> String {
    s.split('_').map(first_upper).collect::<Vec<_>>().join("_")
}

/// Bounded edit-distance scan (<= 2 edits, ASCII case-insensitive): typo
/// tolerance when every other tier came up empty.
fn fuzzy_scan(g: &Graph, needle: &str) -> Vec<u32> {
    use rayon::prelude::*;
    let needle = needle.to_ascii_lowercase();
    let nb = needle.as_bytes();
    (0..g.node_count() as u32)
        .into_par_iter()
        .filter(|&v| {
            let t = g.title_bytes(v);
            t.len().abs_diff(nb.len()) <= 2 && edit_within_2(t, nb)
        })
        .collect()
}

/// Banded Levenshtein, ASCII case-insensitive, early-out above 2.
fn edit_within_2(a: &[u8], b: &[u8]) -> bool {
    let (m, n) = (a.len(), b.len());
    let mut prev: Vec<u16> = (0..=n as u16).collect();
    let mut cur = vec![0u16; n + 1];
    for i in 1..=m {
        cur[0] = i as u16;
        let mut row_min = cur[0];
        for j in 1..=n {
            let cost = u16::from(!a[i - 1].eq_ignore_ascii_case(&b[j - 1]));
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            row_min = row_min.min(cur[j]);
        }
        if row_min > 2 {
            return false;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[n] <= 2
}

/// Case-insensitive (ASCII) substring scan over every title, in parallel.
fn substring_scan(g: &Graph, needle: &str) -> Vec<u32> {
    use rayon::prelude::*;
    let needle = needle.as_bytes();
    let mut hits: Vec<u32> = (0..g.node_count() as u32)
        .into_par_iter()
        .filter(|&v| {
            let t = g.title_bytes(v);
            t.len() >= needle.len()
                && t.windows(needle.len()).any(|w| w.eq_ignore_ascii_case(needle))
        })
        .collect();
    hits.truncate(MAX_SEARCH_SCAN);
    hits
}

async fn node(
    State(app): State<Arc<App>>,
    Path(id): Path<u32>,
) -> Result<Json<Node>, (StatusCode, Json<Value>)> {
    Ok(Json(app.node(app.check(id)?)))
}

#[derive(Deserialize)]
struct LinksParams {
    #[serde(default)]
    dir: Dir,
    #[serde(default = "default_links_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn default_links_limit() -> usize {
    30
}

#[derive(Deserialize, Default, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Dir {
    #[default]
    Out,
    In,
}

/// Neighbors ordered by pagerank descending, paged by offset/limit.
async fn links(
    State(app): State<Arc<App>>,
    Path(id): Path<u32>,
    Query(p): Query<LinksParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let id = app.check(id)?;
    let neighbors = match p.dir {
        Dir::Out => app.g.out(id),
        Dir::In => app.g.incoming(id),
    };
    let limit = p.limit.min(200);
    let want = (p.offset + limit).min(MAX_PAGE);
    if p.offset >= want {
        return Err(err(StatusCode::BAD_REQUEST, "offset too deep"));
    }
    // top-(offset+limit) by rank via bounded min-heap, then slice the page
    let mut heap: BinaryHeap<(std::cmp::Reverse<u64>, u32)> = BinaryHeap::with_capacity(want + 1);
    for &v in neighbors {
        let k = (app.g.rank(v) as f64).to_bits();
        if heap.len() < want {
            heap.push((std::cmp::Reverse(k), v));
        } else if heap.peek().unwrap().0 .0 < k {
            heap.pop();
            heap.push((std::cmp::Reverse(k), v));
        }
    }
    let mut ranked = heap.into_vec();
    ranked.sort_by_key(|&(std::cmp::Reverse(k), _)| std::cmp::Reverse(k));
    let items: Vec<Node> = ranked
        .into_iter()
        .skip(p.offset)
        .map(|(_, v)| app.node(v))
        .collect();
    Ok(Json(json!({ "total": neighbors.len(), "items": items })))
}

#[derive(Deserialize)]
struct NeighborsBinParams {
    #[serde(default)]
    dir: Dir,
}

/// The complete, uncapped neighbor set as packed little-endian binary:
/// [count: u32][ids: count x u32][ranks: count x f32]. This is close to a
/// straight copy out of the mmap'd CSR slice, so a 1.6M-degree hub is one
/// ~13MB response with zero serialization cost.
async fn neighbors_bin(
    State(app): State<Arc<App>>,
    Path(id): Path<u32>,
    Query(p): Query<NeighborsBinParams>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<Value>)> {
    let id = app.check(id)?;
    let app2 = app.clone();
    let body = tokio::task::spawn_blocking(move || {
        let neighbors = match p.dir {
            Dir::Out => app2.g.out(id),
            Dir::In => app2.g.incoming(id),
        };
        let n = neighbors.len();
        let mut buf = Vec::with_capacity(4 + n * 8);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        for &v in neighbors {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for &v in neighbors {
            buf.extend_from_slice(&app2.g.rank(v).to_le_bytes());
        }
        buf
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "gather failed"))?;
    Ok((
        [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
        body,
    ))
}

/// The precomputed whole-graph layout, ~7 bytes per article.
async fn atlas_bin(
    State(app): State<Arc<App>>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<Value>)> {
    let atlas = app
        .atlas
        .as_ref()
        .ok_or_else(|| err(StatusCode::NOT_FOUND, "no atlas built"))?;
    Ok((
        [
            (axum::http::header::CONTENT_TYPE, "application/octet-stream"),
            (axum::http::header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        atlas.to_vec(),
    ))
}

#[derive(Deserialize)]
struct LinksSearchParams {
    q: String,
    #[serde(default)]
    dir: Dir,
    #[serde(default = "default_limit")]
    limit: usize,
}

/// Search within one node's neighbor set (case-insensitive substring over
/// titles), rank-ordered. Even a 1.6M-link hub is a fast linear scan.
async fn links_search(
    State(app): State<Arc<App>>,
    Path(id): Path<u32>,
    Query(p): Query<LinksSearchParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let id = app.check(id)?;
    let needle = p.q.trim().replace(' ', "_").to_lowercase();
    if needle.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "empty query"));
    }
    let limit = p.limit.min(50);
    let app2 = app.clone();
    let ids = tokio::task::spawn_blocking(move || {
        let neighbors = match p.dir {
            Dir::Out => app2.g.out(id),
            Dir::In => app2.g.incoming(id),
        };
        let mut hits: Vec<u32> = neighbors
            .iter()
            .copied()
            .filter(|&v| app2.g.title(v).to_lowercase().contains(&needle))
            .collect();
        hits.sort_by(|&a, &b| app2.g.rank(b).total_cmp(&app2.g.rank(a)));
        hits.truncate(limit);
        hits
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "search failed"))?;
    let items: Vec<Node> = ids.into_iter().map(|v| app.node(v)).collect();
    Ok(Json(json!({ "items": items })))
}

#[derive(Deserialize)]
struct MeshBinParams {
    #[serde(default)]
    dir: Dir,
    #[serde(default = "default_mesh_limit")]
    limit: usize,
}

fn default_mesh_limit() -> usize {
    2_000_000
}

/// All directed edges among {center} ∪ neighbors(center, dir), as packed LE
/// [count u32][from: count x u32][to: count x u32]. When the induced set
/// exceeds `limit`, the top edges by min-endpoint rank are kept. Membership
/// tests use a bitmap over the id space, so even ISBN-sized sets stay fast.
async fn mesh_bin(
    State(app): State<Arc<App>>,
    Path(id): Path<u32>,
    Query(p): Query<MeshBinParams>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<Value>)> {
    let id = app.check(id)?;
    let limit = p.limit.clamp(1_000, 5_000_000);
    let _permit = app.heavy_permit().await?;
    let app2 = app.clone();
    let body = tokio::task::spawn_blocking(move || {
        let neighbors = match p.dir {
            Dir::Out => app2.g.out(id),
            Dir::In => app2.g.incoming(id),
        };
        let n = app2.g.node_count();
        let mut bits = vec![0u64; n.div_ceil(64)];
        let set = |bits: &mut [u64], v: u32| bits[v as usize / 64] |= 1 << (v % 64);
        let test = |bits: &[u64], v: u32| bits[v as usize / 64] >> (v % 64) & 1 == 1;
        set(&mut bits, id);
        for &v in neighbors {
            set(&mut bits, v);
        }
        let mut edges: Vec<(f32, u32, u32)> = Vec::new();
        for &from in std::iter::once(&id).chain(neighbors.iter()) {
            for &to in app2.g.out(from) {
                if test(&bits, to) {
                    edges.push((app2.g.rank(from).min(app2.g.rank(to)), from, to));
                }
            }
        }
        if edges.len() > limit {
            edges.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            edges.truncate(limit);
        }
        let mut buf = Vec::with_capacity(4 + edges.len() * 8);
        buf.extend_from_slice(&(edges.len() as u32).to_le_bytes());
        for &(_, from, _) in &edges {
            buf.extend_from_slice(&from.to_le_bytes());
        }
        for &(_, _, to) in &edges {
            buf.extend_from_slice(&to.to_le_bytes());
        }
        buf
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "mesh failed"))?;
    Ok((
        [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
        body,
    ))
}

#[derive(Deserialize)]
struct TitlesParams {
    /// Comma-separated node ids, at most 1000.
    ids: String,
}

async fn titles(
    State(app): State<Arc<App>>,
    Query(p): Query<TitlesParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let mut items = serde_json::Map::new();
    for part in p.ids.split(',').take(1000) {
        let Ok(id) = part.trim().parse::<u32>() else {
            return Err(err(StatusCode::BAD_REQUEST, "bad id list"));
        };
        if (id as usize) < app.g.node_count() {
            items.insert(id.to_string(), json!(app.g.title(id)));
        }
    }
    Ok(Json(json!({ "titles": items })))
}

#[derive(Deserialize)]
struct PathParams {
    from: u32,
    to: u32,
}

async fn path(
    State(app): State<Arc<App>>,
    Query(p): Query<PathParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let from = app.check(p.from)?;
    let to = app.check(p.to)?;
    let _permit = app.heavy_permit().await?;
    let app2 = app.clone();
    let found = tokio::task::spawn_blocking(move || {
        algo::shortest_path(&app2.g, from, to, MAX_BFS_VISITED)
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "search failed"))?;
    match found {
        Some(ids) => {
            let items: Vec<Node> = ids.into_iter().map(|v| app.node(v)).collect();
            Ok(Json(json!({ "path": items })))
        }
        None => Err(err(StatusCode::NOT_FOUND, "no path found")),
    }
}

#[derive(Deserialize)]
struct PathsParams {
    from: u32,
    to: u32,
    #[serde(default = "default_paths_max_nodes")]
    max_nodes: usize,
}

fn default_paths_max_nodes() -> usize {
    500
}

/// ALL shortest paths between two articles, as a DAG. When the DAG exceeds
/// max_nodes, the highest-ranked nodes are kept (endpoints always included)
/// and the edge set is filtered to survivors; totals report the full size.
async fn paths(
    State(app): State<Arc<App>>,
    Query(p): Query<PathsParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let from = app.check(p.from)?;
    let to = app.check(p.to)?;
    let max_nodes = p.max_nodes.clamp(2, 2000);
    let _permit = app.heavy_permit().await?;
    let app2 = app.clone();
    let found = tokio::task::spawn_blocking(move || {
        algo::shortest_path_dag(&app2.g, from, to, MAX_BFS_VISITED)
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "search failed"))?;
    let Some((d, mut nodes, edges)) = found else {
        return Err(err(StatusCode::NOT_FOUND, "no path found"));
    };
    let (total_nodes, total_edges) = (nodes.len(), edges.len());
    if nodes.len() > max_nodes {
        nodes.sort_unstable_by(|&a, &b| app.g.rank(b).total_cmp(&app.g.rank(a)));
        nodes.truncate(max_nodes);
        for v in [from, to] {
            if !nodes.contains(&v) {
                nodes.push(v);
            }
        }
    }
    let keep: std::collections::HashSet<u32> = nodes.iter().copied().collect();
    let edges: Vec<[u32; 2]> = edges
        .into_iter()
        .filter(|&(u, v)| keep.contains(&u) && keep.contains(&v))
        .map(|(u, v)| [u, v])
        .collect();
    let items: Vec<Node> = nodes.into_iter().map(|v| app.node(v)).collect();
    Ok(Json(json!({
        "distance": d,
        "totalNodes": total_nodes,
        "totalEdges": total_edges,
        "nodes": items,
        "edges": edges,
    })))
}

#[derive(Deserialize)]
struct EdgeParams {
    from: u32,
    to: u32,
}

/// Whether a direct wikilink from -> to exists (neighbor lists are sorted).
async fn edge(
    State(app): State<Arc<App>>,
    Query(p): Query<EdgeParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let from = app.check(p.from)?;
    let to = app.check(p.to)?;
    let exists = app.g.out(from).binary_search(&to).is_ok();
    Ok(Json(json!({ "exists": exists })))
}

/// Hop count of the shortest path, without revealing the path itself.
async fn distance(
    State(app): State<Arc<App>>,
    Query(p): Query<EdgeParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let from = app.check(p.from)?;
    let to = app.check(p.to)?;
    let _permit = app.heavy_permit().await?;
    let app2 = app.clone();
    let d = tokio::task::spawn_blocking(move || {
        algo::distance(&app2.g, from, to, MAX_BFS_VISITED)
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "search failed"))?;
    match d {
        Some(d) => Ok(Json(json!({ "distance": d }))),
        None => Err(err(StatusCode::NOT_FOUND, "no path found")),
    }
}

#[derive(Deserialize)]
struct EdgesBinParams {
    #[serde(default = "default_edges_limit")]
    limit: usize,
}

fn default_edges_limit() -> usize {
    2_000_000
}

/// The graph's skeleton: the top edges scored by the weaker endpoint's rank,
/// as packed LE [count u32][from: count x u32][to: count x u32]. Pairs with
/// atlas.bin positions to draw the whole graph's structure in one overlay.
async fn edges_bin(
    State(app): State<Arc<App>>,
    Query(p): Query<EdgesBinParams>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<Value>)> {
    let limit = p.limit.clamp(100_000, SKELETON_MAX);
    let app2 = app.clone();
    let body = tokio::task::spawn_blocking(move || {
        let skeleton = app2.skeleton.get_or_init(|| build_skeleton(&app2.g));
        let take = limit.min(skeleton.len());
        let mut buf = Vec::with_capacity(4 + take * 8);
        buf.extend_from_slice(&(take as u32).to_le_bytes());
        for &(from, _) in &skeleton[..take] {
            buf.extend_from_slice(&from.to_le_bytes());
        }
        for &(_, to) in &skeleton[..take] {
            buf.extend_from_slice(&to.to_le_bytes());
        }
        buf
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "skeleton failed"))?;
    Ok((
        [
            (axum::http::header::CONTENT_TYPE, "application/octet-stream"),
            (axum::http::header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        body,
    ))
}

#[derive(Deserialize)]
struct BfsParams {
    from: u32,
    #[serde(default)]
    dir: BfsDir,
    #[serde(default = "default_bfs_depth")]
    max_depth: u8,
}

#[derive(Deserialize, Default, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
enum BfsDir {
    #[default]
    Out,
    In,
    Both,
}

fn default_bfs_depth() -> u8 {
    12
}

/// Whole-graph BFS depth map from one seed: one u8 per node (255 = beyond
/// max_depth or unreachable). ~7MB; a precomputed input for code-mode scripts
/// ("color everything by distance from X") that a browser could not derive.
async fn bfs_bin(
    State(app): State<Arc<App>>,
    Query(p): Query<BfsParams>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<Value>)> {
    let from = app.check(p.from)?;
    let max_depth = p.max_depth.min(254);
    let _permit = app.heavy_permit().await?;
    let app2 = app.clone();
    let body = tokio::task::spawn_blocking(move || {
        let n = app2.g.node_count();
        let mut depth = vec![255u8; n];
        depth[from as usize] = 0;
        let mut frontier = vec![from];
        let mut d = 0u8;
        while !frontier.is_empty() && d < max_depth {
            d += 1;
            let mut next = Vec::new();
            for &v in &frontier {
                let mut push = |u: u32| {
                    if depth[u as usize] == 255 {
                        depth[u as usize] = d;
                        next.push(u);
                    }
                };
                if p.dir != BfsDir::In {
                    app2.g.out(v).iter().copied().for_each(&mut push);
                }
                if p.dir != BfsDir::Out {
                    app2.g.incoming(v).iter().copied().for_each(&mut push);
                }
            }
            frontier = next;
        }
        depth
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "bfs failed"))?;
    Ok((
        [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
        body,
    ))
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct CommunityInfo {
    id: u16,
    size: u32,
    /// Title of the community's highest-ranked member: its de facto name.
    label: String,
    top_id: u32,
    /// Centroid in atlas u16 coordinates.
    x: u16,
    y: u16,
}

#[derive(Deserialize)]
struct CommunitiesParams {
    #[serde(default = "default_communities_limit")]
    limit: usize,
}

fn default_communities_limit() -> usize {
    200
}

/// Communities ordered by size, for semantic zoom: far out, the map shows
/// named community blobs instead of raw articles.
async fn communities(
    State(app): State<Arc<App>>,
    Query(p): Query<CommunitiesParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if app.atlas.is_none() {
        return Err(err(StatusCode::NOT_FOUND, "no atlas built"));
    }
    let app2 = app.clone();
    let list = tokio::task::spawn_blocking(move || {
        app2.communities
            .get_or_init(|| build_communities(&app2))
            .clone()
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "communities failed"))?;
    let items: Vec<&CommunityInfo> = list.iter().take(p.limit.min(2000)).collect();
    Ok(Json(json!({ "items": items })))
}

fn build_communities(app: &App) -> Vec<CommunityInfo> {
    #[derive(Default, Clone)]
    struct Acc {
        size: u32,
        sum_x: u64,
        sum_y: u64,
        top: u32,
        top_rank: f32,
    }
    let mut accs: std::collections::HashMap<u16, Acc> = std::collections::HashMap::new();
    for v in 0..app.g.node_count() as u32 {
        let (Some(c), Some((x, y))) = (app.community_of(v), app.pos_of(v)) else {
            continue;
        };
        let a = accs.entry(c).or_default();
        a.size += 1;
        a.sum_x += x as u64;
        a.sum_y += y as u64;
        let r = app.g.rank(v);
        if r > a.top_rank {
            a.top_rank = r;
            a.top = v;
        }
    }
    let mut list: Vec<CommunityInfo> = accs
        .into_iter()
        .map(|(id, a)| CommunityInfo {
            id,
            size: a.size,
            label: app.g.title(a.top).to_string(),
            top_id: a.top,
            x: (a.sum_x / a.size as u64) as u16,
            y: (a.sum_y / a.size as u64) as u16,
        })
        .collect();
    list.sort_unstable_by_key(|c| std::cmp::Reverse(c.size));
    list
}

#[derive(Deserialize)]
struct RegionParams {
    x0: u16,
    y0: u16,
    x1: u16,
    y1: u16,
    #[serde(default = "default_region_limit")]
    limit: usize,
}

fn default_region_limit() -> usize {
    200_000
}

/// Viewport virtualization for the skeleton: the highest-ranked edges with an
/// endpoint inside the given atlas-coordinate rect. The skeleton is already
/// rank-sorted, so this is a linear scan with early exit — a few ms.
async fn edges_region(
    State(app): State<Arc<App>>,
    Query(p): Query<RegionParams>,
) -> Result<impl axum::response::IntoResponse, (StatusCode, Json<Value>)> {
    if app.atlas.is_none() {
        return Err(err(StatusCode::NOT_FOUND, "no atlas built"));
    }
    let limit = p.limit.clamp(1_000, 1_000_000);
    let _permit = app.heavy_permit().await?;
    let app2 = app.clone();
    let body = tokio::task::spawn_blocking(move || {
        let skeleton = app2.skeleton.get_or_init(|| build_skeleton(&app2.g));
        let inside = |v: u32| {
            app2.pos_of(v).is_some_and(|(x, y)| {
                x >= p.x0 && x <= p.x1 && y >= p.y0 && y <= p.y1
            })
        };
        let mut edges = Vec::with_capacity(limit.min(65_536));
        for &(from, to) in skeleton.iter() {
            if inside(from) || inside(to) {
                edges.push((from, to));
                if edges.len() >= limit {
                    break;
                }
            }
        }
        let mut buf = Vec::with_capacity(4 + edges.len() * 8);
        buf.extend_from_slice(&(edges.len() as u32).to_le_bytes());
        for &(from, _) in &edges {
            buf.extend_from_slice(&from.to_le_bytes());
        }
        for &(_, to) in &edges {
            buf.extend_from_slice(&to.to_le_bytes());
        }
        buf
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "region failed"))?;
    Ok((
        [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
        body,
    ))
}

/// Threshold-sample then collect: estimate the score cutoff for the top
/// SKELETON_MAX edges from a stride sample, gather everything above it in
/// parallel, then sort and truncate exactly.
fn build_skeleton(g: &Graph) -> Vec<(u32, u32)> {
    use rayon::prelude::*;
    let n = g.node_count() as u32;
    let m = g.edge_count();
    let score = |a: u32, b: u32| g.rank(a).min(g.rank(b));

    let mut sample: Vec<f32> = (0..n)
        .into_par_iter()
        .flat_map_iter(|v| {
            g.out(v)
                .iter()
                .step_by(128)
                .map(move |&to| score(v, to))
                .collect::<Vec<_>>()
        })
        .collect();
    let want_in_sample = ((SKELETON_MAX as f64 / m as f64) * sample.len() as f64) as usize;
    let cut_idx = sample.len().saturating_sub(want_in_sample.max(1));
    sample.par_sort_unstable_by(f32::total_cmp);
    let threshold = sample.get(cut_idx).copied().unwrap_or(0.0);

    let mut edges: Vec<(f32, u32, u32)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|v| {
            g.out(v)
                .iter()
                .filter_map(move |&to| {
                    let s = score(v, to);
                    (s >= threshold).then_some((s, v, to))
                })
                .collect::<Vec<_>>()
        })
        .collect();
    edges.par_sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    edges.truncate(SKELETON_MAX);
    eprintln!("skeleton built: {} edges (threshold {:.3e})", edges.len(), threshold);
    edges.into_iter().map(|(_, f, t)| (f, t)).collect()
}

#[derive(Deserialize)]
struct LinksAmongParams {
    /// Comma-separated node ids, at most 1000.
    ids: String,
}

/// Every directed edge among the given nodes: the induced subgraph. Lets the
/// canvas show cross-links between articles that arrived via different
/// expansions.
async fn links_among(
    State(app): State<Arc<App>>,
    Query(p): Query<LinksAmongParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let mut ids = Vec::new();
    for part in p.ids.split(',').take(1000) {
        let Ok(id) = part.trim().parse::<u32>() else {
            return Err(err(StatusCode::BAD_REQUEST, "bad id list"));
        };
        if (id as usize) < app.g.node_count() {
            ids.push(id);
        }
    }
    let set: std::collections::HashSet<u32> = ids.iter().copied().collect();
    let mut edges = Vec::new();
    for &from in &ids {
        for &to in app.g.out(from) {
            if set.contains(&to) {
                edges.push([from, to]);
            }
        }
    }
    Ok(Json(json!({ "edges": edges })))
}

#[derive(Deserialize)]
struct NeighborhoodParams {
    id: u32,
    #[serde(default = "default_depth")]
    depth: usize,
    #[serde(default = "default_max_nodes")]
    max_nodes: usize,
}

fn default_depth() -> usize {
    2
}

fn default_max_nodes() -> usize {
    50
}

/// One-shot subgraph drop: rank-prioritized BFS out from a seed to `depth`
/// hops, capped at `max_nodes`, plus the induced edge set.
async fn neighborhood(
    State(app): State<Arc<App>>,
    Query(p): Query<NeighborhoodParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let seed = app.check(p.id)?;
    let depth = p.depth.clamp(1, 3);
    let max_nodes = p.max_nodes.clamp(2, 300);

    let mut picked = vec![seed];
    let mut seen: std::collections::HashSet<u32> = [seed].into();
    let mut frontier = vec![seed];
    for _ in 0..depth {
        if picked.len() >= max_nodes {
            break;
        }
        // gather candidates one hop out, keep the highest-ranked first
        let mut candidates: Vec<u32> = Vec::new();
        for &v in &frontier {
            for &u in app.g.out(v) {
                if !seen.contains(&u) {
                    candidates.push(u);
                }
            }
        }
        candidates.sort_unstable_by(|&a, &b| app.g.rank(b).total_cmp(&app.g.rank(a)));
        candidates.dedup();
        frontier = Vec::new();
        for u in candidates {
            if picked.len() >= max_nodes {
                break;
            }
            if seen.insert(u) {
                picked.push(u);
                frontier.push(u);
            }
        }
    }

    let set: std::collections::HashSet<u32> = picked.iter().copied().collect();
    let mut edges = Vec::new();
    for &from in &picked {
        for &to in app.g.out(from) {
            if set.contains(&to) {
                edges.push([from, to]);
            }
        }
    }
    let nodes: Vec<Node> = picked.into_iter().map(|v| app.node(v)).collect();
    Ok(Json(json!({ "nodes": nodes, "edges": edges })))
}

#[derive(Deserialize)]
struct GameParams {
    #[serde(default = "default_min_par")]
    min_par: usize,
    #[serde(default = "default_max_par")]
    max_par: usize,
    #[serde(default)]
    difficulty: Difficulty,
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Difficulty {
    Easy,
    #[default]
    Medium,
    Hard,
}

impl Difficulty {
    /// How deep into the fame-ordered pool puzzles may draw from.
    fn pool_len(self, full: usize) -> usize {
        match self {
            Difficulty::Easy => full.min(10_000),
            Difficulty::Medium => full.min(50_000),
            Difficulty::Hard => full,
        }
    }
}

fn default_min_par() -> usize {
    3
}

fn default_max_par() -> usize {
    5
}

/// A link-game puzzle: two well-known articles whose shortest chain is
/// min_par..=max_par clicks. Well-known = top ~2% by pagerank with a real
/// out-degree, so every step has reasonable choices.
async fn game_new(
    State(app): State<Arc<App>>,
    Query(p): Query<GameParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let (min_par, max_par) = (p.min_par.clamp(2, 8), p.max_par.clamp(2, 8));
    if min_par > max_par {
        return Err(err(StatusCode::BAD_REQUEST, "min_par > max_par"));
    }
    let _permit = app.heavy_permit().await?;
    let app2 = app.clone();
    let found = tokio::task::spawn_blocking(move || {
        let pool = &app2.top_rank_pool[..p.difficulty.pool_len(app2.top_rank_pool.len())];
        for _ in 0..200 {
            let a = pool[app2.next_rand() as usize % pool.len()];
            let b = pool[app2.next_rand() as usize % pool.len()];
            if a == b {
                continue;
            }
            // par computed on the real graph; cap visits so misses stay cheap
            if let Some(d) = algo::distance(&app2.g, a, b, 2_000_000) {
                if (min_par..=max_par).contains(&d) {
                    return Some((a, b, d));
                }
            }
        }
        None
    })
    .await
    .map_err(|_| err(StatusCode::INTERNAL_SERVER_ERROR, "search failed"))?;
    match found {
        Some((a, b, d)) => Ok(Json(json!({
            "from": app.node(a),
            "to": app.node(b),
            "par": d,
        }))),
        None => Err(err(StatusCode::SERVICE_UNAVAILABLE, "no puzzle found, retry")),
    }
}

#[derive(Deserialize)]
struct RandomParams {
    /// Reject articles with fewer total links; keeps Random interesting.
    #[serde(default)]
    min_links: usize,
}

async fn random(State(app): State<Arc<App>>, Query(p): Query<RandomParams>) -> Json<Node> {
    let min = p.min_links.min(1_000);
    for _ in 0..200 {
        let v = (app.next_rand() % app.g.node_count() as u64) as u32;
        if app.g.out_degree(v) + app.g.in_degree(v) >= min {
            return Json(app.node(v));
        }
    }
    let v = (app.next_rand() % app.g.node_count() as u64) as u32;
    Json(app.node(v))
}

#[derive(Deserialize)]
struct TopParams {
    #[serde(default)]
    by: TopBy,
    #[serde(default = "default_limit")]
    limit: usize,
}

#[derive(Deserialize, Default, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum TopBy {
    In,
    Out,
    #[default]
    Rank,
}

async fn top(State(app): State<Arc<App>>, Query(p): Query<TopParams>) -> Json<Value> {
    let ids = match p.by {
        TopBy::In => &app.top_in,
        TopBy::Out => &app.top_out,
        TopBy::Rank => &app.top_rank,
    };
    let items: Vec<Node> = ids
        .iter()
        .take(p.limit.min(100))
        .map(|&v| app.node(v))
        .collect();
    Json(json!({ "items": items }))
}
