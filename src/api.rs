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

    /// xorshift64*, good enough for picking pages
    fn next_rand(&self) -> u64 {
        let mut x = self.rng.load(Ordering::Relaxed);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng.store(x, Ordering::Relaxed);
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    fn check(&self, v: u32) -> Result<u32, (StatusCode, Json<Value>)> {
        if (v as usize) < self.g.node_count() {
            Ok(v)
        } else {
            Err(err(StatusCode::NOT_FOUND, "no such node"))
        }
    }
}

fn err(code: StatusCode, msg: &str) -> (StatusCode, Json<Value>) {
    (code, Json(json!({ "error": msg })))
}

pub fn router(app: Arc<App>, tiles_dir: Option<std::path::PathBuf>) -> Router {
    let mut router = Router::new();
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
        .route("/titles", get(titles))
        .route("/atlas.bin", get(atlas_bin))
        .route("/edges.bin", get(edges_bin))
        .route("/edge", get(edge))
        .route("/links_among", get(links_among))
        .route("/neighborhood", get(neighborhood))
        .route("/distance", get(distance))
        .route("/game/new", get(game_new))
        .route("/path", get(path))
        .route("/random", get(random))
        .route("/top", get(top))
        .layer(CorsLayer::permissive())
        .with_state(app)
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

/// Titles are stored in underscore form with an uppercase first letter;
/// normalize the query the same way, trying the exact form first.
async fn search(
    State(app): State<Arc<App>>,
    Query(p): Query<SearchParams>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let q = p.q.trim().replace(' ', "_");
    if q.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "empty query"));
    }
    let limit = p.limit.min(50);
    let mut candidates = prefix_candidates(&app.g, q.as_bytes());
    if candidates.is_empty() {
        let mut chars = q.chars();
        let upper: String = chars.next().unwrap().to_uppercase().chain(chars).collect();
        candidates = prefix_candidates(&app.g, upper.as_bytes());
    }
    candidates.sort_by(|a, b| app.g.rank(*b).total_cmp(&app.g.rank(*a)));
    candidates.truncate(limit);
    let items: Vec<Node> = candidates.into_iter().map(|v| app.node(v)).collect();
    Ok(Json(json!({ "items": items })))
}

fn prefix_candidates(g: &Graph, prefix: &[u8]) -> Vec<u32> {
    let range = g.prefix_range(prefix);
    let end = range.end.min(range.start + MAX_SEARCH_SCAN as u32);
    (range.start..end).collect()
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

async fn random(State(app): State<Arc<App>>) -> Json<Node> {
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
