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
    top_in: Vec<u32>,
    top_out: Vec<u32>,
    top_rank: Vec<u32>,
    rng: AtomicU64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    id: u32,
    title: String,
    rank: f32,
    out_degree: usize,
    in_degree: usize,
}

impl App {
    pub fn new(g: Graph) -> Arc<Self> {
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
        let seed = std::time::UNIX_EPOCH.elapsed().unwrap().as_nanos() as u64 | 1;
        Arc::new(App {
            g,
            top_in,
            top_out,
            top_rank,
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
        }
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

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .route("/stats", get(stats))
        .route("/search", get(search))
        .route("/node/{id}", get(node))
        .route("/node/{id}/links", get(links))
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

async fn random(State(app): State<Arc<App>>) -> Json<Node> {
    // xorshift64*, good enough for picking a page
    let mut x = app.rng.load(Ordering::Relaxed);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    app.rng.store(x, Ordering::Relaxed);
    let v = (x.wrapping_mul(0x2545F4914F6CDD1D) % app.g.node_count() as u64) as u32;
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
