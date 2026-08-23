//! Graph algorithms over the mmap'd store.

use std::collections::HashMap;

use crate::graph::Graph;

/// Bidirectional BFS over directed edges. Returns the node sequence from
/// `from` to `to` inclusive, or None if no path within the limits.
pub fn shortest_path(g: &Graph, from: u32, to: u32, max_visited: usize) -> Option<Vec<u32>> {
    if from == to {
        return Some(vec![from]);
    }
    // parent maps double as visited sets
    let mut fwd: HashMap<u32, u32> = HashMap::from([(from, from)]);
    let mut bwd: HashMap<u32, u32> = HashMap::from([(to, to)]);
    let mut fwd_frontier = vec![from];
    let mut bwd_frontier = vec![to];

    while !fwd_frontier.is_empty() && !bwd_frontier.is_empty() {
        if fwd.len() + bwd.len() > max_visited {
            return None;
        }
        // expand the cheaper side (sum of degrees)
        let fwd_cost: usize = fwd_frontier.iter().map(|&v| g.out_degree(v)).sum();
        let bwd_cost: usize = bwd_frontier.iter().map(|&v| g.in_degree(v)).sum();
        let forward = fwd_cost <= bwd_cost;
        let (frontier, own, other) = if forward {
            (&mut fwd_frontier, &mut fwd, &mut bwd)
        } else {
            (&mut bwd_frontier, &mut bwd, &mut fwd)
        };
        let mut next = Vec::new();
        for &v in frontier.iter() {
            let neighbors = if forward { g.out(v) } else { g.incoming(v) };
            for &u in neighbors {
                if own.contains_key(&u) {
                    continue;
                }
                own.insert(u, v);
                if other.contains_key(&u) {
                    return Some(stitch(&fwd, &bwd, u, from, to));
                }
                next.push(u);
            }
        }
        *frontier = next;
    }
    None
}

/// Number of hops from `from` to `to`, or None (same limits as shortest_path).
pub fn distance(g: &Graph, from: u32, to: u32, max_visited: usize) -> Option<usize> {
    shortest_path(g, from, to, max_visited).map(|p| p.len() - 1)
}

/// (distance, nodes on any shortest path, edges on any shortest path)
pub type PathDag = (usize, Vec<u32>, Vec<(u32, u32)>);

/// The union of ALL shortest paths from `from` to `to`, as a DAG.
/// A node is on some shortest path iff df[v] + db[v] == D; an edge (u,v) iff
/// df[u] + 1 + db[v] == D. Returns None if unreachable.
pub fn shortest_path_dag(g: &Graph, from: u32, to: u32, max_visited: usize) -> Option<PathDag> {
    let d = distance(g, from, to, max_visited)?;
    let df = bfs_depths(g, from, d as u8, false);
    let db = bfs_depths(g, to, d as u8, true);
    let on_path = |v: u32| {
        let (a, b) = (df[v as usize], db[v as usize]);
        a != u8::MAX && b != u8::MAX && a as usize + b as usize == d
    };
    let nodes: Vec<u32> = (0..g.node_count() as u32).filter(|&v| on_path(v)).collect();
    let mut edges = Vec::new();
    for &u in &nodes {
        for &v in g.out(u) {
            if db[v as usize] != u8::MAX
                && df[u as usize] as usize + 1 + db[v as usize] as usize == d
            {
                edges.push((u, v));
            }
        }
    }
    Some((d, nodes, edges))
}

/// Depth-limited BFS depth map; `reverse` walks incoming edges.
fn bfs_depths(g: &Graph, from: u32, max_depth: u8, reverse: bool) -> Vec<u8> {
    let mut depth = vec![u8::MAX; g.node_count()];
    depth[from as usize] = 0;
    let mut frontier = vec![from];
    let mut d = 0u8;
    while !frontier.is_empty() && d < max_depth {
        d += 1;
        let mut next = Vec::new();
        for &v in &frontier {
            let neighbors = if reverse { g.incoming(v) } else { g.out(v) };
            for &u in neighbors {
                if depth[u as usize] == u8::MAX {
                    depth[u as usize] = d;
                    next.push(u);
                }
            }
        }
        frontier = next;
    }
    depth
}

fn stitch(
    fwd: &HashMap<u32, u32>,
    bwd: &HashMap<u32, u32>,
    meet: u32,
    from: u32,
    to: u32,
) -> Vec<u32> {
    let mut path = Vec::new();
    let mut v = meet;
    while v != from {
        path.push(v);
        v = fwd[&v];
    }
    path.push(from);
    path.reverse();
    let mut v = meet;
    while v != to {
        v = bwd[&v];
        path.push(v);
    }
    path
}
