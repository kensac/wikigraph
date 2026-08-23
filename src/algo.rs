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
