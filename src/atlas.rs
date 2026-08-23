//! Whole-graph atlas: community detection + a 2D layout of every article,
//! packed into a sidecar file whose bytes are also the wire format:
//!
//!   [magic u64][n u32][x: n u16][y: n u16][community: n u16][rank: n u8]
//!
//! Layout strategy: label propagation finds communities, the community
//! meta-graph gets a small force layout (the "continents"), and members are
//! placed in a rank-ordered sunflower spiral inside their community's disc.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;

use crate::graph::Graph;

pub const ATLAS_MAGIC: u64 = 0x31_53_4c_54_41_4b_57; // "WKATLS1"
/// Communities beyond u16 range collapse into this bucket.
const OTHER: u16 = u16::MAX;

pub fn build(g: &Graph, out_path: &Path, lp_iters: usize) -> Result<()> {
    let n = g.node_count();
    let labels = label_propagation(g, lp_iters);

    // Rank communities by member count; keep the largest 65535 as-is.
    let mut counts: Vec<(u32, u32)> = {
        let mut c = vec![0u32; n];
        for &l in &labels {
            c[l as usize] += 1;
        }
        c.into_iter()
            .enumerate()
            .filter(|&(_, cnt)| cnt > 0)
            .map(|(l, cnt)| (l as u32, cnt))
            .collect()
    };
    counts.sort_unstable_by_key(|&(_, cnt)| std::cmp::Reverse(cnt));
    eprintln!("communities: {}", counts.len());

    let mut compact = vec![OTHER; n]; // original label -> compact community id
    for (i, &(l, _)) in counts.iter().enumerate().take(OTHER as usize) {
        compact[l as usize] = i as u16;
    }
    let comm: Vec<u16> = labels.par_iter().map(|&l| compact[l as usize]).collect();
    let n_comm = counts.len().min(OTHER as usize) + 1;

    // Meta-graph: inter-community edge weights, sampled from the real edges.
    let mut weights = std::collections::HashMap::<(u16, u16), u64>::new();
    for v in 0..n as u32 {
        let a = comm[v as usize];
        for &u in g.out(v).iter().step_by(7) {
            let b = comm[u as usize];
            if a != b && a != OTHER && b != OTHER {
                *weights.entry((a.min(b), a.max(b)), ).or_insert(0) += 1;
            }
        }
    }
    let sizes: Vec<u32> = {
        let mut s = vec![0u32; n_comm];
        for &c in &comm {
            let idx = if c == OTHER { n_comm - 1 } else { c as usize };
            s[idx] += 1;
        }
        s
    };
    let centers = meta_layout(&sizes, &weights, 300);

    // Members in a golden-angle spiral inside their community disc, ordered by
    // rank so hubs sit at the center.
    let mut order: Vec<u32> = (0..n as u32).collect();
    let ranks = g.ranks();
    order.par_sort_unstable_by(|&a, &b| ranks[b as usize].total_cmp(&ranks[a as usize]));
    let mut slot = vec![0u32; n_comm]; // next spiral index per community
    let mut xy = vec![(0.0f32, 0.0f32); n];
    const GOLDEN: f32 = 2.399_963; // radians
    for &v in &order {
        let c = comm[v as usize];
        let ci = if c == OTHER { n_comm - 1 } else { c as usize };
        let k = slot[ci] as f32;
        slot[ci] += 1;
        let disc_r = (sizes[ci] as f32).sqrt();
        let r = disc_r * (k / sizes[ci].max(1) as f32).sqrt();
        let theta = k * GOLDEN;
        let (cx, cy) = centers[ci];
        xy[v as usize] = (cx + r * theta.cos(), cy + r * theta.sin());
    }

    // Quantize to u16 grid.
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &(x, y) in &xy {
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let sx = 65535.0 / (max_x - min_x).max(1e-6);
    let sy = 65535.0 / (max_y - min_y).max(1e-6);
    let quant = |v: f32, min: f32, s: f32| -> u16 { ((v - min) * s).round().clamp(0.0, 65535.0) as u16 };

    // Rank as a percentile byte (255 = top).
    let mut rank_order: Vec<u32> = (0..n as u32).collect();
    rank_order.par_sort_unstable_by(|&a, &b| ranks[a as usize].total_cmp(&ranks[b as usize]));
    let mut rank_pct = vec![0u8; n];
    for (i, &v) in rank_order.iter().enumerate() {
        rank_pct[v as usize] = ((i * 255) / n.max(1)) as u8;
    }

    let file = std::fs::File::create(out_path)
        .with_context(|| format!("create {}", out_path.display()))?;
    let mut w = std::io::BufWriter::with_capacity(1 << 22, file);
    w.write_all(&ATLAS_MAGIC.to_le_bytes())?;
    w.write_all(&(n as u32).to_le_bytes())?;
    for &(x, _) in &xy {
        w.write_all(&quant(x, min_x, sx).to_le_bytes())?;
    }
    for &(_, y) in &xy {
        w.write_all(&quant(y, min_y, sy).to_le_bytes())?;
    }
    for &c in &comm {
        w.write_all(&c.to_le_bytes())?;
    }
    w.write_all(&rank_pct)?;
    w.flush()?;
    eprintln!("wrote {} ({} nodes, {} communities)", out_path.display(), n, n_comm);
    Ok(())
}

/// Synchronous parallel label propagation over both edge directions.
fn label_propagation(g: &Graph, iters: usize) -> Vec<u32> {
    let n = g.node_count();
    let mut labels: Vec<u32> = (0..n as u32).collect();
    for it in 0..iters {
        let next: Vec<u32> = (0..n as u32)
            .into_par_iter()
            .map(|v| {
                let mut counts = std::collections::HashMap::<u32, u32>::with_capacity(16);
                for &u in g.out(v) {
                    *counts.entry(labels[u as usize]).or_insert(0) += 1;
                }
                for &u in g.incoming(v) {
                    *counts.entry(labels[u as usize]).or_insert(0) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|&(l, c)| (c, std::cmp::Reverse(l)))
                    .map(|(l, _)| l)
                    .unwrap_or(labels[v as usize])
            })
            .collect();
        let changed = labels
            .par_iter()
            .zip(&next)
            .filter(|(a, b)| a != b)
            .count();
        labels = next;
        eprintln!("label propagation iter {}: {} changed", it + 1, changed);
        if changed < n / 1000 {
            break;
        }
    }
    labels
}

/// Small force layout of the community meta-graph. Sizes set disc radii; the
/// result is centers spaced so discs rarely overlap.
fn meta_layout(
    sizes: &[u32],
    weights: &std::collections::HashMap<(u16, u16), u64>,
    iters: usize,
) -> Vec<(f32, f32)> {
    let k = sizes.len();
    let radii: Vec<f32> = sizes.iter().map(|&s| (s as f32).sqrt()).collect();
    // seed on an area-packed spiral, biggest at the center: position radius
    // grows with the square root of cumulative disc area, so discs start out
    // nearly disjoint instead of piled up
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_unstable_by(|&a, &b| sizes[b].cmp(&sizes[a]));
    let mut pos = vec![(0.0f32, 0.0f32); k];
    const GOLDEN: f32 = 2.399_963;
    let mut cum_area = 0.0f32;
    for (i, &c) in order.iter().enumerate() {
        cum_area += radii[c] * radii[c];
        if i == 0 {
            continue; // biggest community anchors the center
        }
        let r = (cum_area / std::f32::consts::PI).sqrt() * 1.9 + radii[c];
        let t = i as f32 * GOLDEN;
        pos[c] = (r * t.cos(), r * t.sin());
    }
    let edges: Vec<(u16, u16, f32)> = weights
        .iter()
        .map(|(&(a, b), &w)| (a, b, (w as f32).ln_1p()))
        .collect();
    let max_w = edges.iter().map(|e| e.2).fold(1.0f32, f32::max);
    for _ in 0..iters {
        let mut force = vec![(0.0f32, 0.0f32); k];
        // attraction along weighted meta-edges
        for &(a, b, w) in &edges {
            let (ax, ay) = pos[a as usize];
            let (bx, by) = pos[b as usize];
            let (dx, dy) = (bx - ax, by - ay);
            let dist = (dx * dx + dy * dy).sqrt().max(1e-3);
            let target = radii[a as usize] + radii[b as usize];
            let f = 0.02 * (w / max_w) * (dist - target) / dist;
            force[a as usize].0 += f * dx;
            force[a as usize].1 += f * dy;
            force[b as usize].0 -= f * dx;
            force[b as usize].1 -= f * dy;
        }
        // pairwise disc repulsion (k is at most ~65k but usually far smaller;
        // only push when overlapping, sweep over a sorted axis to stay cheap)
        let mut by_x: Vec<usize> = (0..k).collect();
        by_x.sort_unstable_by(|&a, &b| pos[a].0.total_cmp(&pos[b].0));
        for i in 0..k {
            let a = by_x[i];
            for &b in by_x[i + 1..].iter() {
                let sep = radii[a] + radii[b];
                if pos[b].0 - pos[a].0 > sep {
                    break;
                }
                let (dx, dy) = (pos[b].0 - pos[a].0, pos[b].1 - pos[a].1);
                let dist = (dx * dx + dy * dy).sqrt().max(1e-3);
                if dist < sep {
                    let f = 0.5 * (sep - dist) / dist;
                    force[a].0 -= f * dx;
                    force[a].1 -= f * dy;
                    force[b].0 += f * dx;
                    force[b].1 += f * dy;
                }
            }
        }
        for c in 0..k {
            pos[c].0 += force[c].0.clamp(-radii[c], radii[c]);
            pos[c].1 += force[c].1.clamp(-radii[c], radii[c]);
        }
    }
    pos
}
