//! Renders ALL edges into a map-style tile pyramid using the atlas layout.
//!
//! Every edge is drawn as a sampled additive line (constant energy per edge)
//! into a base raster, tone-mapped, then cut into 256px tiles for zooms
//! 0..=BASE_Z. Deep zoom is handled live by the frontend's point/edge layers.
//!
//! Usage: render [--graph data/graph.bin] [--atlas data/atlas.bin]
//!               [--out data/tiles] [--size 8192]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{bail, ensure, Context, Result};
use rayon::prelude::*;
use wikigraph::graph::Graph;

const TILE: usize = 256;
/// Fixed-point scale for atomic accumulation.
const FP: f32 = 4096.0;
/// Max additive samples along one line; keeps total work bounded.
const MAX_SAMPLES: usize = 48;

fn main() -> Result<()> {
    let mut graph_path = PathBuf::from("data/graph.bin");
    let mut atlas_path = PathBuf::from("data/atlas.bin");
    let mut out_dir = PathBuf::from("data/tiles");
    let mut size = 8192usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut val = || {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))
        };
        match arg.as_str() {
            "--graph" => graph_path = PathBuf::from(val()?),
            "--atlas" => atlas_path = PathBuf::from(val()?),
            "--out" => out_dir = PathBuf::from(val()?),
            "--size" => size = val()?.parse()?,
            other => bail!("unknown flag {other}"),
        }
    }
    ensure!(size.is_power_of_two() && size >= TILE, "size must be a power of two >= 256");

    let g = Graph::open(&graph_path)?;
    let n = g.node_count();
    let atlas = std::fs::read(&atlas_path).with_context(|| format!("read {}", atlas_path.display()))?;
    ensure!(atlas.len() == 12 + n * 7, "atlas does not match graph");
    let u16_at = |off: usize, i: usize| -> u16 {
        u16::from_le_bytes([atlas[off + 2 * i], atlas[off + 2 * i + 1]])
    };
    let (x_off, y_off, c_off) = (12, 12 + 2 * n, 12 + 4 * n);

    // node -> raster position and community color
    let scale = (size - 1) as f32 / 65535.0;
    let px: Vec<f32> = (0..n).into_par_iter().map(|i| u16_at(x_off, i) as f32 * scale).collect();
    let py: Vec<f32> = (0..n).into_par_iter().map(|i| u16_at(y_off, i) as f32 * scale).collect();
    let colors: Vec<[f32; 3]> = (0..n)
        .into_par_iter()
        .map(|i| community_color(u16_at(c_off, i)))
        .collect();

    // additive accumulation in fixed point
    eprintln!("accumulating {} edges into {size}x{size}...", g.edge_count());
    let t0 = std::time::Instant::now();
    let acc: Vec<AtomicU32> = (0..size * size * 3).map(|_| AtomicU32::new(0)).collect();
    (0..n as u32).into_par_iter().for_each(|v| {
        let (x0, y0) = (px[v as usize], py[v as usize]);
        let col = colors[v as usize];
        for &to in g.out(v) {
            let (x1, y1) = (px[to as usize], py[to as usize]);
            let steps = ((x1 - x0).abs().max((y1 - y0).abs()) as usize).clamp(1, MAX_SAMPLES);
            let energy = 1.0 / steps as f32;
            for s in 0..steps {
                let t = (s as f32 + 0.5) / steps as f32;
                let x = (x0 + (x1 - x0) * t) as usize;
                let y = (y0 + (y1 - y0) * t) as usize;
                let base = (y * size + x) * 3;
                for c in 0..3 {
                    acc[base + c].fetch_add((col[c] * energy * FP) as u32, Ordering::Relaxed);
                }
            }
        }
    });
    eprintln!("accumulated in {:.0?}", t0.elapsed());

    // auto-expose: map the 99th percentile of nonzero energy near white,
    // instead of a fixed constant that saturates dense regions
    let mut sample: Vec<f32> = acc
        .par_iter()
        .step_by(997)
        .filter_map(|a| {
            let e = a.load(Ordering::Relaxed) as f32 / FP;
            (e > 0.0).then_some(e)
        })
        .collect();
    sample.par_sort_unstable_by(f32::total_cmp);
    let p99 = sample
        .get(sample.len().saturating_sub(sample.len() / 100 + 1))
        .copied()
        .unwrap_or(1.0);
    let denom = (1.0 + p99).ln();
    eprintln!("exposure: p99 energy {p99:.2}, log denom {denom:.3}");
    drop(sample);

    // log tone map to 8-bit RGBA: keeps faint filaments visible while the
    // densest hubs still saturate gracefully
    let img: Vec<u8> = (0..size * size)
        .into_par_iter()
        .flat_map_iter(|i| {
            let mut out = [0u8; 4];
            let mut lum = 0.0f32;
            for c in 0..3 {
                let e = acc[i * 3 + c].load(Ordering::Relaxed) as f32 / FP;
                let v = ((1.0 + e).ln() / denom).min(1.0);
                out[c] = (v.powf(1.0 / 2.2) * 255.0) as u8;
                lum += v;
            }
            out[3] = ((lum / 3.0).powf(0.4) * 255.0).min(255.0) as u8;
            out
        })
        .collect();
    drop(acc);

    // pyramid: base level down to a single tile
    let base_z = (size / TILE).ilog2() as usize;
    let mut level = img;
    let mut level_size = size;
    for z in (0..=base_z).rev() {
        write_tiles(&out_dir, z, level_size, &level)?;
        if z > 0 {
            level = downsample(&level, level_size);
            level_size /= 2;
        }
    }
    eprintln!("tiles written to {} (z 0..={base_z})", out_dir.display());
    Ok(())
}

/// Stable community hue: golden-angle spacing, "other" bucket gets grey.
fn community_color(comm: u16) -> [f32; 3] {
    if comm == u16::MAX {
        return [0.5, 0.5, 0.55];
    }
    let hue = (comm as f32 * 137.508) % 360.0;
    hsl_to_rgb(hue, 0.85, 0.55)
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = l - c / 2.0;
    [r + m, g + m, b + m]
}

fn downsample(img: &[u8], size: usize) -> Vec<u8> {
    let half = size / 2;
    (0..half * half)
        .into_par_iter()
        .flat_map_iter(|i| {
            let (x, y) = (i % half, i / half);
            let mut out = [0u8; 4];
            for c in 0..4 {
                let sum: u32 = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .iter()
                    .map(|&(dx, dy)| img[((2 * y + dy) * size + 2 * x + dx) * 4 + c] as u32)
                    .sum();
                out[c] = (sum / 4) as u8;
            }
            out
        })
        .collect()
}

fn write_tiles(out_dir: &std::path::Path, z: usize, level_size: usize, img: &[u8]) -> Result<()> {
    let tiles = level_size / TILE;
    (0..tiles * tiles).into_par_iter().try_for_each(|i| -> Result<()> {
        let (tx, ty) = (i % tiles, i / tiles);
        let dir = out_dir.join(z.to_string()).join(tx.to_string());
        std::fs::create_dir_all(&dir)?;
        let mut buf = Vec::with_capacity(TILE * TILE * 4);
        for row in 0..TILE {
            let start = ((ty * TILE + row) * level_size + tx * TILE) * 4;
            buf.extend_from_slice(&img[start..start + TILE * 4]);
        }
        let file = std::fs::File::create(dir.join(format!("{ty}.png")))?;
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), TILE as u32, TILE as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header()?;
        writer.write_image_data(&buf)?;
        Ok(())
    })
}
