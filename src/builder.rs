//! Builds graph.bin from enwiki SQL dumps: page, redirect, linktarget, pagelinks.
//!
//! Pipeline: collect ns0 pages -> resolve redirects -> map link targets ->
//! stream pagelinks into an edge list -> CSR both directions -> pagerank -> write.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Result};
use flate2::bufread::MultiGzDecoder;

use crate::graph::{Sections, Writer};
use crate::sql;

#[derive(Default, Clone, Copy)]
pub struct Limits {
    pub pages: Option<usize>,
    pub links: Option<u64>,
}

pub struct Stats {
    pub nodes: usize,
    pub edges: usize,
}

/// A dump table file: `<dir>/enwiki-latest-<table>.sql(.gz)`.
fn open_dump(dir: &Path, table: &str) -> Result<Box<dyn BufRead>> {
    for ext in [".sql.gz", ".sql"] {
        let path = dir.join(format!("enwiki-latest-{table}{ext}"));
        if path.exists() {
            let file = BufReader::with_capacity(1 << 20, File::open(&path)?);
            let reader: Box<dyn Read> = if ext.ends_with(".gz") {
                Box::new(MultiGzDecoder::new(file))
            } else {
                Box::new(file)
            };
            return Ok(Box::new(BufReader::with_capacity(1 << 20, reader)));
        }
    }
    bail!("missing dump {table} in {}", dir.display())
}

/// All ns0 pages: titles in one arena, records referencing it.
struct Pages {
    blob: Vec<u8>,
    recs: Vec<PageRec>,     // sorted by title after load
    dense_of_rec: Vec<u32>, // rec index -> dense article id, or REDIRECT/UNRESOLVED
}

#[derive(Clone, Copy)]
struct PageRec {
    page_id: u32,
    off: u32,
    len: u32,
    redirect: bool,
}

const UNRESOLVED: u32 = u32::MAX;

impl Pages {
    fn title(&self, r: &PageRec) -> &[u8] {
        &self.blob[r.off as usize..(r.off + r.len) as usize]
    }

    /// Binary search by title over the sorted recs.
    fn rec_by_title(&self, title: &[u8]) -> Option<usize> {
        self.recs
            .binary_search_by(|r| self.title(r).cmp(title))
            .ok()
    }
}

pub fn build(data_dir: &Path, out_path: &Path, limits: Limits) -> Result<Stats> {
    let t0 = Instant::now();

    // page.sql: (page_id, ns, title, is_redirect, ...)
    let mut blob = Vec::with_capacity(1 << 28);
    let mut recs = Vec::with_capacity(20_000_000);
    sql::for_each_row(open_dump(data_dir, "page")?, |row| {
        if row.len() < 4 || row[1].as_int() != Some(0) {
            return true;
        }
        let (Some(id), Some(title), Some(redir)) =
            (row[0].as_int(), row[2].as_bytes(), row[3].as_int())
        else {
            return true;
        };
        recs.push(PageRec {
            page_id: id as u32,
            off: blob.len() as u32,
            len: title.len() as u32,
            redirect: redir != 0,
        });
        blob.extend_from_slice(title);
        limits.pages.is_none_or(|max| recs.len() < max)
    })?;
    eprintln!("[{:.0?}] pages: {} ns0 rows", t0.elapsed(), recs.len());

    let mut pages = Pages {
        blob,
        recs,
        dense_of_rec: Vec::new(),
    };
    // sort by title; article order defines dense ids
    {
        let blob = &pages.blob;
        pages.recs.sort_unstable_by(|a, b| {
            blob[a.off as usize..(a.off + a.len) as usize]
                .cmp(&blob[b.off as usize..(b.off + b.len) as usize])
        });
        pages.recs.dedup_by(|a, b| {
            blob[a.off as usize..(a.off + a.len) as usize]
                == blob[b.off as usize..(b.off + b.len) as usize]
        });
    }
    let mut dense = 0u32;
    pages.dense_of_rec = pages
        .recs
        .iter()
        .map(|r| {
            if r.redirect {
                UNRESOLVED
            } else {
                dense += 1;
                dense - 1
            }
        })
        .collect();
    let n = dense as usize;
    eprintln!(
        "[{:.0?}] articles: {n}, redirects: {}",
        t0.elapsed(),
        pages.recs.len() - n
    );

    // redirect.sql: (rd_from, rd_namespace, rd_title, ...) -> target title per redirect
    let mut rd: Vec<(u32, u32, u32)> = Vec::new(); // (page_id, off, len) into rd_blob
    let mut rd_blob = Vec::with_capacity(1 << 26);
    sql::for_each_row(open_dump(data_dir, "redirect")?, |row| {
        if row.len() >= 3 && row[1].as_int() == Some(0) {
            if let (Some(from), Some(title)) = (row[0].as_int(), row[2].as_bytes()) {
                rd.push((from as u32, rd_blob.len() as u32, title.len() as u32));
                rd_blob.extend_from_slice(title);
            }
        }
        true
    })?;
    rd.sort_unstable_by_key(|&(id, ..)| id);
    eprintln!("[{:.0?}] redirect targets: {}", t0.elapsed(), rd.len());

    // resolve every redirect rec to a dense id, chasing up to 3 hops
    let rd_target = |page_id: u32| -> Option<&[u8]> {
        rd.binary_search_by_key(&page_id, |&(id, ..)| id)
            .ok()
            .map(|i| &rd_blob[rd[i].1 as usize..(rd[i].1 + rd[i].2) as usize])
    };
    for i in 0..pages.recs.len() {
        if !pages.recs[i].redirect {
            continue;
        }
        let mut cur = i;
        for _ in 0..3 {
            let Some(target) = rd_target(pages.recs[cur].page_id) else {
                break;
            };
            let Some(j) = pages.rec_by_title(target) else {
                break;
            };
            if !pages.recs[j].redirect {
                pages.dense_of_rec[i] = pages.dense_of_rec[j];
                break;
            }
            cur = j;
        }
    }

    // page_id -> dense id (redirects resolved), sorted for binary search
    let mut by_page_id: Vec<(u32, u32, bool)> = pages
        .recs
        .iter()
        .zip(&pages.dense_of_rec)
        .filter(|&(_, &d)| d != UNRESOLVED)
        .map(|(r, &d)| (r.page_id, d, r.redirect))
        .collect();
    by_page_id.sort_unstable_by_key(|&(id, ..)| id);
    let dense_of_page_id = |page_id: u32| -> Option<(u32, bool)> {
        by_page_id
            .binary_search_by_key(&page_id, |&(id, ..)| id)
            .ok()
            .map(|i| (by_page_id[i].1, by_page_id[i].2))
    };

    // linktarget.sql: (lt_id, lt_namespace, lt_title) -> dense id, redirects resolved
    let mut lt: Vec<(u64, u32)> = Vec::with_capacity(40_000_000);
    sql::for_each_row(open_dump(data_dir, "linktarget")?, |row| {
        if row.len() >= 3 && row[1].as_int() == Some(0) {
            if let (Some(id), Some(title)) = (row[0].as_int(), row[2].as_bytes()) {
                if let Some(i) = pages.rec_by_title(title) {
                    if pages.dense_of_rec[i] != UNRESOLVED {
                        lt.push((id as u64, pages.dense_of_rec[i]));
                    }
                }
            }
        }
        true
    })?;
    lt.sort_unstable_by_key(|&(id, _)| id);
    eprintln!("[{:.0?}] link targets resolved: {}", t0.elapsed(), lt.len());

    // pagelinks.sql: (pl_from, pl_from_namespace, pl_target_id) -> edges.
    // Capacity sized above the observed ~90% keep rate on ~1.6B rows so the
    // vec never doubles mid-stream (a realloc at this size would spike peak
    // memory past what the machine has).
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(1_700_000_000);
    let mut scanned = 0u64;
    sql::for_each_row(open_dump(data_dir, "pagelinks")?, |row| {
        scanned += 1;
        if scanned.is_multiple_of(100_000_000) {
            eprintln!(
                "[{:.0?}] pagelinks: {scanned} rows, {} edges",
                t0.elapsed(),
                edges.len()
            );
        }
        if row.len() < 3 || row[1].as_int() != Some(0) {
            return true;
        }
        let (Some(from_id), Some(target)) = (row[0].as_int(), row[2].as_int()) else {
            return true;
        };
        // links *from* redirect pages are collapsed away
        let Some((from, false)) = dense_of_page_id(from_id as u32) else {
            return true;
        };
        let Ok(i) = lt.binary_search_by_key(&(target as u64), |&(id, _)| id) else {
            return true;
        };
        let to = lt[i].1;
        if from != to {
            edges.push((from, to));
        }
        limits.links.is_none_or(|max| (edges.len() as u64) < max)
    })?;
    drop(lt);
    drop(rd);
    drop(rd_blob);
    eprintln!("[{:.0?}] raw edges: {}", t0.elapsed(), edges.len());

    rayon::slice::ParallelSliceMut::par_sort_unstable(edges.as_mut_slice());
    edges.dedup();
    edges.shrink_to_fit();
    let m = edges.len();
    eprintln!("[{:.0?}] deduped edges: {m}", t0.elapsed());

    // out CSR
    let mut out_offsets = vec![0u64; n + 1];
    for &(from, _) in &edges {
        out_offsets[from as usize + 1] += 1;
    }
    for i in 0..n {
        out_offsets[i + 1] += out_offsets[i];
    }
    let out_edges: Vec<u32> = edges.iter().map(|&(_, to)| to).collect();

    // in CSR by counting sort
    let mut in_offsets = vec![0u64; n + 1];
    for &(_, to) in &edges {
        in_offsets[to as usize + 1] += 1;
    }
    for i in 0..n {
        in_offsets[i + 1] += in_offsets[i];
    }
    let mut cursor = in_offsets.clone();
    let mut in_edges = vec![0u32; m];
    for &(from, to) in &edges {
        in_edges[cursor[to as usize] as usize] = from;
        cursor[to as usize] += 1;
    }
    drop(cursor);
    drop(edges);
    eprintln!("[{:.0?}] CSR built", t0.elapsed());

    let pagerank = pagerank(n, &out_offsets, &in_offsets, &in_edges, 20, 0.85);
    eprintln!("[{:.0?}] pagerank done", t0.elapsed());

    // titles in dense (sorted) order
    let mut title_offsets = Vec::with_capacity(n + 1);
    let mut titles = Vec::with_capacity(pages.blob.len());
    title_offsets.push(0u32);
    for (r, &d) in pages.recs.iter().zip(&pages.dense_of_rec) {
        if r.redirect {
            continue;
        }
        debug_assert_eq!(d as usize + 1, title_offsets.len());
        titles.extend_from_slice(pages.title(r));
        title_offsets.push(titles.len() as u32);
    }

    Writer::write(
        out_path,
        &Sections {
            out_offsets: &out_offsets,
            in_offsets: &in_offsets,
            title_offsets: &title_offsets,
            pagerank: &pagerank,
            out_edges: &out_edges,
            in_edges: &in_edges,
            titles: &titles,
        },
    )?;
    eprintln!("[{:.0?}] done: {n} nodes, {m} edges", t0.elapsed());
    Ok(Stats { nodes: n, edges: m })
}

/// Power iteration with uniform teleport and dangling-mass redistribution.
/// Gather form over the in-CSR: each node sums its in-neighbors' contributions,
/// so iterations parallelize with no write contention.
fn pagerank(
    n: usize,
    out_offsets: &[u64],
    in_offsets: &[u64],
    in_edges: &[u32],
    iters: usize,
    d: f32,
) -> Vec<f32> {
    use rayon::prelude::*;
    if n == 0 {
        return Vec::new();
    }
    let inv_out: Vec<f32> = (0..n)
        .map(|v| {
            let deg = out_offsets[v + 1] - out_offsets[v];
            if deg == 0 {
                0.0
            } else {
                1.0 / deg as f32
            }
        })
        .collect();
    let mut rank = vec![1.0f32 / n as f32; n];
    let mut next = vec![0.0f32; n];
    let mut contrib = vec![0.0f32; n];
    for _ in 0..iters {
        let dangling: f64 = rank
            .par_iter()
            .zip(&inv_out)
            .filter(|&(_, &inv)| inv == 0.0)
            .map(|(&r, _)| r as f64)
            .sum();
        contrib
            .par_iter_mut()
            .zip(&rank)
            .zip(&inv_out)
            .for_each(|((c, &r), &inv)| *c = r * inv);
        let base = (1.0 - d) / n as f32 + d * (dangling as f32 / n as f32);
        next.par_iter_mut().enumerate().for_each(|(v, out)| {
            let (s, e) = (in_offsets[v] as usize, in_offsets[v + 1] as usize);
            let mut sum = 0.0f32;
            for &u in &in_edges[s..e] {
                sum += contrib[u as usize];
            }
            *out = base + d * sum;
        });
        std::mem::swap(&mut rank, &mut next);
    }
    rank
}
