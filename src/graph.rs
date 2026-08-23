//! On-disk graph format and zero-copy mmap reader.
//!
//! Little-endian, sections 8-byte aligned, in fixed order after a 64-byte
//! header so offsets derive from (n, m) alone:
//!
//!   header | out_offsets: (n+1) u64 | in_offsets: (n+1) u64
//!   | title_offsets: (n+1) u32 | pagerank: n f32
//!   | out_edges: m u32 | in_edges: m u32 | titles blob
//!
//! Node ids are dense u32 assigned in sorted-title order, so title lookup
//! and prefix search are binary searches over the id space itself.

use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::ops::Range;
use std::path::Path;

use anyhow::{ensure, Context, Result};
use memmap2::Mmap;

pub const MAGIC: u64 = 0x31_46_52_47_49_4b_57; // "WKIGRF1"
const HEADER_LEN: usize = 64;

pub struct Graph {
    mmap: Mmap,
    n: usize,
    m: usize,
}

struct Layout {
    out_offsets: Range<usize>,
    in_offsets: Range<usize>,
    title_offsets: Range<usize>,
    pagerank: Range<usize>,
    out_edges: Range<usize>,
    in_edges: Range<usize>,
    titles: usize, // start; runs to EOF
}

fn align8(v: usize) -> usize {
    (v + 7) & !7
}

fn layout(n: usize, m: usize) -> Layout {
    let mut at = HEADER_LEN;
    let mut take = |len: usize| {
        let r = at..at + len;
        at = align8(at + len);
        r
    };
    Layout {
        out_offsets: take((n + 1) * 8),
        in_offsets: take((n + 1) * 8),
        title_offsets: take((n + 1) * 4),
        pagerank: take(n * 4),
        out_edges: take(m * 4),
        in_edges: take(m * 4),
        titles: at,
    }
}

/// Casts an 8-aligned byte range of the mmap to a typed slice.
fn cast<T: Copy>(bytes: &[u8], range: Range<usize>) -> &[T] {
    let src = &bytes[range];
    debug_assert_eq!(src.as_ptr() as usize % std::mem::align_of::<T>(), 0);
    debug_assert_eq!(src.len() % std::mem::size_of::<T>(), 0);
    unsafe { std::slice::from_raw_parts(src.as_ptr().cast(), src.len() / std::mem::size_of::<T>()) }
}

impl Graph {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        ensure!(mmap.len() >= HEADER_LEN, "file too short");
        let word = |i: usize| u64::from_le_bytes(mmap[i * 8..i * 8 + 8].try_into().unwrap());
        ensure!(word(0) == MAGIC, "bad magic (not a wikigraph file)");
        let n = word(1) as usize;
        let m = word(2) as usize;
        let l = layout(n, m);
        ensure!(mmap.len() >= l.titles, "file truncated");
        Ok(Graph { mmap, n, m })
    }

    pub fn node_count(&self) -> usize {
        self.n
    }

    pub fn edge_count(&self) -> usize {
        self.m
    }

    fn l(&self) -> Layout {
        layout(self.n, self.m)
    }

    fn out_offsets(&self) -> &[u64] {
        cast(&self.mmap, self.l().out_offsets)
    }

    fn in_offsets(&self) -> &[u64] {
        cast(&self.mmap, self.l().in_offsets)
    }

    fn title_offsets(&self) -> &[u32] {
        cast(&self.mmap, self.l().title_offsets)
    }

    pub fn ranks(&self) -> &[f32] {
        cast(&self.mmap, self.l().pagerank)
    }

    pub fn out(&self, v: u32) -> &[u32] {
        let o = self.out_offsets();
        let edges: &[u32] = cast(&self.mmap, self.l().out_edges);
        &edges[o[v as usize] as usize..o[v as usize + 1] as usize]
    }

    pub fn incoming(&self, v: u32) -> &[u32] {
        let o = self.in_offsets();
        let edges: &[u32] = cast(&self.mmap, self.l().in_edges);
        &edges[o[v as usize] as usize..o[v as usize + 1] as usize]
    }

    pub fn out_degree(&self, v: u32) -> usize {
        let o = self.out_offsets();
        (o[v as usize + 1] - o[v as usize]) as usize
    }

    pub fn in_degree(&self, v: u32) -> usize {
        let o = self.in_offsets();
        (o[v as usize + 1] - o[v as usize]) as usize
    }

    pub fn rank(&self, v: u32) -> f32 {
        self.ranks()[v as usize]
    }

    pub fn title_bytes(&self, v: u32) -> &[u8] {
        let o = self.title_offsets();
        let base = self.l().titles;
        &self.mmap[base + o[v as usize] as usize..base + o[v as usize + 1] as usize]
    }

    pub fn title(&self, v: u32) -> &str {
        std::str::from_utf8(self.title_bytes(v)).unwrap_or("\u{fffd}")
    }

    /// Exact title match (underscore form).
    pub fn find(&self, title: &[u8]) -> Option<u32> {
        let mut lo = 0u32;
        let mut hi = self.n as u32;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.title_bytes(mid) < title {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo < self.n as u32 && self.title_bytes(lo) == title).then_some(lo)
    }

    /// Id range of titles starting with `prefix`.
    pub fn prefix_range(&self, prefix: &[u8]) -> Range<u32> {
        let lower = self.partition(|t| t < prefix);
        let upper = self.partition(|t| t < prefix || t.starts_with(prefix));
        lower..upper
    }

    fn partition(&self, pred: impl Fn(&[u8]) -> bool) -> u32 {
        let mut lo = 0u32;
        let mut hi = self.n as u32;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if pred(self.title_bytes(mid)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }
}

/// Section slices of a built graph, ready to stream to disk.
pub struct Sections<'a> {
    pub out_offsets: &'a [u64],
    pub in_offsets: &'a [u64],
    pub title_offsets: &'a [u32],
    pub pagerank: &'a [f32],
    pub out_edges: &'a [u32],
    pub in_edges: &'a [u32],
    pub titles: &'a [u8],
}

pub struct Writer;

impl Writer {
    pub fn write(path: &Path, s: &Sections) -> Result<()> {
        let Sections {
            out_offsets,
            in_offsets,
            title_offsets,
            pagerank,
            out_edges,
            in_edges,
            titles,
        } = *s;
        let n = pagerank.len();
        let m = out_edges.len();
        assert_eq!(out_offsets.len(), n + 1);
        assert_eq!(in_offsets.len(), n + 1);
        assert_eq!(title_offsets.len(), n + 1);
        assert_eq!(in_edges.len(), m);

        let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
        let mut w = BufWriter::with_capacity(1 << 22, file);
        let mut header = [0u8; HEADER_LEN];
        header[0..8].copy_from_slice(&MAGIC.to_le_bytes());
        header[8..16].copy_from_slice(&(n as u64).to_le_bytes());
        header[16..24].copy_from_slice(&(m as u64).to_le_bytes());
        w.write_all(&header)?;

        let mut written = HEADER_LEN;
        let mut put = |w: &mut BufWriter<File>, bytes: &[u8]| -> Result<()> {
            w.write_all(bytes)?;
            written += bytes.len();
            let pad = align8(written) - written;
            w.write_all(&[0u8; 8][..pad])?;
            written += pad;
            Ok(())
        };
        put(&mut w, as_bytes(out_offsets))?;
        put(&mut w, as_bytes(in_offsets))?;
        put(&mut w, as_bytes(title_offsets))?;
        put(&mut w, as_bytes(pagerank))?;
        put(&mut w, as_bytes(out_edges))?;
        put(&mut w, as_bytes(in_edges))?;
        w.write_all(titles)?;
        w.flush()?;
        let size = w.into_inner()?.stream_position()?;
        eprintln!("wrote {} ({:.2} GB)", path.display(), size as f64 / 1e9);
        Ok(())
    }
}

fn as_bytes<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr().cast(), std::mem::size_of_val(slice)) }
}
