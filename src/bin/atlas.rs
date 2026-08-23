//! Computes communities + whole-graph layout from an existing graph.bin.
//!
//! Usage: atlas [--graph data/graph.bin] [--out data/atlas.bin] [--iters 8]

use std::path::PathBuf;

use anyhow::{bail, Result};
use wikigraph::{atlas, graph::Graph};

fn main() -> Result<()> {
    let mut graph_path = PathBuf::from("data/graph.bin");
    let mut out: Option<PathBuf> = None;
    let mut iters = 8usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut val = || {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))
        };
        match arg.as_str() {
            "--graph" => graph_path = PathBuf::from(val()?),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--iters" => iters = val()?.parse()?,
            other => bail!("unknown flag {other}"),
        }
    }
    let out = out.unwrap_or_else(|| graph_path.with_file_name("atlas.bin"));
    let g = Graph::open(&graph_path)?;
    eprintln!("loaded {} nodes, {} edges", g.node_count(), g.edge_count());
    atlas::build(&g, &out, iters)
}
