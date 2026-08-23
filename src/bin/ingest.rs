//! Builds data/graph.bin from enwiki SQL dumps.
//!
//! Usage: ingest [--data-dir data] [--out data/graph.bin]
//!               [--limit-pages N] [--limit-links N]

use std::path::PathBuf;

use anyhow::{bail, Result};
use wikigraph::builder::{build, Limits};

fn main() -> Result<()> {
    let mut data_dir = PathBuf::from("data");
    let mut out: Option<PathBuf> = None;
    let mut limits = Limits::default();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut val = || {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))
        };
        match arg.as_str() {
            "--data-dir" => data_dir = PathBuf::from(val()?),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--limit-pages" => limits.pages = Some(val()?.parse()?),
            "--limit-links" => limits.links = Some(val()?.parse()?),
            other => bail!("unknown flag {other}"),
        }
    }
    let out = out.unwrap_or_else(|| data_dir.join("graph.bin"));
    let stats = build(&data_dir, &out, limits)?;
    println!("{} nodes, {} edges", stats.nodes, stats.edges);
    Ok(())
}
