//! Serves the graph API.
//!
//! Usage: serve [--graph data/graph.bin] [--addr 0.0.0.0:8080]
//!        serve --check   (healthcheck probe: GET /healthz on localhost)

use std::path::PathBuf;

use anyhow::{bail, Result};
use wikigraph::{api, graph::Graph};

/// Shell-free healthcheck for distroless images.
fn check(addr: &str) -> Result<()> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect(addr)?;
    stream.write_all(b"GET /healthz HTTP/1.0\r\n\r\n")?;
    let mut buf = String::new();
    stream.read_to_string(&mut buf)?;
    if buf.starts_with("HTTP/1.0 200") || buf.starts_with("HTTP/1.1 200") {
        Ok(())
    } else {
        bail!("unhealthy: {}", buf.lines().next().unwrap_or(""))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut graph_path = PathBuf::from("data/graph.bin");
    let mut addr = "0.0.0.0:8080".to_string();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut val = || {
            args.next()
                .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))
        };
        match arg.as_str() {
            "--check" => return check("127.0.0.1:8080"),
            "--graph" => graph_path = PathBuf::from(val()?),
            "--addr" => addr = val()?,
            other => bail!("unknown flag {other}"),
        }
    }

    let g = Graph::open(&graph_path)?;
    eprintln!(
        "loaded {}: {} nodes, {} edges",
        graph_path.display(),
        g.node_count(),
        g.edge_count()
    );
    // atlas.bin is optional; without it /atlas.bin 404s and everything else works
    let atlas_path = graph_path.with_file_name("atlas.bin");
    let atlas = std::fs::File::open(&atlas_path)
        .ok()
        .and_then(|f| unsafe { memmap2::Mmap::map(&f).ok() });
    eprintln!(
        "atlas: {}",
        if atlas.is_some() { "loaded" } else { "absent" }
    );
    let app = api::App::new(g, atlas);
    let tiles_dir = graph_path.with_file_name("tiles");
    let tiles = tiles_dir.is_dir().then_some(tiles_dir);
    eprintln!("tiles: {}", if tiles.is_some() { "serving" } else { "absent" });
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    eprintln!("listening on {addr}");
    axum::serve(listener, api::router(app, tiles))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
