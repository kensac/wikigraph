# wikigraph

The English Wikipedia link graph as a queryable service: every article a node,
every wikilink a directed edge. Built from the enwiki SQL dumps into a single
memory-mapped binary, served over a small HTTP API, explored interactively at
[kanishksachdev.com/experiments](https://www.kanishksachdev.com/experiments).

## How it works

- `ingest` streams `enwiki-latest-{page,redirect,linktarget,pagelinks}.sql.gz`
  (no full decompression to disk), collapses redirects, and writes `graph.bin`:
  CSR adjacency in both directions, titles sorted so node id order *is* title
  order (lookup and prefix search are binary searches over the id space), and
  precomputed PageRank.
- `atlas` computes a whole-graph map from graph.bin: label-propagation
  communities, a force layout of the community meta-graph, rank-ordered
  sunflower placement within each community. Output `atlas.bin` (~50MB,
  7 bytes/node) is both the file format and the wire format.
- `serve` mmaps both files (near-zero startup, OS page cache does the caching)
  and exposes:
  - `/search`, `/node/{id}`, `/node/{id}/links`, `/path`, `/distance`, `/edge`,
    `/stats`, `/top`, `/random`, `/healthz` — JSON
  - `/node/{id}/neighbors.bin` — the complete, uncapped neighbor set as packed
    binary (ISBN's 1.66M in-links: one 13MB response, ~27ms)
  - `/atlas.bin` — the whole-graph layout, `/titles?ids=` — batch labels
  - `/game/new` — link-game puzzles (two articles + par = true shortest path)

## Usage

```sh
# fetch dumps (~11GB)
cd data && curl -O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-{page,redirect,linktarget,pagelinks}.sql.gz

cargo run --release --bin ingest            # data/*.sql.gz -> data/graph.bin (~9 min on an M1 Pro)
cargo run --release --bin atlas             # data/graph.bin -> data/atlas.bin (~4 min)
cargo run --release --bin serve             # api on :8080
```

`ingest --limit-pages N --limit-links N` builds a small sample for development.

## Deploy

Image `ghcr.io/kensac/wikigraph` (built by CI) runs on the homelab behind
Traefik as `wiki-api.kanishksachdev.com`; `graph.bin` is shipped to the host's
data volume out of band. Re-ingest and re-ship to update the snapshot.
