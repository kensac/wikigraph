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
- `serve` mmaps the file (near-zero startup, OS page cache does the caching)
  and exposes: `/search`, `/node/{id}`, `/node/{id}/links`, `/path` (bidirectional
  BFS), `/stats`, `/top`, `/random`, `/healthz`.

## Usage

```sh
# fetch dumps (~11GB)
cd data && curl -O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-{page,redirect,linktarget,pagelinks}.sql.gz

cargo run --release --bin ingest            # data/*.sql.gz -> data/graph.bin
cargo run --release --bin serve             # api on :8080
```

`ingest --limit-pages N --limit-links N` builds a small sample for development.

## Deploy

Image `ghcr.io/kensac/wikigraph` (built by CI) runs on the homelab behind
Traefik as `wiki-api.kanishksachdev.com`; `graph.bin` is shipped to the host's
data volume out of band. Re-ingest and re-ship to update the snapshot.
