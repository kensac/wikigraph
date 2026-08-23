//! End-to-end: fixture dumps -> graph.bin -> mmap reader -> queries.

use std::fs;
use std::path::PathBuf;

use wikigraph::algo::shortest_path;
use wikigraph::builder::{build, Limits};
use wikigraph::graph::Graph;

/// Articles: Apple(1), Banana(2), Cherry(3), Durian(4, dangling). Fruit(5) is a
/// redirect to Apple. Elderberry(6) links to Apple via the Fruit redirect.
/// Links: Apple->Banana, Banana->Cherry, Cherry->Apple, Elderberry->Fruit(=Apple),
/// Fruit->Cherry (dropped: from a redirect).
fn fixtures(dir: &PathBuf) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("enwiki-latest-page.sql"),
        "INSERT INTO `page` VALUES \
         (1,0,'Apple',0),(2,0,'Banana',0),(3,0,'Cherry',0),(4,0,'Durian',0),\
         (5,0,'Fruit',1),(6,0,'Elderberry',0),(7,2,'Talk_page',0);\n",
    )
    .unwrap();
    fs::write(
        dir.join("enwiki-latest-redirect.sql"),
        "INSERT INTO `redirect` VALUES (5,0,'Apple','',NULL);\n",
    )
    .unwrap();
    fs::write(
        dir.join("enwiki-latest-linktarget.sql"),
        "INSERT INTO `linktarget` VALUES \
         (10,0,'Apple'),(11,0,'Banana'),(12,0,'Cherry'),(13,0,'Fruit'),(14,0,'Missing'),(15,2,'Talk_page');\n",
    )
    .unwrap();
    fs::write(
        dir.join("enwiki-latest-pagelinks.sql"),
        "INSERT INTO `pagelinks` VALUES \
         (1,0,11),(2,0,12),(3,0,10),(6,0,13),(5,0,12),(1,0,14),(7,2,10),(1,0,11);\n",
    )
    .unwrap();
}

#[test]
fn build_and_query() {
    let dir = std::env::temp_dir().join(format!("wikigraph-test-{}", std::process::id()));
    fixtures(&dir);
    let out = dir.join("graph.bin");
    let stats = build(&dir, &out, Limits::default()).unwrap();
    assert_eq!(stats.nodes, 5); // Apple Banana Cherry Durian Elderberry
    assert_eq!(stats.edges, 4); // A->B, B->C, C->A, E->A (dedup + redirect collapse)

    let g = Graph::open(&out).unwrap();
    assert_eq!(g.node_count(), 5);

    // dense ids follow sorted title order
    let titles: Vec<&str> = (0..5).map(|v| g.title(v)).collect();
    assert_eq!(
        titles,
        ["Apple", "Banana", "Cherry", "Durian", "Elderberry"]
    );

    let apple = g.find(b"Apple").unwrap();
    let banana = g.find(b"Banana").unwrap();
    let cherry = g.find(b"Cherry").unwrap();
    let durian = g.find(b"Durian").unwrap();
    let elderberry = g.find(b"Elderberry").unwrap();
    assert!(g.find(b"Fruit").is_none(), "redirects are not nodes");

    assert_eq!(g.out(apple), [banana]);
    assert_eq!(g.out(banana), [cherry]);
    assert_eq!(g.incoming(apple), [cherry, elderberry]);
    assert_eq!(g.out_degree(durian), 0);
    assert_eq!(g.in_degree(durian), 0);

    // prefix search
    let r = g.prefix_range(b"B");
    assert_eq!(r, banana..banana + 1);
    assert_eq!(g.prefix_range(b"Z"), 5..5);

    // pagerank: cycle members outrank the dangling node
    assert!(g.rank(apple) > g.rank(durian));

    // paths
    assert_eq!(
        shortest_path(&g, apple, cherry, 1000),
        Some(vec![apple, banana, cherry])
    );
    assert_eq!(
        shortest_path(&g, elderberry, cherry, 1000),
        Some(vec![elderberry, apple, banana, cherry])
    );
    assert_eq!(
        shortest_path(&g, apple, elderberry, 1000),
        None,
        "edges are directed"
    );
    assert_eq!(shortest_path(&g, apple, apple, 1000), Some(vec![apple]));

    fs::remove_dir_all(&dir).ok();
}
