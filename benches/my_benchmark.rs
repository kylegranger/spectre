use std::fs;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode};
use serde::Deserialize;
use spectre::{edge::Edge, graph::Graph};

#[derive(Default, Clone, Deserialize, Debug)]
pub struct Sample {
    pub node_ips: Vec<String>,
    pub indices: Vec<Vec<usize>>,
}

fn do_medium_graph(num_threads: usize) -> bool {
    let jstring = fs::read_to_string("testdata/sample.json").unwrap();
    let sample: Sample = serde_json::from_str(&jstring).unwrap();
    // let sample = load_sample("testdata/sample.json");
    let mut graph = Graph::new();
    let mut n = 0;
    for node in &sample.indices {
        for connection in node {
            if *connection > n {
                graph.insert(Edge::new(n, *connection));
            }
        }
        n += 1;
    }

    _ = graph.betweenness_centrality(num_threads, false);
    _ = graph.closeness_centrality(num_threads);

    true
}

fn from_elem(c: &mut Criterion) {
    let mut group = c.benchmark_group("from_elem");
    for nthreads in [1, 2, 3, 4, 6, 8, 12, 16].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(nthreads),
            nthreads,
            |b, &nthreads| {
                b.iter(|| do_medium_graph(black_box(nthreads)));
            },
        );
        group.sampling_mode(SamplingMode::Flat);
        group.significance_level(0.1).sample_size(10);
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = from_elem,
}

criterion_main!(benches);
