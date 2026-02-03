//! Criterion microbenches for panlabel format parsing and writing.
//!
//! Run with: `cargo bench`
//!
//! These benchmarks measure the performance of:
//! - COCO JSON parsing (from_coco_str, from_coco_slice)
//! - TFOD CSV writing (to_tfod_csv_string)

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use std::hint::black_box;

use panlabel::ir::io_coco_json::{from_coco_slice, from_coco_str};
use panlabel::ir::io_tfod_csv::{from_tfod_csv_str, to_tfod_csv_string};

// Include test fixtures at compile time (no file I/O during benchmark)
const COCO_FIXTURE: &str = include_str!("../tests/fixtures/sample_valid.coco.json");

// Small inline TFOD CSV for benchmarking (assets/ is gitignored, not available in CI)
const TFOD_FIXTURE: &str = "filename,width,height,class,xmin,ymin,xmax,ymax
image001.jpg,640,480,person,0.1,0.2,0.5,0.8
image001.jpg,640,480,car,0.3,0.1,0.7,0.4
image002.jpg,800,600,dog,0.2,0.3,0.6,0.9
image002.jpg,800,600,cat,0.1,0.1,0.4,0.5
image003.jpg,640,480,person,0.0,0.0,0.3,0.6
";

/// Benchmark COCO JSON parsing from string.
fn bench_coco_parse_str(c: &mut Criterion) {
    let mut group = c.benchmark_group("coco_parse");
    group.throughput(Throughput::Bytes(COCO_FIXTURE.len() as u64));

    group.bench_function("from_coco_str", |b| {
        b.iter(|| {
            let ds = from_coco_str(black_box(COCO_FIXTURE)).unwrap();
            black_box(ds)
        })
    });

    group.finish();
}

/// Benchmark COCO JSON parsing from byte slice.
fn bench_coco_parse_slice(c: &mut Criterion) {
    let bytes = COCO_FIXTURE.as_bytes();
    let mut group = c.benchmark_group("coco_parse");
    group.throughput(Throughput::Bytes(bytes.len() as u64));

    group.bench_function("from_coco_slice", |b| {
        b.iter(|| {
            let ds = from_coco_slice(black_box(bytes)).unwrap();
            black_box(ds)
        })
    });

    group.finish();
}

/// Benchmark TFOD CSV writing.
///
/// We parse the TFOD fixture once to get a Dataset, then benchmark
/// writing it back to CSV string format.
fn bench_tfod_write(c: &mut Criterion) {
    // Parse TFOD CSV fixture once (outside the timed region)
    let dataset = from_tfod_csv_str(TFOD_FIXTURE).expect("Failed to parse TFOD fixture");

    let mut group = c.benchmark_group("tfod_write");
    // Throughput based on number of annotations
    group.throughput(Throughput::Elements(dataset.annotations.len() as u64));

    group.bench_function("to_tfod_csv_string", |b| {
        b.iter(|| {
            let csv = to_tfod_csv_string(black_box(&dataset)).unwrap();
            black_box(csv)
        })
    });

    group.finish();
}

/// Benchmark TFOD CSV parsing (for comparison).
fn bench_tfod_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("tfod_parse");
    group.throughput(Throughput::Bytes(TFOD_FIXTURE.len() as u64));

    group.bench_function("from_tfod_csv_str", |b| {
        b.iter(|| {
            let ds = from_tfod_csv_str(black_box(TFOD_FIXTURE)).unwrap();
            black_box(ds)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_coco_parse_str,
    bench_coco_parse_slice,
    bench_tfod_write,
    bench_tfod_parse,
);
criterion_main!(benches);
