//! A module for performing the multi-threaded computation of betweenness

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

use crate::graph::GraphIndex;

const MIN_NUM_THREADS: usize = 1;
const MAX_NUM_THREADS: usize = 128;

fn betweenness_for_node(s: usize, indices: &Vec<Vec<GraphIndex>>, betweenness_count: &mut [f64]) {
    let num_nodes = indices.len();
    // println!("\nDOING s {s}");

    let mut sp: Vec<f64> = vec![0.0; num_nodes];
    let mut d: Vec<usize> = vec![num_nodes + 1; num_nodes];
    let mut totals: Vec<Vec<usize>> = vec![Vec::<usize>::new(); num_nodes];
    let mut delta: Vec<f64> = vec![0.0; num_nodes];
    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut v: usize;
    let mut w: usize;

    sp[s] = 1.0;
    d[s] = 0;
    queue.push_back(s);

    while !queue.is_empty() {
        v = *queue.front().unwrap();
        queue.pop_front();
        stack.push(v);

        let z: usize = indices[v].len();
        // println!("v {v}, z {z}");
        for i in 0..z {
            let w: usize = indices[v][i] as usize;
            if d[w] == num_nodes + 1 {
                d[w] = d[v] + 1;
                // println!("  push w {w} tp Q");
                queue.push_back(w);
            }
            if d[w] == d[v] + 1 {
                sp[w] += sp[v];
                totals[w].push(v);
            }
        }
    }

    // println!("s {s}, S size is {}", stack.len());
    while !stack.is_empty() {
        w = stack[stack.len() - 1];
        stack.pop();

        // println!("  w {w}, P[w].len() {}", totals[w].len());
        for j in 0..totals[w].len() {
            v = totals[w][j];
            delta[v] += sp[v] / sp[w] * (1.0 + delta[w]);
        }
        if w != s {
            betweenness_count[w] += delta[w];
        }
    }

    // println!("s {s}, total {total}");
    // total_path_length[s] = total as u32;
}

fn closeness_for_node(s: usize, indices: &Vec<Vec<GraphIndex>>, total_path_length: &mut [u32]) {
    let num_nodes = indices.len();

    let mut q: VecDeque<usize> = VecDeque::new();
    let mut deltas: Vec<i32> = vec![-1; num_nodes];

    deltas[s] = 0;
    q.push_back(s);

    while !q.is_empty() {
        let current = *q.front().unwrap();
        q.pop_front();
        let z = indices[current].len();
        for j in 0..z {
            if deltas[indices[current][j] as usize] == -1 {
                deltas[indices[current][j] as usize] = deltas[current] + 1;
                q.push_back(indices[current][j] as usize);
                // total += dd[indices[current][j] as usize];
                // println!("  inc index {} by {}", indices[current][j], dd[indices[current][j] as usize]);
                total_path_length[indices[current][j] as usize] +=
                    deltas[indices[current][j] as usize] as u32;
            }
        }
    }
}

/// this function is the thread task
/// grabs next unprocessed node
/// if no more nodes, exits
/// returning betweenness, total path lengths, and num paths
fn betweenness_task(acounter: Arc<Mutex<usize>>, aindices: Arc<Vec<Vec<GraphIndex>>>) -> Vec<f64> {
    let start = Instant::now();
    let indices = &aindices;
    let num_nodes = indices.len();

    // each worker thread keeps its own cache of data
    // these are returned when the thread finishes
    // and then summed by the caller
    let mut betweenness_count: Vec<f64> = vec![0.0; num_nodes];
    // let mut total_path_length: Vec<u32> = vec![0; num_nodes];

    let mut finished = false;
    while !finished {
        let mut counter = acounter.lock().unwrap();
        let index: usize = *counter;
        *counter += 1;
        drop(counter);
        if index < num_nodes {
            if index % 100 == 0 {
                println!("node: {}, time: {:?}", index, start.elapsed());
            }
            betweenness_for_node(index, indices, &mut betweenness_count);
        } else {
            finished = true;
        }
    }
    betweenness_count
}

/// this function is the thread task
/// grabs next unprocessed node
/// if no more nodes, exits
/// returning betweenness, total path lengths, and num paths
fn closeness_task(acounter: Arc<Mutex<usize>>, aindices: Arc<Vec<Vec<GraphIndex>>>) -> Vec<u32> {
    let start = Instant::now();
    let indices = &aindices;
    let num_nodes = indices.len();

    // each worker thread keeps its own cache of data
    // these are returned when the thread finishes
    // and then summed by the caller
    // let mut betweenness_count: Vec<f64> = vec![0.0; num_nodes];
    let mut total_path_length: Vec<u32> = vec![0; num_nodes];

    let mut finished = false;
    while !finished {
        let mut counter = acounter.lock().unwrap();
        let index: usize = *counter;
        *counter += 1;
        drop(counter);
        if index < num_nodes {
            if index % 100 == 0 {
                println!("node: {}, time: {:?}", index, start.elapsed());
            }
            closeness_for_node(index, indices, &mut total_path_length);
        } else {
            finished = true;
        }
    }
    total_path_length
}

/// This public function is called by the graph method
/// betweenness_and_closeness_centrality.  It does all
/// the heavy lifting with processing the data via
/// multiple threads
/// It is reponsibility for:
/// - setting up the data to be passed to the threads
/// - instantiating and spawning the threads
/// - collecting the results when each is finished
/// - added the results together, and returning them
pub fn compute_betweenness(indices: Vec<Vec<GraphIndex>>, mut num_threads: usize) -> Vec<f64> {
    let start = Instant::now();
    num_threads = num_threads.clamp(MIN_NUM_THREADS, MAX_NUM_THREADS);
    println!("\ncompute_betweenness: num_threads {:?}", num_threads);

    let num_nodes = indices.len();

    let mut betweenness_count: Vec<f64> = vec![0.0; num_nodes];
    // let mut total_path_length: Vec<u32> = vec![0; num_nodes];

    let mut handles = Vec::with_capacity(num_threads);
    let wrapped_indices = Arc::new(indices);
    let wrapped_counter = Arc::new(Mutex::new(0));

    for _ in 0..num_threads {
        let acounter = Arc::clone(&wrapped_counter);
        let aindices = Arc::clone(&wrapped_indices);
        let handle = thread::spawn(move || betweenness_task(acounter, aindices));
        handles.push(handle);
    }

    for h in handles {
        let b = h.join().unwrap();

        for i in 0..num_nodes {
            betweenness_count[i] += b[i] / 2.0;
            // total_path_length[i] += t[i];
        }
    }

    println!("compute: done {:?}", start.elapsed());

    betweenness_count
}

pub fn compute_closeness(indices: Vec<Vec<GraphIndex>>, mut num_threads: usize) -> Vec<u32> {
    let start = Instant::now();
    num_threads = num_threads.clamp(MIN_NUM_THREADS, MAX_NUM_THREADS);
    println!("\ncompute_closeness: num_threads {:?}", num_threads);

    let num_nodes = indices.len();

    // let mut betweenness_count: Vec<f64> = vec![0.0; num_nodes];
    let mut total_path_length: Vec<u32> = vec![0; num_nodes];

    let mut handles = Vec::with_capacity(num_threads);
    let wrapped_indices = Arc::new(indices);
    let wrapped_counter = Arc::new(Mutex::new(0));

    for _ in 0..num_threads {
        let acounter = Arc::clone(&wrapped_counter);
        let aindices = Arc::clone(&wrapped_indices);
        let handle = thread::spawn(move || closeness_task(acounter, aindices));
        handles.push(handle);
    }

    for h in handles {
        let t = h.join().unwrap();

        for i in 0..num_nodes {
            // betweenness_count[i] += b[i] / 2.0;
            total_path_length[i] += t[i];
        }
    }

    println!("compute: done {:?}", start.elapsed());

    total_path_length
}
