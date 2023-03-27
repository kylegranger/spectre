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

/// this does a BFS (breadth first search) for a given node.
/// it finds the shortest paths to all other nodes, summing
/// the lengths in a node-indexed array that is passed in
fn closeness_for_node(index: usize, indices: &Vec<Vec<GraphIndex>>, total_path_length: &mut [u32]) {
    let num_nodes = indices.len();

    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut deltas: Vec<i32> = vec![-1; num_nodes];

    deltas[index] = 0;
    queue.push_back(index);

    while !queue.is_empty() {
        let current = *queue.front().unwrap();
        queue.pop_front();
        let len = indices[current].len();
        for j in 0..len {
            if deltas[indices[current][j] as usize] == -1 {
                deltas[indices[current][j] as usize] = deltas[current] + 1;
                queue.push_back(indices[current][j] as usize);
                total_path_length[indices[current][j] as usize] +=
                    deltas[indices[current][j] as usize] as u32;
            }
        }
    }
}

/// this function is the thread task
/// grabs next unprocessed node
/// if no more nodes, exits
/// returning total path lengths
fn closeness_task(acounter: Arc<Mutex<usize>>, aindices: Arc<Vec<Vec<GraphIndex>>>) -> Vec<u32> {
    let start = Instant::now();
    let indices = &aindices;
    let num_nodes = indices.len();

    // each worker thread keeps its own cache of data
    // these are returned when the thread finishes
    // and then summed by the caller
    let mut total_path_length: Vec<u32> = vec![0; num_nodes];

    let mut finished = false;
    while !finished {
        let mut counter = acounter.lock().unwrap();
        let index: usize = *counter;
        *counter += 1;
        drop(counter);
        if index < num_nodes {
            closeness_for_node(index, indices, &mut total_path_length);
        } else {
            finished = true;
        }
    }
    total_path_length
}

/// This function is called by the graph method
/// closeness_centrality.  It does all
/// the heavy lifting with processing the data via
/// multiple threads.
/// It is reponsibility for:
/// - setting up the data to be passed to the threads
/// - instantiating and spawning the threads
/// - collecting the results when each is finished
/// - added the results together, and returning them
/// It public for graph, but is not exposed in the public library interface.
pub fn compute_closeness(indices: Vec<Vec<GraphIndex>>, mut num_threads: usize) -> Vec<u32> {
    let start = Instant::now();
    num_threads = num_threads.clamp(MIN_NUM_THREADS, MAX_NUM_THREADS);

    let num_nodes = indices.len();

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
            total_path_length[i] += t[i];
        }
    }

    total_path_length
}
