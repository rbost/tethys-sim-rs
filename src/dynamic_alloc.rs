#![allow(dead_code)]

// use std::collections::VecDeque;
extern crate rand;
use rand::prelude::*;

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use std::{slice::Iter, u64};

use std::vec::Vec;

pub use crate::alloc_experiments_types::{AllocExperimentParams, ListGenerationMethod};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

pub use crate::alloc_experiments_types::*;
pub use crate::utils::*;

#[derive(Debug, Clone)]
struct Edge {
    pub index: usize,
    pub label: u64,
    pub start: usize, // pointer to the edge's start
    pub end: usize,   // pointer to the edge's end
                      // pub capacity: i64,
}

impl Edge {
    fn capacity(&self) -> i64 {
        1i64
    }
}
#[derive(Debug, Clone)]
struct Vertex {
    pub label: u64,
    pub in_edges: Vec<usize>,
    pub out_edges: Vec<usize>,
    pub component: Option<usize>,
}

#[derive(Debug, Clone)]
struct Graph {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub connected_components_count: usize,
    pub max_vertex_capacity: usize,
}

impl Graph {
    fn new(max_cap: usize) -> Graph {
        Graph {
            vertices: Vec::new(),
            edges: Vec::new(),
            connected_components_count: 0,
            max_vertex_capacity: max_cap,
        }
    }

    fn new_with_vertices(max_cap: usize, n_vertices: usize) -> Graph {
        let mut vertices: Vec<Vertex> = Vec::with_capacity(n_vertices);
        for i in 0..n_vertices {
            let v = Vertex {
                label: i as u64,
                in_edges: Vec::new(),
                out_edges: Vec::new(),
                component: None,
            };
            vertices.push(v);
        }
        Graph {
            vertices: vertices,
            edges: Vec::new(),
            connected_components_count: 0,
            max_vertex_capacity: max_cap,
        }
    }

    fn add_vertex(&mut self, l: u64) -> usize {
        self.vertices.push(Vertex {
            label: l,
            in_edges: vec![],
            out_edges: vec![],
            component: None,
        });
        self.vertices.len() - 1
    }

    fn push_edge(&mut self, edge_index: usize) {
        self.push_edge_cuckoo(edge_index, 0);
    }

    fn push_edge_cuckoo(&mut self, edge_index: usize, iteration_depth: usize) {
        if iteration_depth > 10 {
            println!("iteration depth {}", iteration_depth);
        }

        let edge = &self.edges[edge_index];
        let cap_start = self.out_edge_capacity(edge.start);

        if cap_start > self.max_vertex_capacity as u64 {
            // we need to reverse one outgoing edge of the starting vertex

            let v = &self.vertices[edge.start];
            // position of the edge to be reversed in the outgoing edges array
            // pick that position randomly
            let reversed_edge_loc_pos = rand::thread_rng().gen_range(0, v.out_edges.len());

            let rev_edge_index = v.out_edges[reversed_edge_loc_pos];

            // reverse the edge
            self.reverse_edge(rev_edge_index);

            // iterate the eviction
            self.push_edge_cuckoo(rev_edge_index, iteration_depth + 1);
        }
    }

    fn push_edge_min_cap(&mut self, edge_index: usize) {
        let edge = &self.edges[edge_index];
        let cap_start = self.out_edge_capacity(edge.start);
        let cap_end = self.out_edge_capacity(edge.end);

        // we want to choose the direction that minimizes the vertex load
        if cap_end < (cap_start as i64 - edge.capacity()) as u64 {
            // reverse the edge
            self.reverse_edge(edge_index);
        }
    }

    fn push_edge_min_cap_aux(&mut self, edge_index: usize, iteration_depth: usize) {
        let edge = &self.edges[edge_index];
        let cap_start = self.out_edge_capacity(edge.start);

        if iteration_depth > 100 {
            println!("iteration depth {}", iteration_depth);
        }

        if cap_start > self.max_vertex_capacity as u64 {
            // we need to reverse one outgoing edge of the starting vertex

            let v = &self.vertices[edge.start];
            // position of the edge to be reversed in the outgoing edges array
            // take the least charged vertex
            let reversed_edge_loc_pos = v
                .out_edges
                .iter()
                .map(|&e| self.out_edge_capacity(self.edges[e].start))
                .enumerate()
                .min_by(|(_, v1), (_, v2)| v1.cmp(v2))
                .unwrap()
                .0;

            let rev_edge_index = v.out_edges[reversed_edge_loc_pos];

            // reverse the edge
            self.reverse_edge(rev_edge_index);

            // iterate the eviction
            self.push_edge_min_cap_aux(rev_edge_index, iteration_depth + 1);
        }
    }

    fn add_edge(&mut self, label: u64, start: usize, end: usize, cap: u64) {
        if start >= self.vertices.len() {
            panic!("Invalid starting vertex");
        }
        if end >= self.vertices.len() {
            panic!("Invalid ending vertex");
        }

        let mut e = if self.out_edge_capacity(end) < self.out_edge_capacity(start) {
            Edge {
                index: usize::max_value(),
                label,
                start,
                end,
                // capacity: cap as i64,
            }
        } else {
            Edge {
                index: usize::max_value(),
                label,
                end,
                start,
                // capacity: cap as i64,
            }
        };

        let e_start = e.start;
        let e_end = e.end;
        let e_index = self.edges.len();
        e.index = e_index;
        self.vertices[e_start].out_edges.push(e_index);
        self.vertices[e_end].in_edges.push(e_index);
        self.edges.push(e);

        //
        self.push_edge(e_index);

        if cap > 1 {
            self.add_edge(label, start, end, cap - 1)
        }
    }

    fn reverse_edge(&mut self, edge_index: usize) {
        assert!(edge_index < self.edge_count());

        let edge = &mut self.edges[edge_index];
        let old_start = edge.start;
        let old_end = edge.end;

        edge.start = old_end;
        edge.end = old_start;

        // remove from the old starting vertex outgoing edges
        let pos: usize = self.vertices[old_start]
            .out_edges
            .iter()
            .position(|&x| x == edge_index)
            .unwrap();

        self.vertices[old_start].out_edges.swap_remove(pos);

        // remove from the old ending vertex ingoing edges
        let pos: usize = self.vertices[old_end]
            .in_edges
            .iter()
            .position(|&x| x == edge_index)
            .unwrap();

        self.vertices[old_end].in_edges.swap_remove(pos);

        // reinsert the edge
        self.vertices[edge.start].out_edges.push(edge_index);
        self.vertices[edge.end].in_edges.push(edge_index);
    }

    fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    fn in_edge_count(&self, vertex: usize) -> usize {
        assert!(vertex < self.vertex_count());

        self.vertices[vertex].in_edges.len()
    }

    fn in_edge_capacity(&self, vertex: usize) -> u64 {
        assert!(vertex < self.vertex_count());

        self.vertices[vertex]
            .in_edges
            .iter()
            .map(|&e| self.edges[e].capacity())
            .sum::<i64>() as u64
    }

    fn out_edge_count(&self, vertex: usize) -> usize {
        assert!(vertex < self.vertex_count());

        self.vertices[vertex].out_edges.len()
    }

    fn out_edge_capacity(&self, vertex: usize) -> u64 {
        assert!(vertex < self.vertex_count());

        self.vertices[vertex]
            .out_edges
            .iter()
            .map(|&e| self.edges[e].capacity())
            .sum::<i64>() as u64
    }

    // fn out_edge_capacity_debug(&self, vertex: usize) -> u64 {
    //     assert!(vertex < self.vertex_count());

    //     self.vertices[vertex]
    //         .out_edges
    //         .iter()
    //         .map(|&e| {
    //             println!("{:?}", self.edges[e]);
    //             self.edges[e].capacity
    //         })
    //         .sum::<i64>() as u64
    // }

    fn edge_iterator(&self) -> Iter<Edge> {
        self.edges.iter()
    }

    fn check_graph_correctness(&self) -> bool {
        let mut res = true;
        for v in 0..self.vertices.len() {
            for e in &self.vertices[v].in_edges {
                if v != self.edges[*e].end {
                    println!("Invalid end for in edge");
                    res = false;
                }
            }
            for e in &self.vertices[v].out_edges {
                if v != self.edges[*e].start {
                    println!("Invalid start for out edge");
                    res = false;
                }
            }
        }

        for e in 0..self.edges.len() {
            let v_start = self.edges[e].start;
            let v_end = self.edges[e].end;
            assert_eq!(
                self.vertices[v_start]
                    .out_edges
                    .iter()
                    .filter(|&&o_e| o_e == e)
                    .collect::<Vec<&usize>>()
                    .len(),
                1,
                "Invalid number of matching out edges"
            );
            assert_eq!(
                self.vertices[v_end]
                    .in_edges
                    .iter()
                    .filter(|&&o_e| o_e == e)
                    .collect::<Vec<&usize>>()
                    .len(),
                1,
                "Invalid number of matching in edges"
            );
        }
        res
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EvictionStrategy {
    Cuckoo,
    LeastCharged,
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LocationGeneration {
    FullyRandom,
    HalfRandom,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DynamicAllocExperimentParams {
    #[serde(flatten)]
    pub alloc_params: AllocExperimentParams,
    pub location_generation: LocationGeneration,
    pub eviction_strategy: EvictionStrategy,
}

#[derive(Debug, Clone, Serialize)]
pub struct DynamicExperimentResult {
    pub size: usize,
    pub max_load: usize,
    // pub load_modes: Vec<usize>,
    pub stash_size: usize,
    // pub connected_components: usize,
    // pub timings: FlowAllocTimings,
}

fn generate_location<T>(rng: &mut T, m: usize, loc_gen: LocationGeneration) -> (usize, usize)
where
    T: rand::Rng,
{
    match loc_gen {
        LocationGeneration::FullyRandom => (rng.gen_range(0, m), rng.gen_range(0, m)),
        LocationGeneration::HalfRandom => {
            (rng.gen_range(0, m / 2), m / 2 + rng.gen_range(0, m - m / 2))
        }
    }
}

fn dynamic_alloc(
    params: DynamicAllocExperimentParams,
    // timings: Option<&mut FlowAllocTimings>,
    // connected_components_count: Option<&mut usize>,
) -> Vec<usize> {
    // let mut times: FlowAllocTimings = Default::default();
    // let start_gen = std::time::Instant::now();

    let mut remaining_elements = params.alloc_params.n;

    let mut rng = thread_rng();

    // create a new graph with m vertices, each of capacity
    let mut graph =
        Graph::new_with_vertices(params.alloc_params.bucket_capacity, params.alloc_params.m);

    let mut list_index: u64 = 0;

    while remaining_elements != 0 {
        let l: usize = match params.alloc_params.generation_method {
            ListGenerationMethod::RandomGeneration => {
                rng.gen_range(0, params.alloc_params.list_max_len.min(remaining_elements)) + 1
            }
            ListGenerationMethod::WorstCaseGeneration => {
                params.alloc_params.list_max_len.min(remaining_elements)
            }
        };

        let (h1, h2) =
            generate_location(&mut rng, params.alloc_params.m, params.location_generation);

        let start = h1;
        let end = h2;

        graph.add_edge(list_index, start, end, l as u64);

        remaining_elements -= l;
        list_index += 1;
    }

    // if let Some(timings) = timings {
    //     *timings = times;
    // }

    // if let Some(components_count) = connected_components_count {
    //     *components_count = rff.connected_components_count;
    // }
    // Now, we can easily compute the load of each bucket.
    // We must be careful to remove the edges whose end are the sink or the
    // source from the load computation
    let res: Vec<usize> = (0..params.alloc_params.m)
        .map(|v| {
            graph.vertices[v]
                .out_edges
                .iter()
                // .filter(
                //     |&&e| graph.edges[e].end < params.alloc_params.m, // this predicates returns true iff rff.edges[e] is an edge whose end is in the graph
                // )
                .map(|&e| graph.edges[e].capacity())
                .sum::<i64>() as usize
        })
        .collect();

    res
}

pub fn run_experiment(params: DynamicAllocExperimentParams) -> DynamicExperimentResult {
    // let mut timings: FlowAllocTimings = Default::default();
    // let mut connected_components: usize = 0;
    // let rand_alloc = flow_alloc(params, Some(&mut timings), Some(&mut connected_components));
    let rand_alloc = dynamic_alloc(params);
    let size = rand_alloc.iter().sum();
    let max_load = rand_alloc.iter().fold(0, |max, x| max.max(*x));
    let load_modes = compute_modes(rand_alloc.into_iter(), max_load);
    let stash_size = compute_overflow_stat(load_modes.iter(), params.alloc_params.bucket_capacity);

    DynamicExperimentResult {
        size,
        max_load,
        // load_modes,
        stash_size,
        // connected_components,
        // timings,
    }
}

pub fn iterated_experiment<F>(
    params: DynamicAllocExperimentParams,
    iterations: usize,
    show_progress: bool,
    iteration_progress_callback: F,
) -> Vec<DynamicExperimentResult>
where
    F: Fn(usize) + Send + Sync,
{
    // println!(
    // "{} one choice allocation iterations with N={}, m={}, max_len={}",
    // iterations, n, m, max_len
    // );

    let elements_pb = ProgressBar::new((iterations * params.alloc_params.n) as u64);
    if show_progress {
        elements_pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {msg} [{bar:40.cyan/blue}] ({pos}/{len} elts - {percent}%) | ETA: {eta_precise}")
        .progress_chars("##-"));
        elements_pb.set_draw_delta(1_000_000);
    }

    let mut iter_completed = AtomicUsize::new(0);

    if show_progress {
        elements_pb.set_position(0);
        elements_pb.set_message(&format!(
            "{}/{} iterations",
            *iter_completed.get_mut(),
            iterations
        ));
    }

    let results: Vec<DynamicExperimentResult> = (0..iterations)
        .into_par_iter()
        .map(|_| {
            let r: DynamicExperimentResult = run_experiment(params);
            if show_progress {
                elements_pb.inc(params.alloc_params.n as u64);
            }
            iteration_progress_callback(params.alloc_params.n);

            let previous_count = iter_completed.fetch_add(1, Ordering::SeqCst);
            if show_progress {
                elements_pb.set_message(&format!(
                    "{}/{} iterations",
                    previous_count + 1,
                    iterations
                ));
            }
            r
        })
        .collect();

    if show_progress {
        elements_pb.finish_with_message("Done!");
    }

    results
}
