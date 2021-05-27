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
    pub start: usize,        // pointer to the edge's start
    pub end: usize,          // pointer to the edge's end
    pub total_capacity: u64, // total capacity
    pub in_capacity: u64,    // capacity affected to the end vertex
    pub out_capacity: u64,   // capacity affected to the start vertex
}

impl Edge {
    fn in_capacity(&self) -> u64 {
        self.in_capacity
    }
    fn out_capacity(&self) -> u64 {
        self.out_capacity
    }
    fn stashed_capacity(&self) -> u64 {
        self.total_capacity - self.in_capacity - self.out_capacity
    }
}
#[derive(Debug, Clone)]
struct Vertex {
    pub label: u64,
    pub in_edges: Vec<usize>,
    pub out_edges: Vec<usize>,
    pub component: Option<usize>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EvictionStrategy {
    LeastChargedFullNonRec,
    LeastChargedSplitNonRec,
}

#[derive(Debug, Clone)]
struct Graph {
    pub vertices: Vec<Vertex>,
    pub edges: Vec<Edge>,
    pub connected_components_count: usize,
    pub max_vertex_load: usize,
    pub eviction_strategy: EvictionStrategy,
}

impl Graph {
    fn new(max_cap: usize, eviction_strategy: EvictionStrategy) -> Graph {
        Graph {
            vertices: Vec::new(),
            edges: Vec::new(),
            connected_components_count: 0,
            max_vertex_load: max_cap,
            eviction_strategy,
        }
    }

    fn new_with_vertices(
        max_cap: usize,
        n_vertices: usize,
        eviction_strategy: EvictionStrategy,
    ) -> Graph {
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
            max_vertex_load: max_cap,
            eviction_strategy,
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
        // self.push_edge_cuckoo(edge_index, 0);
        match self.eviction_strategy {
            EvictionStrategy::LeastChargedFullNonRec => self.push_edge_min_cap_full(edge_index),
            EvictionStrategy::LeastChargedSplitNonRec => self.push_edge_min_cap_split(edge_index),
        }
    }

    // fn push_edge_cuckoo(&mut self, edge_index: usize, iteration_depth: usize) {
    //     if iteration_depth > 10 {
    //         println!("cuckoo iteration depth {}", iteration_depth);
    //     }

    //     let edge = &self.edges[edge_index];
    //     let cap_start = self.vertex_load(edge.start);

    //     if cap_start > self.max_vertex_load as u64 {
    //         // we need to reverse one outgoing edge of the starting vertex

    //         let v = &self.vertices[edge.start];
    //         // position of the edge to be reversed in the outgoing edges array
    //         // pick that position randomly
    //         let reversed_edge_loc_pos = rand::thread_rng().gen_range(0, v.out_edges.len());

    //         let rev_edge_index = v.out_edges[reversed_edge_loc_pos];

    //         // reverse the edge
    //         todo!("Missing implementation");
    //         // self.reverse_edge(rev_edge_index);

    //         // iterate the eviction
    //         self.push_edge_cuckoo(rev_edge_index, iteration_depth + 1);
    //     }
    // }

    fn push_edge_min_cap_full(&mut self, edge_index: usize) {
        let edge = &self.edges[edge_index];
        let load_start = self.vertex_load(edge.start);
        let load_end = self.vertex_load(edge.end);

        assert!(load_start <= self.max_vertex_load as u64);
        assert!(load_end <= self.max_vertex_load as u64);

        let cap_start = self.max_vertex_load as u64 - load_start;
        let cap_end = self.max_vertex_load as u64 - load_end;

        let mut rem_cap = edge.total_capacity;

        // we want to choose the direction that minimizes the vertex load
        let edge = &mut self.edges[edge_index]; // mutable borrow starting from here

        if edge.start != edge.end {
            if cap_start >= cap_end {
                // try to fit everything in the least charged vertex (outgoing) ...
                let out_cap = rem_cap.min(cap_start);
                edge.out_capacity = out_cap;
                rem_cap -= out_cap;

                // and then in the most charged one (incoming)
                let in_cap = rem_cap.min(cap_end);
                edge.in_capacity = in_cap;
                rem_cap -= in_cap;
            } else {
                // try to fit everything in the least charged vertex (incoming) ...
                let in_cap = rem_cap.min(cap_end);
                edge.in_capacity = in_cap;
                rem_cap -= in_cap;

                // and then in the most charged one (outgoing)
                let out_cap = rem_cap.min(cap_start);
                edge.out_capacity = out_cap;
                rem_cap -= out_cap;
            }
        } else {
            // looping edge
            let out_cap = rem_cap.min(cap_start);
            edge.out_capacity = out_cap;
        }
        // self.push_edge_min_cap_aux(edge_index, 0)
    }

    fn push_edge_min_cap_split(&mut self, edge_index: usize) {
        let edge = &self.edges[edge_index];

        let mut load_start = self.vertex_load(edge.start);
        let mut load_end = self.vertex_load(edge.end);

        assert!(load_start <= self.max_vertex_load as u64);
        assert!(load_end <= self.max_vertex_load as u64);

        let mut rem_cap = edge.total_capacity;

        // we want to choose the direction that minimizes the vertex load
        let edge = &mut self.edges[edge_index]; // mutable borrow starting from here

        // println!("\nStart assign cap {}", rem_cap);
        // println!("Load start {}", load_start);
        // println!("Load end {}", load_end);

        if edge.start != edge.end {
            if load_start > load_end {
                // need to put some weight at the end
                let load_diff = load_start - load_end;
                let assigned_cap = rem_cap.min(load_diff);

                // equalize the loads
                edge.in_capacity = assigned_cap;
                rem_cap -= assigned_cap;

                // update the load
                load_end += assigned_cap;
            } else if load_end > load_start {
                // need to put some weight at the start
                let load_diff = load_end - load_start;
                let assigned_cap = rem_cap.min(load_diff);

                // equalize the loads
                edge.out_capacity = assigned_cap;
                rem_cap -= assigned_cap;

                // update the load
                load_start += assigned_cap;
            }

            // println!("Remaining cap 1 {}", rem_cap);
            // println!("Load start {}", load_start);
            // println!("Load end {}", load_end);
            // println!("In cap {}", edge.in_capacity());
            // println!("Out cap {}", edge.out_capacity());
            // assert!(load_start <= self.max_vertex_load as u64);
            // assert!(load_end <= self.max_vertex_load as u64);

            let cap_start = self.max_vertex_load as u64 - load_start;
            let cap_end = self.max_vertex_load as u64 - load_end;

            if rem_cap > 0 {
                assert_eq!(cap_start, cap_end);
            }

            let in_cap = cap_end.min(rem_cap / 2);
            edge.in_capacity += in_cap;

            rem_cap -= in_cap;
            load_end += in_cap;

            // println!("Remaining cap 2 {}", rem_cap);
            // println!("Load start {}", load_start);
            // println!("Load end {}", load_end);
            // println!("In cap {}", edge.in_capacity());
            // println!("Out cap {}", edge.out_capacity());

            let out_cap = cap_start.min(rem_cap);
            edge.out_capacity += out_cap;
            rem_cap -= out_cap;
            load_start += out_cap;

            // println!("Remaining cap 3 {}", rem_cap);
            // println!("Load start {}", load_start);
            // println!("Load end {}", load_end);
            // println!("In cap {}", edge.in_capacity());
            // println!("Out cap {}", edge.out_capacity());

            assert!(load_start <= self.max_vertex_load as u64);
            assert!(load_end <= self.max_vertex_load as u64);

            assert!(edge.out_capacity + edge.in_capacity <= edge.total_capacity);

            // self.push_edge_min_cap_aux(edge_index, 0)
        } else {
            // looping edge
            let cap = rem_cap.min(self.max_vertex_load as u64 - load_start);

            edge.out_capacity += cap;
        }
    }
    // fn push_edge_min_cap_aux(&mut self, edge_index: usize, iteration_depth: usize) {
    //     let edge = &self.edges[edge_index];
    //     let cap_start = self.vertex_load(edge.start);

    //     if iteration_depth > 100 {
    //         println!("push_edge_min_cap iteration depth {}", iteration_depth);
    //     }

    //     if cap_start > self.max_vertex_load as u64 {
    //         // we need to reverse one outgoing edge of the starting vertex

    //         let v = &self.vertices[edge.start];
    //         // position of the edge to be reversed in the outgoing edges array
    //         // take the least charged vertex
    //         let reversed_edge_loc_pos = v
    //             .out_edges
    //             .iter()
    //             .map(|&e| self.vertex_load(self.edges[e].start))
    //             .enumerate()
    //             .min_by(|(_, v1), (_, v2)| v1.cmp(v2))
    //             .unwrap()
    //             .0;

    //         let rev_edge_index = v.out_edges[reversed_edge_loc_pos];

    //         // reverse the edge
    //         self.reverse_edge(rev_edge_index);

    //         // iterate the eviction
    //         self.push_edge_min_cap_aux(rev_edge_index, iteration_depth + 1);
    //     }
    // }

    // fn add_edge(&mut self, label: u64, start: usize, end: usize, cap: u64) {
    //     if start >= self.vertices.len() {
    //         panic!("Invalid starting vertex");
    //     }
    //     if end >= self.vertices.len() {
    //         panic!("Invalid ending vertex");
    //     }

    //     let mut e = if self.vertex_load(end) < self.vertex_load(start) {
    //         Edge {
    //             index: usize::max_value(),
    //             label,
    //             start,
    //             end,
    //             total_capacity: cap,
    //             out_capacity: cap,
    //             in_capacity: 0u64,
    //         }
    //     } else {
    //         Edge {
    //             index: usize::max_value(),
    //             label,
    //             end,
    //             start,
    //             total_capacity: cap,
    //             out_capacity: cap,
    //             in_capacity: 0u64,
    //         }
    //     };

    //     let e_start = e.start;
    //     let e_end = e.end;
    //     let e_index = self.edges.len();
    //     e.index = e_index;
    //     self.vertices[e_start].out_edges.push(e_index);
    //     self.vertices[e_end].in_edges.push(e_index);
    //     self.edges.push(e);

    //     //
    //     self.push_edge(e_index);

    //     // if cap > 1 {
    //     // self.add_edge(label, start, end, cap - 1)
    //     // }
    // }

    fn add_edge(&mut self, label: u64, start: usize, end: usize, cap: u64) {
        if start >= self.vertices.len() {
            panic!("Invalid starting vertex");
        }
        if end >= self.vertices.len() {
            panic!("Invalid ending vertex");
        }

        // println!("\n\nLoad start {}", self.vertex_load(start));
        // println!("Load end {}", self.vertex_load(end));

        let e_index = self.edges.len();
        let e = Edge {
            index: e_index,
            label,
            start,
            end,
            total_capacity: cap,
            out_capacity: 0u64,
            in_capacity: 0u64,
        };

        let e_start = e.start;
        let e_end = e.end;
        self.vertices[e_start].out_edges.push(e_index);
        self.vertices[e_end].in_edges.push(e_index);
        self.edges.push(e);

        //
        self.push_edge(e_index);

        // println!("\nNew load start {}", self.vertex_load(start));
        // println!("New Load end {}", self.vertex_load(end));
        // println!("In cap {}", self.edges[e_index].in_capacity());
        // println!("Out cap {}", self.edges[e_index].out_capacity());

        // assert!(
        //     self.vertex_load(start) <= self.max_vertex_load as u64,
        //     "{}",
        //     self.vertex_load(start)
        // );
        // assert!(
        //     self.vertex_load(end) <= self.max_vertex_load as u64,
        //     "{}",
        //     self.vertex_load(end)
        // );

        // if cap > 1 {
        // self.add_edge(label, start, end, cap - 1)
        // }
    }

    // fn reverse_edge(&mut self, edge_index: usize) {
    //     assert!(edge_index < self.edge_count());

    //     let e = &mut self.edges[edge_index];
    //     let c = e.in_capacity;
    //     e.in_capacity = e.out_capacity;
    //     e.out_capacity = c;
    // }

    //     let edge = &mut self.edges[edge_index];

    // fn reverse_edge(&mut self, edge_index: usize) {
    //     assert!(edge_index < self.edge_count());

    //     let edge = &mut self.edges[edge_index];
    //     let old_start = edge.start;
    //     let old_end = edge.end;

    //     edge.start = old_end;
    //     edge.end = old_start;

    //     // remove from the old starting vertex outgoing edges
    //     let pos: usize = self.vertices[old_start]
    //         .out_edges
    //         .iter()
    //         .position(|&x| x == edge_index)
    //         .unwrap();

    //     self.vertices[old_start].out_edges.swap_remove(pos);

    //     // remove from the old ending vertex ingoing edges
    //     let pos: usize = self.vertices[old_end]
    //         .in_edges
    //         .iter()
    //         .position(|&x| x == edge_index)
    //         .unwrap();

    //     self.vertices[old_end].in_edges.swap_remove(pos);

    //     // reinsert the edge
    //     self.vertices[edge.start].out_edges.push(edge_index);
    //     self.vertices[edge.end].in_edges.push(edge_index);
    // }

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

    fn vertex_load(&self, vertex_index: usize) -> u64 {
        let v = &self.vertices[vertex_index];
        v.out_edges
            .iter()
            .map(|&ei| self.edges[ei].out_capacity())
            .sum::<u64>()
            + v.in_edges
                .iter()
                .map(|&ei| self.edges[ei].in_capacity())
                .sum::<u64>()
    }

    // fn in_edge_capacity(&self, vertex: usize) -> u64 {
    //     assert!(vertex < self.vertex_count());

    //     self.vertices[vertex]
    //         .in_edges
    //         .iter()
    //         .map(|&e| self.edges[e].capacity())
    //         .sum::<i64>() as u64
    // }

    fn out_edge_count(&self, vertex: usize) -> usize {
        assert!(vertex < self.vertex_count());

        self.vertices[vertex].out_edges.len()
    }

    // fn out_edge_capacity(&self, vertex: usize) -> u64 {
    //     assert!(vertex < self.vertex_count());

    //     self.vertices[vertex]
    //         .out_edges
    //         .iter()
    //         .map(|&e| self.edges[e].capacity())
    //         .sum::<i64>() as u64
    // }

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
) -> (Vec<usize>, u64) {
    // let mut times: FlowAllocTimings = Default::default();
    // let start_gen = std::time::Instant::now();

    let mut remaining_elements = params.alloc_params.n;

    let mut rng = thread_rng();

    // create a new graph with m vertices, each of capacity
    let mut graph = Graph::new_with_vertices(
        params.alloc_params.bucket_capacity,
        params.alloc_params.m,
        params.eviction_strategy,
    );

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
            graph.vertex_load(v) as usize
            // graph.vertices[v]
            //     .out_edges
            //     .iter()
            //     // .filter(
            //     //     |&&e| graph.edges[e].end < params.alloc_params.m, // this predicates returns true iff rff.edges[e] is an edge whose end is in the graph
            //     // )
            //     .map(|&e| graph.edges[e].capacity())
            //     .sum::<i64>() as usize
        })
        .collect();

    let stash_size: u64 = graph.edges.iter().map(|e| e.stashed_capacity()).sum();

    (res, stash_size)
}

pub fn run_experiment(params: DynamicAllocExperimentParams) -> DynamicExperimentResult {
    // let mut timings: FlowAllocTimings = Default::default();
    // let mut connected_components: usize = 0;
    // let rand_alloc = flow_alloc(params, Some(&mut timings), Some(&mut connected_components));
    let (rand_alloc, stash_size) = dynamic_alloc(params);
    let size = rand_alloc.iter().sum();
    let max_load = rand_alloc.iter().fold(0, |max, x| max.max(*x));
    // let load_modes = compute_modes(rand_alloc.into_iter(), max_load);
    // let stash_size = compute_overflow_stat(load_modes.iter(), params.alloc_params.bucket_capacity);

    assert!(size <= params.alloc_params.n);
    assert_eq!(size + stash_size as usize, params.alloc_params.n);

    DynamicExperimentResult {
        size,
        max_load,
        // load_modes,
        stash_size: stash_size as usize,
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
