#![allow(dead_code)]

use std::collections::VecDeque;
extern crate rand;
use rand::prelude::*;

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use std::slice::Iter;

use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

pub use crate::utils::*;

#[derive(Debug, Clone)]
struct Edge {
    pub label: u64,
    pub start: usize, // pointer to the edge's start
    pub end: usize,   // pointer to the edge's end
    pub capacity: i64,
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
}

#[derive(Debug, Clone, Copy)]
enum TraversalAlgorithm {
    DepthFirstSearch,
    BreadthFirstSearch,
}

impl Graph {
    fn new() -> Graph {
        Graph {
            vertices: Vec::new(),
            edges: Vec::new(),
            connected_components_count: 0,
        }
    }

    fn new_with_vertices(n_vertices: usize) -> Graph {
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
        }
    }

    // Construct a residual graph for the Ford-Fulkerson algorithm
    // from the input graph
    // An important invariant is that an edge with index e in the original
    // graph correspond the the edge with index e in the residual graph and,
    // the reversed edge in the residual graph has index 2e.
    fn new_residual_graph(graph: &Graph) -> Graph {
        let mut res_graph = graph.clone();
        let mut inv_edges = res_graph.edges.clone();

        let n_edges = inv_edges.len();

        // create inverted edges
        for e in inv_edges.iter_mut() {
            let v = e.end;
            e.end = e.start;
            e.start = v;
            e.capacity = 0; // these newly created edges have no real capacity
        }

        // and append them to the regular edges
        res_graph.edges.append(&mut inv_edges);

        for v in res_graph.vertices.iter_mut() {
            let mut reversed_out: Vec<usize> = v.out_edges.iter().map(|e| e + n_edges).collect();
            let mut reversed_in: Vec<usize> = v.in_edges.iter().map(|e| e + n_edges).collect();
            v.out_edges.append(&mut reversed_in);
            v.in_edges.append(&mut reversed_out);
        }

        res_graph
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

    fn add_edge(&mut self, label: u64, start: usize, end: usize, cap: u64) {
        if start < self.vertices.len() && end < self.vertices.len() {
            let e = Edge {
                label,
                start,
                end,
                capacity: cap as i64,
            };
            self.edges.push(e);
            let e_index = self.edges.len() - 1;
            self.vertices[start].out_edges.push(e_index);
            self.vertices[end].in_edges.push(e_index);
        } else if start >= self.vertices.len() {
            panic!("Invalid starting vertex");
        } else {
            panic!("Invalid ending vertex");
        }
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
            .map(|&e| self.edges[e].capacity)
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
            .map(|&e| self.edges[e].capacity)
            .sum::<i64>() as u64
    }

    fn out_edge_capacity_debug(&self, vertex: usize) -> u64 {
        assert!(vertex < self.vertex_count());

        self.vertices[vertex]
            .out_edges
            .iter()
            .map(|&e| {
                println!("{:?}", self.edges[e]);
                self.edges[e].capacity
            })
            .sum::<i64>() as u64
    }

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

    fn compute_connected_components(&mut self, source: usize, sink: usize) {
        if source >= self.vertices.len() {
            panic!("Invalid source");
        }
        if sink >= self.vertices.len() {
            panic!("Invalid sink");
        }
        self.vertices[source].component = Some(usize::MAX);
        self.vertices[sink].component = Some(usize::MAX);

        let mut component_index = 0;

        for v_index in 0..self.vertices.len() {
            if self.vertices[v_index].component.is_some() {
                continue;
            }
            // no component
            let mut queue = VecDeque::<usize>::with_capacity(self.vertices.len());
            queue.push_front(v_index);

            while let Some(vi) = queue.pop_front() {
                if self.vertices[vi].component.is_some() {
                    continue;
                }
                self.vertices[vi].component = Some(component_index);

                let out_edges_index = &(self.vertices[vi].out_edges);

                for e in out_edges_index {
                    let dest = self.edges[*e].end;
                    if self.vertices[dest].component.is_none() {
                        queue.push_front(dest)
                    }
                }
            }
            component_index += 1;
        }
        self.connected_components_count = component_index;
        // println!("Number of connected components: {}", component_index);
    }

    // Takes a source and a sink and finds a path from the source to the sink.
    // The path is (optionally) returned as the list of edges leading from the
    // source to the sink, together with the capacity of the path
    fn find_path(
        &self,
        source: usize,
        sink: usize,
        alg: TraversalAlgorithm,
    ) -> Option<(Vec<usize>, i64)> {
        if source >= self.vertices.len() {
            panic!("Invalid source");
        }
        if sink >= self.vertices.len() {
            panic!("Invalid sink");
        }
        let mut visited = vec![false; self.vertices.len()];
        let mut parent_edge: Vec<Option<usize>> = vec![None; self.vertices.len()];
        let mut queue = VecDeque::<usize>::with_capacity(self.vertices.len());

        queue.push_front(source);
        visited[source] = true;

        // println!("Source {}", source);
        // println!("Sink {}", sink);

        let mut found_sink = false; // flag for early exit
                                    // as we have two nested loops (and we do not want to use gotos -- this
                                    // is Rust after all), this flag is necessary
        while !found_sink {
            match queue.pop_front() {
                None => {
                    // println!("\nEmpty queue");
                    break;
                }
                Some(v) => {
                    // println!("\nVertex {}", v);

                    let out_edges_index = &(self.vertices[v].out_edges);

                    for e in out_edges_index {
                        // println!("Edge {}", *e);
                        if self.edges[*e].capacity > 0 {
                            let dest = self.edges[*e].end;
                            if !visited[dest] {
                                match alg {
                                    TraversalAlgorithm::BreadthFirstSearch => queue.push_back(dest),
                                    TraversalAlgorithm::DepthFirstSearch => queue.push_front(dest),
                                }
                                parent_edge[dest] = Some(*e);
                                visited[dest] = true;
                                if dest == sink {
                                    found_sink = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        if visited[sink] {
            // it is faster to compute the size of the vector, and then
            // construct it, than it is to rely on multiple dynamic allocations
            let mut capacity = std::i64::MAX;
            let mut v = sink;
            let mut size = 0;
            loop {
                match parent_edge[v] {
                    None => break,
                    Some(e) => {
                        v = self.edges[e].start;
                        capacity = capacity.min(self.edges[e].capacity);
                        size += 1;
                    }
                }
            }
            assert_ne!(size, 0);
            let mut path = vec![0; size];

            v = sink;
            let mut i = 0;

            loop {
                match parent_edge[v] {
                    None => break,
                    Some(e) => {
                        path[size - 1 - i] = e;
                        v = self.edges[e].start;

                        i += 1;
                    }
                }
            }

            Some((path, capacity))
        } else {
            None
        }
    }

    fn find_path_bfs(&self, source: usize, sink: usize) -> Option<(Vec<usize>, i64)> {
        self.find_path(source, sink, TraversalAlgorithm::BreadthFirstSearch)
    }

    fn find_path_dfs(&self, source: usize, sink: usize) -> Option<(Vec<usize>, i64)> {
        self.find_path(source, sink, TraversalAlgorithm::DepthFirstSearch)
    }

    fn compute_residual_max_flow(
        &mut self,
        source: usize,
        sink: usize,
        traversal_alg: TraversalAlgorithm,
    ) {
        // let mut max_flow = 0;
        let n_original_edges = self.edges.len() / 2;
        loop {
            match self.find_path(source, sink, traversal_alg) {
                None => break,
                Some((path, path_flow)) => {
                    for e in path {
                        // update capacities
                        self.edges[e].capacity -= path_flow;
                        // get the reversed edge
                        let rev_edge = if e < n_original_edges {
                            e + n_original_edges
                        } else {
                            e - n_original_edges
                        };
                        self.edges[rev_edge].capacity += path_flow;
                    }
                    // max_flow += path_flow;
                }
            }
        }

        // println!("Max flow {}", max_flow);
    }

    fn compute_max_flow(
        &self,
        source: usize,
        sink: usize,
        traversal_alg: TraversalAlgorithm,
    ) -> Graph {
        let mut g = Graph::new_residual_graph(self);
        g.compute_residual_max_flow(source, sink, traversal_alg);
        g.transform_residual_to_flow();
        g
    }

    fn transform_residual_to_flow(&mut self) {
        assert_eq!(self.edges.len() % 2, 0);

        let n_real_edges = self.edges.len() / 2;

        for v in self.vertices.iter_mut() {
            v.in_edges = v
                .in_edges
                .iter()
                .filter(|e| n_real_edges > **e)
                .map(|e| *e) // why do we need that ??
                .collect();

            v.out_edges = v
                .out_edges
                .iter()
                .filter(|e| n_real_edges > **e)
                .map(|e| *e)
                .collect();
        }

        // we would like to do this, but due to double mutable borrow in the closure, we have an issue
        // self.edges[..n_real_edges]
        //     .iter_mut()
        //     .enumerate()
        //     .for_each(|(i, mut e)| {
        //         e.capacity = self.edges[i + n_real_edges].capacity;
        //     });

        for i in 0..n_real_edges {
            self.edges[i].capacity = self.edges[i + n_real_edges].capacity;
        }

        // remove the now unnecessary edges
        self.edges.truncate(n_real_edges);
    }
}

fn generate_random_graph(n_vertices: usize, n_edges: usize) -> Graph {
    let mut graph = Graph::new();

    for i in 0..n_vertices {
        graph.add_vertex(i as u64);
    }

    let mut rng = thread_rng();

    for i in 0..n_edges {
        let s = rng.gen_range(0, n_vertices);
        let mut e = rng.gen_range(0, n_vertices);

        while e == s {
            // avoid loops
            e = rng.gen_range(0, n_vertices);
        }
        graph.add_edge(i as u64, s, e, 1);
    }

    graph
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ListGenerationMethod {
    RandomGeneration,
    WorstCaseGeneration,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EdgeOrientation {
    RandomOrientation,
    LeastChargedOrientation,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum LocationGeneration {
    FullyRandom,
    HalfRandom,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MaxFlowAllocExperimentParams {
    pub n: usize,
    pub m: usize,
    pub list_max_len: usize,
    pub bucket_capacity: usize,
    pub generation_method: ListGenerationMethod,
    pub edge_orientation: EdgeOrientation,
    pub location_generation: LocationGeneration,
}

#[derive(Debug, Clone, Copy, Serialize, Default)]
pub struct FlowAllocTimings {
    pub generation: u128,
    pub sink_source: u128,
    pub residual: u128,
    pub connected_components: u128,
    pub max_flow: u128,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaxFlowExperimentResult {
    pub size: usize,
    pub max_load: usize,
    // pub load_modes: Vec<usize>,
    pub stash_size: usize,
    pub connected_components: usize,
    pub timings: FlowAllocTimings,
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
fn generate_alloc_graph(params: MaxFlowAllocExperimentParams) -> (Graph, u64) {
    let mut remaining_elements = params.n;

    let mut rng = thread_rng();

    // create a new graph with m+2 vertices: one per bucket + a source and a sink
    // by convention, the source has index m and the sink has index m+1
    let mut graph = Graph::new_with_vertices(params.m + 2);

    let mut list_index: u64 = 0;

    while remaining_elements != 0 {
        let l: usize = match params.generation_method {
            ListGenerationMethod::RandomGeneration => {
                rng.gen_range(0, params.list_max_len.min(remaining_elements)) + 1
            }
            ListGenerationMethod::WorstCaseGeneration => {
                params.list_max_len.min(remaining_elements)
            }
        };

        let (h1, h2) = generate_location(&mut rng, params.m, params.location_generation);

        let mut start = h1;
        let mut end = h2;

        if params.edge_orientation == EdgeOrientation::LeastChargedOrientation
            && graph.out_edge_capacity(h1) < graph.out_edge_capacity(h2)
        {
            start = h2;
            end = h1;
        }
        // Adding a list of size l consists in adding an edge of capacity l
        // between to random vertices
        graph.add_edge(list_index, start, end, l as u64);

        remaining_elements -= l;
        list_index += 1;
    }

    (graph, list_index)
}

fn flow_alloc(
    params: MaxFlowAllocExperimentParams,
    timings: Option<&mut FlowAllocTimings>,
    connected_components_count: Option<&mut usize>,
) -> Vec<usize> {
    let mut times: FlowAllocTimings = Default::default();
    let start_gen = std::time::Instant::now();
    let (mut graph, list_count) = generate_alloc_graph(params);
    // assert!(graph.check_graph_correctness());

    // let base_graph = graph.clone();
    // println!("All edges:\n {:?} \n\n\n", base_graph.edges);

    let start_sink_source = std::time::Instant::now();
    // OK, now we have to add the edges originating from the source and the
    // ones ending at the sink to 'encode' overflowing and underflowing nodes.

    let mut additional_label = 2 * list_count;
    let bucket_capacity: u64 = params.bucket_capacity as u64;

    for v in 0..params.m {
        let out_count = graph.out_edge_capacity(v);

        if out_count > bucket_capacity {
            // this is an overflowing vertex
            // add capacity from the source
            graph.add_edge(additional_label, params.m, v, out_count - bucket_capacity);
            additional_label += 1;
        } else if out_count < bucket_capacity {
            // this is a vertex with remaining space
            // add capacity to the sink
            graph.add_edge(
                additional_label,
                v,
                params.m + 1,
                bucket_capacity - out_count,
            );
            additional_label += 1;
        }
    }
    // assert!(graph.check_graph_correctness());

    let start_residual = std::time::Instant::now();

    let mut rff = Graph::new_residual_graph(&graph);

    let start_connected_components = std::time::Instant::now();

    rff.compute_connected_components(params.m, params.m + 1);

    // It is time to max flow!
    // We could do
    // let ff = graph.compute_max_flow(params.m, params.m + 1, TraversalAlgorithm::DepthFirstSearch);
    // and then flip the edges in base_graph that carry flow (in ff)
    // Yet, this is exactly reconstructing the last residual graph of the
    // Ford-Fulkerson algorithm. So, we only compute this graph for now
    let start_ff = std::time::Instant::now();

    rff.compute_residual_max_flow(params.m, params.m + 1, TraversalAlgorithm::DepthFirstSearch);

    let end_ff = std::time::Instant::now();

    times.generation = start_sink_source.duration_since(start_gen).as_nanos();
    times.sink_source = start_residual.duration_since(start_sink_source).as_nanos();
    times.residual = start_connected_components
        .duration_since(start_residual)
        .as_nanos();
    times.connected_components = start_ff
        .duration_since(start_connected_components)
        .as_nanos();
    times.max_flow = end_ff.duration_since(start_ff).as_nanos();
    // assert!(rff.check_graph_correctness());

    if let Some(timings) = timings {
        *timings = times;
    }

    if let Some(components_count) = connected_components_count {
        *components_count = rff.connected_components_count;
    }
    // Now, we can easily compute the load of each bucket.
    // We must be careful to remove the edges whose end are the sink or the
    // source from the load computation
    let res: Vec<usize> = (0..params.m)
        .map(|v| {
            rff.vertices[v]
                .out_edges
                .iter()
                .filter(
                    |&&e| rff.edges[e].end < params.m, // this predicates returns true iff rff.edges[e] is an edge whose end is in the graph
                )
                .map(|&e| rff.edges[e].capacity)
                .sum::<i64>() as usize
        })
        .collect();

    res
}

pub fn run_experiment(params: MaxFlowAllocExperimentParams) -> MaxFlowExperimentResult {
    let mut timings: FlowAllocTimings = Default::default();
    let mut connected_components: usize = 0;
    let rand_alloc = flow_alloc(params, Some(&mut timings), Some(&mut connected_components));
    let size = rand_alloc.iter().sum();
    let max_load = rand_alloc.iter().fold(0, |max, x| max.max(*x));
    let load_modes = compute_modes(rand_alloc.into_iter(), max_load);
    let stash_size = compute_overflow_stat(load_modes.iter(), params.bucket_capacity);

    MaxFlowExperimentResult {
        size,
        max_load,
        // load_modes,
        stash_size,
        connected_components,
        timings,
    }
}

pub fn iterated_experiment<F>(
    params: MaxFlowAllocExperimentParams,
    iterations: usize,
    show_progress: bool,
    iteration_progress_callback: F,
) -> Vec<MaxFlowExperimentResult>
where
    F: Fn(usize) + Send + Sync,
{
    // println!(
    // "{} one choice allocation iterations with N={}, m={}, max_len={}",
    // iterations, n, m, max_len
    // );

    let elements_pb = ProgressBar::new((iterations * params.n) as u64);
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

    let results: Vec<MaxFlowExperimentResult> = (0..iterations)
        .into_par_iter()
        .map(|_| {
            let r: MaxFlowExperimentResult = run_experiment(params);
            if show_progress {
                elements_pb.inc(params.n as u64);
            }
            iteration_progress_callback(params.n);

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
