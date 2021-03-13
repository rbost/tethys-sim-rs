#![allow(dead_code)]

// use std::collections::VecDeque;
extern crate rand;
use rand::prelude::*;

// use rayon::prelude::*;
// use std::sync::atomic::{AtomicUsize, Ordering};

use std::{slice::Iter, u64};

use std::vec::Vec;

// use indicatif::{ProgressBar, ProgressStyle};
// use serde::{Deserialize, Serialize};

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
            self.push_edge_cuckoo(rev_edge_index, iteration_depth + 1);
        }
    }

    fn add_edge(
        &mut self,
        label: u64,
        start: usize,
        end: usize, // , cap: u64
    ) {
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
