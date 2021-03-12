#![allow(dead_code)]

// use std::collections::VecDeque;
extern crate rand;
// use rand::prelude::*;

// use rayon::prelude::*;
// use std::sync::atomic::{AtomicUsize, Ordering};

use std::slice::Iter;

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

    fn push_edge(&mut self, edge_index: usize) {}

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

        self.vertices[vertex].in_edges.iter().map(|_| 1u64).sum()
        // .map(|&e| self.edges[e].capacity)
        // .sum::<i64>() as u64
    }

    fn out_edge_count(&self, vertex: usize) -> usize {
        assert!(vertex < self.vertex_count());

        self.vertices[vertex].out_edges.len()
    }

    fn out_edge_capacity(&self, vertex: usize) -> u64 {
        assert!(vertex < self.vertex_count());

        self.vertices[vertex].out_edges.iter().map(|_| 1u64).sum()
        // .map(|&e| self.edges[e].capacity)
        // .sum::<i64>() as u64
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
