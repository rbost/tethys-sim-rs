#![allow(dead_code)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ListGenerationMethod {
    RandomGeneration,
    WorstCaseGeneration,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AllocExperimentParams {
    pub n: usize,
    pub m: usize,
    pub list_max_len: usize,
    pub bucket_capacity: usize,
    pub generation_method: ListGenerationMethod,
}
