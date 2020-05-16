#![allow(dead_code)]

mod utils;
pub use crate::utils::*;

// use gnuplot::*;

extern crate csv;
extern crate serde_json;
use serde::{Deserialize, Serialize};

use rayon::prelude::*;

use std::convert::TryInto;
use std::fs::File;
use std::io;
use std::io::{BufReader, BufWriter};
use std::path::Path;

extern crate structopt;
use structopt::StructOpt;

mod max_flow;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "max_flow_alloc",
    about = "Max flow-base variable length list allocation."
)]
struct CliArgs {
    #[structopt(parse(from_os_str), short = "c", long = "config")]
    /// Path to a JSON configuration file. See "example_config.json" for an example
    config_path: std::path::PathBuf,
    #[structopt(
        parse(from_os_str),
        short = "o",
        long = "output",
        default_value = "results"
    )]
    /// Path for the output statistics of the experiments. A JSON and two CSV files (one for the load, the other for the space) will be generated
    output_path: std::path::PathBuf,
    #[structopt(short = "g", long = "gnuplot")]
    gnuplot: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InteratedMaxFlowAllocExperimentParams {
    pub exp_params: max_flow::MaxFlowAllocExperimentParams,
    pub iterations: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MaxFlowAllocTimingStats {
    pub generation: crate::utils::Stats,
    pub sink_source: crate::utils::Stats,
    pub max_flow: crate::utils::Stats,
}
#[derive(Debug, Clone, Serialize)]
pub struct MaxFlowAllocStats {
    pub parameters: InteratedMaxFlowAllocExperimentParams,
    pub size: crate::utils::Stats,
    pub load: crate::utils::Stats,
    pub stash_size: crate::utils::Stats,
    // pub load_modes: Vec<crate::utils::ModeStats>,
    pub stash_modes: Vec<usize>,
    pub connected_components: crate::utils::Stats,
    pub timings: MaxFlowAllocTimingStats,
}

fn read_config_file<P: AsRef<Path>>(
    path: P,
) -> io::Result<Vec<InteratedMaxFlowAllocExperimentParams>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let params = serde_json::from_reader(reader)?;

    Ok(params)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::from_args();

    println!("{:?}", args);

    let inputs = read_config_file(args.config_path)?;

    let stats = run_experiments_stats(&inputs);

    // write_load_stats_csv(&stats, args.output_path.with_extension("load.csv"))?;
    // write_size_stats_csv(&stats, args.output_path.with_extension("size.csv"))?;
    write_stats_json(&stats, args.output_path.with_extension("json"))?;

    Ok(())
}

fn run_experiments_stats(
    inputs: &[InteratedMaxFlowAllocExperimentParams],
) -> Vec<MaxFlowAllocStats> {
    let tot_iterations: usize = inputs.iter().map(|p| p.iterations).sum();
    let tot_elements: usize = inputs.iter().map(|p| p.iterations * p.exp_params.n).sum();

    let iter_completed = std::sync::atomic::AtomicUsize::new(0);

    let pb = indicatif::ProgressBar::new(tot_elements as u64);

    pb.set_style(indicatif::ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] {msg} [{bar:40.cyan/blue}] ({pos}/{len} elts - {percent}%) | ETA: {eta_precise}")
        .progress_chars("##-"));
    pb.set_message(&format!("{}/{} iterations", 0, tot_iterations));
    pb.enable_steady_tick(1000);

    let iteration_progress_callback = |n: usize| {
        let previous_count = iter_completed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        pb.set_message(&format!(
            "{}/{} iterations",
            previous_count + 1,
            tot_iterations
        ));
        pb.inc(n as u64);
    };

    let results = inputs
        .into_par_iter()
        .map(|p| {
            (
                p,
                max_flow::iterated_experiment(
                    p.exp_params,
                    p.iterations,
                    false,
                    iteration_progress_callback,
                ),
            )
        })
        .map(|(p, results)| {
            let load_stat = compute_stats(results.iter().map(|x| x.max_load));
            let stash_stat = compute_stats(results.iter().map(|x| x.stash_size));
            MaxFlowAllocStats {
                parameters: *p,
                size: compute_stats(results.iter().map(|x| x.size)),
                load: load_stat,
                stash_size: stash_stat,
                // load_modes: compute_modes_stat(
                // results.iter().map(|x| &x.load_modes),
                // load_stat.max.try_into().unwrap(),
                // ),
                stash_modes: compute_modes(
                    results.iter().map(|x| x.stash_size),
                    stash_stat.max.try_into().unwrap(),
                ),
                connected_components: compute_stats(results.iter().map(|x| x.connected_components)),
                timings: MaxFlowAllocTimingStats {
                    generation: compute_stats_u128(results.iter().map(|x| x.timings.generation)),
                    sink_source: compute_stats_u128(results.iter().map(|x| x.timings.sink_source)),
                    max_flow: compute_stats_u128(results.iter().map(|x| x.timings.max_flow)),
                },
            }
        })
        .collect();

    pb.finish();
    results
}

fn write_stats_json<P: AsRef<Path>>(stats: &[MaxFlowAllocStats], path: P) -> io::Result<()> {
    let f_json = File::create(path)?;
    serde_json::to_writer_pretty(&f_json, &stats)?;
    Ok(())
}
