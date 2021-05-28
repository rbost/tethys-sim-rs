# Dataless simulation of the TethysDIP algorithm

[![Build Status](https://img.shields.io/github/workflow/status/rbost/tethys-sim-rs/Rust)](https://github.com/rbost/tethys-sim-rs/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



This is the supplementary material of _SSE and SSD: Page-Efficient Searchable Symmetric Encryption_ by [Bossuat](https://people.irisa.fr/Angele.Bossuat/), [Bost](https://raphael.bost.fyi/), [Fouque](https://www.di.ens.fr/~fouque/), [Minaud](https://www.di.ens.fr/~bminaud) and Reichle published at [Crypto 2021](https://crypto.iacr.org/2021/).
This archive contains some of the code used for the evaluation of the algorithms in the paper, notably the _TethysCore_ allocation algorithm.

This implementation of TethysCore is dataless because it does not use data to run: it is just a simulation to evaluate the distribution of the stash size and the asymptotical running time of the algorithm.

## Build & Use

You will need a functional rust installation to build the code. See the [Rust's documentation](<https://www.rust-lang.org/tools/install>) for details on how to install.

To compile (in dev mode):
```
cargo build
```

To run (in release mode):
```
cargo run --release --bin tethys_core_allocation -- -c input_configuration.json -o output_file
```
This will use the JSON configuration file `input_configuration.json` and run several experiments, whose results will be written to `output_file.json`. The JSON file contains all the results of the experiments.

The configuration file is a list of the following key-value pairs:
```json
{
    "exp_params": {
      "n": 4096,
      "m": 512,
      "list_max_len": 32,
      "bucket_capacity": 32,
      "generation_method": "WorstCaseGeneration",
      "edge_orientation": "RandomOrientation",
      "location_generation": "FullyRandom"
    },
    "iterations": 100
}
```
where `n` is the number of elements to insert, `m` the number of buckets, `list_max_len` the maximum list length, `bucket_capacity` is the maximum number of entries per bucket, `generation_method` is the way the data set is generated (for now, only the `"WorstCaseGeneration"`, `"RandomGeneration"` algorithms are supported), `edge_orientation` parametrizes the way edges are oriented (choose between `"RandomOrientation"` and `"LeastChargedOrientation"`), `location_generation` changes the way the hash functions behave (`"FullyRandom"` picks two locations fully at random, and `"HalfRandom"` picks one location in each half of the allocation array) and `iterations` is the number of experiments to run with these parameters.
The file [`max_flow_config.json`](max_flow_config.json
) gives an example of such configuration file.


## Python plotting scripts

In the `python` directory, will find scripts which can process the experiments' JSON output and plot different graphs (stash modes, stash size, allocation timing).
They essentially all have the same usage and options:
* pass the path of the JSON file to process as the first input
* `--logx`, `--logy`: use log scale.
* `--label label` or `-l label`: use `label` as the `x` values. `label` can be `n`, `m`, `n/m`, or `epsilon`.
* `--normalize` or `-n`: divide the value of interest by `n`.
* `--out stats.csv` or `-o stats.csv`: output the statistics as as CSV file (put the result in `stats.csv`).

## Configuration generation scripts

In the `python` directory, you will also find two scripts to generate the configuration files used in the experiments (`tethys_config_gen_const_eps.py` and `tethys_config_gen_epsilon.py`).

## `fio` configuration file

`sse_workloads.fio` is the [`fio`](https://github.com/axboe/fio) configuration file we used for the evaluation. To run it on your computer, just call `fio sse_workloads.fio`. Be careful: it will write a 32GB on your disk.