#! python3

"""
Generate JSON configuration files for the Tethys simulation code.
This script generates experimental settings with a constant p and m, as
well as a fixed number of iterations. Other parameters, such as the generation
method of the list (more precisely the lists' length), the initial orientation
of the edges in the graph, and the choice of the starting and ending edges, are
also fixed and can be customized.
The values for epsilon are given as a list (see other comments in the code) and
the values for n are derived from these.
"""

import json
import math
import copy
import argparse

m = 2**10
p = 512
iterations = 6000000

# the different values for epsilon
epsilons = [-0.35, - 0.3, - 0.25, - 0.2, - 0.15, - 0.1, - 0.05, 0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14,
            0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5]

# values of n computed from the above list and parameters
n_values = [math.ceil(m * p / (2 + e)) for e in epsilons]

print("Values for epsilon:")
print(epsilons)

print("Values for n:")
print(n_values)


base_params_dict = dict()
base_params_dict["m"] = m
base_params_dict["bucket_capacity"] = p
base_params_dict["list_max_len"] = p
base_params_dict["generation_method"] = "WorstCaseGeneration"
base_params_dict["edge_orientation"] = "RandomOrientation"
base_params_dict["location_generation"] = "HalfRandom"

base_dict = dict()
base_dict["exp_params"] = base_params_dict
base_dict["iterations"] = iterations

data = list()

for n in n_values:
    exp_dict = copy.deepcopy(base_dict)
    exp_dict["exp_params"]["n"] = n
    data.append(exp_dict)


parser = argparse.ArgumentParser(
    description='Maxflow configuration generator (variable epsilon)')
parser.add_argument('filename', metavar='path',
                    help='Path of the output JSON file')

args = parser.parse_args()

with open(args.filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
