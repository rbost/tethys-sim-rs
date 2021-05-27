#! python3


import json
import math
import copy
import argparse

p = 512
iterations = 100000
epsilon = 0.1

n_values = [2**22, 2**21, 2**20, 2**19, 2**18, 2**17, 2**16, 2**15]

n_m_pairs = [(n, math.ceil(n * (2 + epsilon)/p)) for n in n_values]

print("Values for n:")
print(n_values)

print("Values for m:")
print([m for (_, m) in n_m_pairs])


base_params_dict = dict()
base_params_dict["bucket_capacity"] = p
base_params_dict["list_max_len"] = p
base_params_dict["generation_method"] = "RandomGeneration"
base_params_dict["eviction_strategy"] = "LeastChargedFullNonRec"
base_params_dict["location_generation"] = "HalfRandom"

base_dict = dict()
base_dict["exp_params"] = base_params_dict
base_dict["iterations"] = iterations

data = list()

for (n, m) in n_m_pairs:
    exp_dict = copy.deepcopy(base_dict)
    exp_dict["exp_params"]["n"] = n
    exp_dict["exp_params"]["m"] = m
    data.append(exp_dict)


parser = argparse.ArgumentParser(
    description='Maxflow configuration generator (constant epsilon, variable n)')
parser.add_argument('filename', metavar='path',
                    help='Path of the output JSON file')

args = parser.parse_args()

with open(args.filename, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
