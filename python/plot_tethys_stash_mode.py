#! python3

##
# Plot the experimental results of the buckets overflows
# The script takes a mandatory argument: the input file
# You can also specify the label to use when plotting by using the -l (or
# --label) option. The acceptable values are "m", "n", "n/m", "max_len" and
# "algorithm"

import argparse
import matplotlib.pyplot as plt
import json
import math
import csv
import itertools


def plot_file(filename, label, normalize=False, plot=True, logx=False, logy=False):
    with open(filename) as source:
        data = json.load(source)
        x = []
        y_max = []
        y_avg = []

        rows = []

        for experiment in data:  # iterate over the experiments
            n = float(experiment["parameters"]["exp_params"]["n"])
            m = float(experiment["parameters"]["exp_params"]["m"])
            p = int(experiment["parameters"]
                    ["exp_params"]["bucket_capacity"])

            iterations = int(experiment["parameters"]["iterations"])

            if label == "n/m":
                l = n / m
            elif label == "epsilon":
                l = (m*p/n) - 2
            else:
                l = float(experiment["parameters"]["exp_params"][label])

            norm_factor = 1

            if (normalize):
                norm_factor = n

            probs = [i/(norm_factor*iterations)
                     for i in experiment["stash_modes"][0:: p]]
            plt.plot(probs, label="n=%d" % (n))

            probs.insert(0, int(l))

            rows.append(probs)
            # x.append(l)

            # stash_max = float(experiment["stash_size"]["max"])
            # stash_avg = float(experiment["stash_size"]["mean"])

            # if normalize:
            # stash_avg /= n
            # stash_max /= n

            # y_max.append(stash_max)
            # y_avg.append(stash_avg)

        if plot:
            # plt.plot(x, y_max, label="Max stash size")
            # plt.plot(x, y_avg, label="Average stash size")
            plt.legend(loc='upper right')

            if logx:
                plt.semilogx()
            if logy:
                plt.semilogy()

            plt.show()

        return rows


parser = argparse.ArgumentParser(description='Plot allocation overflows.')
parser.add_argument('filename', metavar='path',
                    help='Path to a JSON file')
parser.add_argument('--label', '-l', default='n',
                    help='Define the used label')
parser.add_argument('--normalize', '-n', action='store_true')
parser.add_argument('--logx', action='store_true')
parser.add_argument('--logy', action='store_true')
parser.add_argument('--no-plot', action='store_true')
parser.add_argument('--out', '-o', metavar='path', default=None,
                    help='Output csv data file', required=False)

args = parser.parse_args()
# print(args)

rows = plot_file(args.filename, args.label, normalize=args.normalize,
                 logx=args.logx, logy=args.logy,  plot=(not args.no_plot))
transposed_rows = list(
    map(list, itertools.zip_longest(*rows, fillvalue=0)))


if args.out:
    with open(args.out, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(transposed_rows)
