#! python3

##
# Plot the experimental results of the buckets overflows
# The script takes a mandatory argument: the input file
# You can also specify the label to use when plotting by using the -l (or
# --label) option. The acceptable values are "m", "n", "n/m", "max_len" and
# "algorithm"

import argparse
import json
import math
import csv

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def f(x, a):
    return a/np.log2(x)


def f_norm(x, a):
    return a/(x*np.log2(x))


def exp_fit(x):
    return 72684 / (x*np.log2(x))


def process(filename, label, normalize=False, plot=True, logx=False, logy=False):
    with open(filename) as source:
        data = json.load(source)

        confidence_param = 3
        x = []
        y_max = []
        y_avg = []
        y_std_dev = []
        y_upper_hull = []
        y_lower_hull = []

        rows = []

        for experiment in data:  # iterate over the experiments
            n = float(experiment["parameters"]["exp_params"]["n"])
            m = float(experiment["parameters"]["exp_params"]["m"])
            p = float(experiment["parameters"]
                      ["exp_params"]["bucket_capacity"])

            if label == "n/m":
                l = n / m
            elif label == "epsilon":
                l = (m*p/n) - 2
            else:
                l = float(experiment["parameters"]["exp_params"][label])

            x.append(l)

            stash_max = float(experiment["stash_size"]["max"])
            stash_avg = float(experiment["stash_size"]["mean"])
            stash_avg_std_dev = math.sqrt(
                float(experiment["stash_size"]["variance"]))

            if normalize:
                stash_avg /= n
                stash_max /= n
                stash_avg_std_dev /= n

            y_max.append(stash_max)
            y_avg.append(stash_avg)
            y_std_dev.append(stash_avg_std_dev)

            y_up = stash_avg + confidence_param * \
                stash_avg_std_dev / math.sqrt(n)
            y_down = max(0, stash_avg - confidence_param *
                         stash_avg_std_dev / math.sqrt(n))

            y_upper_hull.append(y_up)
            y_lower_hull.append(y_down)

            rows.append(
                [l, stash_avg, stash_avg_std_dev, y_up, y_down, stash_max])

        fit_func = f

        if normalize:
            fit_func = f_norm

        popt, pcov = curve_fit(fit_func, x[:-1], y_max[:-1])

        if plot:
            plt.plot(x, y_max, label="Max stash size")
            plt.plot(x, y_avg, label="Average stash size")
            # plt.errorbar(x, y_avg, yerr=y_std_dev,
            #  label="Average stash size")
            # plt.plot(x, y_upper_hull, label="Average stash size")
            # plt.plot(x, y_lower_hull, label="Average stash size")
            plt.fill_between(x, y_upper_hull, y_lower_hull, alpha=0.2)

            fit_label = 'fit: y=a/log(x), a=%5.3f' % tuple(popt)

            if normalize:
                fit_label = 'fit: y=a/(x log(x)), a=%5.3f' % tuple(popt)

            # plt.plot(x, f(x, *popt), 'g--',
                #  label=fit_label)

            # plt.plot(x, exp_fit(x), 'r--',
            #  label="expected")

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

rows = process(args.filename, args.label, normalize=args.normalize,
               logx=args.logx, logy=args.logy, plot=(not args.no_plot))

if args.out:
    with open(args.out, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(rows)
