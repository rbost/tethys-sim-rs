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

from scipy.optimize import curve_fit


def f(x, a, b):
    return b*(x**a)


def plot_file(filename, label, normalize=False, logx=False, logy=False):
    with open(filename) as source:
        data = json.load(source)
        x = []
        y_max = []
        y_avg = []
        x_square = []
        x_alpha = []
        alpha = 1.5
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

            time_max = float(experiment["timings"]["max_flow"]["max"])
            time_avg = float(experiment["timings"]["max_flow"]["mean"])

            if normalize:
                time_avg /= n
                time_max /= n

            x_square.append(l*l)
            x_alpha.append(l**alpha)
            y_max.append(time_max)
            y_avg.append(time_avg)

        popt, pcov = curve_fit(f, x, y_max, bounds=([1., 0.], [2., 5000.]))

        # plt.plot(x, x_alpha, label="x^alpha")
        # plt.plot(x, x_square, label="x^2")
        plt.plot(x, f(x, *popt), 'g--',
                 label='fit: y=b*(x^a), a=%5.3f, b=%5.3f' % tuple(popt))
        plt.plot(x, y_max, label="Max max flow time")
        plt.plot(x, y_avg, label="Average max flow time")
        plt.legend(loc='lower right')

        if logx:
            plt.semilogx()
        if logy:
            plt.semilogy()

        plt.show()


parser = argparse.ArgumentParser(description='Plot allocation overflows.')
parser.add_argument('filename', metavar='path',
                    help='Path to a JSON file')
parser.add_argument('--label', '-l', default='n',
                    help='Define the used label')
parser.add_argument('--normalize', '-n', action='store_true')
parser.add_argument('--logx', action='store_true')
parser.add_argument('--logy', action='store_true')


args = parser.parse_args()
# print(args)

plot_file(args.filename, args.label, normalize=args.normalize,
          logx=args.logx, logy=args.logy)
