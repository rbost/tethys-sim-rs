#! python3

"""
Plot the experimental results of the buckets overflows
The script takes a mandatory argument: the input file
You can also specify the label to use when plotting by using the -l (or
--label) option. The acceptable values are "m", "n", "n/m", "max_len" and
"algorithm"
"""

import argparse
import json
import csv
import itertools

import matplotlib.pyplot as plt


def get_plot_rows(filename, label, normalize=False, plot=True, logx=False, logy=False):
    """Compute the stash mode expectancies, plot them and return them as rows
    (one row per experiment)."""
    with open(filename) as source:
        data = json.load(source)

        rows = []

        for experiment in data:  # iterate over the experiments
            n = float(experiment["parameters"]["exp_params"]["n"])
            m = float(experiment["parameters"]["exp_params"]["m"])
            p = int(experiment["parameters"]
                    ["exp_params"]["bucket_capacity"])

            iterations = int(experiment["parameters"]["iterations"])

            # 'Compute' the label
            if label == "n/m":
                l = n / m
            elif label == "epsilon":
                l = (m*p/n) - 2
            else:
                l = float(experiment["parameters"]["exp_params"][label])

            # Compute the normalization factor
            norm_factor = 1

            if normalize:
                norm_factor = n

            # Compute the modes as expectancies (or probabilities when using a
            # normalization factor)
            probs = [i/(norm_factor*iterations)
                     for i in experiment["stash_modes"][0:: p]]

            # Add the probabilites to the graph (labeled by the n variable)
            plt.plot(probs, label="n=%d" % (n))

            # Put the label at the beginning of the row
            # (this is done for the CSV output)
            probs.insert(0, int(l))

            rows.append(probs)

        # Display the plot if requested
        if plot:
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

plot_rows = get_plot_rows(args.filename, args.label, normalize=args.normalize,
                          logx=args.logx, logy=args.logy,  plot=(not args.no_plot))

if args.out:
    # Output the result as a CSV file
    # In that case, rows need to be turned into columns, all of the same length
    transposed_rows = list(
        map(list, itertools.zip_longest(*plot_rows, fillvalue=0)))

    with open(args.out, 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(transposed_rows)
