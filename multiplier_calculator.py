#!/usr/bin/env python3

import argparse
import ast
import csv
import json
import os.path
from datetime import datetime, timedelta
import re
from decimal import Decimal as D
import logging as log
import math
from prettytable import PrettyTable
from urllib.request import urlopen
import psycopg2
import getpass

COUNTRY_MAPPING_URL = "https://git.xx.network/xx_network/primitives/-/raw/release/region/country.go"
DATE_FORMAT_STRING = "%Y-%m-%d %H:%M"


def get_args():
    """
    get_args controls the argparse usage for the script.  It sets up and parses
    arguments and returns them in dict format
    """
    parser = argparse.ArgumentParser(description="Options for raw point parsing script")
    parser.add_argument("--verbose", action="store_true",
                        help="Print debug logs", default=False)
    parser.add_argument("--log", type=str,
                        help="Path to output log information",
                        default="/tmp/mult_calc.log")
    parser.add_argument("--raw-points-log", type=str,
                        help="Path to output log information",
                        default="/tmp/raw-points.log")
    default_lower = (datetime.now() - timedelta(days=14)).strftime(DATE_FORMAT_STRING)
    parser.add_argument("--lower-bound", type=str,
                        help="Timestamp for lower bound of logs to ingest",
                        default=default_lower)
    default_upper = datetime.now().strftime(DATE_FORMAT_STRING)
    parser.add_argument("--upper-bound", type=str,
                        help="Timestamp for upper bound of logs to ingest",
                        default=default_upper)
    parser.add_argument("--wallet-country-supplement", type=str,
                        help="Supplemental file containing old mappings of wallet to country code")
    parser.add_argument("--json", type=str, help="Path for JSON output")
    parser.add_argument("--csv", type=str, help="Path for CSV output")
    parser.add_argument("--historical", action='store_true', help="Run historical rounds")
    parser.add_argument("-a", "--host", metavar="host", type=str,
                        help="Database server host for attempted connection")
    parser.add_argument("-p", "--port", type=int,
                        help="Port for database connection")
    parser.add_argument("-d", "--db", type=str,
                        help="Database name")
    parser.add_argument("-U", "--user",  type=str,
                        help="Username for connecting to database")

    args = vars(parser.parse_args())
    log.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',
                    level=log.DEBUG if args['verbose'] else log.INFO,
                    datefmt='%d-%b-%y %H:%M:%S',
                    filename=args["log"])

    print(args)
    if args["host"]:
        args["pass"] = getpass.getpass("Enter database password: ")

    if args["lower_bound"]:
        args["lower_bound"] = datetime.strptime(args["lower_bound"], DATE_FORMAT_STRING)
    if args["upper_bound"]:
        args["upper_bound"] = datetime.strptime(args["upper_bound"], DATE_FORMAT_STRING)

    return args


def get_raw_points_lines(raw_points_path):
    """
    Gets list of all lines from raw points logs at path
    Can accept either single log file or directory of log files
    :param str raw_points_path: path to raw points logs
    :return: list[str]
    """
    if os.path.isfile(raw_points_path):
        with open(raw_points_path, 'r') as f:
            raw_points_lines = f.readlines()
    elif os.path.isdir(raw_points_path):
        raw_points_lines = []
        for entry in os.listdir(raw_points_path):
            ep = os.path.join(raw_points_path, entry)
            if os.path.isfile(ep):
                with open(ep, 'r') as f:
                    raw_points_lines = raw_points_lines + f.readlines()
    return raw_points_lines


def get_countrycode_bin_map():
    """
    Parse a dict of country code to geo bins from raw git url containing the list
    :return: dict[country]geobin
    """
    page = urlopen(COUNTRY_MAPPING_URL)
    html_bytes = page.read()
    contents = html_bytes.decode('utf-8')
    l = contents.find("var countryBins")
    r = contents.find("}", l)
    lines = contents[l:r - 1].split("\n\t")[1:]
    countrycode_bin_dict = {}
    for line in lines:
        parts = line.split(":")
        countrycode_bin_dict[parts[0].strip('" ')] = parts[1].strip(', ')
    return countrycode_bin_dict


def get_wallet_bin_map(filepath):
    """
    get_wallet_bin_map reads a csv with lines in the format {wallet},{country}
    and creates a dict mapping wallets to country codes for use later in the process
    :param str filepath: path to supplemental wallet -> country mapping
    :return: dict[wallet]country
    """
    ret = {}
    with open(filepath, "r") as f:
        wallet_bin_lines = f.readlines()
        for line in wallet_bin_lines:
            split = line.split(",")
            wallet = split[0].strip()
            country = split[1].strip()
            ret[wallet] = country
    return ret


def first_pass(raw_points_lines, lower_bound, upper_bound, wallet_country_map, countrycode_bin_map):
    """
    first_pass runs preliminary calculations on the contents of the raw
    points log, using eras between lower_bound and upper_bound only
    :param list[string] raw_points_lines: Log lines from raw points logs to parse
    :param datetime lower_bound: discard any lines with timestamp before lower_bound
    :param datetime upper_bound: discard any lines with timestamp after upper_bound
    :param dict[str]str wallet_country_map: supplemental mapping of wallet to country code
    :param dict[str]str countrycode_bin_map: country code to geo bin associations
    :return: dict[bin]quantized average nodes, dict[bin]normalized average points
    """
    # Sanity check on bounds
    if lower_bound >= upper_bound:
        print(f"Lower bound {lower_bound} cannot be after upper bound {upper_bound}")
        exit(1)

    # Establish dictionaries to fill in first pass
    # The goal of this pass is averages, so the data needed for the end step is totals per bin & era count
    total_nodes_in_bin = {}
    total_points_in_bin = {}
    era_count = 0

    # Keep list of unmatched wallets & print a warning at the end if not empty
    unmatched_wallets = set()

    # Do first pass through lines
    for line in raw_points_lines:
        match = re.search(r"^\[\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d\d\d\d\d\d]", line)
        if not match:
            continue

        # Split line into timestamp & dictionary
        split_ind = match.span()[1]
        ts_raw = match.group(0).strip("[]")
        points_dict_raw = line[split_ind + 1:-1]

        # Parse timestamp to datetime object for easier use
        era = datetime.strptime(ts_raw, "%Y-%m-%d %H:%M:%S.%f")

        # Check that the timestamp is within established bounds
        if not lower_bound <= era <= upper_bound:
            continue

        # Use python abstract syntax tree lib to parse rest of line to dict
        points_dict = ast.literal_eval(points_dict_raw)
        era_nodes_in_bin = {"total": 0}  # Create era_bins entry for this period
        era_points_in_bin = {"total": 0}  # Create era_bin_points entry for this period
        era_bins = set()
        era_count += 1

        # There are some empty dicts at the beginning of the log, skip those if they would be in the period
        if points_dict == {}:
            continue

        # Loop through points dict for this period
        for wallet, val in points_dict.items():
            if type(val) == list:
                era_points = val[0]
                country = val[1]
            elif type(val) == int:
                era_points = val
                if wallet not in wallet_country_map:
                    unmatched_wallets.add(wallet)
                    country = "??"
                else:
                    country = wallet_country_map[wallet]
            else:
                print(f"Unknown type in points dict {val}")
                exit(1)

            if era_points == 0:
                continue
            # Get bin for country from dict parsed from git
            country_bin = countrycode_bin_map[country]

            era_bins.add(country_bin)

            # Increment era bins for this wallet
            if country_bin not in era_nodes_in_bin:  # Make sure entry is there
                era_nodes_in_bin[country_bin] = 0
            era_nodes_in_bin[country_bin] += 1  # Increment era bins for this wallet
            era_nodes_in_bin["total"] += 1

            # Increment total nodes per bin
            if country_bin not in total_nodes_in_bin:
                total_nodes_in_bin[country_bin] = 0
            total_nodes_in_bin[country_bin] += 1

            # Add this wallet's points to its era_bin_points total
            if country_bin not in era_points_in_bin:  # Make sure entry is there
                era_points_in_bin[country_bin] = 0
            era_points_in_bin[country_bin] += era_points  # Add val to country bin total
            era_points_in_bin["total"] += era_points

        # Do adjusted nodes & points calculations for each bin
        adjusted_nodes_in_bin = {}
        size_adjusted_points_in_bin = {"total": 0}
        total = era_nodes_in_bin["total"]
        for geo_bin in era_bins:
            # Fill adjusted nodes in bin for this era
            # {size-adjusted nodes in bin} = {nodes in bin} / {total nodes in era}
            node_count = era_nodes_in_bin[geo_bin]
            adjusted_nodes = node_count / total
            adjusted_nodes_in_bin[geo_bin] = adjusted_nodes

            # Do size adjustion for era points per bin
            # {adjusted bin points} = {points in era} / {adjusted nodes in bin}
            bin_points = era_points_in_bin[geo_bin]
            adjusted_points = bin_points / adjusted_nodes
            size_adjusted_points_in_bin[geo_bin] = adjusted_points
            size_adjusted_points_in_bin["total"] += adjusted_points

        # Normalize size adjusted numbers, accumulate totals for final average
        for geo_bin, era_points in size_adjusted_points_in_bin.items():
            if geo_bin == "total":
                continue
            era_points = size_adjusted_points_in_bin[geo_bin]
            # {normalized era points} = {adjusted era points} / {total adjusted era points}
            normalized_adjusted_era_points = era_points / size_adjusted_points_in_bin["total"]
            if geo_bin not in total_points_in_bin:
                total_points_in_bin[geo_bin] = 0
            total_points_in_bin[geo_bin] += normalized_adjusted_era_points

    # Warn if any unmatched wallets
    if len(unmatched_wallets) > 0:
        print(f"Unmatched wallets in log: {unmatched_wallets}")

    # Do base averages for nodes & points in bin
    # {average for bin} = {total for bin across eras} / {era count}
    bin_node_averages = {geo_bin: round(total_nodes / era_count) for geo_bin, total_nodes in total_nodes_in_bin.items()}
    bin_point_averages = {geo_bin: total_points / era_count for geo_bin, total_points in total_points_in_bin.items()}

    # Normalize bin point averages
    # {normalized average} = {average for bin} / {max bin average}
    max_avg = max(bin_point_averages.values())
    bin_point_averages_normalized = {geo_bin: point_avg / max_avg for geo_bin, point_avg in bin_point_averages.items()}

    # Return bin node averages & normalized bin point averages
    return bin_node_averages, bin_point_averages_normalized


def calculate_multipliers(bin_node_averages, bin_point_averages_normalized):
    """
    calculate_multipliers performs the main multiplier calculation on the two dicts returned by first_pass
    :param bin_node_averages:
    :param bin_point_averages_normalized:
    :return:
    """
    # Establish a few constants for use in this step of calculation
    total_nodes = sum(bin_node_averages.values())
    team_size = 5
    max_combin = math.comb(total_nodes, team_size)

    # Order bins by normalized point averages
    sorted_normalized_averages = dict(sorted(bin_point_averages_normalized.items(), key=lambda item: item[1]))
    ordered_bins = list(sorted_normalized_averages.keys())

    # Create dictionaries to store intermediary steps of calculation
    cumulative_bins = {}
    probs = {}
    p_sums = {}
    factors = {}

    # Dict of multipliers to be returned by this step
    multipliers = {}

    # Calculation happens in order of bins, from lowest normalized avg points to greatest
    for i in range(len(ordered_bins)):
        bin_name = ordered_bins[i]

        # Number of total nodes minus this bin & all previous bins

        # {cumulative nodes in bin} = {total nodes} - {nodes in this bin} - {nodes in previous bins}
        if i == 0:
            cumulative_nodes = total_nodes - bin_node_averages[bin_name]
        else:
            cumulative_nodes = cumulative_bins[ordered_bins[i-1]] - bin_node_averages[bin_name]
        cumulative_bins[bin_name] = cumulative_nodes

        # Run combinations for all possible node counts from 1 to team_size
        # NOTE: math.comb(n, k) evaluates to n! / (k! * (n - k)!) when k <= n and evaluates to zero when k > n
        # for team sizes through 5:
        #   comb(nodes in bin, team size) * comb(cumulative nodes in bin, team size - 1)
        combs = [math.comb(bin_node_averages[bin_name], j) * math.comb(cumulative_nodes, team_size - j)
                 for j in range(1, team_size + 1)]

        # Get probability of selecting >=1 node from this bin, 0 from all previous
        # We start converting floats to Decimal here to avoid floating point errors
        # bin probability = sum(combinations for team sizes 1-5) / comb(total_nodes, team_size)
        probs[bin_name] = D(sum(combs)) / D(max_combin)

        # Calculate bin multiplier
        # See https://xx.network/archive/regionalmultipliers/ for more information on this calculation
        bin_point_avg = D(bin_point_averages_normalized[bin_name])
        if i == 0:
            # First bin has special calculations for these values, since rest rely on previous results
            p_sums[bin_name] = probs[bin_name]
            multipliers[bin_name] = D(1) / bin_point_avg
            factors[bin_name] = multipliers[bin_name] * probs[bin_name]
        else:
            # Get p_sum & factor from previous bin
            previous_bin_factor = factors[ordered_bins[i - 1]]
            previous_bin_p_sum = p_sums[ordered_bins[i - 1]]

            # Cumulative probability of selecting >=1 node from this bin, 0 from all previous
            p_sums[bin_name] = p_sums[ordered_bins[i - 1]] + probs[bin_name]

            # Multiplier & factor calculation
            multipliers[bin_name] = ((D(2) / bin_point_avg) - previous_bin_factor) / (D(2) - previous_bin_p_sum)
            factors[bin_name] = multipliers[bin_name] * probs[bin_name] + previous_bin_factor

    return multipliers, probs


def average_multipliers(bin_multipliers, bin_probabilities):
    """
    calculate average & maximum multipliers across bins based on team composition probabilities
    :param dict[str]Decimal bin_multipliers: computed multipliers for each bin
    :param dict[str]Decimal bin_probabilities: computed probability of selecting nodes from each bin
    """
    possible_mults = {}  # Multipliers averaged with other regions, weighted by probability
    # Get average multipliers for geo bins
    for geo_bin, multiplier in bin_multipliers.items():
        for other_bin, other_multiplier in bin_multipliers.items():
            if geo_bin not in possible_mults:
                possible_mults[geo_bin] = []
            possible_mults[geo_bin] = possible_mults[geo_bin] + [max(multiplier, other_multiplier) * bin_probabilities[other_bin]]

    avg_summed_mults = {k: sum(v) for k, v in possible_mults.items()}  # Sum the possible_mults for each bin
    # Average of sum of possible and mult for this bin
    avg_mults = {k: (v + bin_multipliers[k]) / 2 for k, v in avg_summed_mults.items()}
    max_mult = max(avg_mults.values())
    # Max possible mult for bin
    max_mults = {k: (v + max_mult) / 2 for k, v in avg_mults.items()}
    return avg_mults, max_mults


def tabulate_data(avg_mults, max_mults, bin_multipliers, bin_probabilities,
                  bin_node_averages, bin_point_averages_normalized):
    """
    Accept data to output, returning a list of data for each bin
    :param avg_mults: computed average multipliers for each bin
    :param max_mults: computed maximum multipliers for each bin
    :param bin_multipliers: computed multipliers for each bin
    :param bin_probabilities: probability of selecting nodes from bin
    :param bin_node_averages: quantized average nodes per bin
    :param bin_point_averages_normalized: normalized adjusted point averages for each bin
    """
    rows = []
    for geo_bin, val in avg_mults.items():
        rows = rows + [[geo_bin, round(bin_multipliers[geo_bin], 5), round(val, 5),
                        round(max_mults[geo_bin], 5), round(bin_probabilities[geo_bin], 5),
                        bin_node_averages[geo_bin], round(bin_point_averages_normalized[geo_bin], 5)]]
    return rows


output_headers = ["Bin", "Multiplier", "Average Multiplier", "Max Multiplier",
                  "Bin Probability", "Avg Nodes/Bin", "Avg Points/Bin"]


def output_json(rows, jsonpath):
    """
    outputs data to a json file
    :param list[list] rows: list of rows for each geo bin
    :param str jsonpath: path to output json
    """
    json_object = {}
    for row in rows:
        json_object[row[0]] = {output_headers[i]: row[i] if type(row[i]) != D else float(row[i])
                               for i in range(1, len(output_headers))}

    with open(jsonpath, "w") as f:
        f.write(json.dumps(json_object))


def output_csv(rows, csvpath):
    """
    outputs data to a csv file
    :param list[list] rows: list of rows for each geo bin
    :param str csvpath: path to output csv
    """
    with open(csvpath, "w") as f:
        w = csv.writer(f)
        w.writerow(output_headers)
        for row in rows:
            w.writerow(row)


def output_raw(rows):
    """
    prints output to console in human-readable format
    :param list[list] rows: list of rows for each geo bin
    """
    # Output data in readable format
    table = PrettyTable()
    table.field_names = output_headers
    table.add_rows(rows)

    print(table)


def main():
    args = get_args()

    raw_points_lines = get_raw_points_lines(args['raw_points_log'])

    # Get map of country codes to geo_bins
    countrycode_bin_map = get_countrycode_bin_map()
    countrycode_bin_map["??"] = "NorthAmerica"

    # Read in map of wallets to country codes
    wallet_bin_map = get_wallet_bin_map(args["wallet_country_supplement"])

    # Do first pass over raw points data, returning average nodes in each bin & average normalized points in each bin
    bin_node_averages, bin_point_averages_normalized = first_pass(raw_points_lines, args["lower_bound"], args["upper_bound"],
                                                                  wallet_bin_map, countrycode_bin_map)

    # Run multiplier calculation on data accumulated by first pass
    bin_multipliers, bin_probabilities = calculate_multipliers(bin_node_averages, bin_point_averages_normalized)

    avg_mults, max_mults = average_multipliers(bin_multipliers, bin_probabilities)

    print(f"Multipliers calculated from {args['lower_bound']} to {args['upper_bound']}")

    rows = tabulate_data(avg_mults, max_mults, bin_multipliers, bin_probabilities, bin_node_averages, bin_point_averages_normalized)

    if args['json']:
        output_json(rows, args['json'])
    elif args['csv']:
        output_csv(rows, args['csv'])
    else:
        output_raw(rows)


if __name__ == "__main__":
    main()
