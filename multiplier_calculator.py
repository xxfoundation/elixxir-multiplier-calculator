#!/usr/bin/env python3

import argparse
import ast
from datetime import datetime, timedelta
from decimal import Decimal as D
import logging as log
import math
from urllib.request import urlopen

COUNTRY_MAPPING_URL = "https://git.xx.network/xx_network/primitives/-/raw/release/region/country.go"
DATE_FORMAT_STRING = "%Y-%m-%d %H:%M"


def main():
    args = get_args()

    # Get bounds for calculation
    if not args["lower_bound"]:
        lower_bound = datetime.strptime("2022-06-09 18:17", DATE_FORMAT_STRING)  # datetime.now() - timedelta(days=14)
    else:
        lower_bound = args["lower_bound"]

    if not args["upper_bound"]:
        upper_bound = datetime.now()
    else:
        upper_bound = args["upper_bound"]

    # Sanity check on bounds
    if lower_bound >= upper_bound:
        print(f"Lower bound {lower_bound} cannot be after upper bound {upper_bound}")
        exit(1)

    # Read in raw points log
    with open(args['raw_points_log'], 'r') as f:
        raw_points_lines = f.readlines()

    # Get map of country codes to geo_bins
    countrycode_bin_map = get_countrycode_bin_map()
    countrycode_bin_map["??"] = "NorthAmerica"  # TODO: check wiht ben on this

    # Read in map of wallets to country codes
    wallet_bin_map = get_wallet_bin_map(args["wallet_country_supplement"])

    # Do first pass over raw points data, returning average nodes in each bin & average normalized points in each bin
    bin_node_averages, bin_point_averages_normalized = first_pass(raw_points_lines, lower_bound, upper_bound,
                                                                  wallet_bin_map, countrycode_bin_map)

    # Run multiplier calculation on data accumulated by first pass
    bin_multipliers, bin_probabilities = calculate_multipliers(bin_node_averages, bin_point_averages_normalized)

    possible_mults = {}  # Multipliers averaged with other regions, weighted by probability
    # Get average multipliers for geo bins
    for key, value in bin_multipliers.items():
        for okey, ovalue in bin_multipliers.items():
            if key not in possible_mults:
                possible_mults[key] = []
            possible_mults[key] = possible_mults[key] + [max(value, ovalue) * bin_probabilities[okey]]

    avg_summed_mults = {k: sum(v) for k, v in possible_mults.items()}  # Sum the possible_mults for each bin
    # Average of sum of possible and mult for this bin
    avg_mults = {k: (v + bin_multipliers[k]) / 2 for k, v in avg_summed_mults.items()}
    max_mult = max(avg_mults.values())
    # Max possible mult for bin
    max_mults = {k: (v + max_mult) / 2 for k, v in avg_mults.items()}

    print(f"Multipliers calculated from {lower_bound.strftime(DATE_FORMAT_STRING)} to {upper_bound.strftime(DATE_FORMAT_STRING)}")
    # Output data in readable format
    header_cols = {"Bin": max([len(i) for i in avg_mults.keys()]) + 3,
                   "Multiplier": 15, "Avg Mult": 15, "Max mult": 15,
                   "Bin prob": 15, "Avg Nodes/bin": 15, "Avg Pts/bin": 15}
    headers = [f"{key}{' ' * (val - len(key))}" for key, val in header_cols.items()]

    header_row = "|  ".join(headers)
    print(header_row)
    print("-"*len(header_row))
    col_lens = list(header_cols.values())
    for key, val in avg_mults.items():

        cols = [f"{key}", f"{round(bin_multipliers[key], 5)}", f"{round(val, 5)}",
                f"{round(max_mults[key], 5)}", f"{round(bin_probabilities[key], 5)}",
                f"{bin_node_averages[key]}", f"{round(bin_point_averages_normalized[key], 5)}"]

        row = [f"{cols[i]}{' ' * (col_lens[i] - len(cols[i]))}" for i in range(len(cols))]

        print("|  ".join(row))


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
        if i == 0:
            cumulative_nodes = total_nodes - bin_node_averages[bin_name]
        else:
            cumulative_nodes = cumulative_bins[ordered_bins[i-1]] - bin_node_averages[bin_name]
        cumulative_bins[bin_name] = cumulative_nodes

        # Run combinations for all possible node counts from 1 to team_size
        combs = [math.comb(bin_node_averages[bin_name], j) * math.comb(cumulative_nodes, team_size - j)
                 for j in range(1, team_size + 1)]

        # Get probability of selecting >=1 node from this bin, 0 from all previous
        # We start converting floats to Decimal here to avoid floating point errors
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


def first_pass(raw_points_lines, lower_bound, upper_bound, wallet_bin_map, countrycode_bin_map):
    """
    first_pass runs preliminary calculations on the contents of the raw
    points log, using eras between lower_bound and upper_bound only
    :param list[string] raw_points_lines:
    :param datetime lower_bound:
    :param datetime upper_bound:
    :param dict[str]str wallet_bin_map:
    :param dict[str]str countrycode_bin_map:
    :return:
    """
    # Establish dictionaries to fill in first pass
    # The goal of this pass is averages, so the data needed for the end step is totals per bin & era count
    total_nodes_in_bin = {}
    total_points_in_bin = {}
    era_count = 0

    # Keep list of unmatched wallets & print a warning at the end if not empty
    unmatched_wallets = set()

    # Do first pass through lines
    for line in raw_points_lines:
        # Split line into timestamp & dictionary
        split_ind = line.find(']')
        ts_raw = line[1:split_ind]
        points_dict_raw = line[split_ind + 2:-1]

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
        for key, era_points in points_dict.items():
            parts = key.split(",")
            wallet = parts[0]
            if len(parts) == 1:
                if wallet not in wallet_bin_map:
                    unmatched_wallets.add(wallet)
                    country = "??"
                else:
                    country = wallet_bin_map[wallet]
            else:
                country = parts[2]

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
            node_count = era_nodes_in_bin[geo_bin]
            adjusted_nodes = node_count / total
            adjusted_nodes_in_bin[geo_bin] = adjusted_nodes

            # Do size adjustion for era points per bin
            bin_points = era_points_in_bin[geo_bin]
            adjusted_points = bin_points / adjusted_nodes
            size_adjusted_points_in_bin[geo_bin] = adjusted_points
            size_adjusted_points_in_bin["total"] += adjusted_points

        # Normalize size adjusted numbers, accumulate totals for final average
        for geo_bin, era_points in size_adjusted_points_in_bin.items():
            if geo_bin == "total":
                continue
            era_points = size_adjusted_points_in_bin[geo_bin]
            adjusted_era_points = era_points / size_adjusted_points_in_bin["total"]
            if geo_bin not in total_points_in_bin:
                total_points_in_bin[geo_bin] = 0
            total_points_in_bin[geo_bin] += adjusted_era_points

    # Warn if any unmatched wallets
    if len(unmatched_wallets) > 0:
        print(f"Unmatched wallets in log: {unmatched_wallets}")

    # Do base averages for nodes & points in bin
    bin_node_averages = {geo_bin: round(total_nodes / era_count) for geo_bin, total_nodes in total_nodes_in_bin.items()}
    bin_point_averages = {geo_bin: total_points / era_count for geo_bin, total_points in total_points_in_bin.items()}

    # Normalize bin point averages
    max_avg = max(bin_point_averages.values())
    bin_point_averages_normalized = {geo_bin: point_avg / max_avg for geo_bin, point_avg in bin_point_averages.items()}

    # Return bin node averages & normalized bin point averages
    return bin_node_averages, bin_point_averages_normalized


def get_wallet_bin_map(filepath):
    """
    get_wallet_bin_map reads a csv with lines in the format {wallet},{country}
    and creates a dict mapping wallets to country codes for use later in the process
    :param filepath:
    :return:
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


def get_countrycode_bin_map():
    """
    Parse a dict of country code to geo bins from raw git url containing the list
    :return:
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
    parser.add_argument("--lower-bound", type=str,
                        help="Timestamp for lower bound of logs to ingest")
    parser.add_argument("--upper-bound", type=str,
                        help="Timestamp for upper bound of logs to ingest")
    parser.add_argument("--wallet-country-supplement", type=str,
                        help="Supplemental file containing old mappings of wallet to country code")

    args = vars(parser.parse_args())
    log.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s',
                    level=log.DEBUG if args['verbose'] else log.INFO,
                    datefmt='%d-%b-%y %H:%M:%S',
                    filename=args["log"])

    if args["lower_bound"]:
        args["lower_bound"] = datetime.strptime(args["lower_bound"], DATE_FORMAT_STRING)
    if args["upper_bound"]:
        args["upper_bound"] = datetime.strptime(args["upper_bound"], DATE_FORMAT_STRING)

    return args


if __name__ == "__main__":
    main()
