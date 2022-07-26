#!/usr/bin/env python3
import csv

import prettytable
import psycopg2
import json
import logging as log
from datetime import datetime, timedelta
import argparse
import getpass
from urllib.request import urlopen

COUNTRY_MAPPING_URL = "https://git.xx.network/xx_network/primitives/-/raw/release/region/country.go"
DATE_FORMAT_STRING = "%Y-%m-%d %H:%M"


def get_args():
    """
    get_args controls the argparse usage for the script.  It sets up and parses
    arguments and returns them in dict format
    """
    parser = argparse.ArgumentParser(description="Options for raw point parsing script")
    default_lower = (datetime.now() - timedelta(days=14)).strftime(DATE_FORMAT_STRING)
    parser.add_argument("--lower-bound", type=str,
                        help="Timestamp for lower bound of logs to ingest",
                        default=default_lower)
    default_upper = datetime.now().strftime(DATE_FORMAT_STRING)
    parser.add_argument("--upper-bound", type=str,
                        help="Timestamp for upper bound of logs to ingest",
                        default=default_upper)
    parser.add_argument("--input-json", type=str,
                        help="Input multipliers json")
    parser.add_argument("-a", "--host", metavar="host", type=str,
                        help="Database server host for attempted connection")
    parser.add_argument("-p", "--port", type=int,
                        help="Port for database connection")
    parser.add_argument("-d", "--db", type=str,
                        help="Database name")
    parser.add_argument("-U", "--user",  type=str,
                        help="Username for connecting to database")
    parser.add_argument("--out-path", type=str, help="Path to output csv")

    args = vars(parser.parse_args())

    print(args)
    if args["host"]:
        args["pass"] = getpass.getpass("Enter database password: ")

    if args["lower_bound"]:
        args["lower_bound"] = datetime.strptime(args["lower_bound"], DATE_FORMAT_STRING)
    if args["upper_bound"]:
        args["upper_bound"] = datetime.strptime(args["upper_bound"], DATE_FORMAT_STRING)

    return args


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


def get_historical_points(lower_bound, upper_bound, success_points, fail_points, bin_map, multiplier_dict, out_path, conn):
    """
    get_historical_points attempts to reconstitute the points calculation of the
    points script, using historical data from the scheduling database.  it
    accepts bounds, success & fail point values (which should be taken from the
    explorer), a countrycode - bin map, a multiplier dict (loaded from json
    output of multiplier_calculator), output path and a db connection
    :param lower_bound: lower bound of query
    :param upper_bound: upper bound of query
    :param success_points: raw point value for successful round
    :param fail_points: raw point value for round error
    :param bin_map: map of countrycode to geo bin
    :param multiplier_dict: dict parsed from multiplier calculator json output
    :param out_path: output path
    :param conn: database connection object
    :return:
    """
    cur = lower_bound
    round_info = []
    next_cur = cur + timedelta(days=5)
    while next_cur < upper_bound:
        part = get_historical_round_info(cur, next_cur, conn)
        round_info += part
        cur = next_cur
        next_cur = cur + timedelta(days=5)
    final = get_historical_round_info(cur, upper_bound, conn)
    round_info += final

    simulated_log_lines = {}
    simulated_points_with_multipliers = {}
    node_points_with_multipliers = {}

    points_per_bin = {}
    node_eras = {}
    node_bins = {}
    total_raw_points = 0
    total_mult_points = 0

    total_nodes = sum([i['Avg Nodes/Bin'] for i in multiplier_dict.values()])

    avg_mult = sum([i['Multiplier'] * i['Avg Nodes/Bin'] for i in multiplier_dict.values()]) / total_nodes
    multiplier_dict['MiddleEast'] = {'Multiplier': avg_mult}

    for row in round_info:
        nodes = row[0]
        rid = row[1]
        error = row[2]
        era = row[3]

        node_mults = [multiplier_dict[bin_map[i[1]]]['Multiplier'] for i in nodes]
        max_multiplier = max(node_mults)
        if era not in simulated_log_lines:
            simulated_log_lines[era] = {}
            simulated_points_with_multipliers[era] = {}

        for node_list in nodes:
            node = node_list[0]
            country = node_list[1]
            node_bin = bin_map[country]
            node_bins[node] = node_bin
            node_multiplier = multiplier_dict[node_bin]['Multiplier']
            mult = (max_multiplier + node_multiplier) / 2

            if node_bin not in points_per_bin:
                points_per_bin[node_bin] = 0

            if node not in node_eras:
                node_eras[node] = set()
            node_eras[node].add(era)

            if node not in simulated_log_lines[era]:
                simulated_log_lines[era][node] = [0, country]
                simulated_points_with_multipliers[era][node] = 0

            if node not in node_points_with_multipliers:
                node_points_with_multipliers[node] = 0

            raw_points = success_points if not error else fail_points
            total_raw_points += raw_points
            simulated_log_lines[era][node][0] += raw_points

            multiplied_points = success_points * mult if not error else fail_points
            total_mult_points += multiplied_points
            simulated_points_with_multipliers[era][node] += multiplied_points
            node_points_with_multipliers[node] += multiplied_points
            points_per_bin[node_bin] += multiplied_points

    total_eras = len(simulated_log_lines.keys())

    total_nodes_per_bin = {}
    for era, points_dict in simulated_log_lines.items():
        for node, val in points_dict.items():
            bin = bin_map[val[1]]
            if bin not in total_nodes_per_bin:
                total_nodes_per_bin[bin] = 0
            total_nodes_per_bin[bin] += 1

    normalized_points = {key: points_per_bin[key] / total_nodes_per_bin[key] for key in sorted(points_per_bin.keys())}

    print(points_per_bin)
    # table = prettytable.PrettyTable()
    # table.add_row(["Node", "%Points", "Online", "Points"])
    with open(out_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["Node", "Bin", "Points", "%Points", "Online"])
        for node_id in node_points_with_multipliers.keys():
            multiplied_points = node_points_with_multipliers[node_id]
            row = [node_id, node_bins[node_id], multiplied_points, multiplied_points / total_mult_points, len(node_eras[node_id]) / total_eras]
            # table.add_row(row + [multiplied_points])
            writer.writerow(row)

    print(normalized_points)
    # print(table)

    return normalized_points, simulated_points_with_multipliers


def get_historical_round_info(lower_bound, upper_bound, conn):
    """
    get_historical_round_info retrieves historical round information from the
    scheduling db.  It fetches standard data from the round_metrics table,
    along with an aggregate array of the round topology & an extracted
    interval_alias which splits the rounds into 5-minute intervals.
    :param lower_bound: timestamp of lower bound of query
    :param upper_bound: timestamp of upper bound of query
    :param conn: database connection object
    :return: list of round info rows [topology, round id, error, interval alias]
    """
    print(f"Getting round info {lower_bound} to {upper_bound}")
    sql = """select array_agg(ARRAY[t1.nid, t1.sequence]), round_metrics.id, 
    exists (select * from round_errors where round_errors.round_metric_id = round_metrics.id),
    to_timestamp(floor((extract('epoch' from round_metrics.precomp_start) / 300 )) * 300) AT TIME ZONE 'UTC' as interval_alias
    from round_metrics inner join (select encode(topologies.node_id, 'base64') as nid, 
    nodes.sequence, topologies.round_metric_id from topologies 
    inner join nodes on topologies.node_id = nodes.id) as t1 
    on round_metrics.id = t1.round_metric_id where %s < round_metrics.precomp_start 
    and round_metrics.precomp_start <= %s group by round_metrics.id order by round_metrics.id desc;"""
    with conn.cursor() as cur:
        cur.execute(sql, (lower_bound, upper_bound,))
        return cur.fetchall()


def get_conn(host, port, db, user, pw):
    """
    Create a database connection object for use in the rest of the script
    :param host: Hostname for database connection
    :param port: port for database connection
    :param db: database name
    :param user: database user
    :param pw: database password
    :return: connection object for the database
    """
    conn_str = "dbname={} user={} password={} host={} port={}".format(db, user, pw, host, port)
    try:
        conn = psycopg2.connect(conn_str)
    except Exception as e:
        log.error(f"Failed to get database connection: {conn_str}")
        raise e
    log.info("Connected to {}@{}:{}/{}".format(user, host, port, db))
    return conn


def main():
    args = get_args()

    bin_map = get_countrycode_bin_map()

    # Read multiplier dict
    with open(args['input_json']) as f:
        raw_multiplier_dict = f.read()
        multiplier_dictionary = json.loads(raw_multiplier_dict)

    conn = get_conn(args['host'], args['port'], args['db'], args['user'], args['pass'])
    normalized, multiplied = get_historical_points(args['lower_bound'], args['upper_bound'], 10, -20, bin_map, multiplier_dictionary, args['out_path'], conn)

    print("done")


if __name__ == "__main__":
    main()
