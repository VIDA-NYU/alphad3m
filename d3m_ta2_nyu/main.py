"""Entrypoints for the TA2.

This contains the multiple entrypoints for the TA2, that get called from
different commands. They spin up a D3mTa2 object and use it.
"""

import json
import logging
import os
import sys
import uuid

from d3m_ta2_nyu.ta2 import D3mTa2


logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:TA2:%(name)s:%(message)s")


def main_search():
    setup_logging()

    if len(sys.argv) != 2:
        sys.stderr.write(
            "Invalid usage, use:\n"
            "    ta2_search <config_file.json>\n"
            "        Run the TA2 system standalone, solving the given problem "
            "as per official schemas\n")
        sys.exit(2)
    else:
        with open(sys.argv[1]) as config_file:
            config = json.load(config_file)
        ta2 = D3mTa2(
            storage_root=config['temp_storage_root'],
            logs_root=config['pipeline_logs_root'],
            executables_root=config['executables_root'])
        ta2.run_search(
            dataset=config['dataset_schema'],
            problem=config['problem_root'])


def main_serve():
    setup_logging()

    if len(sys.argv) not in (1, 2):
        sys.stderr.write(
            "Invalid usage, use:\n"
            "    ta2_serve [port_number]\n"
            "        Runs in server mode, waiting for a TA3 to connect on the "
            "given port\n"
            "        (default: 45042)\n"
            "        The configuration file is read from $CONFIG_JSON_PATH\n"
            "        Alternatively, the JSON *contents* can be read from "
            "$CONFIG_JSON\n")
        sys.exit(2)
    else:
        if 'CONFIG_JSON_PATH' in os.environ:
            if 'CONFIG_JSON' in os.environ:
                logger.warning("Both $CONFIG_JSON_PATH and CONFIG_JSON are "
                               "set, preferring $CONFIG_JSON_PATH")
            with open(os.environ['CONFIG_JSON_PATH']) as config_file:
                config = json.load(config_file)
        elif 'CONFIG_JSON' in os.environ:
            config = json.loads(os.environ['CONFIG_JSON'])
        elif 'JSON_CONFIG' is os.environ:
            logger.warning("The correct environment variable is now "
                           "CONFIG_JSON. Please update your configuration")
            config = json.loads(os.environ['CONFIG_JSON'])
        else:
            logger.critical("Neither $CONFIG_JSON_PATH nor CONFIG_JSON are "
                            "set!")
            sys.exit(2)

        port = None
        if len(sys.argv) == 2:
            port = int(sys.argv[1])
        ta2 = D3mTa2(
            storage_root=config['temp_storage_root'],
            logs_root=config['pipeline_logs_root'],
            executables_root=config['executables_root'])
        ta2.run_server(config['problem_root'], port)


def main_test():
    setup_logging()

    if len(sys.argv) != 3:
        sys.exit(2)
    else:
        with open(sys.argv[2]) as config_file:
            config = json.load(config_file)
        ta2 = D3mTa2(
            storage_root=config['temp_storage_root'])
        ta2.run_test(
            dataset=config['test_data_root'],
            problem=config['problem_root'],
            pipeline_id=uuid.UUID(hex=sys.argv[1]),
            results_root=config['results_root'])
