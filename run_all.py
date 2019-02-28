import os
import csv
import logging
import subprocess
from datetime import datetime
from os.path import dirname, join


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
PIPELINES_INFO_PATH = '/Users/rlopez/D3M/tmp/pipelines_considered/pipelines_info.txt'
DATASETS_PATH = '/Users/rlopez/D3M/datasets/seed_datasets_current/'

def run_all_datasets():
    statistics_path = join(dirname(__file__), 'run_all_statistics.csv')
    processing_output_path = join(dirname(__file__), 'processing_output/')
    datasets = [x for x in sorted([x for x in os.listdir(DATASETS_PATH) if not x.startswith('.')]) if x not in {'uu5_heartstatlog', '56_sunspots_monthly'}]
    size = len(datasets)

    for i, dataset in enumerate(datasets):
        if os.path.exists(PIPELINES_INFO_PATH):
            os.remove(PIPELINES_INFO_PATH)

        logging.info('Processing dataset "%s" (%d/%d)' % (dataset, i+1, size))
        start_time = datetime.now()
        command = './docker.sh search seed_datasets_current/%s/TRAIN ta2-test:latest' % dataset
        #output = subprocess.check_output([command], shell=True, universal_newlines=True)
        subprocess.run([command], shell=True)
        end_time = datetime.now()
        count, metric, value, first_time = get_pipelines_info()
        row = [dataset, count, metric, value, first_time, str(end_time - start_time)]
        save_row(statistics_path, row)
        #save_file(join(processing_output_path, dataset + '.txt'), output)


def save_row(file_path, row):
    with open(file_path, 'a') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerow(row)


def save_file(file_path, text):
    with open(file_path, 'w') as fout:
        fout.write(text)


def get_pipelines_info():
    info_list = []

    if not os.path.exists(PIPELINES_INFO_PATH):
        return ['None', 'None', 'None', 'None']

    with open(PIPELINES_INFO_PATH, 'r') as fin:
        for line in fin:
            info_list.append(line.rstrip())

    if len(info_list) >= 4:#found pipelines
        return info_list[-3:] + [info_list[0]]
    else:
        return info_list[-2:] + ['None', 'None']


run_all_datasets()
