import os
import csv
import json
import grpc
import logging
import pandas as pd
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
from datetime import datetime
from os.path import dirname, join
from d3m_ta2_nyu.grpc_logger import LoggingStub
from d3m_ta2_nyu.common import SCORES_FROM_SCHEMA, SCORES_TO_SKLEARN
from client import do_search, do_train, do_test


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DATASETS_PATH = '/Users/rlopez/D3M/datasets/seed_datasets_current/'
D3MINPUTDIR = '/Users/rlopez/D3M/tmp/'


def run_all_datasets():
    channel = grpc.insecure_channel('localhost:45042')
    core = LoggingStub(pb_core_grpc.CoreStub(channel), logger)
    statistics_path = join(dirname(__file__), 'statistics_datasets.csv')
    datasets = sorted([x for x in os.listdir(DATASETS_PATH) if os.path.isdir(join(DATASETS_PATH, x))])
    old_datasets = ['1491_one_hundred_plants_margin', '1491_one_hundred_plants_margin_clust', '1567_poker_hand', '185_baseball', '196_autoMpg', '22_handgeometry', '26_radon_seed', '27_wordLevels', '299_libras_move', '30_personae', '313_spectrometer', '31_urbansound', '32_wikiqa', '38_sick', '4550_MiceProtein', '49_facebook', '534_cps_85_wages', '56_sunspots', '56_sunspots_monthly', '57_hypothyroid', '59_umls', '60_jester', '66_chlorineConcentration', '6_70_com_amazon', '6_86_com_DBLP', 'DS01876', 'LL0_1100_popularkids', 'LL0_186_braziltourism', 'LL0_207_autoPrice', 'LL0_acled', 'LL0_acled_reduced', 'LL1_336_MS_Geolife_transport_mode_prediction', 'LL1_336_MS_Geolife_transport_mode_prediction_separate_lat_lon', 'LL1_3476_HMDB_actio_recognition', 'LL1_726_TIDY_GPS_carpool_bus_service_rating_prediction', 'LL1_736_stock_market', 'LL1_EDGELIST_net_nomination_seed', 'LL1_crime_chicago', 'LL1_net_nomination_seed', 'LL1_penn_fudan_pedestrian', 'uu1_datasmash', 'uu2_gp_hyperparameter_estimation', 'uu2_gp_hyperparameter_estimation_v2', 'uu3_world_development_indicators', 'uu4_SPECT', 'uu5_heartstatlog', 'uu6_hepatitis', 'uu7_pima_diabetes']
    datasets = ['SEMI_1053_jm1']#[x for x in datasets if x not in old_datasets][11:]
    size = len(datasets)

    for i, dataset in enumerate(datasets):
        logger.info('Processing dataset "%s" (%d/%d)' % (dataset, i+1, size))
        start_time = datetime.now()

        train_dataset_path = '/input/%s/TRAIN/dataset_TRAIN/datasetDoc.json' % dataset
        test_dataset_path = '/input/%s/TEST/dataset_TEST/datasetDoc.json' % dataset
        problem_path = join(DATASETS_PATH, dataset, 'TRAIN/problem_TRAIN/problemDoc.json')

        if not os.path.isfile(problem_path):
            logger.error('Problem file doesnt exist in the format expected')
            continue

        with open(problem_path) as fin:
            problem = json.load(fin)

        metric = SCORES_FROM_SCHEMA[problem['inputs']['performanceMetrics'][0]['metric']]
        best_time, score = 'None', 'None'
        solutions = do_search(core, problem, train_dataset_path, time_bound=10.0)
        search_time = str(datetime.now() - start_time)
        number_solutions = len(solutions)

        if number_solutions > 0:
            best_time = sorted(solutions.values(), key=lambda x: x[2])[0][2]
            best_solution = sorted(solutions.items(), key=lambda x: x[1][0])[-1][0]
            logger.info('Best pipeline: solution_id=%s' % best_solution)

            fitted_solution = do_train(core, [best_solution], train_dataset_path)
            tested_solution = do_test(core, fitted_solution, test_dataset_path)

            if len(tested_solution) > 0:
                true_file_path = join(DATASETS_PATH, dataset, '%s_dataset/tables/learningData.csv' % dataset)
                pred_file_path = join(D3MINPUTDIR, 'predictions', os.path.basename(list(tested_solution.values())[0]))
                try:
                    labels, predictions = get_ytrue_ypred(true_file_path, pred_file_path)
                    score = calculate_performance(metric, labels, predictions)
                    logger.info('Best pipeline scored: %s=%.2f' % (metric, score))
                except Exception as e:
                    logger.error('Error calculating test score')
                    logger.error(e)

        row = [dataset, number_solutions, best_time, search_time, score]
        save_row(statistics_path, row)


def get_ytrue_ypred(true_file_path, pred_file_path):
    labels = pd.read_csv(true_file_path)
    predictions = pd.read_csv(pred_file_path)
    col_index, col_label = list(predictions.columns)
    labels = labels[[col_index, col_label]]
    labels = pd.merge(labels, predictions[[col_index]], on=[col_index], how='inner')

    return labels[col_label].values, predictions[col_label].values


def calculate_performance(metric, labels, predictions):
    score = SCORES_TO_SKLEARN[metric](labels, predictions)

    return score


def save_row(file_path, row):
    with open(file_path, 'a') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerow(row)


if __name__ == '__main__':
    run_all_datasets()


