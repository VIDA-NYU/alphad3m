import os
import re
import logging
import pandas as pd
from bs4 import BeautifulSoup
from os.path import join, exists, dirname
from prettytable import PrettyTable

logger = logging.getLogger(__name__)

ALL_TA2S = {'NYU-TA2', 'CMU-TA2', 'UCB-TA2', 'Uncharted-TA2', 'SRI-TA2', 'Texas A&M-TA2', 'D3M ENSEMBLE-TA2',
            'NEW-NYU-TA2'}
SKIP_DATASETS = {'LL1_FB15k_237', 'LL1_FB15k_237_V2'}  # These datasets use unsupported  metrics, so skip them


def get_task_name(task_keywords):
    task_name = None
    if 'clustering' in task_keywords:
        task_name = 'CLUSTERING'
    elif 'semi-supervised' in task_keywords:
        task_name = 'SEMISUPERVISED_CLASSIFICATION'
    elif 'collaborative' in task_keywords:
        task_name = 'COLLABORATIVE_FILTERING'
    elif 'forecasting' in task_keywords:
        task_name = 'TIME_SERIES_FORECASTING'
    elif 'lupi' in task_keywords:
        task_name = 'LUPI'
    elif 'community' in task_keywords:
        task_name = 'COMMUNITY_DETECTION'
    elif 'link' in task_keywords:
        task_name = 'LINK_PREDICTION'
    elif 'object' in task_keywords:
        task_name = 'OBJECT_DETECTION'
    elif 'matching' in task_keywords:
        task_name = 'GRAPH_MATCHING'
    elif 'series' in task_keywords:
        task_name = 'TIME_SERIES_CLASSIFICATION'
    elif 'vertex' in task_keywords:
        task_name = 'VERTEX_CLASSIFICATION'
    elif 'multipleInstanceLearning' in task_keywords:
        task_name = 'TABULAR_CLASSIFICATION'  # There are no primitives for multi instance classification
    elif 'text' in task_keywords:
        task_name = 'TEXT_CLASSIFICATION'
    elif 'image' in task_keywords and 'classification' in task_keywords:
        task_name = 'IMAGE_CLASSIFICATION'
    elif 'image' in task_keywords and 'regression' in task_keywords:
        task_name = 'IMAGE_REGRESSION'
    elif 'audio' in task_keywords:
        task_name = 'AUDIO_CLASSIFICATION'
    elif 'video' in task_keywords:
        task_name = 'VIDEO_CLASSIFICATION'
    elif 'classification' in task_keywords:
        task_name = 'TABULAR_CLASSIFICATION'
    elif 'regression' in task_keywords:
        task_name = 'TABULAR_REGRESSION'

    return task_name


def get_leaderboard(leaderboard_path):
    leaderboard = {}

    with open(leaderboard_path) as fin:
        html_doc = fin.read()

    soup = BeautifulSoup(html_doc, 'html.parser')
    items = soup.find_all('div', {'class': 'dropdown-menu'})[1].find_all('li')
    logger.info('Found %d datasets', len(items))
    datasets = []
    task_types = {}

    for item in items:
        dataset_description = item.get_text().replace('\n', ' ')
        match = re.search(r'(.+) \((.+)\)', dataset_description)
        dataset_name, task_keywords = match.group(1), match.group(2)
        dataset_name = dataset_name.rstrip()
        task_keywords = re.split(r'\s+', task_keywords.strip())
        datasets.append(dataset_name)
        task_types[dataset_name] = get_task_name(task_keywords)

    tables = soup.find_all('table', {'class': 'dataTable no-footer'})
    tables = tables[1:]
    logger.info('Found %d tables', len(tables))

    for index, table in enumerate(tables):
        dataset_name = datasets[index]
        if dataset_name in SKIP_DATASETS:
            continue
        rows = table.find('tbody').find_all('tr')
        ranking = []
        metric = None
        for index_row, row in enumerate(rows):
            cells = row.find_all('td')
            team = cells[1].get_text()
            score = cells[6].get_text()
            metric = cells[9].get_text()
            # baseline = cells[7].get_text()

            if team == 'NYU-TA2' and task_types[dataset_name] not in {'TABULAR_CLASSIFICATION', 'TABULAR_REGRESSION'}:
                team = None  # We consider NYU-TA2 as the system that supports only classification and regression

            if team in ALL_TA2S:  # Remove TA1 performances
                ranking.append((team, round(float(score), 3)))

        new_ranking = add_rank(ranking)
        leaderboard[dataset_name] = {'ranking': new_ranking, 'task': task_types[dataset_name], 'metric': metric}

    return leaderboard


def add_rank(ranking, worst_rank=len(ALL_TA2S)):
    new_ranking = {}
    previous_score = ranking[0][1]
    rank = 1

    for team, score in ranking:
        if score != previous_score:
            rank += 1
        new_ranking[team] = {'rank': rank, 'score': score}
        previous_score = score

    for team in ALL_TA2S:
        if team not in new_ranking:
            new_ranking[team] = {'rank': worst_rank, 'score': None}  # Add the worse rank

    return new_ranking


def collect_new_scores(folder_path):
    new_scores = {}
    datasets = sorted([x for x in os.listdir(folder_path) if os.path.isdir(join(folder_path, x))])

    for dataset in datasets:
        csv_path = join(folder_path, dataset, 'output/temp/statistics_datasets.csv')
        if exists(csv_path):
            data = pd.read_csv(csv_path, header=None, sep='\t')
            data = data.replace({'None': None})
            score = data.iloc[0][4]
            metric = data.iloc[0][5]
            if score is not None:
                score = round(float(score), 3)
                new_scores[dataset] = {'score': score, 'metric': metric}

    return new_scores


def update_leaderboard(leaderboard, new_scores, new_team):
    for dataset, ranking_info in leaderboard.items():
        ranking = ranking_info['ranking']
        ranking = [(t, s['score']) for t, s in ranking.items() if s['score'] is not None]
        if dataset in new_scores:
            new_score = new_scores[dataset]['score']
            metric = new_scores[dataset]['metric']
            ranking.append((new_team, new_score))
            is_reverse = 'ERROR' not in metric
            ranking = sorted(ranking, key=lambda x: x[1], reverse=is_reverse)
        else:
            logger.warning('No new score found for dataset %s', dataset)

        new_ranking = add_rank(ranking)
        leaderboard[dataset] = {'ranking': new_ranking, 'task': ranking_info['task'], 'metric': ranking_info['metric']}

    return leaderboard


def calculate_statistics(leaderboard):
    team_statistics = {x: {'winner_pipelines': 0, 'avg_rank': 0} for x in ALL_TA2S}
    for dataset in leaderboard:
        dataset_ranking = leaderboard[dataset]['ranking']
        for team in ALL_TA2S:
            team_rank = dataset_ranking[team]['rank']
            if team_rank == 1:
                team_statistics[team]['winner_pipelines'] += 1
            team_statistics[team]['avg_rank'] += team_rank

    total_datasets = float(len(leaderboard))

    for team in team_statistics:
        team_statistics[team]['avg_rank'] = round(team_statistics[team]['avg_rank'] / total_datasets, 3)

    team_statistics = sorted(team_statistics.items(), key=lambda x: x[1]['winner_pipelines'], reverse=True)

    table_team = PrettyTable()
    table_team.field_names = ['Team', 'Winner Pipelines', 'Avg. Rank']
    for team, statistics in team_statistics:
        table_team.add_row([team, statistics['winner_pipelines'], statistics['avg_rank']])

    print(table_team)

    task_statistics = {}
    for dataset in leaderboard:
        task = leaderboard[dataset]['task']
        if task not in task_statistics:
            task_statistics[task] = {'teams': {}, 'total': 0}
        task_statistics[task]['total'] += 1

        for team in ALL_TA2S:
            team_score = leaderboard[dataset]['ranking'][team]['score']
            if team not in task_statistics[task]['teams']:
                task_statistics[task]['teams'][team] = 0
            if team_score is not None:
                task_statistics[task]['teams'][team] += 1

    ta2s = sorted(ALL_TA2S)
    table_task = PrettyTable()
    table_task.field_names = ['Tasks', 'Total'] + ta2s
    for task in task_statistics:
        table_task.add_row([task, task_statistics[task]['total']] + [task_statistics[task]['teams'][x] for x in ta2s])

    print(table_task)


def to_csv(leaderboard, file_path):
    teams = ['AlphaD3M', 'AutonML', 'Ensemble', 'Aika', 'Distil AutoML', 'Autoflow', 'Axolotl',  'Drori']
    team_name_mapper = {'NEW-NYU-TA2': 'AlphaD3M',  'CMU-TA2': 'AutonML', 'UCB-TA2': 'Aika', 'SRI-TA2': 'Autoflow',
                        'Uncharted-TA2': 'Distil AutoML', 'Texas A&M-TA2': 'Axolotl', 'D3M ENSEMBLE-TA2': 'Ensemble',
                        'NYU-TA2': 'Drori'}

    data = {'Dataset': [], 'Task': [], 'Metric': []}

    for team in teams:
        data[team] = []

    for dataset in leaderboard:
        task = leaderboard[dataset]['task']
        metric = leaderboard[dataset]['metric']
        data['Dataset'].append(dataset)
        data['Task'].append(task)
        data['Metric'].append(metric)
        for team, team_data in leaderboard[dataset]['ranking'].items():
            team = team_name_mapper[team]
            score = team_data['score']
            data[team].append(score)

    data = pd.DataFrame(data)
    data.to_csv(file_path, index=False)
    logger.info('CSV file created!')


logger.info('Top 1 pipeline')
leaderboard_path = join(dirname(__file__), '../../evaluations/leaderboard_results.html')
leaderboard = get_leaderboard(leaderboard_path)
new_results_path = join(dirname(__file__), '../../evaluations/new_results')
new_scores = collect_new_scores(new_results_path)
leaderboard = update_leaderboard(leaderboard, new_scores, 'NEW-NYU-TA2')
to_csv(leaderboard, join(dirname(__file__), './paper_automlconference/resource/d3m_leaderboard.csv'))
