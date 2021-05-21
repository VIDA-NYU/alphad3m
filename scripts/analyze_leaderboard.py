import logging
from prettytable import PrettyTable
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ALL_TA2S = {'NYU-TA2', 'CMU-TA2', 'UCB-TA2', 'Uncharted-TA2', 'SRI-TA2', 'Texas A&M-TA2'}


def get_new_leaderboard(leaderboard_path):
    leaderboard = {}

    with open(leaderboard_path) as fin:
        html_doc = fin.read()

    soup = BeautifulSoup(html_doc, 'html.parser')
    items = soup.find_all('div', {'class': 'dropdown-menu'})[1].find_all('li')
    logger.info('Found %d datasets', len(items))
    datasets = []

    for item in items:
        dataset_name = item.get_text()
        datasets.append(dataset_name[:dataset_name.find('(')].rstrip())

    tables = soup.find_all('table', {'class': 'dataTable no-footer'})
    tables = tables[1:]
    logger.info('Found %d tables', len(tables))

    for index, table in enumerate(tables):
        dataset_name = datasets[index]
        if dataset_name in {'LL1_FB15k_237', 'LL1_FB15k_237_V2'}:
            continue
        rows = table.find('tbody').find_all('tr')
        dataset_ranking = []
        for index_row, row in enumerate(rows):
            cells = row.find_all('td')
            team = cells[1].get_text()
            score = cells[6].get_text()
            baseline = cells[7].get_text()
            metric = cells[9].get_text()

            #if 'TA1' in team:
            #    team = 'TA1'
            if team in ALL_TA2S:  # Remove TA1 performances
                dataset_ranking.append((team, score))

        new_dataset_ranking = {}
        previous_score = dataset_ranking[0][1]
        rank = 1
        for team, score in dataset_ranking:
            if score != previous_score:
                rank += 1
            new_dataset_ranking[team] = {'rank': rank, 'score': score}
            previous_score = score

        worst_rank = len(ALL_TA2S)
        for team in ALL_TA2S:
            if team not in new_dataset_ranking:
                new_dataset_ranking[team] = {'rank': worst_rank, 'score': None}  # Add the worse rank

        leaderboard[dataset_name] = new_dataset_ranking

    return leaderboard


def calculate_statistics(leaderboard):
    team_statistics = {x: {'winner_pipelines': 0, 'avg_rank': 0} for x in ALL_TA2S}
    for dataset in leaderboard:
        dataset_ranking = leaderboard[dataset]
        for team in ALL_TA2S:
            team_rank = dataset_ranking[team]['rank']
            if team_rank == 1:
                team_statistics[team]['winner_pipelines'] += 1
            team_statistics[team]['avg_rank'] += team_rank

    total_datasets = float(len(leaderboard))

    for team in team_statistics:
        team_statistics[team]['avg_rank'] = round(team_statistics[team]['avg_rank'] / total_datasets, 3)

    team_statistics = sorted(team_statistics.items(), key=lambda x:x[1]['winner_pipelines'], reverse=True)

    table = PrettyTable()
    table.field_names = ['Team', 'Winner Pipelines', 'Avg. Rank']
    for team, statistics in team_statistics:
        table.add_row([team, statistics['winner_pipelines'], statistics['avg_rank']])

    print(table)


logger.info('Top 1 pipeline')
leaderboard_path = '../../leaderboarddecember2020dryrun_rank1.html'
new_leaderboard = get_new_leaderboard(leaderboard_path)
calculate_statistics(new_leaderboard)

logger.info('Top 20 pipeline')
leaderboard_path = '../../leaderboarddecember2020dryrun_rank20.html'
new_leaderboard = get_new_leaderboard(leaderboard_path)
calculate_statistics(new_leaderboard)
