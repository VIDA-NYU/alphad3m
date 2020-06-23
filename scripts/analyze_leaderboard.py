import logging
import csv
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

with open('../../leaderboardwinter2020.html') as fin:
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
target_team = 'NYU-TA2'
leaderboard_data = []

for index, table in enumerate(tables):
    rows = table.find('tbody').find_all('tr')
    best_teams = []
    best_score = target_score = baseline = exline = metric = 'None'
    for index_row, row in enumerate(rows):
        cells = row.find_all('td')
        position = cells[0].get_text()
        team = cells[1].get_text()
        score = cells[6].get_text()
        baseline = cells[7].get_text()
        exline = cells[9].get_text()
        metric = cells[11].get_text()
        if position == '1':
            best_teams.append(team)
            best_score = score
        if team == target_team:
            target_score = score
    if len(best_teams) > 0:
        leaderboard_data.append([datasets[index], ', '.join(best_teams), best_score, target_team, target_score, baseline, exline, metric])
    else:
        leaderboard_data.append([datasets[index]] + ['N/A'] * 7)


with open('../../leaderboard_data.csv', mode='w') as fout:
    writer = csv.writer(fout, delimiter='\t')
    writer.writerow(['Dataset', 'Best team', 'Best score', 'Target team', 'Target score', 'Baseline', 'Exline', 'Metric'])
    for row in leaderboard_data:
        writer.writerow(row)
    logger.info('Leaderboard data saved on file')
