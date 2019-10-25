import logging
import csv
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

with open('leaderboardsummer.html') as fin:
    html_doc = fin.read()

soup = BeautifulSoup(html_doc, 'html.parser')
items = soup.find_all('div', {'class': 'dropdown-menu'})[1].find_all('li')
logger.info('Found %d datasets', len(items))
datasets = []

for item in items:
    dataset_name = item.get_text()
    datasets.append(dataset_name[:dataset_name.find('(')].rstrip())

tables = soup.find_all('table', {'class': 'dataTable no-footer'})
logger.info('Found %d tables', len(tables))
target_team = 'NYU-TA2'
leaderboard_data = []

for index, table in enumerate(tables):
    rows = table.find('tbody').find_all('tr')
    best_team = best_score = target_score = baseline = exline = 'None'
    for index_row, row in enumerate(rows):
        cells = row.find_all("td")
        team = cells[1].get_text()
        score = cells[6].get_text()
        baseline = cells[7].get_text()
        exline = cells[9].get_text()
        if index_row == 0:
            best_team = team
            best_score = score
        if team == target_team:
            target_score = score
    if best_team == 'N/A-N/A':
        leaderboard_data.append([datasets[index]] + ['N/A'] * 6)
    else:
        leaderboard_data.append([datasets[index], best_team, best_score, target_team, target_score, baseline, exline])

with open('leaderboard_data.csv', mode='w') as fout:
    writer = csv.writer(fout, delimiter='\t')
    for row in leaderboard_data:
        writer.writerow(row)
    logger.info('Leaderboard data saved on file')
