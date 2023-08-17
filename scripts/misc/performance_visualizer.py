import json
import pandas as pd
import altair as alt


def load_search_performances(file_path, method):
    search_performances = {'dataset': [], 'method': [], 'score': [], 'time': []}

    with open(file_path) as fin:
        search_data = json.load(fin)

    for dataset in search_data.keys():
        for pipeline in search_data[dataset]['all_scores']:
            search_performances['dataset'].append(dataset)
            search_performances['method'].append(method)
            search_performances['score'].append(pipeline['score'])
            search_performances['time'].append(pipeline['time'])

    search_performances = pd.DataFrame.from_dict(search_performances)
    search_performances['time'] = pd.to_datetime(search_performances['time'])

    return search_performances


def plot_search_performances(performances, dataset, max_minutes=None):
    filtered_performances = performances[performances['dataset'] == dataset]

    if max_minutes is not None:
        filtered_performances = filtered_performances[(filtered_performances['time'].dt.minute < max_minutes) &
                                                      (filtered_performances['time'].dt.hour == 0)]

    return alt.Chart(filtered_performances).mark_line(point=True).encode(
        alt.X('hoursminutes(time):T', title='Time'),
        alt.Y('score:Q', title='Score'),
        color='method',
    ).properties(width=800)
