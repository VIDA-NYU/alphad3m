import json
import altair as alt


def load_search_performances(dataset, file_path, mode):
    search_results = {'mode': [], 'score': [], 'time': []}

    with open(file_path) as fin:
        search_data = json.load(fin)

    for pipeline in search_data[dataset]['all_scores']:
        search_results['mode'].append(mode)
        search_results['score'].append(pipeline['score'])
        search_results['time'].append(pipeline['time'])

    return search_results


def plot_search_performances(data):
    return alt.Chart(data).mark_line().encode(
        alt.X('hoursminutes(time):T', title='Time'),
        alt.Y('score:Q', title='Score'),
        color='mode',
    ).properties(width=800)
