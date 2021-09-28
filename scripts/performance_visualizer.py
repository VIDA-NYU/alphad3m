import json
import altair as alt


def load_search_performances(dataset, file_path, method):
    search_results = {'method': [], 'score': [], 'time': []}

    with open(file_path) as fin:
        search_data = json.load(fin)

    for pipeline in search_data[dataset]['all_scores']:
        search_results['method'].append(method)
        search_results['score'].append(pipeline['score'])
        search_results['time'].append(pipeline['time'])

    return search_results


def plot_search_performances(data):
    return alt.Chart(data).mark_line(point=True).encode(
        alt.X('hoursminutes(time):T', title='Time'),
        alt.Y('score:Q', title='Score'),
        color='method',
    ).properties(width=800)
