import logging
import argparse
from d3m_ta2_nyu.ta2 import D3mTa2


logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:TA2:%(name)s:%(message)s')


def search_pipelines(output, dataset, target, metric, task_keywords, timeout):
    setup_logging()
    problem_config = {'target_name': target, 'metric': metric, 'task_keywords': task_keywords}
    ta2 = D3mTa2(output)
    ta2.run_search(dataset, problem_config=problem_config, timeout=timeout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", help="path to a output folder")
    parser.add_argument("--dataset", "-d", help="path to a dataset")
    parser.add_argument("--target", "-t", help="target")
    parser.add_argument("--metric", "-m", help="metric")
    parser.add_argument("--task_keywords", "-k", nargs='+', help="task keywords")
    parser.add_argument("--timeout", "-b", type=int, help="timeout")
    args = parser.parse_args()

    if args.output:
        output = args.output
    if args.dataset:
        dataset = args.dataset
    if args.target:
        target = args.target
    if args.metric:
        metric = args.metric
    if args.task_keywords:
        task_keywords = args.task_keywords
    if args.timeout:
        timeout = args.timeout

    search_pipelines(output, dataset, target, metric, task_keywords, timeout)
