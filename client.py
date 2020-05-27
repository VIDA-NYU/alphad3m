import grpc
import json
import logging
import sys
import datetime
import d3m_ta2_nyu.grpc_api.core_pb2 as pb_core
import d3m_ta2_nyu.grpc_api.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.grpc_api.value_pb2 as pb_value
from ta3ta2_api.utils import encode_problem_description, encode_performance_metric, decode_performance_metric, decode_value
from d3m.metadata.problem import parse_problem_description


logger = logging.getLogger(__name__)


class Cliente:
    def __init__(self):
        channel = grpc.insecure_channel('localhost:45042')
        self.core = pb_core_grpc.CoreStub(channel)

    def do_hello(self):
        self.core.Hello(pb_core.HelloRequest())

    def do_listprimitives(self):
        self.core.ListPrimitives(pb_core.ListPrimitivesRequest())

    def do_search(self, dataset_path, problem_path, time_bound=30.0, pipelines_limit=0, pipeline_template=None):
        try:
            problem = parse_problem_description(problem_path)
        except:
            logger.exception('Error parsing problem')

        version = pb_core.DESCRIPTOR.GetOptions().Extensions[pb_core.protocol_version]

        search = self.core.SearchSolutions(pb_core.SearchSolutionsRequest(
            user_agent='ta3_stub',
            version=version,
            time_bound_search=time_bound,
            rank_solutions_limit=pipelines_limit,
            allowed_value_types=[pb_value.CSV_URI],
            problem=encode_problem_description(problem),
            template=pipeline_template,
            inputs=[pb_value.Value(
                dataset_uri='file://%s' % dataset_path,
            )],
        ))

        start_time = datetime.datetime.now()
        results = self.core.GetSearchSolutionsResults(
            pb_core.GetSearchSolutionsResultsRequest(
                search_id=search.search_id,
            )
        )

        for result in results:
            if result.solution_id:
                time_search = str(datetime.datetime.now() - start_time)
                pipeline_id = result.solution_id
                score = round(decode_value(result.scores[0].scores[0].value)['value'], 5)
                metric = decode_performance_metric(result.scores[0].scores[0].metric)['metric'].name.lower()
                internal_score = result.internal_score

                yield {'id': pipeline_id, 'score': score, 'normalized_score': internal_score, 'metric': metric,
                       'time': time_search}

    def do_score(self, problem, solutions, dataset_path):
        metrics = []

        for metric in problem['problem']['performance_metrics']:
            metrics.append(encode_performance_metric(metric))

        for solution in solutions:
            try:
                response = self.core.ScoreSolution(pb_core.ScoreSolutionRequest(
                    solution_id=solution,
                    inputs=[pb_value.Value(
                        dataset_uri='file://%s' % dataset_path,
                    )],
                    performance_metrics=metrics,
                    users=[],
                    configuration=pb_core.ScoringConfiguration(
                        method=pb_core.EvaluationMethod.Value('K_FOLD'),
                        folds=4,
                        train_test_ratio=0.75,
                        shuffle=True,
                        random_seed=0
                    ),
                ))
                results = self.core.GetScoreSolutionResults(
                    pb_core.GetScoreSolutionResultsRequest(
                        request_id=response.request_id,
                    )
                )
                for _ in results:
                    pass
            except Exception:
                logger.exception("Exception during scoring %r", solution)

    def do_train(self, solution_id, dataset_path):
        fitted_solution = None

        try:
            response = self.core.FitSolution(pb_core.FitSolutionRequest(
                solution_id=solution_id,
                inputs=[pb_value.Value(
                    dataset_uri='file://%s' % dataset_path,
                )],
                expose_outputs=[],
                expose_value_types=[pb_value.CSV_URI],
                users=[],
            ))
            results = self.core.GetFitSolutionResults(
                pb_core.GetFitSolutionResultsRequest(
                    request_id=response.request_id,
                )
            )
            for result in results:
                if result.progress.state == pb_core.COMPLETED:
                    fitted_solution = result.fitted_solution_id
        except Exception:
            logger.exception("Exception training %r", solution_id)

        return fitted_solution

    def do_test(self, fitted_solution_id, dataset_path):
        tested = None
        try:
            response = self.core.ProduceSolution(pb_core.ProduceSolutionRequest(
                fitted_solution_id=fitted_solution_id,
                inputs=[pb_value.Value(
                    dataset_uri='file://%s' % dataset_path,
                )],
                expose_outputs=[],
                expose_value_types=[pb_value.CSV_URI],
                users=[],
            ))
            results = self.core.GetProduceSolutionResults(
                pb_core.GetProduceSolutionResultsRequest(
                    request_id=response.request_id,
                )
            )
            for result in results:
                if result.progress.state == pb_core.COMPLETED:
                    tested = result.exposed_outputs['outputs.0'].csv_uri
        except Exception:
            logger.exception("Exception testing %r", fitted_solution_id)

        return tested

    def do_export(self, fitted):
        for i, fitted_solution in enumerate(fitted.values()):
            try:
                self.core.SolutionExport(pb_core.SolutionExportRequest(
                    solution_id=fitted_solution,
                    rank=(i + 1.0) / (len(fitted) + 1.0),
                ))
            except Exception:
                logger.exception("Exception exporting %r", fitted_solution)

    def do_describe(self, solutions):
        for solution in solutions:
            try:
                self.core.DescribeSolution(pb_core.DescribeSolutionRequest(
                    solution_id=solution,
                ))
            except Exception:
                logger.exception("Exception during describe %r", solution)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s")

    channel = grpc.insecure_channel('localhost:45042')
    core = pb_core_grpc.CoreStub(channel)
    train_dataset_path = '/input/TRAIN/dataset_TRAIN/datasetDoc.json'
    test_dataset_path = '/input/TEST/dataset_TEST/datasetDoc.json'

    with open(sys.argv[1]) as problem:
        problem = json.load(problem)

    # Do a hello
    do_hello(core)

    # Do a list primitives
    do_listprimitives(core)
    # Do a search
    solutions = do_search(core, problem, train_dataset_path)

    # Describe the pipelines
    do_describe(core, solutions)

    # Score all found solutions
    do_score(core, problem, solutions, train_dataset_path)

    # Train all found solutions
    fitted = do_train(core, solutions, train_dataset_path)

    # Test all fitted solutions
    do_test(core, fitted, test_dataset_path)

    # Export all fitted solutions
    do_export(core, fitted)


if __name__ == '__main__':
    main()
