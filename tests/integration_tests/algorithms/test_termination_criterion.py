from unittest import TestCase

from reactea import run_reactea, case_study_wrapper
from reactea.optimization.evaluation import QED, NumberOfLargeRings
from reactea.optimization.jmetal.terminators import StoppingByScore, StoppingByEvaluations
from base_test_cases import AlgorithmsBaseTestCase


class TestTerminationCriterion(AlgorithmsBaseTestCase, TestCase):

    def run_algorithm(self, algorithm, termination_criterion):
        multi_objective = algorithm not in ('SA', 'GA')
        configs = dict(self.configs, algorithm=algorithm, generations=10, output_path=str(self.output_root),
                       init_pop_size=1 if algorithm == 'SA' else 4)
        case_study = case_study_wrapper([QED(), NumberOfLargeRings()], multi_objective=multi_objective,
                                        name='termination', weights=[0.5, 0.5])
        run_reactea(configs, case_study, termination_criterion=termination_criterion)

    def test_algorithms(self):
        for algorithm in ['NSGAII', 'GA', 'SA']:
            # reached by the initial population
            criterion = StoppingByScore(0.0)
            self.run_algorithm(algorithm, criterion)
            self.assertEqual(criterion.generations, 1, algorithm)
            # the maximum number of generations of the configurations is used
            self.assertEqual(criterion.max_generations, 10, algorithm)

            # the criterion's maximum number of generations is used
            criterion = StoppingByEvaluations(3)
            self.run_algorithm(algorithm, criterion)
            self.assertEqual(criterion.generations, 3, algorithm)
