from unittest import TestCase

from reactea.optimization.evaluation import AggregatedSum, Caloric, LogP, QED
from reactea.wrappers import evaluation_functions_wrapper
from .test_evaluation_functions import EvaluationFunctionBaseTestCase


class TestParallelEvaluation(EvaluationFunctionBaseTestCase, TestCase):

    def test_evaluation_function(self):
        mols = self.mols
        serial = QED()
        parallel = QED()
        parallel.n_jobs = 2

        self.assertEqual(serial.n_jobs, 1)
        self.assertEqual(serial.get_fitness(mols), parallel.get_fitness(mols))

        serial_sum = AggregatedSum(fevaluation=[Caloric(), LogP()], tradeoffs=[0.5, 0.5])
        parallel_sum = AggregatedSum(fevaluation=[Caloric(), LogP()], tradeoffs=[0.5, 0.5], n_jobs=2)
        self.assertEqual(serial_sum.get_fitness(mols), parallel_sum.get_fitness(mols))

    def test_evaluation_function_with_invalid_mols(self):
        mols = self.mols_w_invalid
        # user-defined function (pickled by value by the loky workers, like functions defined in a notebook)
        serial = evaluation_functions_wrapper(lambda mol: mol.GetNumAtoms(), maximize=True, worst_fitness=-1,
                                              name='n_atoms')
        parallel = evaluation_functions_wrapper(lambda mol: mol.GetNumAtoms(), maximize=True, worst_fitness=-1,
                                                name='n_atoms', n_jobs=2)
        scores = parallel.get_fitness(mols)

        self.assertEqual(serial.get_fitness(mols), scores)
        self.assertEqual(scores.count(-1), 3)