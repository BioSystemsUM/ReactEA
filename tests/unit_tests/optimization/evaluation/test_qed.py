from unittest import TestCase

from reactea.optimization.evaluation import QED
from .test_evaluation_functions import EvaluationFunctionBaseTestCase


class TestQED(EvaluationFunctionBaseTestCase, TestCase):

    def test_evaluation_function(self):
        mols = self.mols[1]
        spds = QED(maximize=True, worst_fitness=0.0)
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), 1)
        self.assertEqual(spds.method_str(), "QED")

    def test_evaluation_function_with_invalid_mols(self):
        mols = self.mols_w_invalid
        spds = QED(maximize=True, worst_fitness=0.0)
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), len(mols))
        self.assertEqual(scores[-1], spds.worst_fitness)
