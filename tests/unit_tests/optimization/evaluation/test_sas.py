import sys
from unittest import TestCase

from reactea.optimization.evaluation import SAS
from .test_evaluation_functions import EvaluationFunctionBaseTestCase


class TestQED(EvaluationFunctionBaseTestCase, TestCase):

    def test_evaluation_function(self):
        mols = self.mols
        spds = SAS(maximize=True, worst_fitness=0.0)
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), len(mols))
        self.assertEqual(spds.method_str(), "SAS")

    def test_evaluation_function_with_invalid_mols(self):
        mols = self.mols_w_invalid
        spds = SAS(maximize=True, worst_fitness=0.0)
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), len(mols))
        self.assertEqual(scores[-1], spds.worst_fitness)

    def test_sys_path_does_not_grow(self):
        spds = SAS()
        spds.get_fitness(self.mols)
        n_paths = len(sys.path)
        spds.get_fitness(self.mols)
        self.assertEqual(len(sys.path), n_paths)
