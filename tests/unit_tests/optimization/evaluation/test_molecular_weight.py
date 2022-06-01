from unittest import TestCase

from reactea.optimization.evaluation import MolecularWeight
from .test_evaluation_functions import EvaluationFunctionBaseTestCase


class TestMolecularWeight(EvaluationFunctionBaseTestCase, TestCase):

    def test_evaluation_function(self):
        mols = self.mols[1]
        spds = MolecularWeight(min_weight=300, max_weight=900, maximize=True, worst_fitness=0.0)
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), 1)
        self.assertEqual(spds.method_str(), "MolecularWeight")

    def test_evaluation_function_with_invalid_mols(self):
        mols = self.mols_w_invalid
        spds = MolecularWeight(min_weight=300, max_weight=900, maximize=True, worst_fitness=0.0)
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), len(mols))
        self.assertEqual(scores[-1], spds.worst_fitness)
