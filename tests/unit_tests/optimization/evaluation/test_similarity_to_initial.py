from unittest import TestCase

from reactea.optimization.evaluation import SimilarityToInitial
from .test_evaluation_functions import EvaluationFunctionBaseTestCase


class TestSimilarityToInitial(EvaluationFunctionBaseTestCase, TestCase):

    def test_evaluation_function(self):
        mols = self.mols[1]
        init_pop = ['C=CCC(=O)C(=O)O', 'Cc1ncc(COP(=O)(O)OP(=O)(O)O)c(=N)[nH]1']
        spds = SimilarityToInitial(initial_population=init_pop, maximize=True, worst_fitness=0.0)
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), 1)
        self.assertEqual(spds.method_str(), "SimilarityToInitial")

    def test_evaluation_function_with_invalid_mols(self):
        mols = self.mols_w_invalid
        init_pop = ['C=CCC(=O)C(=O)O', 'Cc1ncc(COP(=O)(O)OP(=O)(O)O)c(=N)[nH]1']
        spds = SimilarityToInitial(initial_population=init_pop, maximize=True, worst_fitness=0.0)
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), len(mols))
        self.assertEqual(scores[-1], spds.worst_fitness)
