from unittest import TestCase, skip

from reactea.optimization.evaluation import PenalizedSweetness
from .test_evaluation_functions import EvaluationFunctionBaseTestCase


#@skip("Requires DeepSweet.")
class TestPenalizedSweetness(EvaluationFunctionBaseTestCase, TestCase):

    def test_evaluation_function(self):
        mols = self.mols[-1]
        spds = PenalizedSweetness(maximize=True, worst_fitness=0.0)
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), 1)
        self.assertEqual(spds.method_str(), "Penalized Sweetness (DeepSweet-Caloric)")

    def test_evaluation_function_with_invalid_mols(self):
        mols = self.mols_w_invalid
        spds = PenalizedSweetness(maximize=True, worst_fitness=0.0)
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), len(mols))
        self.assertEqual(scores[-1], spds.worst_fitness)
