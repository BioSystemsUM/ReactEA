from unittest import TestCase

from reactea.optimization.evaluation import AggregatedSum, Caloric, LogP
from .test_evaluation_functions import EvaluationFunctionBaseTestCase


class TestAggregatedSum(EvaluationFunctionBaseTestCase, TestCase):

    def test_evaluation_function(self):
        mols = self.mols[1]
        fevaluations = [Caloric(), LogP()]
        tradeoffs = [0.5, 0.5]
        spds = AggregatedSum(fevaluation=fevaluations, tradeoffs=tradeoffs, maximize=True, worst_fitness='mean')
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), 1)
        self.assertEqual(spds.method_str(), "Aggregated Sum =  Caloric logP")

    def test_evaluation_function_with_invalid_mols(self):
        mols = self.mols_w_invalid
        fevaluations = [Caloric(), LogP()]
        tradeoffs = [0.5]
        spds = AggregatedSum(fevaluation=fevaluations, tradeoffs=tradeoffs, maximize=True, worst_fitness='max')
        scores = spds.get_fitness(mols)

        self.assertEqual(len(scores), len(mols))
