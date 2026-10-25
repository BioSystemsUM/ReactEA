from unittest import TestCase

from rdkit.Chem import Mol

from reactea.optimization.evaluation import AggregatedSum, Caloric, LogP, ChemicalEvaluationFunction
from .test_evaluation_functions import EvaluationFunctionBaseTestCase


class FakeSweetness(ChemicalEvaluationFunction):
    """
    Stands in for the DeepSweet functions (which predict a whole batch at once).
    """

    def get_fitness(self, candidates):
        candidates = [candidates] if isinstance(candidates, Mol) else candidates
        return [0.5] * len(candidates)

    def get_fitness_single(self, candidate):
        return self.get_fitness(candidate)

    def method_str(self):
        return "Penalized Sweetness (fake)"


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

    def test_evaluation_function_with_batch_sweetness(self):
        mols = self.mols
        spds = AggregatedSum(fevaluation=[FakeSweetness(), LogP()], tradeoffs=[0.5, 0.5])
        scores = spds.get_fitness(mols)

        expected = [0.5 * 0.5 + 0.5 * LogP().get_fitness_single(mol) for mol in mols]
        self.assertEqual(len(scores), len(mols))
        for score, expected_score in zip(scores, expected):
            self.assertAlmostEqual(score, expected_score)
