from unittest import TestCase, skip

from reactea.optimization.evaluation import Docking
from .test_evaluation_functions import EvaluationFunctionBaseTestCase


@skip("Requires DOCKSTRING.")
class TestDocking(EvaluationFunctionBaseTestCase, TestCase):

    def test_evaluation_function(self):
        smiles = self.smiles
        spds = Docking(target='DRD2', maximize=False, worst_fitness=-50)
        scores = spds.get_fitness(smiles)

        self.assertEqual(len(scores), 5)
        self.assertEqual(spds.method_str(), "Docking")

    def test_evaluation_function_with_invalid_mols(self):
        smiles = self.smiles_w_invalid
        spds = Docking(target='DHFR', maximize=False, worst_fitness=-50)
        scores = spds.get_fitness(smiles)

        self.assertEqual(len(scores), len(smiles))
        self.assertEqual(scores[-1], spds.worst_fitness)
