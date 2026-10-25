from unittest import TestCase

from reactea.chem import Compound
from reactea.optimization.solution import ChemicalSolution, non_dominated_population


class TestNonDominatedPopulation(TestCase):

    def setUp(self):
        self.solutions = [ChemicalSolution(Compound(smiles, i), objectives)
                          for i, (smiles, objectives) in enumerate([('CCO', [1.0, 1.0]),
                                                                    ('CCCO', [2.0, 2.0]),
                                                                    ('CCCCO', [0.5, 0.5])])]

    def test_minimization(self):
        # the best solution is the last one (it must not be skipped)
        front = non_dominated_population(self.solutions, maximize=False)
        self.assertEqual([s.variables.smiles for s in front], ['CCCCO'])

    def test_maximization(self):
        front = non_dominated_population(self.solutions, maximize=True)
        self.assertEqual([s.variables.smiles for s in front], ['CCCO'])