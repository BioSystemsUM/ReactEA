from types import SimpleNamespace
from unittest import TestCase

from reactea.chem import Compound
from reactea.optimization.jmetal.algorithms import ReactorEvolutionStrategy
from reactea.optimization.solution import ChemicalSolution


class TestReactorEvolutionStrategy(TestCase):

    def setUp(self):
        self.parents = [ChemicalSolution(Compound('CCO', 'p0'), [-0.1]), ChemicalSolution(Compound('CCCO', 'p1'), [-0.2])]
        self.offspring = [ChemicalSolution(Compound('CCCCO', 'o0'), [-0.9]),
                          ChemicalSolution(Compound('CCCCCO', 'o1'), [-0.5])]

    def test_replacement_non_elitist(self):
        # the new population is made of the offspring only (sorted, objectives are minimized)
        strategy = SimpleNamespace(elitist=False, mu=2)
        population = ReactorEvolutionStrategy.replacement(strategy, list(self.parents), list(self.offspring))
        self.assertEqual([s.variables.smiles for s in population], ['CCCCO', 'CCCCCO'])

    def test_replacement_elitist(self):
        strategy = SimpleNamespace(elitist=True, mu=3)
        population = ReactorEvolutionStrategy.replacement(strategy, list(self.parents), list(self.offspring))
        self.assertEqual([s.variables.smiles for s in population], ['CCCCO', 'CCCCCO', 'CCCO'])