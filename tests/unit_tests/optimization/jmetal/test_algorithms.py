from types import SimpleNamespace
from unittest import TestCase

from reactea.chem import Compound
from reactea.optimization.evaluation import ChemicalEvaluationFunction
from reactea.optimization.jmetal.algorithms import ReactorEvolutionStrategy, ReactorRandomSearch
from reactea.optimization.jmetal.problem import JmetalProblem
from reactea.optimization.jmetal.terminators import StoppingByEvaluations
from reactea.optimization.problem import ChemicalProblem
from reactea.optimization.solution import ChemicalSolution


class NumberOfAtoms(ChemicalEvaluationFunction):

    def get_fitness_single(self, candidate):
        return candidate.GetNumAtoms()

    def method_str(self):
        return "NumberOfAtoms"


class AddCarbon:
    """
    Mutation that adds a carbon to the compound (stands in for a ReactorMutation).
    """

    def execute(self, solution):
        solution.variables = Compound(solution.variables.smiles + 'C', 'mutant')
        return solution


class TestReactorRandomSearch(TestCase):

    def test_random_walk(self):
        # each step adds an atom, so each new compound dominates the previous one
        problem = JmetalProblem(ChemicalProblem([NumberOfAtoms()]), [Compound('CCO', 'id0')])
        algorithm = ReactorRandomSearch(problem=problem, mutation=AddCarbon(),
                                        termination_criterion=StoppingByEvaluations(5))
        algorithm.run()

        self.assertEqual(len(algorithm.solutions), 1)
        self.assertEqual(algorithm.solutions[0].variables.mol.GetNumAtoms(), 3 + 4)
        # the seed is not changed by the mutations
        self.assertEqual(problem.initial_pop[0].smiles, 'CCO')


class TestReactorEvolutionStrategy(TestCase):

    def setUp(self):
        self.parents = [ChemicalSolution(Compound('CCO', 'p0'), [-0.1]),
                        ChemicalSolution(Compound('CCCO', 'p1'), [-0.2])]
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
