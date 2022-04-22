from typing import List

from jmetal.util.generator import Generator

from reactea.chem.compounds import Compound
from reactea.optimization.problem import ChemicalProblem
from reactea.optimization.solution import Solution


class ChemicalGenerator(Generator):
    """"""

    def __init__(self, initial_population: List[Compound]):
        """"""
        super(ChemicalGenerator, self).__init__()
        self.initial_population = initial_population
        self.current = 0

    def new(self, problem: ChemicalProblem):
        if self.current == len(self.initial_population):
            self.current = 0
        individual = self.initial_population[self.current]
        new_solution = Solution(individual)
        self.current += 1
        return new_solution
