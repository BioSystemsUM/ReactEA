from typing import TypeVar, List

import cytoolz
from jmetal.algorithm.singleobjective import GeneticAlgorithm
S = TypeVar("S")
R = TypeVar("R")


class ReactorGeneticAlgorithm(GeneticAlgorithm):
    """"""

    def __init__(self, **kwarg):
        """"""
        super(ReactorGeneticAlgorithm, self).__init__(**kwarg)

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        """"""
        population.extend(offspring_population)

        population.sort(key=lambda s: s.objectives[0])

        unique_population = list(cytoolz.unique(population, key=lambda x: x.variables))

        if len(unique_population) >= self.population_size:
            unique_population = unique_population[:self.population_size]
        else:
            unique_population.extend(population[:len(offspring_population) - len(unique_population)])

        return unique_population

    def get_result(self):
        return self.solutions
