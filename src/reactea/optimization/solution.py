from abc import ABC, abstractmethod
from typing import Sequence, Union

Num = Union[int, float]

class SolutionInterface(ABC):
    """"""

    @abstractmethod
    def get_fitness(self):
        """"""
        raise NotImplementedError

    @abstractmethod
    def get_representation(self):
        """"""
        raise NotImplementedError


class Solution(SolutionInterface):
    """"""

    def __init__(self, values: str, fitness: Sequence[Num], is_maximize: bool = True):
        """"""
        self.values = values
        self.fitness = fitness
        self._is_maximize = is_maximize

    def get_fitness(self):
        return self.fitness

    def get_representation(self):
        return self.values

    def __str__(self):
        return f"{self.fitness};{self.values}"

    def __repr__(self):
        return f"{self.fitness};{self.values}"

    def __eq__(self, solution):
        return set(self.values) == set(solution.values)

    def __gt__(self, solution):
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=self._is_maximize) == 1
        return False

    def __lt__(self, solution):
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=self._is_maximize) == -1
        return False

    def __ge__(self, solution):
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=self._is_maximize) != -1
        return False

    def __le__(self, solution):
        if isinstance(solution, self.__class__):
            return dominance_test(self, solution, maximize=self._is_maximize) != 1
        return False

    def __copy__(self):
        import copy
        values = copy.copy(self.values)
        fitness = self.fitness.copy()
        new_solution = Solution(values, fitness)
        return new_solution

    def __hash__(self):
        return hash(str(self.values))


#TODO: best place to put this functions
def dominance_test(solution1: Solution, solution2: Solution, maximize: bool = True):
    """
    Testes Pareto dominance

    :param solution1: The first solution.
    :param solution2: The second solution.
    :param maximize: (bool) maximization (True) or minimization (False)
    :returns:   1 : if the first solution dominates the second;
               -1 : if the second solution dominates the first;
                0 : if non of the solutions dominates the other.

    """
    best_is_one = 0
    best_is_two = 0

    values1 = solution1.get_fitness()
    values2 = solution2.get_fitness()

    for i in range(len(values1)):
        value1 = values1[i]
        value2 = values2[i]
        if value1 != value2:
            if value1 < value2:
                best_is_two = 1
            if value1 > value2:
                best_is_one = 1

    if best_is_one > best_is_two:
        if maximize:
            result = 1
        else:
            result = -1
    elif best_is_two > best_is_one:
        if maximize:
            result = -1
        else:
            result = 1
    else:
        result = 0

    return result

def non_dominated_population(population, maximize=True, filter_duplicate=True):
    """
    Returns the non dominated solutions from the population.
    """
    # population.sort(reverse = True)
    non_dominated = []
    for i in range(len(population) - 1):
        individual = population[i]
        j = 0
        dominates = True
        while j < len(population) and dominates:
            if dominance_test(individual, population[j], maximize=maximize) == -1:
                dominates = False
            else:
                j += 1
        if dominates:
            non_dominated.append(individual)

    if filter_duplicate:
        result = filter_duplicates(non_dominated)
    else:
        result = non_dominated
    return result


def filter_duplicates(population):
    """ Filters equal solutions from a population
    """

    def remove_equal(individual, population):
        filtered = []
        for other in population:
            if individual != other:
                filtered.append(other)
        return filtered

    fitered_list = []
    l = population
    while len(l) > 1:
        individual = l[0]
        fitered_list.append(individual)
        l = remove_equal(individual, l)
    if l:
        fitered_list.extend(l)
    return fitered_list


def cmetric(pf1, pf2, maximize=True):
    """
    Computes the c-metric quality indicator.

    :param pf1: The first pareto front.
    :param pf2: The second pareto front.
    :param maximize: (bool) maximization (True) or minimization (False).
    :returns: r1,r2,pf1_2,pf2_1
                r1: percentage of solutions on pf2 dominated by some solution on pf1;
                r2: percentage of solutions on pf1 dominated by some solution on pf2;
                pf1_2: solutions on pf2 dominated by some solution on pf1;
                pf2_1: solutions on pf1 dominated by some solution on pf2.

    """
    # solutions on pf2 dominated by some solution on pf1
    pf1_2 = set()
    # solutions on pf1 dominated by some solution on pf2
    pf2_1 = set()
    for s1 in pf1:
        for s2 in pf2:
            d = dominance_test(s1, s2, maximize=maximize)
            if d == 1:
                pf1_2.add(s2)
            elif d == -1:
                pf2_1.add(s1)
    r1 = len(pf1_2) / len(pf2)
    r2 = len(pf2_1) / len(pf1)
    return r1, r2, pf1_2, pf2_1
