from abc import ABC, abstractmethod
from unittest import TestCase

from reactea.chem import Compound
from reactea.optimization.comparators import ParetoDominanceComparator
from reactea.optimization.solution import ChemicalSolution


class TestComparators(ABC):

    def setUp(self):
        compound1 = Compound('CCCCCCCCCCCCCCCCCC(=O)OC(CO)COP(=O)(O)OCCN', 'id1')
        compound2 = Compound('CCCCCCCCCCCCCC(=O)OC(CO)COP(=O)(O)OCC(O)CO', 'id2')
        # Dominated solution
        self.solution1 = ChemicalSolution(compound1, [0.0, 0.5], True)
        # Dominant solution
        self.solution2 = ChemicalSolution(compound2, [0.75, 0.6], True)

    @abstractmethod
    def test_comparator(self):
        pass


class TestParetoDominanceComparator(TestComparators, TestCase):

    def test_comparator(self):
        res1 = ParetoDominanceComparator().compare(self.solution1, self.solution2)
        self.assertTrue(res1 == -1)
        res2 = ParetoDominanceComparator().compare(self.solution2, self.solution1)
        self.assertTrue(res2 == 1)
        res3 = ParetoDominanceComparator().compare(self.solution1, self.solution1)
        self.assertTrue(res3 == 0)
