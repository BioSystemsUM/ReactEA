from abc import ABC, abstractmethod
from unittest import TestCase

from rdkit.Chem import Mol

from reactea.chem.compounds import Compound
from reactea.chem.standardization import ChEMBLStandardizer


class TestMolecularStandardizer(ABC):

    def setUp(self):
        smiles = [('CCCCCCCCCCCCCCCCCC(=O)OC(CO)COP(=O)(O)OCCN', 'id1'), ('CC)(CC=', 'id3')]
        self.compounds = [Compound(sm[0], sm[1]) for sm in smiles]

    @abstractmethod
    def test_molecular_standardizer(self):
        pass


class TestChEMBLStandardizer(TestMolecularStandardizer, TestCase):

    def test_molecular_standardizer(self):
        cmp1 = ChEMBLStandardizer().standardize(self.compounds[0])
        cmp2 = ChEMBLStandardizer().standardize(self.compounds[1])
        self.assertIsInstance(cmp1, Compound)
        self.assertIsInstance(cmp1.smiles, str)
        self.assertIsInstance(cmp1.mol, Mol)

        self.assertTrue(cmp2.mol is None)
