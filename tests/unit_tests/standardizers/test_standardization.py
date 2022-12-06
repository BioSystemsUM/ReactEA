from unittest import TestCase

from rdkit.Chem import Mol

from reactea.chem import Compound
from reactea.standardizers import ChEMBLStandardizer


class TestChEMBLStandardizer(TestCase):

    def test_molecular_standardizer(self):
        smiles = [('CCCCCCCCCCCCCCCCCC(=O)OC(CO)COP(=O)(O)OCCN', 'id1'), ('CC)(CC=', 'id3')]
        compounds = [Compound(sm[0], sm[1]) for sm in smiles]
        cmp1 = ChEMBLStandardizer().standardize(compounds[0])
        cmp2 = ChEMBLStandardizer().standardize(compounds[1])
        self.assertIsInstance(cmp1, Compound)
        self.assertIsInstance(cmp1.smiles, str)
        self.assertIsInstance(cmp1.mol, Mol)

        self.assertTrue(cmp2.mol is None)
