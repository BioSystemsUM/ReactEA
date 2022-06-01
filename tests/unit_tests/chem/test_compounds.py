from unittest import TestCase

from reactea.chem.compounds import Compound


class TestCompound(TestCase):

    def test_compound(self):
        compound1 = 'CCCCCCCCCCCCCCCCCC(=O)OC(CO)COP(=O)(O)OCCN'
        id1 = 0
        compound2 = 'CC)(CC='
        id2 = 'id2'

        cmp1 = Compound(compound1, id1)
        cmp2 = Compound(compound2, id2)

        self.assertEqual(cmp1.cmp_id, id1)
        self.assertEqual(cmp1.smiles, compound1)
        self.assertEqual(cmp2.cmp_id, id2)
        self.assertEqual(cmp2.smiles, compound2)
        self.assertTrue(cmp2.mol is None)

        cmp2.mol = cmp1.mol
        self.assertEqual(cmp2.smiles, cmp1.smiles)
        self.assertTrue(cmp2.mol is not None)
        self.assertTrue(cmp2.mol == cmp1.mol)

        cmp2.smiles = 'CO'
        self.assertEqual(cmp2.smiles, 'CO')
        self.assertTrue(cmp2.mol != cmp1.mol)
