import random
from itertools import chain
from unittest import TestCase

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

from reactea.chem import ChemUtils


class TestReact(TestCase):

    def test_react_beyond_max_products(self):
        # joining any two carbons of two copies of a 41-atom molecule has 40 * 40 matches (more than the 1000 kept)
        random.seed(0)
        mol = MolFromSmiles('C' * 40 + 'O')
        reaction = ReactionFromSmarts('[C:1].[C:2]>>[C:1]-[C:2]')
        first_matches = {MolToSmiles(p) for p in chain.from_iterable(reaction.RunReactants((mol, mol)))}

        products = ChemUtils.react([mol, mol], reaction)

        self.assertTrue(products)
        # the products are not always the ones of the first atoms
        self.assertTrue(set(products) - first_matches)

    def test_react_below_max_products(self):
        mol = MolFromSmiles('CCO')
        reaction = ReactionFromSmarts('[C:1][O:2]>>[C:1]=[O:2]')
        self.assertEqual(ChemUtils.react(mol, reaction), ['CC=O'])
