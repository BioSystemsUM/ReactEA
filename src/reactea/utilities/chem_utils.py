from itertools import chain
from typing import Union, List

from IPython.core.display import SVG
from rdkit import Chem
from rdkit.Chem import Mol, rdmolfiles, rdmolops, MolFromSmiles, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D, MolToImage, DrawingOptions
from rdkit.Chem.rdChemReactions import ChemicalReaction


class ChemUtils:
    """
    ChemUtils class contains a set of chemical utilities.
    """

    @staticmethod
    def canonicalize_atoms(mol: Mol):
        """
        Returns the canonical atom ranking for each atom of a molecule fragment.
        (see: https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.CanonicalRankAtoms)
        Parameters
        ----------
        mol: Mol
            RDKit Mol object

        Returns
        -------
        Mol:
            canonical mol
        """
        try:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
            return mol
        except Exception:
            return mol

    @staticmethod
    def react(mol: Union[Mol, List[Mol]], rule: ChemicalReaction):
        """
        Apply the reaction to a sequence of reactant molecules and return the products as a tuple of tuples.
        (see: https://www.rdkit.org/docs/source/rdkit.Chem.rdChemReactions.html#rdkit.Chem.rdChemReactions.ChemicalReaction.RunReactants )
        Parameters
        ----------
        mol: Union[Mol, List[Mol]]
            RDKit Mol or list of Mol objects
        rule: ChemicalReaction
            RDKit ChemicalReaction object

        Returns
        -------
        List[Mol]:
            list o products as RDKit Mol objects
        """
        try:
            if not isinstance(mol, list):
                products = rule.RunReactants((mol,))
                return list(chain.from_iterable(products))
            else:
                products = rule.RunReactants(tuple(mol))
                return list(chain.from_iterable(products))
        except Exception:
            return ()

    @staticmethod
    def smiles_to_img(smiles, size=(200, 200), highlightMol=False):
        mol = MolFromSmiles(smiles)
        if not highlightMol:
            return ChemUtils.mol_to_image(mol, size=size)
        else:
            highlight_atoms = list(mol.GetSubstructMatch(mol))
            hit_bonds = []
            for bond in mol.GetBonds():
                aid1 = highlight_atoms[bond.GetBeginAtomIdx()]
                aid2 = highlight_atoms[bond.GetEndAtomIdx()]
                hit_bonds.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())
            return ChemUtils.mol_to_image(mol, size=size, highlightAtoms=highlight_atoms, highlightBonds=hit_bonds)

    @staticmethod
    def mol_to_image(mol, size, highlightAtoms=None, highlightBonds=None):
        if highlightAtoms:
            return MolToImage(mol,
                              size=size,
                              highlightAtoms=highlightAtoms,
                              highlightBonds=highlightBonds,
                              highlightColor=(1, 0.8, 0.79))
        return MolToImage(mol, size)
