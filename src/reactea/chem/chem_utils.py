import random
from itertools import chain
from typing import Union, List

import numpy as np
from rdkit import DataStructs, Chem
from rdkit.Chem import Mol, rdmolfiles, rdmolops, MolFromSmiles, MolToSmiles, RemoveHs
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.Fingerprints.FingerprintMols import FingerprintMol
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
        List[str]:
            list of products SMILES
        """
        try:
            if not isinstance(mol, list):
                products = rule.RunReactants((mol,))
            else:
                products = rule.RunReactants(tuple(mol))
            return list(set([MolToSmiles(s) for s in list(chain.from_iterable(products))]))
        except Exception:
            return ()

    @staticmethod
    def smiles_to_img(smiles: str, size: tuple = (200, 200), highlightMol: bool = False):
        """
        Returns an image of a molecule from a SMILES string.

        Parameters
        ----------
        smiles: str
            SMILES string
        size: tuple
            image size
        highlightMol: bool
            highlight the molecule

        Returns
        -------
        Molecule Image.
        """
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
    def mol_to_image(mol: Mol, size: tuple, highlightAtoms: list = None, highlightBonds: list = None):
        """
        Returns an image of a molecule.

        Parameters
        ----------
        mol: Mol
            RDKit Mol object
        size: tuple
            image size
        highlightAtoms: list
            list of atom to highlight
        highlightBonds: list
            list of bond to highlight

        Returns
        -------
        Molecule Image.
        """
        if highlightAtoms:
            return MolToImage(mol,
                              size=size,
                              highlightAtoms=highlightAtoms,
                              highlightBonds=highlightBonds,
                              highlightColor=(1, 0.8, 0.79))
        return MolToImage(mol, size)

    @staticmethod
    def valid_product(smiles: str):
        """
        Returns True if the SMILES string is a valid reaction product.

        Parameters
        ----------
        smiles: str
            SMILES string
        Returns
        -------
        bool:
            True if the SMILES string is a valid product
        """
        if '*' in smiles:
            return False
        mol = MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol)
        except:
            return False
        if mol.GetNumAtoms() < 4:
            return False
        carbon = MolFromSmiles('C')
        if not mol.HasSubstructMatch(carbon):
            return False
        return True

    @staticmethod
    def calc_fingerprint_similarity(smiles1: str, smiles2: str):
        """
        Calculates the similarity between two molecules based on fingerprints.

        Parameters
        ----------
        smiles1: str
            The first molecule smiles.
        smiles2: str
            The second molecule smiles.
        Returns
        -------
        float
            The similarity between the two molecules.
        """
        mol1 = MolFromSmiles(smiles1)
        mol2 = MolFromSmiles(smiles2)
        if mol1 and mol2:
            fp1 = FingerprintMol(mol1)
            fp2 = FingerprintMol(mol2)
            return DataStructs.FingerprintSimilarity(fp1, fp2)
        return 0.0

    @staticmethod
    def most_similar_compound(smiles: str, smiles_list: List[str], tolerance: float = 0.25):
        """
        Finds the most similar compound in a list of compounds.

        Parameters
        ----------
        smiles: str
            The smiles of the compound to find the most similar compound for.
        smiles_list: List[str]
            The list of compounds to find the most similar compound in.
        tolerance: float
            Compounds between max_similarity and max_similarity - tolerance are considered to be picked.

        Returns
        -------
        str
            The most similar compound SMILES string.
        """
        if len(smiles_list) == 1:
            return smiles_list[0]
        sims = [ChemUtils.calc_fingerprint_similarity(smiles, s) for s in smiles_list]
        idx = [i for i, x in enumerate(sims) if x >= max(sims) - tolerance]
        return smiles_list[random.choice(idx)]
