from itertools import chain
from typing import Union, List

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, Mol, rdmolfiles, rdmolops


class ChemUtils:
    """"""

    @staticmethod
    def canonicalize_atoms(mol: Mol):
        """"""
        try:
            new_order = rdmolfiles.CanonicalRankAtoms(mol)
            mol = rdmolops.RenumberAtoms(mol, new_order)
            return mol
        except Exception:
            return mol

    @staticmethod
    def react(mol: Union[Mol, List[Mol]], rule):
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
    def atomPairsDescriptors(list_mols: List[Mol], n_bits: int = 2048):
        """"""
        invalids = np.zeros(len(list_mols), np.bool)
        arr = np.zeros((len(list_mols), n_bits))

        for i, mol in enumerate(list_mols):
            if mol is None:
                invalids[i] = True
                mol = Chem.MolFromSmiles("")
            res = np.zeros((1,))
            desc = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, n_bits)
            DataStructs.ConvertToNumpyArray(desc, res)
            arr[i, :] = res
        return arr, invalids
