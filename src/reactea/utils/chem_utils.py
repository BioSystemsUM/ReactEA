from itertools import chain

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors


class ChemUtils:
    """"""

    @staticmethod
    def react(mol, rule):
        try:
            if not isinstance(mol, list):
                products = rule.RunReactants((mol,))
                return list(chain.from_iterable(products))
            else:
                products = rule.RunReactants(tuple(mol))
                return list(chain.from_iterable(products))
        except Exception as e:
            return ()

    @staticmethod
    def atomPairsDescriptors(list_mols, nBits: int = 2048):
        """"""
        invalids = np.zeros(len(list_mols), np.bool)
        arr = np.zeros((len(list_mols), nBits))

        for i, mol in enumerate(list_mols):
            if mol is None:
                invalids[i] = True
                mol = Chem.MolFromSmiles("")
            res = np.zeros((1,))
            desc = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits)
            DataStructs.ConvertToNumpyArray(desc, res)
            arr[i, :] = res
        return arr, invalids
