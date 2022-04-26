from typing import Union

from rdkit.Chem import MolFromSmiles, MolToSmiles, Mol


class Compound:
    """"""

    def __init__(self, smiles: str, cmp_id: Union[str, int]):
        """"""
        self._smiles = smiles
        self._cmp_id = cmp_id
        self._mol = self._to_mol()

    @property
    def smiles(self):
        """"""
        return self._smiles

    @smiles.setter
    def smiles(self, new_smiles: str):
        """"""
        self._smiles = new_smiles
        self._mol = self._to_mol()

    @property
    def cmp_id(self):
        """"""
        return self._cmp_id

    @cmp_id.setter
    def cmp_id(self, new_id: Union[str, int]):
        """"""
        self._cmp_id = new_id

    @property
    def mol(self):
        """"""
        return self._mol

    @mol.setter
    def mol(self, new_mol: Mol):
        """"""
        self._mol = new_mol
        self._smiles = self._to_smiles()

    def _to_mol(self):
        """"""
        try:
            return MolFromSmiles(self._smiles)
        except ValueError:
            return None

    def _to_smiles(self):
        """"""
        try:
            return MolToSmiles(self._mol)
        except ValueError:
            return None
