from typing import Union

from rdkit.Chem import MolFromSmiles, MolToSmiles, Mol


class Compound:
    """"""

    def __init__(self, smiles: str, cmp_id: Union[str, int] = None):
        """"""
        self.smiles = smiles
        self.cmp_id = cmp_id
        self.mol = self._to_mol()

    @property
    def smiles(self):
        """"""
        return self.smiles

    @smiles.setter
    def smiles(self, new_smiles: str):
        """"""
        self.smiles = new_smiles
        self.mol = self._to_mol()

    @property
    def cmp_id(self):
        """"""
        return self.cmp_id

    @cmp_id.setter
    def cmp_id(self, new_id: Union[str, int]):
        """"""
        self.cmp_id = new_id

    @property
    def mol(self):
        """"""
        return self.mol

    @mol.setter
    def mol(self, new_mol: Mol):
        """"""
        self.mol = new_mol
        self.smiles = self._to_smiles()

    def _to_mol(self):
        """"""
        try:
            return MolFromSmiles(self.smiles)
        except ValueError:
            return None

    def _to_smiles(self):
        """"""
        try:
            return MolToSmiles(self.smiles)
        except ValueError:
            return None
