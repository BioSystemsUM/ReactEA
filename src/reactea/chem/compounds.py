from typing import Union

from rdkit.Chem import MolFromSmiles, MolToSmiles, Mol

from reactea.chem import ChemUtils


class Compound:
    """
    Class to represent Compounds.
    Each Compound is characterized by an id, SMILES string and Mol object.
    """

    def __init__(self, smiles: str, cmp_id: Union[str, int], canonicalize: bool = True):
        """
        Initializes the Compound.

        Parameters
        ----------
        smiles: str
            compound SMILES string
        cmp_id: Union[str, int]
            compound id
        canonicalize: bool
            whether to canonicalize the SMILES string.
        """
        if canonicalize:
            self._smiles = ChemUtils.canonicalize_smiles(smiles, canonicalize)
        else:
            self._smiles = smiles
        self._cmp_id = cmp_id
        self._mol = self._to_mol()

    @property
    def smiles(self):
        """
        Compound' SMILES property.

        Returns
        -------
        str:
            Compound' SMILES string.
        """
        return self._smiles

    @smiles.setter
    def smiles(self, new_smiles: str):
        """
        Compound' SMILES setter.
        When a new SMILES is defined the mol property is also updated.

        Parameters
        ----------
        new_smiles: str
            Compound new SMILES representation.
        """
        self._smiles = new_smiles
        self._mol = self._to_mol()

    @property
    def cmp_id(self):
        """
        Compound id property.

        Returns
        -------
        Union[str, int]:
            Compound id.
        """
        return self._cmp_id

    @cmp_id.setter
    def cmp_id(self, new_id: Union[str, int]):
        """
        Compound' id setter.

        Parameters
        ----------
        new_id: Union[str, int]
            New compound id.
        """
        self._cmp_id = new_id

    @property
    def mol(self):
        """
        Compound' Mol property.

        Returns
        -------
        Mol:
            Compound' RDKit Mol object.
        """
        return self._mol

    @mol.setter
    def mol(self, new_mol: Mol):
        """
        Compound' Mol setter.
        When a new Mol is defined the SMILES property is also updated.

        Parameters
        ----------
        new_mol: Mol
            New compound' Mol object.
        """
        self._mol = new_mol
        self._smiles = self._to_smiles()

    def _to_mol(self):
        """
        Internal method to convert SMILES strings to RDKit Mol objects.

        Returns
        -------
        Mol:
            Converted Mol object.
        """
        return MolFromSmiles(self._smiles)

    def _to_smiles(self):
        """
        Internal method to convert RDKit Mol objects into SMILES strings.

        Returns
        -------
        str:
            Converted smiles string.
        """
        if self._mol is None:
            return None
        else:
            return MolToSmiles(self._mol)
