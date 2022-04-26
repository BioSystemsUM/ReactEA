from abc import ABC, abstractmethod

from rdkit.Chem import Mol
from chembl_structure_pipeline import standardizer

from reactea.chem.compounds import Compound
from reactea.utilities.chem_utils import ChemUtils


class MolecularStandardizer(ABC):
    """"""

    def standardize(self, mol: Compound):
        """"""
        compound_representation = self._standardize(ChemUtils.canonicalize_atoms(mol.mol))
        mol.mol = compound_representation
        return mol

    @abstractmethod
    def _standardize(self, mol: Mol):
        """"""
        raise NotImplementedError


class ChEMBLStandardizer(MolecularStandardizer):
    """"""

    def _standardize(self, mol: Mol):
        try:
            mol = standardizer.standardize_mol(mol)
            mol, _ = standardizer.get_parent_mol(mol)
            return mol
        except ValueError:
            return mol
