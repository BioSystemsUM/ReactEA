from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from rdkit.Chem import Mol, rdmolops
from chembl_structure_pipeline import standardizer

from reactea.chem import ChemUtils

if TYPE_CHECKING:
    from reactea.chem import Compound


class MolecularStandardizer(ABC):
    """
    Base class for all Molecular Standardizers.
    Molecular Standardizers standardize chemical compounds represented as Compound objects.
    """

    def standardize(self, mol: "Compound"):
        """
        Standardizes a chemical compound represented as a Compound object.
        Performs canonicalization of atoms before standardization.
        Calls the _standardize method implemented by the child classes.

        Parameters
        ----------
        mol: Compound
            Compound object representing the chemical compound to be standardized.

        Returns
        -------
        Compound:
            Standardized Compound object.
        """
        if mol.mol is None:
            return mol
        compound_representation = self._standardize(ChemUtils.canonicalize_atoms(mol.mol))
        mol.mol = compound_representation
        return mol

    @abstractmethod
    def _standardize(self, mol: Mol):
        """
        Standardizes a RDKit Mol object.
        Child classes should implement this method.

        Parameters
        ----------
        mol: Mol
            RDKit Mol object to be standardized.

        Returns
        -------
        Mol:
            Standardized RDKit Mol object.
        """
        raise NotImplementedError


class ChEMBLStandardizer(MolecularStandardizer):
    """
    Wrapper to the ChEMBL protocols used to remove fragments, standardize and salt strip molecules.
    Check the paper[1] for a detailed description of the different processes.

    [1] Bento, A.P., Hersey, A., Félix, E. et al. An open source chemical structure curation pipeline using RDKit.
        J Cheminform 12, 51 (2020). https://doi.org/10.1186/s13321-020-00456-1
    """

    def _standardize(self, mol: Mol):
        """
        Standardizes a RDKit Mol object following the ChEMBL standardization pipeline.

        Parameters
        ----------
        mol: Mol
            Chemical compound RDKit Mol object.

        Returns
        -------
        Mol:
            Standardized RDKit Mol object.
        """
        mol = standardizer.standardize_mol(mol)
        mol, _ = standardizer.get_parent_mol(mol)
        mol_frags = rdmolops.GetMolFrags(mol, asMols=True)
        largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
        return largest_mol
