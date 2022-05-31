from itertools import chain
from typing import Union, List
from urllib import parse

from rdkit.Chem import Mol, rdmolfiles, rdmolops, MolFromSmiles, Draw, AllChem
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
    def smiles_to_svg(smiles: str, size: tuple = (690, 400)):
        """
        Returns the SVG representation of a molecule.

        Parameters
        ----------
        smiles: str
            SMILES string
        size: tuple
            SVG size

        Returns
        -------
        str:
            SVG string
        """
        mol = MolFromSmiles(smiles)
        try:
            rdmolops.Kekulize(mol)
        except:
            pass
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        AllChem.Compute2DCoords(mol)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace("svg:", "")
        return svg

    @staticmethod
    def smiles_to_image(smiles):
        """
        Converts a SMILES string to an image.

        Parameters
        ----------
        smiles: str
            SMILES string
        """
        svg_string = ChemUtils.smiles_to_svg(smiles)
        impath = 'data:image/svg+xml;charset=utf-8,' + parse.quote(svg_string, safe="")
        return impath