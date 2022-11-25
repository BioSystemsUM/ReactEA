import uuid
from typing import Union, List

from rdkit.Chem.rdChemReactions import ReactionFromSmarts, ChemicalReaction, ReactionToSmarts

from reactea.chem import Compound


class ReactionRule:
    """
    Class to represent Reaction Rules.
    Each Reaction Rule is characterized by an id, smarts string, ChemicalReaction object and possible coreactants ids.
    """

    def __init__(self, smarts: str, rule_id: Union[str, int], reactants: Union[str, None] = 'Any'):
        """
        Initializes the Reaction Rule.

        Parameters
        ----------
        smarts: str
            Reaction Rule' SMARTS string.
        rule_id: Union[str, int]
            Reaction Rule id.
        reactants: Union[str, None]
            Reaction Rule reactants.
        """
        self._smarts = smarts
        self._rule_id = rule_id
        if reactants is None:
            self._reactants = 'Any'
        else:
            self._reactants = reactants
        self._reaction = self._to_reaction()

    @property
    def smarts(self):
        """
        Reaction Rule' SMARTS property.

        Returns
        -------
        str:
            Reaction Rule' SMARTS string.
        """
        return self._smarts

    @smarts.setter
    def smarts(self, new_smarts: str):
        """
        Reaction Rule' SMARTS setter.
        When a new SMARTS is defined the reaction property is also updated.

        Parameters
        ----------
        new_smarts: str
            Reaction Rule' SMARTS representation.

        """
        self._smarts = new_smarts
        self._reaction = self._to_reaction()

    @property
    def rule_id(self):
        """
        Reaction Rule id property.

        Returns
        -------
        Union[str, int]:
            Reaction Rule id.
        """
        return self._rule_id

    @rule_id.setter
    def rule_id(self, new_id: Union[str, int]):
        """
        Reaction Rule' id setter.

        Parameters
        ----------
        new_id: Union[str, int]
            New Reaction Rule id.
        """
        self._rule_id = new_id

    @property
    def reaction(self):
        """
        Reaction Rule' reaction property.

        Returns
        -------
        ChemicalReaction:
            Reaction Rule' RDKit ChemicalReaction object.
        """
        return self._reaction

    @reaction.setter
    def reaction(self, new_reaction: ChemicalReaction):
        """
        Reaction Rule' reaction setter.
        When a new ChemicalReaction is defined the SMARTS property is also updated.

        Parameters
        ----------
        new_reaction: ChemicalReaction
            New Reaction Rule' ChemicalReaction object.
        """
        self._reaction = new_reaction
        self._smarts = self._to_smarts()

    @property
    def reactants(self):
        """
        Reaction Rule' coreactants ids.

        Returns
        -------
        str
            Reaction Rule' coreactants ids.
        """
        return self._reactants

    @reactants.setter
    def reactants(self, value):
        """
        Reaction Rule' coreactants ids setter.
        Coreactants ids should not be changed.

        Parameters
        ----------
        value: str
            New coreactants ids.

        Returns
        -------
        ValueError
            Raises a ValueError.
        """
        raise ValueError("Coreactants information should not be modified!")

    def _to_reaction(self):
        """
        Internal method to convert SMARTS strings to ChemicalReaction objects.

        Returns
        -------
        ChemicalReaction:
            Converted ChemicalReaction object.
        """
        try:
            return ReactionFromSmarts(self._smarts)
        except ValueError:
            return None

    def _to_smarts(self):
        """
        Internal method to convert ChemicalReaction objects into SMARTS strings.

        Returns
        -------
        str:
            Converted SMARTS string.
        """
        return ReactionToSmarts(self._reaction)

    def reactants_to_mol_list(self, compound):
        """
        Converts the Reaction Rule' reactants into a list of RDKit molecules. The Any field is replaced by the provided
        compound.

        Parameters
        ----------
        compound: Compound
            The compound to replace the Any field.

        Returns
        -------
        reactants: List[Mol]
            List of reactants as RDKit molecules.
        """
        reactants = []
        for r in self.reactants.split(';'):
            if r == 'Any':
                reactants.append(compound.mol)
            else:
                random_id = uuid.uuid4().hex
                reactants.append(Compound(r, random_id).mol)
        if len(reactants) == 1:
            return reactants[0]
        return reactants

