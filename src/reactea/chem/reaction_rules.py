from typing import Union, List

from rdkit.Chem.rdChemReactions import ReactionFromSmarts, ChemicalReaction, ReactionToSmarts


class ReactionRule:
    """
    Class to represent Reaction Rules.
    Each Reaction Rule is characterized by an id, smarts string, ChemicalReaction object and possible coreactants ids.
    """

    def __init__(self, smarts: str, rule_id: Union[str, int], coreactants_ids: List[Union[str, int]] = None):
        """
        Initializes the Reaction Rule.

        Parameters
        ----------
        smarts: str
            Reaction Rule' SMARTS string.
        rule_id: Union[str, int]
            Reaction Rule id.
        coreactants_ids: List[Union[str, int]]
            Reaction Rule coreactants ids.
        """
        self._smarts = smarts
        self._rule_id = rule_id
        self._coreactants_ids = coreactants_ids
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
    def coreactants_ids(self):
        """
        Reaction Rule' coreactants ids.

        Returns
        -------
        List[Union[str, int]]
            List of the Reaction Rule' coreactants ids.
        """
        return self._coreactants_ids

    @coreactants_ids.setter
    def coreactants_ids(self, value):
        """
        Reaction Rule' coreactants ids setter.
        Coreactants ids should not be changed.

        Parameters
        ----------
        value: List[Union[str, int]]
            New coreactants ids list.

        Returns
        -------
        ValueError
            Raises a ValueError.
        """
        raise ValueError("Correactants information should not be modified!")

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
        except Exception:
            return None

    def _to_smarts(self):
        """
        Internal method to convert ChemicalReaction objects to SMARTS strings.

        Returns
        -------
        str:
            Converted SMARTS string.
        """
        try:
            return ReactionToSmarts(self._reaction)
        except Exception:
            return None
