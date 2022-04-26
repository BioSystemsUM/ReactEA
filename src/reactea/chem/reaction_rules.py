from typing import Union, List

from rdkit.Chem.rdChemReactions import ReactionFromSmarts, ChemicalReaction, ReactionToSmarts


class ReactionRule:
    """"""

    def __init__(self, smarts: str, rule_id: Union[str, int], coreactants_ids: List[Union[str, int]] = None):
        """"""
        self._smarts = smarts
        self._rule_id = rule_id
        self._coreactants_ids = coreactants_ids
        self._reaction = self._to_reaction()

    @property
    def smarts(self):
        """"""
        return self._smarts

    @smarts.setter
    def smarts(self, new_smarts: str):
        """"""
        self._smarts = new_smarts
        self._reaction = self._to_reaction()

    @property
    def rule_id(self):
        """"""
        return self._rule_id

    @rule_id.setter
    def rule_id(self, new_id: Union[str, int]):
        """"""
        self._rule_id = new_id

    @property
    def reaction(self):
        """"""
        return self._reaction

    @reaction.setter
    def reaction(self, new_reaction: ChemicalReaction):
        """"""
        self._reaction = new_reaction
        self._smarts = self._to_smarts()

    @property
    def coreactants_ids(self):
        """"""
        return self._coreactants_ids

    @coreactants_ids.setter
    def coreactants_ids(self, value):
        """"""
        raise ValueError("Correactants information should not be modified!")

    def _to_reaction(self):
        """"""
        try:
            return ReactionFromSmarts(self._smarts)
        except ValueError:
            return None

    def _to_smarts(self):
        """"""
        try:
            return ReactionToSmarts(self._reaction)
        except ValueError:
            return None
