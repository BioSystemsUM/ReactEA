from typing import Union, List

from rdkit.Chem.rdChemReactions import ReactionFromSmarts, ChemicalReaction, ReactionToSmarts


class ReactionRule:
    """"""

    def __init__(self, smarts: str, rule_id: Union[str, int], coreactants_ids: List[Union[str, int]] = None):
        """"""
        self.smarts = smarts
        self.rule_id = rule_id
        self.coreactants_ids = coreactants_ids
        self.reaction = self._to_reaction()

    @property
    def smarts(self):
        """"""
        return self.smarts

    @smarts.setter
    def smarts(self, new_smarts: str):
        """"""
        self.smarts = new_smarts
        self.reaction = self._to_reaction()

    @property
    def rule_id(self):
        """"""
        return self.rule_id

    @rule_id.setter
    def rule_id(self, new_id: Union[str, int]):
        """"""
        self.rule_id = new_id

    @property
    def reaction(self):
        """"""
        return self.reaction

    @reaction.setter
    def reaction(self, new_reaction: ChemicalReaction):
        """"""
        self.reaction = new_reaction
        self.smarts = self._to_smarts()

    @property
    def coreactants_ids(self):
        """"""
        return self.coreactants_ids

    @coreactants_ids.setter
    def coreactants_ids(self, value):
        """"""
        raise ValueError("Correactants information should not be modified!")

    def _to_reaction(self):
        """"""
        try:
            return ReactionFromSmarts(self.smarts)
        except ValueError:
            return None

    def _to_smarts(self):
        """"""
        try:
            return ReactionToSmarts(self.smarts)
        except ValueError:
            return None
