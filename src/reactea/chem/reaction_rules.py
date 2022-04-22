from typing import Union

from rdkit.Chem.rdChemReactions import ReactionFromSmarts, ChemicalReaction, ReactionToSmarts


class ReactionRule:
    """"""

    def __init__(self, smarts: str, rule_id: Union[str, int]):
        """"""
        self.smarts = smarts
        self.rule_id = rule_id
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
