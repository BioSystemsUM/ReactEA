import copy
import random
import time
from typing import List

import pandas as pd
from jmetal.core.operator import Mutation, Crossover
from rdkit.Chem import MolToSmiles

from reactea.chem.compounds import Compound
from reactea.chem.reaction_rules import ReactionRule
from reactea.optimization.solution import Solution
from reactea.utils.chem_utils import ChemUtils
from reactea.utils.constatns import ChemConstants
from reactea.utils.io import Writers


class ReactorMutation(Mutation[Solution]):
    """"""

    def __init__(self,
                 probability: float = 0.1,
                 reaction_rules: List[ReactionRule] = None,
                 coreactants: List[Compound] = None,
                 configs: dict = None):
        """"""
        super(ReactorMutation, self).__init__(probability=probability)
        self.reaction_rules = reaction_rules
        self.coreactants = coreactants
        self.configs = configs

    def execute(self, solution: Solution):
        """"""""
        if random.random() <= self.probability:
            compound = solution.variables
            rule = self.reaction_rules[random.randint(0, len(self.reaction_rules) - 1)]
            products = []
            i = 0
            while len(products) < 1 and i < self.configs["max_rules_by_iter"]:
                if self.coreactants is not None:
                    rule_reactants_ids = rule.coreactants_ids
                    reactants = self.set_coreactants(rule_reactants_ids, compound, self.coreactants)
                    reactants = [reac.mol for reac in reactants]
                else:
                    reactants = compound.mol

                products = ChemUtils.react(reactants, rule.reaction)

                if len(products) > 0:
                    rp = random.randint(0, len(products) - 1)
                    mutant = products[rp]
                    mutant_id = f"{compound.cmp_id}--{rule.rule_id}_"
                    mutant = Compound(MolToSmiles(mutant), mutant_id)
                    mutant = ChemConstants.default_standardizer.standardize(mutant)
                    solution.variables = mutant
                    if 'original_compound' not in solution.attributes.keys():
                        solution.attributes['original_compound'] = [compound.smiles]
                        solution.attributes['rule_id'] = [rule.rule_id]
                    else:
                        solution.attributes['original_compound'].append(compound.smiles)
                        solution.attributes['rule_id'].append(rule.rule_id)

                    Writers.update_operators_logs(self.configs, solution, mutant.smiles, rule.rule_id)
                i += 1
        return solution

    @staticmethod
    def set_coreactants(reactants: str,
                        compound: Compound,
                        coreactants: List[Compound]):
        """"""
        reactants_list = []
        if len(reactants.split(';')) > 1:
            for r in reactants.split(';'):
                if r == 'Any':
                    reactants_list.append(compound)
                else:
                    found = False
                    for cor in coreactants:
                        if cor.cmp_id == r:
                            reactants_list.append(cor)
                            found = True
                            break
                    if not found:
                        return None
            return reactants_list
        else:
            return compound

    def get_name(self):
        return 'Reactor Mutation'


class ReactorOnePointCrossover(Crossover[Solution]):
    """One point Crossover ** performs mutation on both parents (not crossover)

    :param probability: (float) The probability of crossover.
    """

    def __init__(self,
                 probability: float = 1.0,
                 reaction_rules: List[ReactionRule] = None,
                 coreactants: List[Compound] = None,
                 configs: dict = None):
        super(ReactorOnePointCrossover, self).__init__(probability=probability)
        self.reaction_rules = reaction_rules
        self.coreactants = coreactants
        self.configs = configs

    def execute(self, parents: List[Solution]):
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))
        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]

        if random.random() <= self.probability:
            stepbro = ReactorMutation(self.probability,
                                      self.reaction_rules,
                                      self.coreactants,
                                      self.configs).execute(offspring[0])
            stepsis = ReactorMutation(self.probability,
                                      self.reaction_rules,
                                      self.coreactants,
                                      self.configs).execute(offspring[1])
            offspring[0] = stepbro
            offspring[1] = stepsis
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Reactor One Point Crossover'
