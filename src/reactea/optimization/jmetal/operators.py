import copy
import heapq
import random
import time
from typing import List

import pandas as pd
from jmetal.core.operator import Mutation, Crossover

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

from reactea.optimization.solution import Solution
from reactea.utils.chem_utils import ChemUtils


class ReactorMutation(Mutation[Solution]):
    """"""

    def __init__(self, probability: float = 0.1, configs: dict = None):
        """"""
        super(ReactorMutation, self).__init__(probability=probability)
        self.configs = configs

    def execute(self, solution: Solution):
        """"""""
        if random.random() <= self.probability:
            with open(self.configs["rules_path"]) as fin:
                sample = heapq.nlargest(self.configs["max_rules_by_iter"], fin, key=lambda L: random.random())
            random.shuffle(sample)

            candidate_smiles = solution.variables
            candidate_mol = MolFromSmiles(candidate_smiles)

            if 'use_coreactant_info' in self.configs.keys():
                if self.configs['use_coreactant_info']:
                    coreactants = pd.read_csv(self.configs['coreactants_path'], sep='\t', header=0)
                else:
                    coreactants = None
            else:
                coreactants = None

            products = []
            i = 0
            while len(products) < 1 and i < self.configs["max_rules_by_iter"]:
                rule_info = sample[i].split('\t')
                rule = rule_info[self.configs["rule_smarts_column_index"]]
                rule_id = rule_info[self.configs["rule_id_column_index"]]
                reaction = ReactionFromSmarts(rule)

                if coreactants is not None:
                    rule_reactants = rule_info[self.configs["coreactant_info_column_index"]]
                    reactants_smiles = self.set_coreactants(rule_reactants, candidate_smiles, coreactants)
                    reactants = [MolFromSmiles(m) for m in reactants_smiles]
                else:
                    reactants = candidate_mol

                products = ChemUtils.react(reactants, reaction)

                if len(products) > 0:
                    rp = random.randint(0, len(products) - 1)
                    mutant = products[rp]
                    mutant = MolToSmiles(mutant)
                    # keep biggest
                    mutant = max(mutant.split('.'), key=len)
                    solution.variables = mutant
                    if 'original_compound' not in solution.attributes.keys():
                        solution.attributes['original_compound'] = [candidate_smiles]
                        solution.attributes['rule_id'] = [rule_id]
                    else:
                        solution.attributes['original_compound'].append(candidate_smiles)
                        solution.attributes['rule_id'].append(rule_id)
                    file = f"reactea/outputs/{self.configs['exp_name']}/operatorLogs/ReactionMutationLogs_" \
                           f"{time.strftime('%Y_%m_%d')}.txt"
                    objectives = []
                    # TODO: check if abs makes sense for all objectives
                    for obj in solution.objectives:
                        objectives.append(str(abs(round(obj, 3))))
                    with open(file, 'a') as log:
                        log.write(f"{candidate_smiles},{mutant},{rule_id},{','.join(objectives)}\n")
                i += 1
        return solution

    def set_coreactants(self,
                        reactants: str,
                        mol_smiles: str,
                        coreactants: pd.DataFrame):
        """"""
        reactants_list = []
        if len(reactants.split(';')) > 1:
            for r in reactants.split(';'):
                if r == 'Any':
                    reactants_list.append(mol_smiles)
                else:
                    try:
                        coreactant = coreactants[coreactants[self.configs["coreactant_ids_column_label"]] == r]
                        coreactant_smiles = coreactant[self.configs["coreactant_smiles_column_label"]].values[0]
                        reactants_list.append(coreactant_smiles)
                    except Exception:
                        reactants_list.append(None)
        else:
            reactants_list = [mol_smiles]
        return reactants_list

    def get_name(self):
        return 'Reactor Mutation'


class ReactorOnePointCrossover(Crossover[Solution]):
    """One point Crossover ** performs mutation on both parents (not crossover)

    :param probability: (float) The probability of crossover.
    """

    def __init__(self, probability: float = 1.0, configs: dict = None):
        super(ReactorOnePointCrossover, self).__init__(probability=probability)
        self.configs = configs

    def execute(self, parents: List[Solution]):
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))
        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]

        if random.random() <= self.probability:
            stepbro = ReactorMutation(self.probability, self.configs).execute(offspring[0])
            stepsis = ReactorMutation(self.probability, self.configs).execute(offspring[1])
            offspring[0] = stepbro
            offspring[0].number_of_variables = len(stepbro.variables)
            offspring[1] = stepsis
            offspring[1].number_of_variables = len(stepsis.variables)

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Reactor One Point Crossover'
