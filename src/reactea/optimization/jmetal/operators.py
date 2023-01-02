import copy
import random
from typing import List, Union

from jmetal.core.operator import Mutation, Crossover

from reactea.chem.compounds import Compound
from reactea.chem.reaction_rules import ReactionRule
from reactea.standardizers.standardization import MolecularStandardizer
from reactea.optimization.solution import ChemicalSolution
from reactea.chem.chem_utils import ChemUtils


class ReactorMutation(Mutation[ChemicalSolution]):
    """
    Class representing a reactor mutation operator.
    A reactor mutation applies alterations in a ChemicalSolution by transforming a reagent (present solution)
    into a product (mutated solution) using reaction rules.
    """

    def __init__(self,
                 reaction_rules: List[ReactionRule],
                 standardizer: Union[MolecularStandardizer, None],
                 configs: dict,
                 logger: Union[callable, None] = None):
        """
        Initializes a ReactorMutation operator.

        Parameters
        ----------
        reaction_rules: List[ReactionRule]
            pool or reaction rules to use
        standardizer: Union[MolecularStandardizer, None]
            standardizer to standardize new solutions
        configs: dict
            configurations of the experiment
        logger: Union[callable, None]
            function to save all intermediate transformations (accepted and not accepted)
        tolerance: float
        """
        super(ReactorMutation, self).__init__(probability=configs['mutation_probability'])
        self.reaction_rules = reaction_rules
        self.standardizer = standardizer
        self.configs = configs
        self.logger = logger
        self.tolerance = configs['tolerance']

    def execute(self, solution: ChemicalSolution):
        """
        Executes the mutation by trying to apply a set os reaction rules to the compound.
        Random reaction rules are picked until one can match and produce a product using the present compound.
        If a maximum number of tries is reached without a match the mutation doesn't happen and the compound
        remains the same.

        Parameters
        ----------
        solution: ChemicalSolution
            solution to mutate
        Returns
        -------
        ChemicalSolution
            mutated solution
        """
        if random.random() <= self.probability:
            compound = solution.variables
            rules = random.sample(self.reaction_rules, self.configs['max_rules_by_iter'])
            products = []
            i = 0
            while len(products) < 1 and i < self.configs["max_rules_by_iter"]:
                rule = rules[i]
                reactants = rule.reactants_to_mol_list(compound)
                products = ChemUtils.react(reactants, rule.reaction)
                if len(products) > 20:
                    products = random.sample(products, 20)
                products = [pd for pd in products if ChemUtils.valid_product(pd)]
                if len(products) > 0:
                    # keep the most similar compound
                    most_similar_product = ChemUtils.most_similar_compound(compound.smiles, products, self.tolerance)
                    mutant_id = f"{compound.cmp_id}--{rule.rule_id}_"
                    if not isinstance(most_similar_product, str):
                        products = []
                    else:
                        mutant = Compound(most_similar_product, mutant_id)
                        if mutant.mol is not None:
                            mutant = self.standardizer().standardize(mutant)
                            if self.logger:
                                self.logger(self.configs, solution, mutant.smiles, rule.rule_id)
                            solution.variables = mutant
                            if 'original_compound' not in solution.attributes.keys():
                                solution.attributes['original_compound'] = [compound.smiles]
                                solution.attributes['rule_id'] = [rule.rule_id]
                            else:
                                solution.attributes['original_compound'].append(compound.smiles)
                                solution.attributes['rule_id'].append(rule.rule_id)
                        else:
                            products = []
                i += 1
        return solution

    def get_name(self):
        """
        Get the name of the operator.

        Returns
        -------
        str:
            name of the operator.
        """
        return 'Reactor Mutation'


class ReactorPseudoCrossover(Crossover[ChemicalSolution, ChemicalSolution]):
    """
    Class representing a reactor pseudo crossover operator.
    A reactor pseudo crossover applies a ReactorMutation operator to both parents producing two children
    compounds.
    """

    def __init__(self,
                 reaction_rules: List[ReactionRule],
                 standardizer: Union[MolecularStandardizer, None],
                 configs: dict,
                 logger: Union[callable, None] = None):
        """
        Initializes a ReactorPseudoCrossover operator.

        Parameters
        ----------
        reaction_rules: List[ReactionRule]
            pool or reaction rules to use
        standardizer: Union[MolecularStandardizer, None]
            standardizer to standardize new solutions
        configs: dict
            configurations of the experiment
        logger: Union[callable, None]
            function to save all intermediate transformations (accepted and not accepted)
        """
        super(ReactorPseudoCrossover, self).__init__(probability=configs['crossover_probability'])
        self.reaction_rules = reaction_rules
        self.standardizer = standardizer
        self.configs = configs
        self.logger = logger

    def execute(self, parent: List[ChemicalSolution]):
        """
        Executes the operator by trying to apply a set os reaction rules to the parent to produce
        the offspring.

        Parameters
        ----------
        parent: List[ChemicalSolution]
            parent solution to mutate
        Returns
        -------
        List[ChemicalSolution]
            mutated offspring solution
        """
        if len(parent) != self.get_number_of_parents():
            raise Exception('The number of parents is not two: {}'.format(len(parent)))
        offspring = [copy.deepcopy(parent[0])]

        if random.random() <= self.probability:
            m_offspring = ReactorMutation(self.reaction_rules,
                                          self.standardizer,
                                          self.configs,
                                          self.logger).execute(offspring[0])
            offspring[0] = m_offspring
        return offspring

    def get_number_of_parents(self) -> int:
        """
        Number of parent compounds used.

        Returns
        -------
        int
            number of parent compounds
        """
        return 1

    def get_number_of_children(self) -> int:
        """
        Number of children compounds created.

        Returns
        -------
        int
            number of children compounds
        """
        return 1

    def get_name(self):
        """
        Get the name of the operator.

        Returns
        -------
        str:
            name of the operator.
        """
        return 'Reactor One Point PseudoCrossover'
