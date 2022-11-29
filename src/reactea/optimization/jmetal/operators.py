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
                 probability: float,
                 reaction_rules: List[ReactionRule],
                 standardizer: Union[MolecularStandardizer, None],
                 configs: dict,
                 logger: Union[callable, None] = None):
        """
        Initializes a ReactorMutation operator.

        Parameters
        ----------
        probability: float
            probability of mutation to occur
        reaction_rules: List[ReactionRule]
            pool or reaction rules to use
        standardizer: Union[MolecularStandardizer, None]
            standardizer to standardize new solutions
        configs: dict
            configurations of the experiment
        logger: Union[callable, None]
            function to save all intermediate transformations (accepted and not accepted)
        """
        super(ReactorMutation, self).__init__(probability=probability)
        self.reaction_rules = reaction_rules
        self.standardizer = standardizer
        self.configs = configs
        self.logger = logger

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
            products = []
            i = 0
            while len(products) < 1 and i < self.configs["max_rules_by_iter"]:
                i += 1
                rule = self.reaction_rules[random.randint(0, len(self.reaction_rules) - 1)]
                reactants = rule.reactants_to_mol_list(compound)
                products = ChemUtils.react(reactants, rule.reaction)
                if len(products) > 20:
                    products = random.sample(products, 20)
                products = [pd for pd in products if ChemUtils.valid_product(pd)]
                if len(products) > 0:
                    # keep the most similar compound
                    most_similar_product = ChemUtils.most_similar_compound(compound.smiles, products)
                    most_similar_product = ChemUtils.smiles_to_isomerical_smiles(most_similar_product)
                    mutant_id = f"{compound.cmp_id}--{rule.rule_id}_"
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
                 probability: float,
                 reaction_rules: List[ReactionRule],
                 standardizer: Union[MolecularStandardizer, None],
                 configs: dict,
                 logger: Union[callable, None] = None):
        """
        Initializes a ReactorPseudoCrossover operator.

        Parameters
        ----------
        probability: float
            probability of mutation to occur
        reaction_rules: List[ReactionRule]
            pool or reaction rules to use
        standardizer: Union[MolecularStandardizer, None]
            standardizer to standardize new solutions
        configs: dict
            configurations of the experiment
        logger: Union[callable, None]
            function to save all intermediate transformations (accepted and not accepted)
        """
        super(ReactorPseudoCrossover, self).__init__(probability=probability)
        self.reaction_rules = reaction_rules
        self.standardizer = standardizer
        self.configs = configs
        self.logger = logger

    def execute(self, parents: List[ChemicalSolution]):
        """
        Executes the operator by trying to apply a set os reaction rules to both parents to produce
        the offspring.

        Parameters
        ----------
        parents: List[ChemicalSolution]
            parent solutions to mutate
        Returns
        -------
        List[ChemicalSolution]
            mutated offspring solutions
        """
        if len(parents) != 2:
            raise Exception('The number of parents is not two: {}'.format(len(parents)))
        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]

        if random.random() <= self.probability:
            stepbro = ReactorMutation(self.probability,
                                      self.reaction_rules,
                                      self.standardizer,
                                      self.configs,
                                      self.logger).execute(offspring[0])

            stepsis = ReactorMutation(self.probability,
                                      self.reaction_rules,
                                      self.standardizer,
                                      self.configs,
                                      self.logger).execute(offspring[1])
            offspring[0] = stepbro
            offspring[1] = stepsis
        return offspring

    def get_number_of_parents(self) -> int:
        """
        Number of parent compounds used.

        Returns
        -------
        int
            number of parent compounds
        """
        return 2

    def get_number_of_children(self) -> int:
        """
        Number of children compounds created.

        Returns
        -------
        int
            number of children compounds
        """
        return 2

    def get_name(self):
        """
        Get the name of the operator.

        Returns
        -------
        str:
            name of the operator.
        """
        return 'Reactor One Point Crossover'
