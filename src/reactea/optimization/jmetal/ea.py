from typing import List

from jmetal.algorithm.multiobjective import NSGAII, SPEA2, GDE3
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.algorithm.singleobjective import SimulatedAnnealing

from reactea.optimization.ea import AbstractEA
from .algorithms import ReactorGeneticAlgorithm as GeneticAlgorithm
from .evaluators import ChemicalEvaluator
from .generators import ChemicalGenerator
from .observers import PrintObjectivesStatObserver, VisualizerObserver
from .problem import JmetalProblem
from reactea.utilities.constants import EAConstants, ChemConstants, SAConstants, GAConstants, NSGAIIIConstants, \
    GDE3Constants
from ..problem import ChemicalProblem
from ...chem.compounds import Compound
from ...chem.reaction_rules import ReactionRule
from ...chem.standardization import MolecularStandardizer
from ...utilities.io import Writers

soea_map = {
    'GA': GeneticAlgorithm,
    'SA': SimulatedAnnealing
}
# MOEA alternatives
moea_map = {
    'NSGAII': NSGAII,
    'SPEA2': SPEA2,
    'NSGAIII': NSGAIII,
    'GDE3': GDE3
}


class EA(AbstractEA):
    """"""

    def __init__(self,
                 problem: ChemicalProblem,
                 initial_population: List[Compound] = None,
                 reaction_rules: List[ReactionRule] = None,
                 standardizer: MolecularStandardizer = ChemConstants.STANDARDIZER,
                 coreactants: List[Compound] = None,
                 max_generations: int = EAConstants.MAX_GENERATIONS,
                 mp: bool = EAConstants.MP,
                 visualizer: bool = EAConstants.VISUALIZER,
                 algorithm: str = EAConstants.ALGORITHM,
                 batched: bool = EAConstants.BATCHED,
                 configs: dict = None,
                 logger: Writers = Writers.update_operators_logs
                 ):
        super(EA, self).__init__(problem, initial_population, max_generations, mp, visualizer)
        self.algorithm_name = algorithm
        self.ea_problem = JmetalProblem(problem, batched=batched)
        self.reaction_rules = reaction_rules
        self.standardizer = standardizer
        self.coreactants = coreactants
        self.configs = configs
        self.logger = logger
        self.initial_population = ChemicalGenerator(initial_population)
        self.population_evaluator = ChemicalEvaluator()
        if isinstance(initial_population, list):
            self.population_size = len(initial_population)
        else:
            self.population_size = 1
        self.termination_criterion = EAConstants.TERMINATION_CRITERION(self.max_generations * self.population_size)

    def _run_so(self):
        """"""
        if self.algorithm_name == 'SA':
            print("Running SA")
            mutation = SAConstants.MUTATION(SAConstants.MUTATION_PROBABILITY,
                                            self.reaction_rules,
                                            self.standardizer,
                                            self.coreactants,
                                            self.configs,
                                            self.logger)
            algorithm = SimulatedAnnealing(
                problem=self.ea_problem,
                mutation=mutation,
                termination_criterion=self.termination_criterion,
                solution_generator=self.initial_population
            )
            algorithm.temperature = SAConstants.TEMPERATURE
            algorithm.minimum_temperature = SAConstants.MINIMUM_TEMPERATURE
            algorithm.alpha = SAConstants.ALPHA
        else:
            print("Running GA")
            mutation = GAConstants.MUTATION(GAConstants.MUTATION_PROBABILITY,
                                            self.reaction_rules,
                                            self.standardizer,
                                            self.coreactants,
                                            self.configs,
                                            self.logger)
            crossover = GAConstants.CROSSOVER(GAConstants.CROSSOVER_PROBABILITY,
                                              self.reaction_rules,
                                              self.standardizer,
                                              self.coreactants,
                                              self.configs,
                                              self.logger
                                              )
            algorithm = GeneticAlgorithm(
                problem=self.ea_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=mutation,
                crossover=crossover,
                selection=GAConstants.SELECTION(),
                termination_criterion=self.termination_criterion,
                population_generator=self.initial_population,
                population_evaluator=self.population_evaluator
            )

        algorithm.observable.register(observer=PrintObjectivesStatObserver())
        algorithm.run()

        result = algorithm.solutions
        return result

    def _run_mo(self):
        """"""
        if self.algorithm_name in moea_map.keys():
            f = moea_map[self.algorithm_name]
        else:
            if self.ea_problem.number_of_objectives > 2:
                self.algorithm_name = 'NSGAIII'
                f = moea_map['NSGAIII']
            else:
                f = moea_map['SPEA2']

        print(f"Running {self.algorithm_name}")
        if self.algorithm_name == 'NSGAIII':
            mutation = NSGAIIIConstants.MUTATION(GAConstants.MUTATION_PROBABILITY,
                                                 self.reaction_rules,
                                                 self.standardizer,
                                                 self.coreactants,
                                                 self.configs,
                                                 self.logger)
            crossover = NSGAIIIConstants.CROSSOVER(NSGAIIIConstants.CROSSOVER_PROBABILITY,
                                                   self.reaction_rules,
                                                   self.standardizer,
                                                   self.coreactants,
                                                   self.configs,
                                                   self.logger
                                                   )
            algorithm = NSGAIII(
                problem=self.ea_problem,
                population_size=self.population_size,
                mutation=mutation,
                crossover=crossover,
                termination_criterion=self.termination_criterion,
                reference_directions=NSGAIIIConstants.REFERENCE_DIRECTIONS(self.ea_problem.number_of_objectives,
                                                                           n_points=self.population_size - 1),
                population_generator=self.initial_population,
                population_evaluator=self.population_evaluator,
            )
        elif self.algorithm_name == "GDE3":
            algorithm = GDE3(
                problem=self.ea_problem,
                population_size=self.population_size,
                cr=GDE3Constants.CR,
                f=GDE3Constants.F,
                k=GDE3Constants.K,
                termination_criterion=self.termination_criterion,
                population_generator=self.initial_population,
                population_evaluator=self.population_evaluator
            )
        else:
            mutation = NSGAIIIConstants.MUTATION(GAConstants.MUTATION_PROBABILITY,
                                                 self.reaction_rules,
                                                 self.standardizer,
                                                 self.coreactants,
                                                 self.configs,
                                                 self.logger)
            crossover = NSGAIIIConstants.CROSSOVER(NSGAIIIConstants.CROSSOVER_PROBABILITY,
                                                   self.reaction_rules,
                                                   self.standardizer,
                                                   self.coreactants,
                                                   self.configs,
                                                   self.logger
                                                   )
            algorithm = f(
                problem=self.ea_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=mutation,
                crossover=crossover,
                termination_criterion=self.termination_criterion,
                population_generator=self.initial_population,
                population_evaluator=self.population_evaluator
            )

        if self.visualizer:
            algorithm.observable.register(observer=VisualizerObserver())
        algorithm.observable.register(observer=PrintObjectivesStatObserver())

        algorithm.run()
        result = algorithm.solutions
        return result
