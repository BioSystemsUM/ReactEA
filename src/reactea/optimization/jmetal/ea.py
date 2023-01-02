from typing import List

from jmetal.algorithm.multiobjective import NSGAII, SPEA2, IBEA, RandomSearch
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.algorithm.singleobjective import SimulatedAnnealing, LocalSearch

from reactea.optimization.ea import AbstractEA
from .algorithms import ReactorGeneticAlgorithm as GeneticAlgorithm
from .algorithms import ReactorEvolutionStrategy as EvolutionStrategy
from .evaluators import ChemicalEvaluator
from .generators import ChemicalGenerator
from .observers import PrintObjectivesStatObserver, VisualizerObserver
from .problem import JmetalProblem
from reactea.constants import EAConstants, ChemConstants, SAConstants, GAConstants, NSGAIIIConstants, \
    ESConstants, LSConstants, IBEAConstants, NSGAIIConstants, SPEA2Constants
from ..problem import ChemicalProblem
from ...chem.compounds import Compound
from ...chem.reaction_rules import ReactionRule
from reactea.standardizers.standardization import MolecularStandardizer
from ...io_streams import Writers


class ChemicalEA(AbstractEA):
    """
    Class representing an ChemicalEA.
    Runs the Evolutionary Algorithm engine.
    """

    def __init__(self,
                 problem: ChemicalProblem,
                 initial_population: List[Compound],
                 reaction_rules: List[ReactionRule],
                 standardizer: MolecularStandardizer = ChemConstants.STANDARDIZER,
                 max_generations: int = EAConstants.MAX_GENERATIONS,
                 visualizer: bool = EAConstants.VISUALIZER,
                 algorithm: str = EAConstants.ALGORITHM,
                 configs: dict = None,
                 logger: Writers = Writers.update_operators_logs
                 ):
        """
        Initializes the EA object.

        Parameters
        ----------
        problem: ChemicalProblem
            Chemical problem to solve
        initial_population: List[Compound]
            list of compounds that constitute the initial population
        reaction_rules: List[ReactionRule]
            pool of available reaction rules
        standardizer: MolecularStandardizer
            molecular standardizer to use
        max_generations: int
            maximum number of generations
        visualizer: bool
            use visualization of the solutions (true) or not (false)
        algorithm: str
            EA algorithm to use
        configs: dict
            configurations of the experiment
        logger: Writers
            writer to use for the operators logs
        """
        super(ChemicalEA, self).__init__(problem, initial_population, max_generations, visualizer)
        self.algorithm_name = algorithm
        self.reaction_rules = reaction_rules
        self.standardizer = standardizer
        self.configs = configs
        self.logger = logger
        self.initial_population = ChemicalGenerator(initial_population)
        self.population_evaluator = ChemicalEvaluator()
        self.ea_problem = JmetalProblem(problem, initial_population)
        if isinstance(initial_population, list):
            self.population_size = len(initial_population)
        else:
            self.population_size = 1
        self.init_constants()
        self.termination_criterion = EAConstants.TERMINATION_CRITERION(configs['patience'],
                                                                       self.max_generations)

    def init_constants(self):
        if 'mutation_probability' not in self.configs:
            self.configs['mutation_probability'] = EAConstants.MUTATION_PROBABILITY
        if 'crossover_probability' not in self.configs:
            self.configs['crossover_probability'] = EAConstants.CROSSOVER_PROBABILITY
        if 'patience' not in self.configs:
            self.configs['patience'] = EAConstants.PATIENCE
        if 'tolerance' not in self.configs:
            self.configs['tolerance'] = EAConstants.TOLERANCE

        if self.configs['algorithm'] == 'SA':
            if 'temperature' not in self.configs:
                self.configs['temperature'] = SAConstants.TEMPERATURE
            if 'minimum_temperature' not in self.configs:
                self.configs['minimum_temperature'] = SAConstants.MINIMUM_TEMPERATURE
            if 'alpha' not in self.configs:
                self.configs['alpha'] = SAConstants.ALPHA
        elif self.configs['algorithm'] == 'ES':
            if 'elitist' not in self.configs:
                self.configs['elitist'] = ESConstants.ELITIST
        elif self.configs['algorithm'] == 'IBEA':
            if 'kappa' not in self.configs:
                self.configs['kappa'] = IBEAConstants.KAPPA


    def _run_so(self):
        """
        Runs a single-objective optimization.

        Returns
        -------
        List[ChemicalSolution]:
            final Chemical solutions
        """
        mutation = EAConstants.MUTATION(self.reaction_rules,
                                        self.standardizer,
                                        self.configs,
                                        self.logger)
        try:
            crossover = EAConstants.CROSSOVER(self.reaction_rules,
                                              self.standardizer,
                                              self.configs,
                                              self.logger,
                                              )
        except TypeError:
            crossover = EAConstants.CROSSOVER()
        if self.algorithm_name == 'SA':
            print("Running Simulated Annealing!")
            if len(self.initial_population.initial_population) != 1:
                raise ValueError('For running SA, only one initial compound must be provided!')
            algorithm = SimulatedAnnealing(problem=self.ea_problem,
                                           mutation=mutation,
                                           termination_criterion=self.termination_criterion,
                                           solution_generator=self.initial_population
                                           )
            algorithm.temperature = self.configs['temperature']
            algorithm.minimum_temperature = self.configs['minimum_temperature']
            algorithm.alpha = self.configs['alpha']
        elif self.algorithm_name == 'GA':
            print("Running Genetic Algorithm!")
            algorithm = GeneticAlgorithm(problem=self.ea_problem,
                                         population_size=self.population_size,
                                         offspring_population_size=self.population_size,
                                         mutation=mutation,
                                         crossover=crossover,
                                         selection=GAConstants.SELECTION(),
                                         termination_criterion=self.termination_criterion,
                                         population_generator=self.initial_population,
                                         population_evaluator=self.population_evaluator
                                         )
        elif self.algorithm_name == 'ES':
            print("Running Evolutionary Strategy!")
            algorithm = EvolutionStrategy(problem=self.ea_problem,
                                          mu=int(self.population_size),
                                          lambda_=self.population_size,
                                          elitist=self.configs['elitist'],
                                          mutation=mutation,
                                          termination_criterion=self.termination_criterion,
                                          population_generator=self.initial_population,
                                          population_evaluator=self.population_evaluator)
        elif self.algorithm_name == 'LS':
            print("Running Local Search!")
            if len(self.initial_population.initial_population) != 1:
                raise ValueError('For running LS, only one initial compound must be provided!')
            algorithm = LocalSearch(problem=self.ea_problem,
                                    mutation=mutation,
                                    termination_criterion=self.termination_criterion,
                                    comparator=LSConstants.COMPARATOR)
        else:
            raise ValueError('Invalid single-objective algorithm name. Choose from SA, GA, ES and LS!')

        algorithm.observable.register(observer=PrintObjectivesStatObserver())
        algorithm.run()

        result = algorithm.solutions
        return result

    def _run_mo(self):
        """
        Runs a multi-objective optimization.

        Returns
        -------
        List[ChemicalSolution]:
            final Chemical solutions
        """
        mutation = EAConstants.MUTATION(self.reaction_rules,
                                        self.standardizer,
                                        self.configs,
                                        self.logger)
        try :
            crossover = EAConstants.CROSSOVER(self.reaction_rules,
                                              self.standardizer,
                                              self.configs,
                                              self.logger)
        except TypeError:
            crossover = EAConstants.CROSSOVER()
        print(f"Running {self.algorithm_name}")
        if self.algorithm_name == 'NSGAIII':
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
                dominance_comparator=NSGAIIIConstants.DOMINANCE_COMPARATOR,
                selection=NSGAIIIConstants.SELECTION
            )
        elif self.algorithm_name == 'NSGAII':
            algorithm = NSGAII(
                problem=self.ea_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=mutation,
                crossover=crossover,
                selection=NSGAIIConstants.SELECTION,
                termination_criterion=self.termination_criterion,
                population_generator=self.initial_population,
                population_evaluator=self.population_evaluator,
                dominance_comparator=NSGAIIConstants.DOMINANCE_COMPARATOR
            )
        elif self.algorithm_name == "IBEA":
            algorithm = IBEA(
                problem=self.ea_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=mutation,
                crossover=crossover,
                kappa=self.configs['kappa'],
                termination_criterion=self.termination_criterion,
                population_generator=self.initial_population,
                population_evaluator=self.population_evaluator
            )
        # TODO: debug randomsearch
        elif self.algorithm_name == "RandomSearch":
            if len(self.initial_population.initial_population) != 1:
                raise ValueError('For running RandomSearch, only one initial compound must be provided!')
            algorithm = RandomSearch(
                problem=self.ea_problem,
                termination_criterion=self.termination_criterion
            )
        elif self.algorithm_name == "SPEA2":
            algorithm = SPEA2(
                problem=self.ea_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=mutation,
                crossover=crossover,
                termination_criterion=self.termination_criterion,
                population_generator=self.initial_population,
                population_evaluator=self.population_evaluator,
                dominance_comparator=SPEA2Constants.DOMINANCE_COMPARATOR
            )
        else:
            raise ValueError('Invalid multi-objective algorithm name. Choose from NSGAII, NSGAIII, SPEA2, IBEA '
                             'and RandomSearch!')
        if self.visualizer:
            algorithm.observable.register(observer=VisualizerObserver())
        algorithm.observable.register(observer=PrintObjectivesStatObserver())

        algorithm.run()
        results = algorithm.solutions
        return results
