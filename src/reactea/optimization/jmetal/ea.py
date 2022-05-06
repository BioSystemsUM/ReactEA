from typing import List

from jmetal.algorithm.multiobjective import NSGAII, SPEA2, GDE3, IBEA, RandomSearch
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.algorithm.singleobjective import SimulatedAnnealing, EvolutionStrategy, LocalSearch

from reactea.optimization.ea import AbstractEA
from .algorithms import ReactorGeneticAlgorithm as GeneticAlgorithm
from .evaluators import ChemicalEvaluator
from .generators import ChemicalGenerator
from .observers import PrintObjectivesStatObserver, VisualizerObserver
from .problem import JmetalProblem
from reactea.utilities.constants import EAConstants, ChemConstants, SAConstants, GAConstants, NSGAIIIConstants, \
    GDE3Constants, ESConstants, LSConstants, IBEAConstants
from ..problem import ChemicalProblem
from ...chem.compounds import Compound
from ...chem.reaction_rules import ReactionRule
from ...chem.standardization import MolecularStandardizer
from ...utilities.io import Writers


class ChemicalEA(AbstractEA):
    """
    Class representing an ChemicalEA.
    Runs the Evolutionary Algorithm engine.
    """

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
        coreactants: List[Compound]
            list of coreactants if available
        max_generations: int
            maximum number of generations
        mp: bool
            use multiprocessing (true) out not (false)
        visualizer: bool
            use visualization of the solutions (true) or not (false)
        algorithm: str
            EA algorithm to use
        configs: dict
            configurations of the experiment
        logger: Writers
            writer to use for the operators logs
        """
        super(ChemicalEA, self).__init__(problem, initial_population, max_generations, mp, visualizer)
        self.algorithm_name = algorithm
        self.reaction_rules = reaction_rules
        self.standardizer = standardizer
        self.coreactants = coreactants
        self.configs = configs
        self.logger = logger
        self.initial_population = ChemicalGenerator(initial_population)
        self.population_evaluator = ChemicalEvaluator()
        self.ea_problem = JmetalProblem(problem, initial_population)
        if isinstance(initial_population, list):
            self.population_size = len(initial_population)
        else:
            self.population_size = 1
        self.termination_criterion = EAConstants.TERMINATION_CRITERION(self.max_generations * self.population_size)

    def _run_so(self):
        """
        Runs a single-objective optimization.

        Returns
        -------
        List[ChemicalSolution]:
            final Chemical solutions
        """
        mutation = EAConstants.MUTATION(EAConstants.MUTATION_PROBABILITY,
                                        self.reaction_rules,
                                        self.standardizer,
                                        self.coreactants,
                                        self.configs,
                                        self.logger)
        crossover = EAConstants.CROSSOVER(EAConstants.CROSSOVER_PROBABILITY,
                                          self.reaction_rules,
                                          self.standardizer,
                                          self.coreactants,
                                          self.configs,
                                          self.logger
                                          )
        if self.algorithm_name == 'SA':
            print("Running Simulated Annealing!")
            if len(self.initial_population.initial_population) != 1:
                raise ValueError('For running SA, only one initial compound must be provided!')
            algorithm = SimulatedAnnealing(problem=self.ea_problem,
                                           mutation=mutation,
                                           termination_criterion=self.termination_criterion,
                                           solution_generator=self.initial_population
                                           )
            algorithm.temperature = SAConstants.TEMPERATURE
            algorithm.minimum_temperature = SAConstants.MINIMUM_TEMPERATURE
            algorithm.alpha = SAConstants.ALPHA
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
                                          elitist=ESConstants.ELITIST,
                                          mutation=mutation,
                                          termination_criterion=self.termination_criterion,
                                          population_generator=self.initial_population,
                                          population_evaluator=self.population_evaluator)
        # TODO: check if LS is working as supposed, fitness never improves, check comparator problem, if problems are
        #  not solvable remove this algorithm
        elif self.algorithm_name == 'LS':
            print("Running Local Search!")
            if len(self.initial_population.initial_population) != 1:
                raise ValueError('For running SA, only one initial compound must be provided!')
            self.termination_criterion = EAConstants.TERMINATION_CRITERION(self.max_generations)
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

    @property
    def _run_mo(self):
        """
        Runs a multi-objective optimization.

        Returns
        -------
        List[ChemicalSolution]:
            final Chemical solutions
        """
        mutation = EAConstants.MUTATION(EAConstants.MUTATION_PROBABILITY,
                                        self.reaction_rules,
                                        self.standardizer,
                                        self.coreactants,
                                        self.configs,
                                        self.logger)
        crossover = EAConstants.CROSSOVER(EAConstants.CROSSOVER_PROBABILITY,
                                          self.reaction_rules,
                                          self.standardizer,
                                          self.coreactants,
                                          self.configs,
                                          self.logger
                                          )
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
            )
        elif self.algorithm_name == 'NSGAII':
            algorithm = NSGAII(
                problem=self.ea_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=mutation,
                crossover=crossover,
                termination_criterion=self.termination_criterion,
                population_generator=self.initial_population,
                population_evaluator=self.population_evaluator
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
        elif self.algorithm_name == "IBEA":
            algorithm = IBEA(
                problem=self.ea_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=mutation,
                crossover=crossover,
                kappa=IBEAConstants.KAPPA,
                termination_criterion=self.termination_criterion,
                population_generator=self.initial_population,
                population_evaluator=self.population_evaluator
            )
        elif self.algorithm_name == "RandomSearch":
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
                population_evaluator=self.population_evaluator
            )
        else:
            raise ValueError('Invalid multi-objective algorithm name. Choose from NSGAII, NSGAIII, SPEA2, GDE3, IBEA '
                             'and RandomSearch!')
        if self.visualizer:
            algorithm.observable.register(observer=VisualizerObserver())
        algorithm.observable.register(observer=PrintObjectivesStatObserver())

        algorithm.run()
        result = algorithm.solutions
        return result
