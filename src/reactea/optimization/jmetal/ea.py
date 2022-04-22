from typing import List

from jmetal.algorithm.multiobjective import NSGAII, SPEA2, GDE3
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.algorithm.singleobjective import SimulatedAnnealing
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations

from reactea.optimization.ea import AbstractEA
from .algorithms import ReactorGeneticAlgorithm
from .evaluators import ChemicalEvaluator
from .generators import ChemicalGenerator
from .observers import PrintObjectivesStatObserver, VisualizerObserver
from .operators import ReactorMutation, ReactorOnePointCrossover
from .problem import JmetalProblem
from reactea.utils.constatns import EAConstants
from ..problem import ChemicalProblem

soea_map = {
    'GA': ReactorGeneticAlgorithm,
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
                 initial_population: List[str] = None,
                 max_generations: int = EAConstants.MAX_GENERATIONS,
                 mp: bool = True,
                 visualizer: bool = False,
                 algorithm: str = None,
                 batched: bool = True,
                 configs: dict = None
                 ):
        super(EA, self).__init__(problem, initial_population, max_generations, mp, visualizer)
        self.algorithm_name = algorithm
        self.ea_problem = JmetalProblem(problem, batched=batched)
        self.crossover = ReactorOnePointCrossover(1, configs)
        self.mutation = ReactorMutation(1, configs)
        self.configs = configs
        self.initial_population = ChemicalGenerator(initial_population)
        self.population_evaluator = ChemicalEvaluator()
        if isinstance(initial_population, list):
            self.population_size = len(initial_population)
        else:
            self.population_size = 1
        self.termination_criterion = StoppingByEvaluations(self.max_generations * self.population_size)

    def _run_so(self):
        """"""
        if self.algorithm_name == 'SA':
            print("Running SA")
            algorithm = SimulatedAnnealing(
                problem=self.ea_problem,
                mutation=self.mutation,
                termination_criterion=self.termination_criterion,
                solution_generator=self.initial_population
            )
            if "temperature" in self.configs.keys():
                print(f"Temperature: {self.configs['temperature']}")
                algorithm.temperature = self.configs["temperature"]
        else:
            print("Running GA")
            algorithm = GeneticAlgorithm(
                problem=self.ea_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=self.mutation,
                crossover=self.crossover,
                selection=BinaryTournamentSelection(),
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
            algorithm = NSGAIII(
                problem=self.ea_problem,
                population_size=self.population_size,
                mutation=self.mutation,
                crossover=self.crossover,
                termination_criterion=self.termination_criterion,
                reference_directions=UniformReferenceDirectionFactory(self.ea_problem.number_of_objectives,
                                                                      n_points=self.population_size - 1),
                population_generator=self.initial_population,
                population_evaluator=self.population_evaluator
            )
        elif self.algorithm_name == "GDE3":
            algorithm = GDE3(
                problem=self.ea_problem,
                population_size=self.population_size,
                cr=0.5,
                f=0.5,
                termination_criterion=self.termination_criterion,
                population_generator=self.initial_population,
                population_evaluator=self.population_evaluator
            )
        else:
            algorithm = f(
                problem=self.ea_problem,
                population_size=self.population_size,
                offspring_population_size=self.population_size,
                mutation=self.mutation,
                crossover=self.crossover,
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
