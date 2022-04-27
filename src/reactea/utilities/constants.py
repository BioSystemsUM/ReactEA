from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.termination_criterion import StoppingByEvaluations

from reactea.chem.standardization import ChEMBLStandardizer
from reactea.optimization.jmetal.operators import ReactorOnePointCrossover, ReactorMutation


class EAConstants:

    # Maximum number of generations (used as stopping criteria for the EA)
    MAX_GENERATIONS = 100
    # Multiprocessing
    MP = True
    # Default MOEA
    ALGORITHM = 'NSGAIII'
    # Visualizer
    VISUALIZER = False
    # Termination Criterion
    TERMINATION_CRITERION = StoppingByEvaluations


class SAConstants:
    """"""
    MUTATION = ReactorMutation
    MUTATION_PROBABILITY = 1
    TEMPERATURE = 1.0
    MINIMUM_TEMPERATURE = 0.000001
    ALPHA = 0.95


class GAConstants:
    """"""
    MUTATION = ReactorMutation
    MUTATION_PROBABILITY = 1
    CROSSOVER = ReactorOnePointCrossover
    CROSSOVER_PROBABILITY = 1
    SELECTION = BinaryTournamentSelection


class NSGAIIIConstants:
    """"""
    MUTATION = ReactorMutation
    MUTATION_PROBABILITY = 1
    CROSSOVER = ReactorOnePointCrossover
    CROSSOVER_PROBABILITY = 1
    REFERENCE_DIRECTIONS = UniformReferenceDirectionFactory


class GDE3Constants:
    """"""
    CR = 0.5
    F = 0.5
    K = 0.5


class ChemConstants:
    """"""
    # Default standardizer
    STANDARDIZER = ChEMBLStandardizer
