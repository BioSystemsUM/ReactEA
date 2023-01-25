from jmetal.algorithm.multiobjective.nsgaiii import UniformReferenceDirectionFactory
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import DominanceComparator, MultiComparator
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.ranking import FastNonDominatedRanking

from reactea.chem import ChEMBLStandardizer
from reactea.optimization.comparators import ParetoDominanceComparator
from reactea.optimization.jmetal.operators import ReactorPseudoCrossover, ReactorMutation
from reactea.optimization.jmetal.terminators import StoppingByEvaluationsOrImprovement, StoppingByEvaluations


class ExperimentConstants:
    """
    Class containing a set of the experiment constants
    """
    RULES_PATH = '/data/reactionrules/retrorules/retrorules_forward_score.5.tsv'
    MAX_RULES_BY_ITER = 1000


class EAConstants:
    """
    Class containing a set of EA parameters constants
    """
    # Mutation
    MUTATION = ReactorMutation
    # Crossover
    CROSSOVER = ReactorPseudoCrossover


class SAConstants:
    """
    Class containing a set of Simulated Annealing parameters constants
    """
    TERMINATION_CRITERION = StoppingByEvaluations


class GAConstants:
    """
    Class containing a set of Genetic Algorithm parameters constants
    """
    SELECTION = BinaryTournamentSelection


class LSConstants:
    """
    Class containing a set of Local Search parameters constants
    """
    COMPARATOR = ParetoDominanceComparator()


class NSGAIIIConstants:
    """
    Class containing a set of NSGAIII parameters constants
    """
    REFERENCE_DIRECTIONS = UniformReferenceDirectionFactory
    DOMINANCE_COMPARATOR = DominanceComparator()
    SELECTION = BinaryTournamentSelection(MultiComparator([FastNonDominatedRanking.get_comparator(),
                                                           CrowdingDistance.get_comparator()]))


class NSGAIIConstants:
    """
    Class containing a set of NSGAII parameters constants
    """
    DOMINANCE_COMPARATOR = DominanceComparator()
    SELECTION = BinaryTournamentSelection(MultiComparator([FastNonDominatedRanking.get_comparator(),
                                                           CrowdingDistance.get_comparator()]))


class SPEA2Constants:
    """
    Class containing a set of SPEA2 parameters constants
    """
    DOMINANCE_COMPARATOR = DominanceComparator()


class ChemConstants:
    """
    Class containing a set of chemical constants
    """
    # Default standardizer
    STANDARDIZER = ChEMBLStandardizer
