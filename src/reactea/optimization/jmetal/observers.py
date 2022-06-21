import copy
from typing import List, TypeVar

import numpy as np
from jmetal.core.observer import Observer
from jmetal.core.solution import Solution
from jmetal.lab.visualization import StreamingPlot

from reactea.optimization.solution import non_dominated_population

S = TypeVar('S')


class VisualizerObserver(Observer):
    """
    Class representing an observer that plots the pareto front approximation.
    """

    def __init__(self,
                 reference_front: List[S] = None,
                 reference_point: list = None,
                 display_frequency: float = 1.0,
                 non_dominated: bool = True):
        """
        Initializes a VisualizerObserver.

        Parameters
        ----------
        reference_front: List[S]
            Reference front to plot.
        reference_point: list
            Reference point to plot.
        display_frequency: float
            Frequency of updates.
        non_dominated: bool
            If True, only the non-dominated solutions are plotted.
        """
        self.figure = None
        self.display_frequency = display_frequency
        self.reference_point = reference_point
        self.reference_front = reference_front
        self.non_dominated = non_dominated

    def update(self, *args, **kwargs):
        """
        Updates the observer and plots a StreamingPlot with the reference point and front.

        Parameters
        ----------
        args: arguments passed to the observer
        kwargs: keyword arguments passed to the observer.
        """
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']

        if solutions:
            if self.figure is None:

                axis_labels = None
                problem = kwargs['PROBLEM']
                if problem and problem.obj_labels:
                    axis_labels = problem.obj_labels

                self.figure = StreamingPlot(reference_point=self.reference_point,
                                            reference_front=self.reference_front,
                                            axis_labels=axis_labels)
                self.figure.plot(solutions)

            if (evaluations % self.display_frequency) == 0:
                # check if reference point has changed
                reference_point = kwargs.get('REFERENCE_POINT', None)
                # negative fitness values are converted to positive
                population = copy.copy(solutions)
                if self.non_dominated:
                    population = non_dominated_population(population)
                for i in range(len(population)):
                    obj = [abs(x) for x in population[i].objectives]
                    population[i].objectives = obj

                if reference_point:
                    self.reference_point = reference_point
                    self.figure.update(population, reference_point)
                else:
                    self.figure.update(population)

                self.figure.ax.set_title(
                    'Eval: {}'.format(evaluations), fontsize=13)


class PrintObjectivesStatObserver(Observer):
    """
    Class representing objectives statistics observer.
    Outputs the generations step, worst and best solution fitness and the median average and standard deviation
    of all solutions for each objective.
    """

    def __init__(self, frequency: float = 1.0) -> None:
        """
        Initializes a PrintObjectivesStatObserver observer.

        Parameters
        ----------
        frequency: float
            frequency of display
        """
        self.display_frequency = frequency
        self.first = True

    @staticmethod
    def fitness_statistics(solutions: List[Solution], obj_directions: List[int]):
        """

        Parameters
        ----------
        solutions: List[Solution]
            list of solutions
        obj_directions: List[int]
            list with the directions of the objectives (maximize, minimize)

        Returns
        -------
        dict:
            statistics of each objective
        """
        stats = {}
        first = solutions[0].objectives
        n = len(first)
        for i in range(n):
            direction = obj_directions[i]*-1
            f = [p.objectives[i]*direction for p in solutions]

            if direction == 1:  # minimizing
                worst_fit = max(f)
                best_fit = min(f)
            else:
                worst_fit = min(f)
                best_fit = max(f)

            med_fit = np.median(f)
            avg_fit = np.mean(f)
            std_fit = np.std(f)
            stats['obj_{}'.format(i)] = {'best': best_fit, 'worst': worst_fit,
                                         'mean': avg_fit, 'median': med_fit, 'std': std_fit}
        return stats

    @staticmethod
    def stats_to_str(stats: dict, evaluations: int, title: bool = False):
        """

        Parameters
        ----------
        stats: dict
            dictionary with the statistics
        evaluations: int
            evaluations number
        title: bool
            print title (true) or not (only true in the first output)

        Returns
        -------
        str:
            string with the statistics
        """
        if title:
            title = "Eval(s)|"
        values = " {0:>6}|".format(evaluations)

        for key in stats:
            s = stats[key]
            if title:
                title = title + "     Worst      Best    Median   Average   Std Dev|"
            values = values + "  {0:.6f}  {1:.6f}  {2:.6f}  {3:.6f}  {4:.6f}|".format(s['worst'],
                                                                                      s['best'],
                                                                                      s['median'],
                                                                                      s['mean'],
                                                                                      s['std'])
        if title:
            return title+"\n"+values
        else:
            return values

    def update(self, *args, **kwargs):
        """
        Update output with new statistics.

        Parameters
        ----------
        args:
            args to use
        kwargs:
            kwargs to use
        """
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']
        obj_directions = kwargs['PROBLEM'].obj_directions
        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                stats = self.fitness_statistics(solutions, obj_directions)
                message = self.stats_to_str(stats, evaluations, self.first)
                self.first = False
            else:
                fitness = solutions.objectives
                res = abs(fitness[0])
                message = 'Evaluations: {}\tFitness: {}'.format(
                    evaluations, res)
            print(message)
