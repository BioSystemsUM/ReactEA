import copy
from abc import ABC, abstractmethod


class CaseStudy(ABC):
    """"""

    @abstractmethod
    def objective(self, configs: dict, multiObjective: bool = False):
        """"""
        raise NotImplementedError

    @abstractmethod
    def name(self):
        """"""
        return NotImplementedError

    @abstractmethod
    def feval_names(self):
        """"""
        return NotImplementedError

    def __str__(self):
        """"""
        return self.name()
