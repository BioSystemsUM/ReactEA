from unittest import TestCase

from reactea.chem.compounds import Compound
from reactea.chem.reaction_rules import ReactionRule
from reactea.chem.standardization import ChEMBLStandardizer
from reactea.optimization.jmetal.operators import ReactorMutation
from reactea.optimization.solution import ChemicalSolution
from reactea.utilities.io import Writers
from .test_operators import OperatorsBaseTestCase


class TestReactorMutation(OperatorsBaseTestCase, TestCase):

    def test_operator(self):
        rrs = [ReactionRule('([#8&v2&H1:1]-[#8&v2&H0:2]-[#6&v4&H1:3])>>([#8&v2&H1:2]-[#6&v4&H1:3].[#8&v2&H2:1])', 'R1'),
               ReactionRule('([#8&v2&H1:1]-[#8&v2&H0:2]-[#6&v4&H1:3])>>([#8&v2&H1:2]-[#6&v4&H1:3].[#8&v2&H2:1])', 'R2')]

        rm1 = ReactorMutation(probability=1.0,
                              reaction_rules=rrs,
                              standardizer=None,
                              coreactants=None,
                              configs=self.configs,
                              logger=None)

        self.assertEqual(rm1.get_name(), 'Reactor Mutation')
        sol1 = rm1.execute(ChemicalSolution(Compound('Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OC(=O)c2cccc(O)c2O)C(O)C1O', 'C1')))
        self.assertIsInstance(sol1, ChemicalSolution)

        rrs2 = [ReactionRule('[#6:1]1=[#6:2]-[#7:3]-[#6:4]=[#6:5]-[#6:6]-1.[#6;$([#6&R]1-&@[#6&R]-&@[#6&R]-&@[#6&R]-&@'
                             '[#6&R]-&@[#6&R]-&@1);!$([#6&R]1-&@[#6&R]-&@[#6&R]-&@[#6&R]-&@[#6&R]-&@[#6&R]-&@1-&!@[#6&!'
                             'R]):7]=[#8:8]>>[#6:7]-[#8:8].[#6:1]1:[#6:6]:[#6:5]:[#6:4]:[#7+:3]:[#6:2]:1',
                             'R1', 'NADH_CoF;Any'),
                ReactionRule('[#6:1]1=[#6:2]-[#7:3]-[#6:4]=[#6:5]-[#6:6]-1.[#6;$([#6&R]1-&@[#6&R]-&@[#6&R]-&@[#6&R]-&@'
                             '[#6&R]-&@[#6&R]-&@1);!$([#6&R]1-&@[#6&R]-&@[#6&R]-&@[#6&R]-&@[#6&R]-&@[#6&R]-&@1-&!@[#6&!'
                             'R]):7]=[#8:8]>>[#6:7]-[#8:8].[#6:1]1:[#6:6]:[#6:5]:[#6:4]:[#7+:3]:[#6:2]:1',
                             'R2', 'NADH_CoF;Any')]
        standardizer = ChEMBLStandardizer()
        coreactants = [Compound('NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP'
                                '(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1', 'NADH_CoF')]
        rm2 = ReactorMutation(probability=1.0,
                              reaction_rules=rrs2,
                              standardizer=standardizer,
                              coreactants=coreactants,
                              configs=self.configs,
                              logger=Writers.update_operators_logs)
        sol2 = rm2.execute(ChemicalSolution(Compound('Nc1ncnc2c1ncn2C1OC(COP(=O)(O)OC(=O)c2cccc(O)c2O)C(O)C1O', 'C1')))
        self.assertEqual(rm2.get_name(), 'Reactor Mutation')
        self.assertIsInstance(sol2, ChemicalSolution)