from unittest import TestCase

from reactea.chem import ReactionRule


class TestReactionRules(TestCase):

    def test_reaction_rules(self):
        rr1 = '([#6&v4&H1:1](=[#8&v2&H0:2])-[#6&v4&H2:3]-[#6&v4&H3:4])>>([#6&v4&H2:1](-[#8&v2&H1:2])-[#6&v4&H1:3](-[#6&v4&H3:4])-[#8&v2&H1])'
        id1 = 0
        reactants1 = 'Any'
        rr2 = '([#6&v4&H2:1](-[#8&v2&H1:2])-[#6&v4&H1:3](-[#6&v4&H3:4])-[#8&v2&H1:5])>>([#6&v4&H1:1](=[#8&v2&H0:2])-[#6&v4&H2:3]-[#6&v4&H3:4].[#8&v2&H2:5])'
        id2 = 'id2'
        reactants2 = None
        # invalid reaction rule
        rr3 = '!!([#6&v4&H2:1](-[#8&v2&H1:2])-[#6&v4&H1:3](-[#6&v4&H3:4])-[#8&v2&H1:5])>>([#6&v4&H1:1](=[#8&v2&H0:2])-[#6&v4&H2:3]-[#6&v4&H3:4].[#8&v2&H2:5])'
        id3 = 'id3'
        reactants3 = 'Any;Any'

        r_r_1 = ReactionRule(rr1, id1, reactants1)
        r_r_2 = ReactionRule(rr2, id2, reactants2)
        r_r_3 = ReactionRule(rr3, id3, reactants3)

        self.assertEqual(r_r_1.smarts, rr1)
        self.assertEqual(r_r_2.smarts, rr2)
        self.assertEqual(r_r_3.smarts, rr3)
        self.assertEqual(r_r_1.rule_id, id1)
        self.assertEqual(r_r_2.rule_id, id2)
        self.assertEqual(r_r_3.rule_id, id3)
        self.assertEqual(r_r_1.reactants, reactants1)
        self.assertEqual(r_r_2.reactants, 'Any')
        self.assertEqual(r_r_3.reactants, reactants3)

        self.assertTrue(r_r_1.reaction, None)

        r_r_3.reaction = r_r_1.reaction
        self.assertEqual(r_r_3.reaction, r_r_1.reaction)
        self.assertFalse(r_r_3.smarts == rr3)

        r_r_2.smarts = r_r_1.smarts
        r_r_2.rule_id = r_r_1.rule_id
        self.assertEqual(r_r_2.smarts, r_r_1.smarts)
        self.assertEqual(r_r_2.rule_id, r_r_1.rule_id)

        with self.assertRaises(ValueError):
            r_r_2.reactants = r_r_1.reactants

        invalid_smarts = r_r_1.smarts[10:15]
        r_r_2.smarts = invalid_smarts
        self.assertEqual(r_r_2.smarts, invalid_smarts)
        self.assertTrue(r_r_2.reaction is None)
