from abc import ABC, abstractmethod

from rdkit.Chem import MolFromSmiles


class EvaluationFunctionBaseTestCase(ABC):

    def setUp(self) -> None:
        smiles = ['CCCCCCCCCCCCCCCCCC(=O)OC(CO)COP(=O)(O)OCCN', 'CCCCCCCCCCCCCC(=O)OC(CO)COP(=O)(O)OCC(O)CO',
                  'O=C(O)C(=O)CC(O)C(O)COP(=O)(O)O', 'C=CCC(=O)C(=O)O',
                  'COC(=O)C(O)C(OC1OC(CO)C(O)C(O)C1O)C(O)C(CO)OC(C)=O']
        self.mols = [MolFromSmiles(smile) for smile in smiles]

        self.mols_w_invalid = self.mols[:2] + ['CC=(', None] + self.mols[2:] + ['CC=)(C']

    @abstractmethod
    def test_evaluation_function(self):
        pass

    @abstractmethod
    def test_evaluation_function_with_invalid_mols(self):
        pass
