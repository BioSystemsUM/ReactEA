import os
import shutil
from abc import ABC, abstractmethod

from rdkit import RDLogger

from tests import TEST_DIR
from reactea.io_streams import Loaders


class OperatorsBaseTestCase(ABC):

    def setUp(self):
        # Mute RDKit logs
        RDLogger.DisableLog("rdApp.*")

        config_path = os.path.join(TEST_DIR, 'configs/base_config.yaml')
        self.configs = Loaders.get_config_from_yaml(config_path)
        self.output_folder = f"{TEST_DIR}/outputs/{self.configs['exp_name']}/"

    def tearDown(self):
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

    @abstractmethod
    def test_operator(self):
        pass
