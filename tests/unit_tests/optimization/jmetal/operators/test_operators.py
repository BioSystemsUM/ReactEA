import os
import shutil
from abc import ABC, abstractmethod

from rdkit import RDLogger

from _utils import ROOT_DIR, load_config, SOURCE_DIR


class OperatorsBaseTestCase(ABC):

    def setUp(self):
        # Mute RDKit logs
        RDLogger.DisableLog("rdApp.*")

        data_folder = os.path.join(ROOT_DIR, 'data')
        config_path = os.path.join(data_folder, 'base_config.yaml')
        self.configs = load_config(config_path)
        self.output_folder = f"{SOURCE_DIR}/src/reactea/outputs/{self.configs['exp_name']}/"

    def tearDown(self):
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)

    @abstractmethod
    def test_operator(self):
        pass
