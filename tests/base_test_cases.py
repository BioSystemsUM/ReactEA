import os
import shutil
from abc import ABC, abstractmethod

from rdkit import RDLogger

from _utils import load_config, ROOT_DIR, SOURCE_DIR


class CaseStudiesBaseTestCase(ABC):

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
    def test_case_study(self):
        pass


class AlgorithmsBaseTestCase(ABC):

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
    def test_algorithms(self):
        pass