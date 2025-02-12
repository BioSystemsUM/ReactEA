import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from rdkit import RDLogger

from reactea.io_streams import Loaders

from tests import TEST_DIR


class CaseStudiesBaseTestCase(ABC):

    def setUp(self):
        # Mute RDKit logs
        RDLogger.DisableLog("rdApp.*")

        config_path = TEST_DIR / 'configs' / 'base_config.yaml'
        self.configs = Loaders.get_config_from_yaml(config_path)
        self.output_folder = TEST_DIR / 'outputs' / self.configs['exp_name']
        self.configs['output_dir'] = self.output_folder
        init_pop_path = TEST_DIR / 'data' / 'compounds' / 'compounds_sample.tsv'
        self.configs['init_pop_path'] = init_pop_path.as_posix()

    def tearDown(self):
        output_folder_path = Path(self.output_folder)
        if output_folder_path.exists():
            shutil.rmtree(output_folder_path)

    @abstractmethod
    def test_case_study(self):
        pass


class AlgorithmsBaseTestCase(ABC):

    def setUp(self):
        # Mute RDKit logs
        RDLogger.DisableLog("rdApp.*")

        config_path = TEST_DIR / 'configs' / 'base_config.yaml'
        self.configs = Loaders.get_config_from_yaml(config_path)
        self.output_folder = TEST_DIR / 'outputs' / self.configs['exp_name']
        self.configs['output_dir'] = self.output_folder
        init_pop_path = TEST_DIR / 'data' / 'compounds' / 'compounds_sample.tsv'
        self.configs['init_pop_path'] = init_pop_path

    def tearDown(self):
        output_folder_path = Path(self.output_folder)
        if output_folder_path.exists():
            shutil.rmtree(output_folder_path)

    @abstractmethod
    def test_algorithms(self):
        pass
