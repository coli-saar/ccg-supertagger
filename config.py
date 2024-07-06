from glob import glob
from typing import Iterable, Optional

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    # debug: bool = False
    batchsize: int
    # ignore_index: int = -100
    epochs: int
    learning_rate: float
    # dropout: float = 0.1
    # transformer_activation: str = "relu"
    # # betas: List[float] = [0.9, 0.999]
    limit_train: int = 1000000000
    limit_dev: int = 1000000000
    training_data: list[str]
    dev_data: list[str]
    test_data: list[str]
    supertag_vocabulary_filename: str = "supertag_vocabulary.txt"
    model_filename: Optional[str] = None

    def expand_filenames(self, dataset: list[str]) -> list[str]:
        """
        Call as e.g. expand_filenames(config.training_data).

        :param dataset:
        :return:
        """
        ret = []
        for globstr in dataset:
            names = glob(globstr)
            ret.extend(names)

        ret.sort()
        return ret

    @staticmethod
    def load(filename) -> "Config":
        with open("config.yml", "r") as f:
            config = Config(**yaml.safe_load(f))
            return config

