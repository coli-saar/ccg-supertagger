import sys
from typing import Optional

import torch
from datasets import Dataset, NamedSplit, DatasetDict
from transformers import BatchEncoding

from config import Config
from supertag_reader import read_corpus


class Vocabulary:
    _cached_vocab: bool
    _filename: str
    _stoi: dict[str,int]
    _itos: list[str]

    def __init__(self):
        self._stoi = {}
        self._itos = []

    def stoi(self, s:str) -> int:
        if s in self._stoi:
            return self._stoi[s]
        else:
            next_ix = len(self._itos)
            self._itos.append(s)
            self._stoi[s] = next_ix
            return next_ix

    def itos(self, i:int) -> Optional[str]:
        if i < len(self._itos):
            return self._itos[i]
        else:
            return None

    def __len__(self):
        return len(self._itos)

    def __str__(self):
        return str(self._itos)

    def __repr__(self):
        return str(self)

    def save(self, filename:str) -> None:
        with open(filename, "w") as f:
            for elem in self._itos:
                print(elem, file=f)


    @staticmethod
    def load(filename:str) -> "Vocabulary":
        ret = Vocabulary()
        with open(filename, "r") as f:
            for elem in f:
                ret.stoi(elem.strip())
        return ret

    @staticmethod
    def load_or_initialize(filename:str) -> "Vocabulary":
        """
        If a cached vocabulary exists under the given filename,
        it is loaded and returned. Otherwise, a fresh empty
        vocabulary is returned.

        :param filename:
        :return:
        """
        try:
            supertag_vocab = Vocabulary.load(filename)
            print(f"Loaded vocabulary from {filename}.")
            supertag_vocab._cached_vocab = True
        except:
            supertag_vocab = Vocabulary()
            print(f"Creating fresh vocabulary.")
            supertag_vocab._cached_vocab = False

        supertag_vocab._filename = filename
        return supertag_vocab

    def save_if_fresh(self) -> None:
        """
        If this vocabulary object was created fresh in a call
        to load_or_initialize, it is written to the filename that
        was specified in the call to load_or_initialize. Otherwise
        (i.e. if the vocabulary was loaded from a cache file),
        nothing is done.

        :return:
        """
        if not self._cached_vocab:
            self.save(self._filename)
            print(f"Cached vocabulary to {self._filename}.")


def prettyprint(words, tags, tokenized_inputs:BatchEncoding) -> None:
    for sentence_ix in range(len(words)):
        word_ids = tokenized_inputs.word_ids(batch_index=sentence_ix)
        tokens = tokenized_inputs[sentence_ix].tokens
        for pos, word_id in enumerate(word_ids):
            print("{0: <15}".format(tokens[pos]), end='')

            if word_id is not None:
                print("{0: <15}".format(words[sentence_ix][word_id]), end="")
                print(tags[sentence_ix][word_id], end="\t")

            print()

        print("---")


def accuracy(predicted, gold, ignore_index) -> (int, int):
    counted = torch.sum(gold != ignore_index) # count entries in gold that are not IGNORE_INDEX
    correct = torch.sum(gold == predicted)    # count entries that are the same
    return int(correct), int(counted)

def create_dataset(config:Config) -> DatasetDict:
    # print(config.expand_filenames(config.training_data))
    # print(config.expand_filenames(config.dev_data))
    # print(config.expand_filenames(config.test_data))
    # sys.exit(0)


    data_dict = {
        "train": read_corpus(config.expand_filenames(config.training_data)).as_dict(),
        "dev": read_corpus(config.expand_filenames(config.dev_data)).as_dict(),
        "test": read_corpus(config.expand_filenames(config.test_data)).as_dict()
    }

    ds_train = Dataset.from_dict(data_dict["train"], split=NamedSplit("train"))
    ds_dev = Dataset.from_dict(data_dict["dev"], split=NamedSplit("dev"))

    return DatasetDict({"train": ds_train, "dev": ds_dev})