
import sys
import torch
import wandb
from datasets import Dataset
from torch import LongTensor
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer, RobertaForTokenClassification, RobertaTokenizer, RobertaTokenizerFast, \
    RobertaModel, XLMRobertaTokenizerFast

import device_selector
from config import Config
from supertag_reader import read_corpus

config = Config.load("config.yml")
device = device_selector.choose_device()


corpus = read_corpus(config.get_training_filenames())
data_dict = corpus.as_dict()
ds = Dataset.from_dict(data_dict)

ds.set_format(type='torch', columns=['words', 'supertags'])
train_dataloader = torch.utils.data.DataLoader(ds, batch_size=1) # https://huggingface.co/docs/datasets/v1.17.0/use_dataset.html

for batch in train_dataloader:
    print(batch)
    sys.exit(0)


# for i in range(10):
#     print(ds[i])

