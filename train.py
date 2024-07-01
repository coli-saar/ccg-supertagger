
import sys
import torch
import wandb
from datasets import Dataset
from torch import LongTensor
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Adam
from transformers import AutoTokenizer, RobertaForTokenClassification, RobertaTokenizer, RobertaTokenizerFast, \
    RobertaModel, XLMRobertaTokenizerFast, BatchEncoding

import device_selector
from config import Config
from supertag_reader import read_corpus
from util import Vocabulary

config = Config.load("config.yml")
IGNORE_INDEX=-100
device = device_selector.choose_device()

corpus = read_corpus(config.get_training_filenames())
data_dict = corpus.as_dict()
dataset = Dataset.from_dict(data_dict)

try:
    supertag_vocab = Vocabulary.load(config.supertag_vocabulary_filename)
    print(f"Loaded vocabulary from {config.supertag_vocabulary_filename}.")
    cached_vocab = True
except:
    supertag_vocab = Vocabulary()
    print(f"Creating fresh vocabulary.")
    cached_vocab = False


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

def tokenize_and_align_labels(examples, label_all_tokens=False, skip_index=IGNORE_INDEX):
    # adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ

    tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True, padding=True) # get "tokenizer" from global variable
    # examples["token_ids"] = tokenized_inputs.input_ids # [sentence_ix][pos] = numeric token ID
    # print(examples)

    # see end of file for an example of what these look like at this point
    # prettyprint(examples["words"], examples["supertags"], tokenized_inputs)

    supertag_ids_whole_batch = []
    for batch_ix, supertags_this_sentence in enumerate(examples["supertags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_ix)
        previous_word_idx = None
        supertag_ids : list[int] = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                supertag_ids.append(skip_index)

            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                supertag_ids.append(supertag_vocab.stoi(supertags_this_sentence[word_idx]))

            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                supertag_ids.append(supertag_vocab.stoi(supertags_this_sentence[word_idx]) if label_all_tokens else skip_index)

            previous_word_idx = word_idx

        supertag_ids_whole_batch.append(supertag_ids)

    tokenized_inputs["supertag_ids"] = supertag_ids_whole_batch
    return tokenized_inputs



tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", add_prefix_space=True)
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True, batch_size=config.batchsize) # this is super cool

if not cached_vocab:
    supertag_vocab.save(config.supertag_vocabulary_filename)
    print(f"Cached vocabulary to {config.supertag_vocabulary_filename}.")



tokenized_datasets.set_format(type='torch', columns=['input_ids', 'supertag_ids', 'attention_mask'])
train_dataloader = torch.utils.data.DataLoader(tokenized_datasets, batch_size=config.batchsize) # https://huggingface.co/docs/datasets/v1.17.0/use_dataset.html

for batch in train_dataloader:
    print(batch)
    sys.exit(0)





#
# [None, 0, 1, 1, 2, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 13, 13, 13, 14, 15, 15, 16, 17, 17, None]
# <s>
# ▁Pierre        Pierre         N/N
# ▁Vin           Vinken         N
# ken            Vinken         N
# ▁              ,              ,
# ,              ,              ,
# ▁61            61             N/N
# ▁years         years          N
# ▁old           old            (S[adj]\NP)\NP
# ▁              ,              ,
# ,              ,              ,
# ▁will          will           (S[dcl]\NP)/(S[b]\NP)
# ▁join          join           ((S[b]\NP)/PP)/NP
# ▁the           the            NP[nb]/N
# ▁board         board          N
# ▁as            as             PP/NP
# ▁a             a              NP[nb]/N
# ▁non           nonexecutive   N/N
# exe            nonexecutive   N/N
# cu             nonexecutive   N/N
# tive           nonexecutive   N/N
# ▁director      director       N
# ▁Nov           Nov.           ((S\NP)\(S\NP))/N[num]
# .              Nov.           ((S\NP)\(S\NP))/N[num]
# ▁29            29             N[num]
# ▁              .              .
# .              .              .
# </s>
# ---
# <s>
# ▁Mr            Mr.            N/N
# .              Mr.            N/N
# ▁Vin           Vinken         N
# ken            Vinken         N
# ▁is            is             (S[dcl]\NP)/NP
# ▁chair         chairman       N
# man            chairman       N
# ▁of            of             (NP\NP)/NP
# ▁El            Elsevier       N/N
# se             Elsevier       N/N
# vier           Elsevier       N/N
# ▁N             N.V.           N
# .              N.V.           N
# V              N.V.           N
# .              N.V.           N
# ▁              ,              ,
# ,              ,              ,
# ▁the           the            NP[nb]/N
# ▁Dutch         Dutch          N/N
# ▁publish       publishing     N/N
# ing            publishing     N/N
# ▁group         group          N
# ▁              .              .
# .              .              .
# </s>
# <pad>
# <pad>