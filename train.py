
import sys
import torch
from tqdm import tqdm

import wandb
from datasets import Dataset, DatasetDict
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import XLMRobertaTokenizerFast

import device_selector
from config import Config
from model import TaggingModel
from supertag_reader import read_corpus
from util import Vocabulary, create_dataset, accuracy

# suppress tokenizer parallelization
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = Config.load("config.yml")
IGNORE_INDEX=-100
device = device_selector.choose_device()

dataset = create_dataset(config)
supertag_vocab = Vocabulary.load_or_initialize(config.supertag_vocabulary_filename)


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


# tokenize and vocabularize the dataset
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", add_prefix_space=True)
tokenized_datasets: DatasetDict = dataset.map(tokenize_and_align_labels, batched=True, batch_size=config.batchsize) # this is super cool
supertag_vocab.save_if_fresh()

# create batched Pytorch datasets
train_dataset = tokenized_datasets["train"]
train_dataset.set_format(type='torch', columns=['input_ids', 'supertag_ids', 'attention_mask'])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batchsize) # https://huggingface.co/docs/datasets/v1.17.0/use_dataset.html

dev_dataset = tokenized_datasets["dev"]
dev_dataset.set_format(type='torch', columns=['input_ids', 'supertag_ids', 'attention_mask'])
dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=config.batchsize) # https://huggingface.co/docs/datasets/v1.17.0/use_dataset.html


# set up training
model = TaggingModel(len(supertag_vocab), roberta_id="xlm-roberta-base").to(device)
loss = CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='mean')
optimizer = Adam(model.parameters(), lr=config.learning_rate)

wandb.init(project="ccg-supertagging",
           config={"learning_rate": config.learning_rate,
                   "batch_size": config.batchsize,
                   "epochs": config.epochs,
                   "optimizer": "Adam",
                   }
)


for epoch in range(config.epochs):
    total_epoch_loss = 0.0
    model.train()

    for batch in tqdm(train_dataloader, desc="Training"):
        input_batch = batch["input_ids"].to(device)
        attention_mask_batch = batch["attention_mask"].to(device)

        logits = model(input_batch, attention_mask_batch) # (bs, seqlen, #tags)

        logits_nbyc = logits.view(-1, len(supertag_vocab)) # (bs * seqlen, #tags)
        labels_nbyc = batch["supertag_ids"].to(device).view(-1) # (bs * seqlen,)
        batch_loss = loss(logits_nbyc, labels_nbyc)

        wandb.log({"batch_loss": batch_loss})

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    # one iteration over data is finished, let's evaluate
    print(f"Epoch {epoch} done, evaluating ...")
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_counted = 0
        total_tokens = 0

        for batch in dev_dataloader:
            input_batch = batch["input_ids"].to(device)
            attention_mask_batch = batch["attention_mask"].to(device)
            logits = model(input_batch, attention_mask_batch)  # (bs, seqlen, #tags)

            predicted = torch.argmax(logits, dim=2).view(-1)  # (bs * seqlen)
            gold = batch["labels"].to(device).view(-1)  # (bs * seqlen)
            correct, counted = accuracy(predicted, gold)

            total_correct += int(correct)
            total_counted += int(counted)
            total_tokens += int(predicted.shape[0])

        acc = float(total_correct) / total_counted
        print(f"Eval: Counted {total_counted}/{total_tokens} tokens, {total_correct} correct (accuracy={acc}).")
        wandb.log({"dev accuracy": acc})

wandb.finish()


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