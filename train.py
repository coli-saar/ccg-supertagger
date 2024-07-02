import sys

import torch
from tqdm import tqdm

import wandb
from datasets import DatasetDict
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from transformers import XLMRobertaTokenizerFast

import device_selector
from config import Config
from model import TaggingModel
from util import Vocabulary, create_dataset, accuracy, tokenize_and_align_labels

# suppress tokenizer parallelization
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = Config.load("config.yml")
IGNORE_INDEX=-100
device = device_selector.choose_device()

dataset = create_dataset(config)
supertag_vocab = Vocabulary.load_or_initialize(config.supertag_vocabulary_filename)

# tokenize and vocabularize the dataset
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", add_prefix_space=True)
tokenized_datasets: DatasetDict = dataset.map(tokenize_and_align_labels(tokenizer, supertag_vocab, IGNORE_INDEX), batched=True, batch_size=config.batchsize) # this is super cool
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

with wandb.init(project="ccg-supertagging",
           config={"learning_rate": config.learning_rate,
                   "batch_size": config.batchsize,
                   "epochs": config.epochs,
                   "optimizer": "Adam",
                   }) as run:

    for epoch in range(config.epochs):
        total_epoch_loss = 0.0
        num_train_instances = 0
        num_dev_instances = 0
        model.train()

        for batch in tqdm(train_dataloader, desc="Training"):
            input_batch = batch["input_ids"].to(device)
            attention_mask_batch = batch["attention_mask"].to(device)
            batchsize = input_batch.shape[0]

            logits = model(input_batch, attention_mask_batch) # (bs, seqlen, #tags)

            logits_nbyc = logits.view(-1, len(supertag_vocab)) # (bs * seqlen, #tags)
            labels_nbyc = batch["supertag_ids"].to(device).view(-1) # (bs * seqlen,)
            batch_loss = loss(logits_nbyc, labels_nbyc)

            wandb.log({"batch_loss": batch_loss})

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # end epoch after requested number of training instances, if specified
            num_train_instances += batchsize
            if num_train_instances >= config.limit_train:
                break

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
                batchsize = input_batch.shape[0]
                logits = model(input_batch, attention_mask_batch)  # (bs, seqlen, #tags)

                predicted = torch.argmax(logits, dim=2).view(-1)  # (bs * seqlen)
                gold = batch["supertag_ids"].to(device).view(-1)  # (bs * seqlen)
                correct, counted = accuracy(predicted, gold, ignore_index=IGNORE_INDEX)

                total_correct += int(correct)
                total_counted += int(counted)
                total_tokens += int(predicted.shape[0])

                # end epoch after requested number of training instances, if specified
                num_dev_instances += batchsize
                if num_dev_instances >= config.limit_dev:
                    break

            acc = float(total_correct) / total_counted
            print(f"Eval: Counted {total_counted}/{total_tokens} tokens, {total_correct} correct (accuracy={acc}).")
            wandb.log({"dev accuracy": acc})

    if config.model_filename:
        filename = config.model_filename.replace("{RUN}", run.name)
        print(f"Saving model parameters to {filename} ...")
        torch.save(model.state_dict(), filename)
        print("Done.")

# wandb.finish()



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