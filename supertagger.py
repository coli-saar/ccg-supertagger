import argparse
import json
import sys

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import XLMRobertaTokenizerFast

import device_selector
from config import Config
from model import TaggingModel

# suppress tokenizer parallelization
import os

from supertag_reader import read_corpus
from util import create_dataset, Vocabulary, tokenize_and_align_labels, accuracy

# process command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('filenames', nargs="+")
parser.add_argument('--output', '-O', required=True, help="Filename of the JSON file in which the supertag scores will be stored.")
parser.add_argument('--num-supertags', '-n', type=int, default=5, help="Number of supertags that will be predicted for each word.")
parser.add_argument("--config", "-c", type=str, default="config.yml", help="Name of the configuration file.")
parser.add_argument("--model", "-m", type=str, default=None, help="Name of Pytorch model file. If no option is specified, use model name from the config file.")
parser.add_argument("--score-model", "-s", choices=["logits", "logsoftmax"], default="logits", help="Function to compute the scores from the logits.")
args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

K = args.num_supertags
config = Config.load(args.config)
IGNORE_INDEX=-100
device = device_selector.choose_device()

# model filename
if args.model:
    model_filename = args.model
else:
    model_filename = config.model_filename


# read test corpus
data_dict = read_corpus(args.filenames).as_dict()
dataset = Dataset.from_dict(data_dict)
supertag_vocab = Vocabulary.load(config.supertag_vocabulary_filename)

# tokenize and vocabularize the dataset
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", add_prefix_space=True)
tokenized_dataset = dataset.map(tokenize_and_align_labels(tokenizer, supertag_vocab, IGNORE_INDEX), batched=True, batch_size=config.batchsize) # this is super cool


# load the model
print(f"Reading model parameters from {model_filename} ...")
model = TaggingModel(len(supertag_vocab), roberta_id="xlm-roberta-base").to(device)
model.load_state_dict(torch.load(model_filename, map_location=device))
model.eval()
print("Done.")

# set up dataloader
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'supertag_ids', 'attention_mask', 'tokens_representing_words'])
test_dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=config.batchsize) # https://huggingface.co/docs/datasets/v1.17.0/use_dataset.html

# print(dataset["words"][0])
# sys.exit(0)

all_words = dataset["words"] # [sequence_ix, word_ix]

supertags = []
skipped_supertags = set(".,;:")

if args.score_model == "logits":
    score_fn = lambda x: x # return raw logits
elif args.score_model == "logsoftmax":
    score_fn = lambda logits: torch.nn.functional.log_softmax(logits, dim=2)

# run supertagger
with torch.no_grad():
    total_correct = 0
    total_counted = 0
    total_tokens = 0
    batch_start_ix = 0

    for batch in tqdm(test_dataloader, desc="Supertagging"):
        input_batch = batch["input_ids"].to(device)
        attention_mask_batch = batch["attention_mask"].to(device) # (bs, seqlen)
        t2w = batch["tokens_representing_words"] # (bs, word-seqlen)
        batchsize = input_batch.shape[0]
        logits = model(input_batch, attention_mask_batch)  # (bs, seqlen, #tags)

        # predicted = torch.argmax(logits, dim=2) # (bs, seqlen)
        seqlen = logits.shape[1]

        # get highest-scoring supertags
        scores = score_fn(logits) # (bs, seqlen, #tags)
        best = torch.topk(scores, args.num_supertags, dim=2)  # (values: (bs, tok-seqlen, k), indices: (bs, tok-seqlen, k))
        assert best.values.shape == (batchsize, seqlen, args.num_supertags)

        # batch["tokens_representing_words"]: (bs, word-seqlen) (give or take one)
        word_seqlen = batch["tokens_representing_words"].shape[1]

        for batch_ix in range(batchsize):
            supertags_here = []
            supertags.append(supertags_here)

            values_by_word = best.values[batch_ix, t2w[batch_ix], :]  # (word-seqlen, K)
            tag_indices_by_word = best.indices[batch_ix, t2w[batch_ix], :]  # (word-seqlen, K)
            assert values_by_word.shape == (word_seqlen, K)

            for word_ix in range(word_seqlen):
                if t2w[batch_ix, word_ix] >= 0:
                    # print(f"\nsequence ix {batch_start_ix+batch_ix}")
                    # print(f"sequence {all_words[batch_start_ix+batch_ix]}")
                    # print(f"len = {len(all_words[batch_start_ix+batch_ix])}")
                    # print(f"word_ix = {word_ix}")

                    word = all_words[batch_start_ix + batch_ix][word_ix]
                    best_supertags = [{"score": values_by_word[word_ix,i].item(), "tag": supertag_vocab.itos(tag_indices_by_word[word_ix, i].item())} for i in range(args.num_supertags)]
                    best_supertags = [elem for elem in best_supertags if not elem["tag"] in skipped_supertags]

                    if len(best_supertags) > 0:
                        supertags_here.append({"word": word, "supertags": best_supertags})

            # values_here = best.values[batch_ix, token_ix, ]

            # for token_ix in range(seqlen):
            #     # TODO get word
            #     # TODO - can we use logits or do we have to normalize?
            #     best_supertags = [{"score": best.values[batch_ix, token_ix, i].item(), "tag": supertag_vocab.itos(best.indices[batch_ix, token_ix, i].item())} for i in range(args.num_supertags)]
            #     supertags_here.append({"word": "?", "supertags": best_supertags})
            #

        batch_start_ix += batchsize

        predicted = torch.argmax(logits, dim=2).view(-1)  # (bs * seqlen)
        gold = batch["supertag_ids"].to(device).view(-1)  # (bs * seqlen)
        correct, counted = accuracy(predicted, gold, ignore_index=IGNORE_INDEX)

        total_correct += int(correct)
        total_counted += int(counted)
        total_tokens += int(predicted.shape[0])

        # if total_tokens > 100:
        #     break


    acc = float(total_correct) / total_counted
    print(f"Eval: Counted {total_counted}/{total_tokens} tokens, {total_correct} correct (accuracy={acc}).")


# output JSON
with open(args.output, "w") as out:
    json.dump(supertags, out)

