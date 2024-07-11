# CCG Supertagger

This is a reimplementation of the neural CCG supertagger by [Lewis et al. (2016)](https://aclanthology.org/N16-1026/), using more modern methods.

## Training

The supertagger is trained on the lexical categories used in the [CCGbank](https://groups.inf.ed.ac.uk/ccg/ccgbank.html) annotations. It uses XLM-RoBERTa encodings from Huggingface plus a simple linear layer to make the supertag predictions.

The top-1 tagging accuracy of this simple model is ~91% when trained on 02-21 and evaluated on 22.

You can instruct `train.py` to save the trained model in a file by specifying the filename in the parameter `model_filename` in `config.yml`.




## Supertagging

You can use such a trained model to run the actual supertagger ([supertagger.py](https://github.com/coli-saar/ccg-supertagger/blob/main/supertagger.py)). The supertagger has a number of useful command-line options, see the source code. It will generate a JSON file with supertag scores for the test corpus, which is suitable as input for the companion project [ccg-parser](https://github.com/coli-saar/ccg-parser).

Note that the supertagger suppresses CCGbank supertags that are not real CCG categories, such as ":" and ",", and removes words whose top-k supertags only contain such suppressed categories. This means that the sentence in the JSON file may not be exactly the same as in the original corpus, making a direct evaluation of parsing accuracy difficult.



## The supertag vocabulary file

During preprocessing, train.py caches a supertag lexicon in a file `supertag_vocabulary.txt`. This file is reused for later runs of train.py. If you want to retrain on a different dataset, delete this file first.

This Github repository already contains a file [supertag_vocabulary.txt](https://github.com/coli-saar/ccg-supertagger/blob/main/supertag_vocabulary.txt), which is the complete supertag vocabulary in Sections 02-21 of the CCGbank. Using this file will allow you to train a supertagger on just a subset of the CCGbank, which might otherwise not contain all the supertags you need for your test set. If you want to train a supertagger from scratch on a different treebank, you should delete this file first.

