
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Iterable

# from attr import dataclass, field

regex = r"<L\s*(\S+)\s*\S+\s*\S+\s+(\S+)[^>]*>"

@dataclass
class Sentence:
    words: list[str] = field(default_factory=list)
    supertags: list[str] = field(default_factory=list)

    def add(self, word, supertag):
        self.words.append(word)
        self.supertags.append(supertag)

    def is_empty(self):
        return len(self.words) == 0

@dataclass
class Corpus:
    sentences: list[Sentence] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[list[str]]]:
        return {
            "words": [sent.words for sent in self.sentences],
            "supertags": [sent.supertags for sent in self.sentences]
        }


def read_corpus(corpus_files:Iterable[str]) -> Corpus:
    """
    Call me on a list of filenames in ccgbank*/data/AUTO.

    :param corpus_files:
    :return:
    """
    ret = Corpus()

    for filename in corpus_files:
        print(f"Processing {filename} ...")
        with open(filename, "r") as f:
            for line in f:
                sentence_here = Sentence()

                matches = re.finditer(regex, line.strip(), re.MULTILINE)
                for match in matches:
                    sentence_here.add(match.group(2), match.group(1))

                if not sentence_here.is_empty(): # skip ID lines
                    ret.sentences.append(sentence_here)

    return ret


if __name__ == "__main__":
    corpus = read_corpus(sys.argv[1:])
    print(f"Read {len(corpus.sentences)} sentences.")
    print(corpus.sentences[0])

