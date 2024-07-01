from typing import Optional


class Vocabulary:
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

