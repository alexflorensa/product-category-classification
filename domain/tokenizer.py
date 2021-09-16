import string
from collections import OrderedDict, defaultdict
from typing import Optional, Sequence, List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import numpy as np


def text_to_word_sequence(
        text: str,
        filters: str = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower: bool = True,
        split: str = " "
) -> List[str]:
    if lower:
        text = text.lower()

    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [e for e in seq if e]


class Tokenizer:
    def __init__(
            self,
            num_words=None,
            filters: str = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower: bool = True,
            split: str = ' ',
            char_level: bool = False,
            oov_token: Optional[str] = None,
            document_count: int = 0
    ) -> None:
        self.num_words = num_words
        self.word_counts = OrderedDict()
        self.word_docs = defaultdict(int)
        self.filters = filters
        self.split = split
        self.lower = lower
        self.document_count = document_count
        self.char_level = char_level
        self.oov_token = oov_token
        self.index_docs = defaultdict(int)
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts: Union[List[str], 'np.ndarray']) -> None:
        for text in texts:
            self.document_count += 1
            seq = self._buid_seq(text)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        self.word_index = dict(
            zip(sorted_voc, list(range(1, len(sorted_voc) + 1))))

        self.index_word = {c: w for w, c in self.word_index.items()}

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def _buid_seq(self, text: str) -> List[str]:
        if self.char_level or isinstance(text, list):
            if self.lower:
                if isinstance(text, list):
                    text = [text_elem.lower() for text_elem in text]
                else:
                    text = text.lower()
            seq = text
        else:
            seq = text_to_word_sequence(text,
                                        filters=self.filters,
                                        lower=self.lower,
                                        split=self.split)
        return seq

    def fit_on_sequences(self, sequences: list) -> None:
        self.document_count += len(sequences)
        for seq in sequences:
            seq = set(seq)
            for i in seq:
                self.index_docs[i] += 1

    def texts_to_sequences(self, texts: List[str], length: Optional[int] = None) -> List[List[int]]:
        return list(self.texts_to_sequences_generator(texts, length))

    def _get_length_of_text(self, text):
        return len(list(filter(lambda w: w.strip() and w not in string.punctuation, text.split(self.split))))

    def texts_to_sequences_generator(self, texts: List[str], length: Optional[int] = None) -> Sequence[List[int]]:
        if length is None:
            length = max(map(lambda text: self._get_length_of_text(text), texts))
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
            seq = self._buid_seq(text)
            seq = seq[:length]
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)

            diff = length - len(vect)
            if diff > 0:
                vect.extend([oov_token_index] * diff)

            yield vect

    def sequences_to_texts(self, sequences: Sequence[List[int]]) -> List[str]:
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences: Sequence[List[int]]) -> Sequence[str]:
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for seq in sequences:
            vect = []
            for num in seq:
                word = self.index_word.get(num)
                if word is not None:
                    if num_words and num >= num_words:
                        if oov_token_index is not None:
                            vect.append(self.index_word[oov_token_index])
                    else:
                        vect.append(word)
                elif self.oov_token is not None:
                    vect.append(self.index_word[oov_token_index])
            vect = ' '.join(vect)
            yield vect
