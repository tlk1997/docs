import logging
from collections import OrderedDict
from typing import Sequence, Optional

logger = logging.getLogger(__name__)

SPECIAL_TOKENS_KEYS = [
    "pad_token",
    "unk_token",
    "mask_token",
    "cls_token",
    "sep_token",
    "bos_token",
    "eos_token",
    "head_token",
    "tail_token",

]

SPECIAL_TOKENS_VALUES = [
    "[PAD]",
    "[UNK]",
    "[MASK]",
    "[CLS]",
    "[SEP]",
    "[BOS]",
    "[EOS]",
    "HEAD",
    "TAIL",
]

SPECIAL_TOKENS = OrderedDict(zip(SPECIAL_TOKENS_KEYS, SPECIAL_TOKENS_VALUES))


class Vocab(object):
    """
        Build vocabulary, add vocabulary,and delete low-frequency vocabulary
    """
    def __init__(self, name: str = 'basic', init_tokens: Sequence = SPECIAL_TOKENS):
        self.name = name
        self.init_tokens = init_tokens
        self.trimed = False
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {}
        self.count = 0
        self._add_init_tokens()

    def _add_init_tokens(self):
        """
            Add origin tokens
        """
        for token in self.init_tokens.values():
            self._add_word(token)

    def _add_word(self, word: str):
        """
            Add single word
            Args:
                word (String) : Word to add
        """
        if word not in self.word2idx:
            self.word2idx[word] = self.count
            self.word2count[word] = 1
            self.idx2word[self.count] = word
            self.count += 1
        else:
            self.word2count[word] += 1

    def add_words(self, words: Sequence):
        """
            Add words
            Args:
                words (List) : List of words
        """
        for word in words:
            self._add_word(word)

    def trim(self, min_freq=2, verbose: Optional[bool] = True):
        """
            When the word frequency of word is lower than min_freq, delete from the thesaurus
            Args:
                min_freq (int): Minimum word frequency
                verbose (bool) : Whether to print log
        """
        assert min_freq == int(min_freq), f'min_freq must be integer, can\'t be {min_freq}'
        min_freq = int(min_freq)
        if min_freq < 2:
            return
        if self.trimed:
            return
        self.trimed = True

        keep_words = []
        new_words = []

        for k, v in self.word2count.items():
            if v >= min_freq:
                keep_words.append(k)
                new_words.extend([k] * v)
        if verbose:
            before_len = len(keep_words)
            after_len = len(self.word2idx) - len(self.init_tokens)
            logger.info('vocab after be trimmed, keep words [{} / {}] = {:.2f}%'.format(
                before_len, after_len, before_len / after_len * 100))

        # Reinitialize dictionaries
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = {}
        self.count = 0
        self._add_init_tokens()
        self.add_words(new_words)
