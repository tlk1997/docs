import re
import unicodedata
import jieba
import logging
from typing import List

logger = logging.getLogger(__name__)
jieba.setLogLevel(logging.INFO)


class Serializer():
    def __init__(self, never_split: List = None, do_lower_case=True, do_chinese_split=False):
        self.never_split = never_split if never_split is not None else []
        self.do_lower_case = do_lower_case
        self.do_chinese_split = do_chinese_split

    def serialize(self, text, never_split: List = None):
        """
        Split a piece of text into a vocabulary list according to the established splitting rules
        Args :
            text (String) : Text for spliting
            never_split (List) : Words not to be split, empty by default
        Rerurn : 
            output_tokens (List): Results after spliting
        """
        never_split = self.never_split + (never_split if never_split is not None else [])
        text = self._clean_text(text)

        if self.do_chinese_split:
            output_tokens = self._use_jieba_cut(text, never_split)
            return output_tokens

        text = self._tokenize_chinese_chars(text)
        orig_tokens = self._orig_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split=never_split))

        output_tokens = self._whitespace_tokenize(" ".join(split_tokens))

        return output_tokens

    def _clean_text(self, text):
        """
        Delete invalid characters and blank characters in the text
        Arg :
            text (String) : Text to be deleted
        Return :
            "".join(output) (String) : Text after deleted
        """
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or self.is_control(char):
                continue
            if self.is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _use_jieba_cut(self, text, never_split):
        """
        Use jieba
        Args :
            text (String) : Text to be splited
            never_split (List) : Words not to be split
        Return :
            tokens (List) : Text after splited
        """
        for word in never_split:
            jieba.suggest_freq(word, True)
        tokens = jieba.lcut(text)
        if self.do_lower_case:
            tokens = [i.lower() for i in tokens]
        try:
            while True:
                tokens.remove(' ')
        except:
            return tokens

    def _tokenize_chinese_chars(self, text):
        """
        Add spaces around CJK characters
        Arg :
            text (String) : Text to be added
        Return :
            "".join(output) (String) : Text after added
        """
        output = []
        for char in text:
            cp = ord(char)
            if self.is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _orig_tokenize(self, text):
        """
        Split text on white space and some punctuation marks (such as commas or periods)
        Arg :
            text (String) : Text to be splited
        Return :
            tokens (List) : Text after splited
        """
        text = text.strip()
        if not text:
            return []
        punc = """,.?!;: 、｜，。？！；：《》「」【】/<>|\“ ”‘ ’"""
        punc_re = '|'.join(re.escape(x) for x in punc)
        tokens = re.sub(punc_re, lambda x: ' ' + x.group() + ' ', text)
        tokens = tokens.split()
        return tokens

    def _whitespace_tokenize(self, text):
        """
        Perform basic whitespace cleaning and segmentation
        Arg :
            text (String) : Text to be splited
        Return :
            tokens (List) : Text after splited
        """
        text = text.strip()
        if not text:
            return []
        tokens = text.split()
        return tokens

    def _run_strip_accents(self, text):
        """
        Delete accent marks from text
        Arg :
            text (String) : Text to be deleted
        Return :
            "".join(output) (String) : Text to after deleted
        """
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """
        Split text by punctuation
        Args :
            text (String) : Text to be splited
            never_split (List) : Words not to be split, empty by default
        Return :
            ["".join(x) for x in output] (List) : Text to after splited
        """
        
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self.is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    @staticmethod
    def is_control(char):
        """
        Determine whether the character is a control character
        Arg :
            char : Character
        Return :
            bool : Result
        """
        if char == "\t" or char == "\n" or char == "\r":
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    @staticmethod
    def is_whitespace(char):
        """
        Determine whether the character is a whitespace character
        Arg :
            char : Character
        Return :
            bool : Result
        """
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False

    @staticmethod
    def is_chinese_char(cp):
        """
        
        Determine whether the character is a chinese character
        Arg :
            cp (char): Character
        Return :
            bool : Result
        
        """
       
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  
            (cp >= 0x3400 and cp <= 0x4DBF) or  
            (cp >= 0x20000 and cp <= 0x2A6DF) or  
            (cp >= 0x2A700 and cp <= 0x2B73F) or  
            (cp >= 0x2B740 and cp <= 0x2B81F) or  
            (cp >= 0x2B820 and cp <= 0x2CEAF) or (cp >= 0xF900 and cp <= 0xFAFF) or  
            (cp >= 0x2F800 and cp <= 0x2FA1F)):
            return True

        return False

    @staticmethod
    def is_punctuation(char):
        """
        Determine whether the character is a punctuation character
        Arg :
            char : Character
        Return :
            bool : Result
        """
        cp = ord(char)
        
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96)
                or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False
