import os
import random
from io import open
import json
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<null>':0}
        self.idx2word = ['<null>']

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, seqlen=25, shuffle_train=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.json'),seqlen=25, shuffle_train=shuffle_train)
        self.valid = self.tokenize(os.path.join(path, 'dev.json'),seqlen=25)

    def tokenize(self, path, seqlen=25, shuffle_train=None):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            routes = [x[0] for x in json.load(f)]
            for line in routes:
                words = ['<sos>']+line + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            routes = [x[0] for x in json.load(f)]
            idss = []
            for line in routes:
                for i in range(len(line)-1):
                    if shuffle_train is None or random.random()>=shuffle_train or line[i]=='start' or line[i+1]=='start':
                        continue
                    line[i],line[i+1]=line[i+1],line[i]
                words = ['<sos>']+line + ['<eos>']
                while len(words)<seqlen:
                    words.append('<null>')
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
