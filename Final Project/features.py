from nltk import ngrams
from collections import Counter
import nltk
import re
import numpy as np

NEGATE = '--n'

NEGATIONS = [
    'no',
    'not',
    'never',
    'isn\'t',
    'didn\'t',
    'don\'t',
    'doesn\'t',
    'aren\'t'
    'hasn\'t',
    'haven\'t',
    'hadn\'t',
    'can\'t',
    'couldn\'t',
    'wouldn\'t',
    'won\'t',
    'shouldn\'t',
    'weren\'t',
    'wasn\'t',
    'ain\'t',
    'needn\'t',
    'dosen\'t',
]

PUNCTUATIONS = [
    '.',
    ',',
    '?',
    '!',
    ';'
]

class TextNegator:
    def getNegated(self, texts):
        negatedTexts = []
        for text in texts:
            words= text.split()
            i = 0
            while i < len(words):
                word = words[i]
                if any(word in s for s in NEGATIONS):
                    i += 1
                    while i < len(words):
                        if any(words[i].endswith(punctuation) for punctuation in PUNCTUATIONS):
                            break;
                        if not words[i].endswith(NEGATE):
                            words[i] += NEGATE
                        i += 1
                i += 1
            negatedTexts.append(' '.join(words))
        return negatedTexts
            
class UnigramFeature:
    def __init__(self):
        self.negations = []
        self.unigrams = None
    
    def process(self, texts):
        wordDict = {}
        
        for text in texts:
            words = text.split()
            for word in words:
                wordDict[word] = wordDict.get(word, 0) + 1
                if word.endswith('n\'t') and word not in self.negations:
                    self.negations.append(word)
        self.unigrams = [k for k,v in wordDict.items() if v >= 4]
    
    def get(self, texts, type='pres'):
        features = np.ones((len(texts), len(self.unigrams)))
        for i in range(len(texts)):
            if type == 'freq':
                counts = Counter(texts[i])
                for k, v in counts.items():
                    try:
                        features[i][self.unigrams.index(k)] = v + 1
                    except:
                        pass
        return features

class BigramFeature:
    def __init__(self):
        self.negations = []
        self.bigrams = None
    
    def process(self, texts):
        wordDict = {}
        for text in texts:
            words = ngrams(text.split(), 2) 
            for word in words:
                wordDict[word] = wordDict.get(word, 0) + 1
                
        bigrams = sorted(wordDict, key=wordDict.get, reverse=True)[:16165]
            
          