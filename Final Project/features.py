from nltk import ngrams
from collections import Counter
import nltk
import re
import numpy as np

NEGATE = '--n'
ADJECTIVE = 'JJ'

NEGATIONS = [
    'no',
    'not',
    'none',
    'nobody',
    'nothing',
    'neither',
    'nowhere'
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
    
class POSTagger:
    def getPOS(self, texts):
        tags = []
        for text in texts:
            tags.append([tuple[1] for tuple in nltk.pos_tag(text.split())])
        return tags
    
class PositionTagger:
    def getPositions(self, texts):
        positions = []
        for text in texts:
            wordsPerQuarter = round(len(text.split())/4)
            wordPositions = [0 for i in range(wordsPerQuarter)]
            wordPositions.extend([1 for i in range(wordsPerQuarter, len(text.split())-wordsPerQuarter)])
            wordPositions.extend([2 for  i in range(len(text)-wordsPerQuarter, len(text))])
            positions.append(wordPositions)
        return positions
            
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
        self.unigrams = {k:v for k,v in wordDict.items() if v >= 4}
        self.unigrams = sorted(self.unigrams, key=self.unigrams.get, reverse=True)
        
    def get(self, texts, type='pres', top=-1):
        unigrams = self.unigrams
        if top != -1:
            unigrams = self.unigrams[:top]
        features = np.zeros((len(texts), len(unigrams)))
        for i in range(len(texts)):
            if type == 'freq':
                counts = Counter(texts[i].split())
                for k, v in counts.items():
                    try:
                        features[i][unigrams.index(k)] = v
                    except:
                        pass
            if type == 'pres':
                counts = Counter(texts[i].split())
                for k, v in counts.items():
                    try:
                        features[i][unigrams.index(k)] = 1
                    except:
                        pass
        return features
    
class UnigramPOSFeature:
    def __init__(self):
        self.unigrams = None
    
    def process(self, negatedTexts, posOfTexts):
        wordDict = {}
        for i in range(len(negatedTexts)):
            words = negatedTexts[i].split()
            for j in range(len(words)):
                words[j] += '--' + posOfTexts[i][j]
            wordDict[words[j]] = wordDict.get(words[j], 0) + 1
        self.unigrams = {k:v for k,v in wordDict.items() if v >= 4}
        self.unigrams = sorted(self.unigrams, key=self.unigrams.get, reverse=True)
        
    def get(self, negatedTexts, posOfTexts):
        features = np.zeros((len(negatedTexts), len(self.unigrams)))
        for i in range(len(negatedTexts)):
            words = negatedTexts[i].split()
            for j in range(len(words)):
                words[j] += '--' + posOfTexts[i][j]
            words = set(words)
            for word in words:
                try:
                    features[i][self.unigrams.index(word)] = 1
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
        self.bigrams = {k:v for k,v in wordDict.items() if v >= 7}        
        self.bigrams = sorted(wordDict, key=wordDict.get, reverse=True)[:16165]
        
    def get(self, texts):
        features = np.zeros((len(texts), len(self.bigrams)))
        for i in range(len(texts)):
            text = texts[i]
            words = ngrams(text.split(), 2)
            words = set(words)
            for word in words:
                try:
                    features[i][self.unigrams.index(word)] = 1
                except:
                    pass
        return features
    
class AdjectiveFeature:
    def __init__(self):
        self.adjectives = set()
        
    def process(self, texts, posOfTexts):
        for i in range(len(texts)):
            foundAdjectives = [word for word, tag in zip(texts[i].split(), posOfTexts[i]) if tag == ADJECTIVE]
            self.adjectives.update(foundAdjectives)
        self.adjectives = list(self.adjectives)  
        
    def get(self, texts):
        features = np.zeros((len(texts), len(self.adjectives)))
        for i in range(len(texts)):
            words = set(texts[i].split())
            for word in words:
                try:
                    features[i][self.adjectives.index(word)] = 1
                except:
                    pass
        return features

class UnigramPositionFeature:
    def __init__(self):
        self.unigrams = None
        
    def process(self, negatedTexts, positionsOfTexts):
        wordDict = {}
        for i in range(len(negatedTexts)):
            words = negatedTexts[i].split()
            for j in range(len(words)):
                words[j] += '--' + str(positionsOfTexts[i][j])
            wordDict[words[j]] = wordDict.get(words[j], 0) + 1
        self.unigrams = {k:v for k,v in wordDict.items() if v >= 4}
        self.unigrams = sorted(self.unigrams, key=self.unigrams.get, reverse=True)
    
    def get(self, negatedTexts, positionsOfTexts):
        features = np.zeros((len(negatedTexts), len(self.unigrams)))
        for i in range(len(negatedTexts)):
            words = negatedTexts[i].split()
            for j in range(len(words)):
                words[j] += '--' + str(positionsOfTexts[i][j])
            words = set(words)
            for word in words:
                try:
                    features[i][self.unigrams.index(word)] = 1
                except:
                    pass
        return features    
    