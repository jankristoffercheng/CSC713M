import os

PUNCTUATIONS = [
    '.',
    ',',
    '?',
    '!',
    ';'
]

class FileReader:
    def getTexts(self, path):
        texts = [open(path + filename, 'r').read() for filename in os.listdir(path)]
        processedTexts = []
        for text in texts:
            words = text.split()
            i = 0
            while i < len(words):
                if words[i] not in PUNCTUATIONS and any(words[i].endswith(punctuation) for punctuation in PUNCTUATIONS):
                    words.insert(i+1, words[i][-1])
                    words[i] = words[i][:-1]
                    i+=1
                i+=1
            processedTexts.append(' '.join(words))
        return processedTexts