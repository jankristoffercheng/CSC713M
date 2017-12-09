import numpy as np

class KFoldBatcher:
    def __init__(self, nFold, featuresNegative, featuresPositive):
        N = len(featuresNegative)
        nPerFold = int(N/nFold)
        
        self.batches = {'batch': [], 'X': [], 'y': []}
        for i in range(nFold):
            self.batches['batch'].append(i)
            self.batches['X'].append(np.concatenate((featuresNegative[nPerFold*i:nPerFold*i+nPerFold], featuresPositive[nPerFold*i:nPerFold*i+nPerFold])))
            self.batches['y'].append(np.append(np.zeros(nPerFold), np.ones(nPerFold)))

        self.batches['batch'] = np.array(self.batches['batch'])
        self.batches['X'] = np.array(self.batches['X'])
        self.batches['y'] = np.array(self.batches['y'])
        
    def getTrainX(self, i):
        trainX = self.batches['X'][self.batches['batch'] != i]
        return trainX.reshape(trainX.shape[0]*trainX.shape[1],-1)
    
    def getTrainY(self, i):
        trainY = self.batches['y'][self.batches['batch'] != i]
        return np.ravel(trainY.reshape(trainY.shape[0]*trainY.shape[1],-1))
    
    def getTestX(self, i):
        return self.batches['X'][i]
    
    def getTestY(self, i):
        return self.batches['y'][i]