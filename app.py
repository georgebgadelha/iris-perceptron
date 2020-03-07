import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron

# Abrindo Database
dataset = pd.read_csv('databases/sonar.all-data', header=None)
dataset.replace(['R', 'M'], [1, 0], inplace=True)

# Embaralhando Database
shuffledDataset = dataset.iloc[:,0:61].values
np.random.shuffle(shuffledDataset)

# Separando Database Teste e Treinamento
trainingDataset = shuffledDataset[0:int(np.floor(len(shuffledDataset)*0.75))]
testDataset = shuffledDataset[int(np.floor(len(shuffledDataset)*0.75)):]

# Separando trainingDataset em valores e resultados
inputs = trainingDataset[:, 0:(len(trainingDataset[0])-1)]
outputs = trainingDataset[:, (len(trainingDataset[0])-1):]

# Criando Perceptron e Adaline
p = Perceptron(len(inputs[0]), epochs=1000)

# Treinando Perceptron e Adaline
oldWeights = p.weights
qntEpochs = p.train(inputs, outputs)
newWeights = p.weights
print(f'Pesos Anterior: {oldWeights}')
print(f'Pesos Atuais: {newWeights}')
print(f'Quantidade Epochs utilizadas: {qntEpochs}')

#plt.xlim(-1,3)
#plt.ylim(-1,3)
#for i in range(len(d)):
#    if d[i] == 1:
#        plt.plot(X[i, 0], X[i, 1], 'ro')
#    else:
#        plt.plot(X[i, 0], X[i, 1], 'bo')
#        
#f = lambda x: (p.weights[0]/p.weights[2]) - (p.weights[1]/p.weights[2] * x)
#xH = list(range(-1,3))
#yH = list(map(f, xH))
#plt.plot(xH, yH, 'y-')





#print(p.predict(X[0]))
#print(p.predict(X[1]))
#print(p.predict(X[2]))
#print(p.predict(X[3]))
#
#plt.xlim(-1,3)
#plt.ylim(-1,3)
#for i in range(len(d)):
#    if d[i] == 1:
#        plt.plot(X[i, 0], X[i, 1], 'ro')
#    else:
#        plt.plot(X[i, 0], X[i, 1], 'bo')
#        
#f = lambda x: (p.weights[0]/p.weights[2]) - (p.weights[1]/p.weights[2] * x)
#xH = list(range(-1,3))
#yH = list(map(f, xH))
#plt.plot(xH, yH, 'g-')
    