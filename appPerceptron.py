import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

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
trainInputs = trainingDataset[:, 0:(len(trainingDataset[0])-1)]
trainOutputs = trainingDataset[:, (len(trainingDataset[0])-1):]

# Separando testDataset em valores e resultados
testInputs = testDataset[:, 0:(len(testDataset[0])-1)]
testOutputs = testDataset[:, (len(testDataset[0])-1):]

# Criando Perceptron
p = Perceptron(len(trainInputs[0]), epochs=1000,learning_rate=0.0025)

# Salvando Pesos Anteriores
oldWeights = ';'.join(['%.8f' % num for num in p.weights])

# Treinando Perceptron
qntEpochs = p.train(trainInputs, trainOutputs)

# Salvando Pesos Novos
newWeights = ';'.join(['%.8f' % num for num in p.weights])

# Adicionando separador por ";"
np.fromstring(oldWeights, sep=';')
np.fromstring(newWeights, sep=';')

# Printando Pesos Antigos e Novos, e Quantidade de epocas utilizadas
print(f'Perceptron - Pesos Anterior: {oldWeights}')
print(f'Perceptron - Pesos Atuais: {newWeights}')
print(f'Perceptron - Quantidade Epochs utilizadas: {qntEpochs}')
print('')

# Prevendo valores
predictedValues = []
for testInput in testInputs:
    predictedValues.append(p.predict(testInput))
    
# Printando matriz de confusão
print('----- Matriz de Confusão -----')
print(confusion_matrix(testOutputs,predictedValues))