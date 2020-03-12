import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from adaline import Adaline

# Abrindo Database
dataset = pd.read_csv('databases/sonar.all-data', header=None)
dataset.replace(['R', 'M'], [1, -1], inplace=True)

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

# Criando Adaline
#a = Adaline(len(trainInputs[0]), epochs=1000,learning_rate=0.0025, precision=0.000001)
a = Adaline(len(trainInputs[0]), epochs=1000,learning_rate=0.0025, precision=0.000001)

# Salvando Pesos Anteriores
oldWeights = ';'.join(['%.8f' % num for num in a.weights])

# Treinando Adaline
<<<<<<< HEAD
oldWeights = a.weights
qntEpochs, erro = a.train(trainInputs, trainOutputs)
newWeights = a.weights
=======
qntEpochs = a.train(trainInputs, trainOutputs)

# Salvando Pesos Novos
newWeights = ';'.join(['%.8f' % num for num in a.weights])

# Printando Pesos Antigos e Novos, e Quantidade de epocas utilizadas
>>>>>>> 51909512688e130709dd516ea800287b75f958d5
print(f'Adaline - Pesos Anterior: {oldWeights}')
print(f'Adaline - Pesos Atuais: {newWeights}')
print(f'Adaline - Quantidade Epochs utilizadas: {qntEpochs}')
print('')

# Prevendo valores
predictedValues = []
for testInput in testInputs:
    predictedValues.append(a.predict_act_func(testInput))

# Printando matriz de confusão
print('----- Matriz de Confusão -----')
confusion_matrix = confusion_matrix(testOutputs,predictedValues)
print(confusion_matrix)
tp, fp, fn, tn = confusion_matrix.ravel()

accuracy = round((tp+tn)/(tp+fp+tn+fn),2)
recall = round(tp/(tp+fn),2)
precision = round(tp/(tp+fp),2)
fscore = round(2 * ((precision * recall)/(precision + recall)),2)

fontDictionary = {
        'color': 'black',
        'size': 16
        }

print(a.error)

plt.figure()
plt.ylabel('Classificados')
plt.xticks([1, 3, 5, 7], ['TN', 'FP', 'FN', 'TP'])
plt.bar([1, 3, 5, 7], [tn, fp, fn, tp], width = 0.7)
plt.title('Matriz de Confusão')
plt.show()

plt.figure()
plt.text(0.1,0.8,f'Acurácia: {accuracy}', fontdict = fontDictionary)
plt.text(0.1,0.6,f'Recall: {recall}',  fontdict = fontDictionary)
plt.text(0.1,0.4,f'Precisão: {precision}',  fontdict = fontDictionary)
plt.text(0.1,0.2,f'FScore: {fscore}',  fontdict = fontDictionary)
plt.show()

plt.figure()
plt.plot(erro)
plt.show()
#print(confusion_matrix(testOutputs,predictedValues))