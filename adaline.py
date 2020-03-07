import numpy as np
from activation_functions import signum_function

class Adaline():
    
    def __init__(self, input_size, act_func=signum_function, epochs=100, learning_rate=0.01, precision=0.000001):
        self.act_func = act_func
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_size + 1)
        self.precision = precision
        
    
    def predict(self, inputs):
        inputs = np.append(-1, inputs)
        u = np.dot(inputs, self.weights)
        return self.act_func(u)
        
    
    def train(self, training_inputs, labels):
        for e in range(self.epochs):
            print(f'>>> Start epoch {e + 1}')
            #print(f'Actual weights {self.weights}')
            eqmAnterior = self.eqm(training_inputs , labels)
            print(f'EQM ANTERIOR: {eqmAnterior}')
            for inputs, label in zip(training_inputs, labels):
                predicton = self.predict(inputs)
                inputs = np.append(-1, inputs)
                self.weights = self.weights + self.learning_rate * (label - predicton) * inputs
            eqmAtual = self.eqm(training_inputs , labels)
            print(f'EQM ATUAL: {eqmAtual}')
            print(f'ABS ATUAL - ANTERIOR: {abs(eqmAtual - eqmAnterior)}')
            print('')
            if abs(0 - eqmAtual) <= self.precision:
#            if eqmAtual <= self.precision:
                break
        return e + 1
    
    def eqm(self, trainInputs, trainOutputs):
        eqmCalc = 0
        for trainInput, trainOutput in zip(trainInputs, trainOutputs):
            prediction2 = self.predict(trainInput)
            eqmCalc += (trainOutput - prediction2) ** 2
        eqmCalc = eqmCalc / len(trainOutputs)
        return eqmCalc

    def restartWeights(self):
        if self.weights.any():
            self.weights = np.random.rand(len(self.weights))
            print(f'Os pesos foram redefinidos: {self.weights}')
        else:
            print('Pesos não foram declarados')
        
    def printWeights(self):
        if self.weights.any():
            print(f'Pesos atuais: {self.weights}')
        else:
            print('Pesos não foram declarados')