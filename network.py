from readData import dataset
import numpy as np
import random
import matplotlib.pyplot as plt

#hyper parameters
learn_rate = 0.9
epochs = 1000
batch_size = 10

class Network:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.rand(y, x) for x, y in zip(self.layers[:-1], self.layers[1:])]
        self.biases = [np.random.rand(x, 1) for x in self.layers[1:]] 
    def Sgd(self, train_data, test_data):
        #random.shuffle(train_data)
        for i in range(epochs):
            for k in range(0, len(train_data), batch_size):
                batch = train_data[k:k + batch_size]
                sum_b = [np.zeros(b.shape) for b in self.biases]
                sum_w = [np.zeros(w.shape) for w in self.weights]
                for x, y in batch:
                    x = np.array(x).reshape(len(x), 1)
                    y = np.array(y).reshape(len(y), 1)
                    delta_b, delta_w = self.BackPropagation(x, y)
                    sum_b = [a + b for a, b in zip(sum_b, delta_b)]
                    sum_w = [a + b for a, b in zip(sum_w, delta_w)]
                self.weights = [w - (learn_rate / batch_size) * sumW for w, sumW in zip(self.weights, sum_w)]
                self.biases = [b - (learn_rate / batch_size) * sumB for b, sumB in zip(self.biases, sum_b)]
            print(f"Epoch {i + 1}: {self.Evaluate(test_data) / len(test_data) * 100 :.2f}%")

    def BackPropagation(self, x, y):
        zs = []
        act = [np.array(x).reshape(len(x), 1)]
        for b, w in zip(self.biases, self.weights):
            zs.append(np.dot(w, act[-1]) + b)
            act.append(Sigmoid(zs[-1]))
        error = act[-1] - y
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
        new_b[-1] = error
        new_w[-1] = np.dot(error, act[-2].transpose())
        for i in range(2, len(self.layers)):
            error = SigmoidDerivate(zs[-i]) * np.dot(self.weights[-i + 1].transpose(), error)
            new_b[-i] = error
            new_w[-i] = np.dot(error, act[-i - 1].transpose())
        return (new_b, new_w)

    def FeedForward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = Sigmoid(np.dot(w, a) + b)
        return a
    
    def Evaluate(self, test_data):
        res = 0
        for x, y in test_data:
            x = np.array(x).reshape(len(x), 1)
            a = self.FeedForward(x)
            #print(a[0][0], y[0],  (y[0] == (1 if a[0][0] >= 0.5 else 0)))
            res += (y[0] == (1 if a[0][0] >= 0.5 else 0))
        #print(res, len(test_data))
        return res

def Sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def SigmoidDerivate(x):
    return Sigmoid(x) * (1 - Sigmoid(x))

if __name__ == '__main__':
    x = dataset.drop(columns = ["diagnosis(1=m, 0=b)"])
    y = dataset["diagnosis(1=m, 0=b)"]
    test_data, train_data = [], []
    x = np.array(x)
    y = np.array(y).reshape(len(y), 1)
    for i in range(len(x)):
        x[i] = Sigmoid(x[i])
    train_data = [[a, b] for a, b in zip(x[:450], y[:450])]
    test_data = [[a, b] for a, b in zip(x[450:], y[450:])]
    net = Network([30, 256, 1])
    net.Sgd(train_data, test_data)
    #df.head(569, 31) 30 layera ce mi imati input layer
    #450 ide na train ostali na test
