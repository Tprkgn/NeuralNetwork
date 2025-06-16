import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

class NeuralNetwork:
    def __init__(self, inputN, hiddenN, outputN, LearningRate):
        self.iNodes = inputN
        self.hNodes = hiddenN
        self.oNodes = outputN
        self.lr = LearningRate

        self.Wih = np.random.rand(self.hNodes, self.iNodes) - 0.5
        self.Who = np.random.rand(self.oNodes, self.hNodes) - 0.5

        self.activationFunction = lambda x: sp.expit(x)

        self.acc = []
        self.err = []
        self.tacc = []
        self.terr = []
        
        pass
    def query(self, inputlist):
        inputs = np.array(inputlist, ndmin=2).T                      # (784,1)

        hidden_inputs = np.dot(self.Wih, inputs)                     # (200,1)
        hidden_outputs = self.activationFunction(hidden_inputs)

        final_inputs = np.dot(self.Who, hidden_outputs)              # (10,1)
        final_outputs = self.activationFunction(final_inputs)

        return final_outputs
    
    def evaluate(self, trainingdata_list):
        scoreCard = []
        for record in tqdm(trainingdata_list):
            all_values = record.split(",")
            correct_label = int(all_values[0])
            inputs = (np.asarray(all_values[1:],dtype=float)/255*0.99)+0.01
            outputs = self.query(inputs)
            predicted_label = np.argmax(outputs)
            if(correct_label==predicted_label):
                scoreCard.append(1)
            else:
                scoreCard.append(0)
            pass
        acc = sum(scoreCard)/len(scoreCard)
        err = 1-acc
        return acc,err
    
    def train(self, inputlist, targetlist):
        inputs = np.array(inputlist, ndmin=2).T
        targets = np.array(targetlist, ndmin=2).T

        hidden_inputs = np.dot(self.Wih,inputs)
        hidden_outputs = self.activationFunction(hidden_inputs)

        final_inputs = np.dot(self.Who, hidden_outputs)
        final_outputs = self.activationFunction(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.Who.T, output_errors)

        self.Who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
        self.Wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))

        pass

    def fit(self, input_data_list, test_data_list=[], epochs=1):
        for e in range(1,epochs+1):
            print("Training epoch :",e)
            for record in tqdm(input_data_list):
                all_values = record.split(",")
                inputs = (np.asarray(all_values[1:],dtype=float) / 255.0 * 0.99) + 0.01
                targets = np.zeros(self.oNodes)+0.01
                targets[int(all_values[0])]=0.99
                self.train(inputs,targets)
            a1,e1=self.evaluate(input_data_list)
            self.acc.append(a1)
            self.err.append(e1)
            if len(test_data_list)>0:
                a1,e1=self.evaluate(test_data_list)
                self.tacc.append(a1)
                self.terr.append(e1)
        plt.plot(self.acc, label="Train Accuracy")
        plt.plot(self.tacc, label="Test Accuracy")
        plt.legend()
        plt.title("Model Accuracy")
        plt.show()

        pass

neural = NeuralNetwork(784, 200, 10, 0.1)

data_file = open("mnist_dataset/mnist_train_100.csv", "r")
data_list = data_file.readlines()
data_file.close()

training_dataset = open("mnist_dataset/mnist_train.csv", "r")
training_data_list = training_dataset.readlines()
training_dataset.close()

test_data_file = open("mnist_dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

neural.fit(training_data_list,test_data_list)

all_values = test_data_list[0].split(',')

all_values[0]

image_array = np.asarray(all_values[1:], dtype=float).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')

scorecard = []
for record in tqdm(test_data_list, desc="Testing"):
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asarray(all_values[1:],dtype=float) / 255.0 * 0.99) + 0.01
    outputs = neural.query(inputs)
    label = np.argmax(outputs)
    if label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

scorecard_array = np.asarray(scorecard)
print("Performance = ", scorecard_array.sum() / scorecard_array.size)
# Visualize some test results
for i in range(10):
    all_values = test_data_list[i].split(',')
    image_array = np.asarray(all_values[1:],dtype=float).reshape((28,28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.title(f"Predicted: {neural.query((np.asarray(all_values[1:],dtype=float) / 255.0 * 0.99) + 0.01).argmax()}")
    plt.show()