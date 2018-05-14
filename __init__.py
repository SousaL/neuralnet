from neural_net import NeuralNet
import random
import numpy as np

instances = 4
hidden = 6
n_hidden = 1

classes = 3
epoch = 500
tx_learning = 0.02
momentum = 1

dict = {'BRICKFACE':0,'SKY':1,'FOLIAGE':2,'CEMENT':3,'WINDOW':4,'PATH':5,'GRASS':6}
dict = {'IRIS-SETOSA':0,'IRIS-VERSICOLOR':1,'IRIS-VIRGINICA':2}
j = 0

net = NeuralNet(instances, hidden, classes, n_hidden, tx_learning, momentum)
data = []
#net.propagation([20,10], prediction=True)
#net.back_propagation(0)
#
#net.propagation([1,1], prediction=True)
#net.back_propagation(1)


with open("iris.txt", "r") as filestream:
    for line in filestream:
        cl = line.split(",")
        for i in range(0,instances):
            cl[i] = float(cl[i])#/1000
        cl[-1] = dict[cl[-1].strip('\n').upper()]
        data.append(cl)

for k in range(0, epoch):
    for line in data:
        p = net.propagation(line[:-1], prediction=True)
        print("-- Interaction: ", j, " Expected - ",line[-1], " - ", p, "\t", end="")
        print(net.output_layer.z)
        net.back_propagation(line[-1])
        #input()
        j+=1
    if k % 100 == 0: print("-- Interaction: ", k)
    #input()
    #print(net.hiddens_layers[-1].w)
    #print(net.output_layer.w)

m = np.zeros((classes, classes))
with open("iris_teste.txt", "r") as filestream:
    for line in filestream:
        cl = line.split(",")
        for i in range(0,instances):
            cl[i] = float(cl[i])#/1000
        cl[-1] = dict[cl[-1].strip('\n').upper()]
        p = net.propagation(cl[:-1], prediction=True)
        print("## Interaction: ", j, " Expected - ",cl[-1], " - ", p, "\t", end="")
        print(net.output_layer.z)
        #print(net.output_layer.w)
        m[cl[-1]][p] += 1
        j+=1

print("\n")
for i in range(0, classes):
    for j in range(0, classes):
        print(m[i][j],"\t", end="")
    print("")
