from neural_net import NeuralNet
import random
import numpy as np

instances = 4
hidden = 200
n_hidden = 1

classes = 3

tx_learning = 0.5
momentum = 0.005

dict = {'BRICKFACE':0,'SKY':1,'FOLIAGE':2,'CEMENT':3,'WINDOW':4,'PATH':5,'GRASS':6}
dict = {'IRIS-SETOSA':0,'IRIS-VERSICOLOR':1,'IRIS-VIRGINICA':2}
j = 0

net = NeuralNet(instances, hidden, classes, n_hidden, tx_learning, momentum)

f = open("out.txt", "w")
for k in range(0, 50):
    with open("bezdekIris.data.txt", "r") as filestream:
        for line in filestream:
            cl = line.split(",")
            for i in range(0,instances):
                cl[i] = float(cl[i])
        #print(cl[-1])
            cl[-1] = dict[cl[-1].strip('\n').upper()]
        #print("\n\n")
            p = net.propagation(cl, prediction=True)
            net.back_propagation(cl[-1])
            print("-- Interaction: ", j, " Expected - ",cl[-1], " - ", p,end="")
        #input()
            j+=1
        #    f.write("input\n")
        #    for neuron in net.input_layer:
        #        for connection in neuron.neurons_connected:
        #            f.write(str(connection[0]) + "\t")
        #        f.write("\n")

        #    for hidden_layer in reversed(net.hiddens_layers):
        #        f.write("========= HIDDEN LAYER =========\n")
        #        for neuron in hidden_layer:
        #            for connection in neuron.neurons_connected:
        #                f.write(str(connection) + "\t")
        #            f.write("\n")
        #print("")

with open("data.txt", "r") as filestream:
    for line in filestream:
        cl = line.split(",")
        for i in range(0,instances):
            cl[i] = float(cl[i])
        #print(cl[-1])
        cl[-1] = dict[cl[-1].strip('\n').upper()]
        #print("\n\n")
        p = net.propagation(cl, prediction=True)
        #net.back_propagation(cl[-1])
        print("## Interaction: ", j, " Expected - ",cl[-1], " - ", p, end = "")
        #input()
        j+=1
        #print("")
"""
with open("bezdekIris.data.txt", "r") as filestream:
    for line in filestream:
        cl = line.split(",")
        for i in range(1,instances):
            cl[i] = float(cl[i])
        cl[0] = dict[cl[0]]
        x = cl[0]
        cl[-1] = float(cl[-1].strip('\n'))
        cl.pop(0)
        cl.append(x)
        #print(cl)
        #input()
        #print("\n\n")
        #input()
        p = net.propagation(cl, prediction=True)
        net.back_propagation(cl[-1])
        print("-- Interaction: ", j, " Expected - ",cl[-1], " - ", p,end="")
        j+=1
        #print("")
        #input()
with open("segmentation.data.txt", "r") as filestream:
    for line in filestream:
        cl = line.split(",")
        for i in range(1,instances):
            cl[i] = float(cl[i])
        cl[0] = dict[cl[0]]
        x = cl[0]
        cl[-1] = float(cl[-1].strip('\n'))
        cl.pop(0)
        cl.append(x)
        #print(cl)
        #input()
        #print("\n\n")
        #print("-- Interaction: ", j, " Expected - ",cl[-1], " ", end="")
        #input()
        p = net.propagation(cl, prediction=True)
        print("Expected - ", cl[-1], " - ", p)
        #net.back_propagation(cl[-1])
        j+=1
"""
