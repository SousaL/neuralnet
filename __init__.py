from neural_net import NeuralNet
import random
import numpy as np
import sys

instances = int(sys.argv[1])
hidden = int(sys.argv[2])
n_hidden = 1

classes = int(sys.argv[3])
tx_learning = float(sys.argv[4])
momentum = float(sys.argv[5])
epoch = int(sys.argv[6])

dict = {'BRICKFACE':0,'SKY':1,'FOLIAGE':2,'CEMENT':3,'WINDOW':4,'PATH':5,'GRASS':6}
#dict = {'IRIS-SETOSA':0,'IRIS-VERSICOLOR':1,'IRIS-VIRGINICA':2}
j = 0

net = NeuralNet(instances, hidden, classes, n_hidden, tx_learning, momentum)
data = []
#net.propagation([20,10], prediction=True)
#net.back_propagation(0)
#
#net.propagation([1,1], prediction=True)
#net.back_propagation(1)

def status():
    print("+-----------------------------------------+")
    print("|Input.:\t", instances)
    print("|Hidden.:\t", hidden)
    print("|Output.:\t", classes)
    print("|L.rate.:\t", tx_learning)
    print("|Moment.:\t", momentum)
    print("|Iterat.:\t", epoch)
    print("+-----------------------------------------+")
def test():
    j = 0
    m = np.zeros((classes, classes))
    with open("teste.txt", "r") as filestream:
        for line in filestream:
            cl = line.split(",")
            for i in range(0,instances):
                cl[i] = float(cl[i])
            #cl[-1] = float(cl[-1].strip('\n'))/1000
            cl[-1] = dict[cl[-1].strip('\n').upper()]
            p = net.propagation(cl[:-1])
            #print("## Interaction: ", j, " Expected - ",cl[0], " - ", p, "\t", end="")
            #print(net.output_layer.z)
            #print(net.output_layer.w)
            print("Expected - ",cl[-1], " - ", p, "\t", end="")
            for k in range(0, classes):
                print(net.output_layer.z[k],"\t", end="")
            print("")
            m[cl[-1]][p] += 1
            j+=1

        print("+",end="")
        for j in range(0, classes):
            print("===============+",end="")
        print("")
        for i in range(0, classes):
            print("|\t",end="")
            for j in range(0, classes):
                print(m[i][j],"\t|\t", end="")
            print("\n+",end="")
            for j in range(0, classes):
                print("---------------+",end="")
            print("")
        status()

test()
with open("treinamento.txt", "r") as filestream:
    for l in filestream:
        c = l.split(",")
        for i in range(0,instances):
            c[i] = float(c[i])
        #c[-1] = float(c[-1].strip('\n'))/1000
        c[-1] = dict[c[-1].strip('\n').upper()]
        data.append(c)

for k in range(epoch):
    for line in data:
        p = net.propagation(line[:-1])
        net.back_propagation(line[-1])
        #print("-- Interaction: ", j, " Expected - ",line[-1], " - ", p, "\t", end="")
        #for t in range(0, classes):
        #    print(net.output_layer.z[t],"\t", end="")
        #print("")
        j+=1
    if k % 1000 == 0: test()
    if k % 100 == 0: print("-- Interaction: ", k)
    #input()
