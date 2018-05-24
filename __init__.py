import matplotlib.pyplot as plt
import sys
import random
import numpy as np
import time
from neural_net import NeuralNet
import matplotlib.pyplot as plt



instances = int(sys.argv[1])
hidden = int(sys.argv[2])
n_hidden = 1

classes = int(sys.argv[3])
tx_learning = float(sys.argv[4])
momentum = float(sys.argv[5])
epoch = int(sys.argv[6])
name = str(instances) + "_" + str(hidden) + "_" + str(classes) + "_" + str(tx_learning) + "_" + str(momentum) + "_.txt"
zeus = open(name, "w")

plt.ion()
plt.imshow(np.zeros((classes, classes)),  cmap='gray', interpolation='none')
plt.pause(1)


dict = {'BRICKFACE':0,'SKY':1,'FOLIAGE':2,'CEMENT':3,'WINDOW':4,'PATH':5,'GRASS':6}


#dict = {'IRIS-SETOSA':0,'IRIS-VERSICOLOR':1,'IRIS-VIRGINICA':2}
j = 0

net = NeuralNet(instances, hidden, classes, n_hidden, tx_learning, momentum)
data = []
#net.propagation([0,0,0], prediction=True)
#net.back_propagation(0)
#
#net.propagation([1,1], prediction=True)
#net.back_propagation(1)

def calculate_confusion_matrix(matrix,k):
    accuracy = np.trace(matrix)/np.sum(matrix)
    error = 1 - accuracy
    print("Accuracy:\t", accuracy)
    print("Error...:\t", error)
    recall_vector = np.sum(matrix, axis=1)
    recall = []
    precision = []
    print("Recall   :\t", end="")
    for i in range(0, recall_vector.shape[0]):
        recall_tmp = np.diag(matrix)[i]/(recall_vector[i])
        recall.append(recall_tmp)
        print(recall_tmp,"\t", end="")
    print("")
    precision_vector = np.sum(matrix, axis=0)
    print("Precision:\t", end="")
    for i in range(0, precision_vector.shape[0]):
        precision_tmp = np.diag(matrix)[i]/(precision_vector[i])
        precision.append(precision_tmp)
        print(precision_tmp,"\t", end="")
    print("")
    specific = np.sum(matrix, axis=0)
    print("Specif   :\t", end="")
    for i in range(0, recall_vector.shape[0]):
        vn = (np.sum(np.diag(matrix)) - np.diag(matrix)[i])
        specific_tmp = vn/(vn + specific[i] - np.diag(matrix)[i])
        print(specific_tmp,"\t", end="")
    print("")
    print("Fmeas    :\t", end="")
    for i in range(0, recall_vector.shape[0]):

        fm = (2*(recall[i]*precision[i]))/(recall[i]+precision[i])
        print(fm,"\t", end="")
    print("")
    zeus.write(str(k) + "\t" + str(accuracy) + "\n")



def status():
    print("+-----------------------------------------+")
    print("|Input.:\t", instances)
    print("|Hidden.:\t", hidden)
    print("|Output.:\t", classes)
    print("|L.rate.:\t", tx_learning)
    print("|Moment.:\t", momentum)
    print("|Iterat.:\t", epoch)
    print("+-----------------------------------------+")

def test(k):
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
            #print("Expected - ",cl[-1], " - ", p, "\t", end="")
            #for k in range(0, classes):
            #    print(net.output_layer.z[k],"\t", end="")
            #print("")
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
        calculate_confusion_matrix(m,k)
        plt.imshow(m, cmap='gray', interpolation='none')
        plt.draw()
        plt.pause(1)
        plt.savefig(str(k)+".png")

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
    if k % 100 == 0: test(k)
    if k % 100 == 0: print("-- Interaction: ", k)
    #input()
