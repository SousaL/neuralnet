from neural_net import NeuralNet
import random
import numpy as np

net = NeuralNet(1, 2, 2, 2)
n = 2
for i in range(0, n):
    x = random.randint(0, 1)
    p = net.propagation([x], prediction=True)
    print(i, "Expected\t", x, "\t\t", p)
    net.back_propagation(x)

for i in range(0, n):
    x = random.randint(0, 1)
    p = net.propagation([x], prediction=True)
    print(i, "OOOOOO\t", x, "\t\t", p)

"""net = NeuralNet(2, 2, 2, 2)

net.propagation([1,1,1])
net.propagation([0,0,1])
net.back_propagation(1)
net.back_propagation(0)
net.propagation([1,0,0])
net.back_propagation(1)
net.propagation([1,0,1])
net.back_propagation(0)
net.propagation([1,1,1])
net.back_propagation(1)
"""


"""
net = NeuralNet(19, 40, 7, 5)
#dict = {'brickface':0,'sky':1,'foliage':2,'cement':3,'window':4,'path':5,'grass':6}
dict = {'BRICKFACE':0,'SKY':1,'FOLIAGE':2,'CEMENT':3,'WINDOW':4,'PATH':5,'GRASS':6}
j = 0
with open("treinamento.txt", "r") as filestream:
    for line in filestream:
        cl = line.split(",")
        for i in range(0,19):
            cl[i] = float(cl[i])
        #print(cl[-1])
        cl[-1] = dict[cl[-1].strip('\n').upper()]
        #print("\n\n")
        print("-- Interaction: ", j, " Expected - ",cl[-1], " ", end="")
        #input()
        net.propagation(cl)
        net.back_propagation(cl[-1])
        j+=1
        print("")


dict = {'BRICKFACE':0,'SKY':1,'FOLIAGE':2,'CEMENT':3,'WINDOW':4,'PATH':5,'GRASS':6}
j = 0
net = NeuralNet(19, 80, 7, 4)
with open("segmentation.test.txt", "r") as filestream:
    for line in filestream:
        cl = line.split(",")
        for i in range(1,19):
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
        print("")
        #input()

with open("segmentation.data.txt", "r") as filestream:
    for line in filestream:
        cl = line.split(",")
        for i in range(1,19):
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
        #print("")
#x = net.propagation([18.0,142.0,9,0.0,0.0,0.77777773,0.65185183,0.2777778,0.15185186,0.5555556,0.11111111,1.3333334,0.22222222,-1.3333334,2.3333333,-1.0,1.3333334,0.41666666,-2.1595633, 2], prediction=True)
#print(x);
#net.propagation([96,138,9,0,0,0.722222,0.64693,0.944444,0.712326,1.14815,0.111111,2.77778,0.555556,-3.11111,4.88889,-1.77778,2.77778,0.977778,-2.21924, 2])
#net.propagation([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 2])
#net.propagation([18.0,142.0,9,0.0,0.0,0.77777773,0.65185183,0.2777778,0.15185186,0.5555556,0.11111111,1.3333334,0.22222222,-1.3333334,2.3333333,-1.0,1.3333334,0.41666666,-2.1595633, 2])
"""
