from neural_net import NeuralNet
import random



net = NeuralNet(19, 30, 7, 1)

dict = {'brickface':0,'sky':1,'foliage':2,'cement':3,'window':4,'path':5,'grass':6}
with open("treinamento.txt", "r") as filestream:
    for line in filestream:
        cl = line.split(",")
        for i in range(0,19):
            cl[i] = float(cl[i])/1000
        print(cl[-1])
        cl[-1] = dict[cl[-1].strip('\n')]
        print(cl)
        #input()
        net.propagation(cl)
        net.back_propagation(cl[-1])
