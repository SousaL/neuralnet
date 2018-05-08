#Author	: Leonardo Pedro da Silva de Sousa
#Subject: AI - 2
#NEURON MODULE
from abc import ABCMeta, abstractmethod
import random
import math
import numpy as np



class Neuron:
	__metaclass__ = ABCMeta

	@abstractmethod
	def summation_unit(self): pass

	@abstractmethod
	def transfer_unit(self): pass

class InputNeuron(Neuron):

	def __init__(self):
		self.weight = random.random()

	def summation_unit(self):
		NotImplementedError()

	def transfer_unit(self):
		NotImplementedError()

	def add_neurons(self, neurons):
		self.neurons_connected = []
		for neuron in neurons:
			weight = random.random()
			self.neurons_connected.append([weight, neuron])

	def synapse(self, input):
		self.result = input
		#print("W: ", end="")
		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			print("\t to neuron", neuron, " - ", input, " * ", weight)
			neuron.summation_unit(weight, input)
		#	print(weight,"\t", end="")
		#print("")

	def calculate_error(self):
		#print("W: ", end="")
		sum_factor = 0
		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			sum_factor += weight * neuron.error
			#print(round(weight,3)," ", end="")
		#print("")
		self.error = self.result * (1 - self.result) * sum_factor


	def update_weights(self, momentum, tx_learning):
		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			connection[0] = weight+tx_learning*self.result*neuron.error


class HiddenNeuron(Neuron):

	def __init__(self):
		self.sum = 0

	def summation_unit(self, weight, data):
		self.sum += (weight * data)
		print("\t\t\t#",weight,"*",data,"=",weight*data," - ",self.sum,"#")

	def transfer_unit(self):
		#self.sum = np.clip(self.sum,-1000,1000)
		self.result = 1.0 / (1.0 + np.exp(-(self.sum)))
		print("\ttransfer unit with activant value: ",self.sum, " ", self.result)
		self.sum = 0


	def add_neurons(self, neurons):
		self.neurons_connected = []
		for neuron in neurons:
			weight = random.random()
			self.neurons_connected.append([weight, neuron])

	def synapse(self):
		#print("W: ", end="")
		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			neuron.summation_unit(weight, self.result)
			#print(weight,"\t", end="")
		#print("")

	def calculate_error(self):
		#print("W: ", end="")
		sum_factor = 0
		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			sum_factor += weight * neuron.error
			#print(round(weight,3)," ", end="")
		#print("")
		self.error = self.result * (1 - self.result) * sum_factor
		#print(self.error)


	def update_weights(self, momentum, tx_learning):
		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			connection[0] = weight*momentum+tx_learning*self.result*neuron.error


class OutputNeuron(Neuron):

	def __init__(self):
		self.sum = 0

	def summation_unit(self, weight, data):
		self.sum = (weight * data) + self.sum

	def transfer_unit(self):
		#self.sum = np.clip(self.sum,-1000,1000)
		self.result = 1 / (1 + np.exp(-self.sum))
		self.sum = 0

	def calculate_error(self, result_expected):
		self.error = self.result * (1- self.result) * (result_expected - self.result)
		#print(self.error)
