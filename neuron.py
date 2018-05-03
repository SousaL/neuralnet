#Author	: Leonardo Pedro da Silva de Sousa
#Subject: AI - 2
#NEURON MODULE
from abc import ABCMeta, abstractmethod
import random




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
		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			neuron.summation_unit(weight, data)


class HiddenNeuron(Neuron):

	def __init__(self):
		self.sum = 0

	def summation_unit(self, weight, data):
		self.sum = (weight * data) + self.sum

	def transfer_unit(self):
		self.result = 1 / (1 + exp(-self.sum))
		self.sum = 0


	def add_neurons(self, neurons):
		self.neurons_connected = []
		for neuron in neurons:
			weight = random.random()
			self.neurons_connected.append([weight, neuron])
	
	def synapse(self):
		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			neuron.summation_unit(weight, self.result)





class OutputNeuron(Neuron):

	def __init__(self):
		self.weight = random.random()
		self.sum = 0

	def summation_unit(self):
		self.sum = (weight * data) + self.sum

	def transfer_unit(self):
		self.result = 1 / (1 + exp(-self.sum))
		self.sum = 0