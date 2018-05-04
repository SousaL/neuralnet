#Author	: Leonardo Pedro da Silva de Sousa
#Subject: AI - 2
#NEURON MODULE
from abc import ABCMeta, abstractmethod
import random
import math




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
		print("W: ", end="")
		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			neuron.summation_unit(weight, input)
			print(round(weight,3)," ", end="")
		print("")

	def calculate_error(self, momentum, tx_learning):
		#print("W: ", end="")
		sum_factor = 0
		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			sum_factor += weight * neuron.error
			#print(round(weight,3)," ", end="")
		#print("")
		self.error = self.result * (1 - self.result) * sum_factor

		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			connection[0] = weight*momentum+tx_learning \
							* self.result * neuron.error


class HiddenNeuron(Neuron):

	def __init__(self):
		self.sum = 0

	def summation_unit(self, weight, data):
		self.sum = (weight * data) + self.sum

	def transfer_unit(self):
		print(self.sum)
		self.result = 1 / (1 + math.exp(-(self.sum)))
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
		#	print(round(weight,3)," ", end="")
		#print("")

	def calculate_error(self, momentum, tx_learning):
		#print("W: ", end="")
		sum_factor = 0
		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			sum_factor += weight * neuron.error
			#print(round(weight,3)," ", end="")
		#print("")
		self.error = self.result * (1 - self.result) * sum_factor

		for connection in self.neurons_connected:
			weight = connection[0]
			neuron = connection[1]
			connection[0] = weight*momentum +tx_learning \
							* self.result * neuron.error


class OutputNeuron(Neuron):

	def __init__(self):
		self.sum = 0

	def summation_unit(self, weight, data):
		self.sum = (weight * data) + self.sum

	def transfer_unit(self):
		self.result = 1 / (1 + math.exp(-self.sum))
		self.sum = 0

	def calculate_error(self, result_expected):
		self.error = result_expected - self.result
