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

	def __init__(self, size):
		self.z = np.zeros(size)
		self.size = size

	def summation_unit(self):
		NotImplementedError()

	def transfer_unit(self):
		NotImplementedError()

	def set_input(self, input):
		self.z = np.array(input)

	def synapse(self):
		#print("input ",self.z.shape)
		#print(self.z)
		self.next.synapse()

	def calculate_error(self, error):
		pass


	def update_weights(self, momentum, tx_learning):
		pass

class HiddenNeuron(Neuron):

	def __init__(self, size, prev):
		self.size = size
		self.prev = prev
		self.prev.next = self
		self.w = np.random.random((prev.size,self.size))
		self.z = np.zeros(self.size)

	def summation_unit(self, weight, data):
		pass
	def transfer_unit(self):
		pass

	def synapse(self):
		#print("hidden")
		##print(self.prev.z)
		#print(self.w)
		r = self.prev.z.dot(self.w)
		#print(r)
		self.z = (1/(1+np.exp(-r)))
		#print(self.z)
		self.next.synapse()

	def calculate_error(self,error, tx_learning, momentum):
		#print("error")
		#print(error)
		self.error = self.z * (1 - self.z) * error
		#print(self.z)
		#print(self.error)
		#self.prev.calculate_error(self.w.dot(self.error))
		#print(self.error)


	def update_weights(self, momentum, tx_learning):
		#print("update_weights - h")
		#print(self.w)
		#print(self.prev.z)
		#print(self.error)
		self.w = self.w * momentum
		tmp = np.tile(self.prev.z, (self.error.size,1))
		self.w = np.add(self.w,((np.transpose(tmp).dot(np.diag(self.error)))*tx_learning))
		self.prev.update_weights(tx_learning, momentum)
		#print(self.w)

class OutputNeuron(Neuron):

	def __init__(self, size, prev):
		self.size = size
		self.prev = prev
		self.prev.next = self
		self.w = np.random.random((prev.size,self.size))
		self.z = np.zeros(self.size)

	def summation_unit(self, weight, data):
		pass

	def synapse(self):
		#print("out")
		#print(self.prev.z)
		#print(self.w)
		r = self.prev.z.dot(self.w)
		#print(r)
		self.z = (1/(1+np.exp(-r)))
		#print(self.z)

	def transfer_unit(self):
		pass
	def calculate_error(self, expected, tx_learning, momentum):
		self.error = self.z * (1-self.z) * (expected - self.z)
		#print(self.error)
		#print(self.error)
		#print(self.w.dot(self.error))
		self.prev.calculate_error(self.w.dot(self.error), tx_learning, momentum)
		self.update_weights(tx_learning, momentum)

	def update_weights(self, tx_learning, momentum):
		#print("update_weights")
		#print(self.w)
		#print(self.prev.z)
		#print(self.error)
		self.w = self.w * momentum
		tmp = np.tile(self.prev.z, (self.error.size,1))
		self.w = np.add(self.w,((np.transpose(tmp).dot(np.diag(self.error)))*tx_learning))
		self.prev.update_weights(tx_learning, momentum)
		#print(self.w)
