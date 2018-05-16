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
		#print("Input neuron call synapse")
		self.next.synapse()

	def calculate_error(self):
		pass


	def update_weights(self, tx_learning, momentum):
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
		#print("In Hidden Synapse")
		#print(self.prev.z, "*", self.w)
		r = self.prev.z.dot(self.w)
		self.z = (1/(1+np.exp(-r)))
		#print("result...", r," and apply function activation", self.z)
		#print("call next synapse")
		self.next.synapse()

	def calculate_error(self):

		#print("Calcuating the error in Hidden layer")
		self.factor_error = self.next.error.dot(np.transpose(self.next.w))
		#print("Error of next layer...", self.next.error)
		#print("Weights...", np.transpose(self.next.w))
		#print("Results in dot...Factor error", self.factor_error)

		self.error = self.z * (1 - self.z) * self.factor_error
		#print("Results in error...", self.error)
		self.prev.calculate_error()


	def update_weights(self, tx_learning, momentum):
		#print("Update Weights in Hidden layer")
		self.w = self.w * momentum
		#print("Multiply w by momentum", self.w, momentum)
		tmp = np.tile(self.prev.z, (self.error.size,1))
		#print("Clonning the columns of z", self.prev.z," in ", self.error, " times")
		#print("multiply the result tmp = ", tmp," by diag of error")
		tmp = np.transpose(tmp).dot(np.diag(self.error))*tx_learning
		#print("add the both")
		self.w = np.add(self.w,tmp)
		#print("the new Weights are... ", self.w)
		self.prev.update_weights(tx_learning, momentum)

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
		#print("In Out synapse")
		r = self.prev.z.dot(self.w)
		#print(self.prev.z, "*", self.w, " = ", r)
		self.z = (1/(1+np.exp(-r)))
		#print("activion sig = ", self.z)

	def transfer_unit(self):
		pass
	def calculate_error(self, expected, tx_learning, momentum):
		#print("Calculating the error")
		#print("Expected...", expected)
		#print("Result...", self.z)
		#print("Difference...", expected - self.z)
		self.error = self.z * (1-self.z) * (expected - self.z)
		#print("Error result in...", self.error)
		self.prev.calculate_error()
		self.update_weights(tx_learning, momentum)

	def update_weights(self, tx_learning, momentum):
		self.w = self.w * momentum
		#print("Multipling Weights...", self.w)
		#print("To momentum...", momentum)
		tmp = np.tile(self.prev.z, (self.error.size,1))
		#print("clone columns", tmp)
		#print("multiply by diagonal of error", self.error, np.diag(self.error))
		#print("and finally mult by learning rate", tx_learning)

		tmp = np.transpose(tmp).dot(np.diag(self.error))*tx_learning
		#print("resulting::", tmp)
		self.w = np.add(self.w,tmp)
		#print("add weights with result...", self.w)
		self.prev.update_weights(tx_learning, momentum)
