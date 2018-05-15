#Author	: Leonardo Pedro da Silva de Sousa
#Subject: AI - 2
#NET MODULE
from neuron import InputNeuron
from neuron import OutputNeuron
from neuron import HiddenNeuron
import numpy as np


class NeuralNet:

	def __init__(self, input_n, hidden_n, output_n, layers_n, tx_learning, momentum):
		#SETA AS CONSTANTES DA REDE NEURAL
		self.NUMBER_INPUT_NEURONS = input_n
		self.NUMBER_HIDDEN_NEURONS = hidden_n
		self.NUMBER_OUTPUT_NEURONS = output_n
		self.NUMBER_OF_LAYERS = layers_n
		self.TX_LEARNING = tx_learning
		self.MOMENTUM = momentum
		self._start_net();
		print("Neural Start!...")

	def _start_net(self):
		#INICIA A REDE NEURAL COM OS PARAMETROS
		#MONTANDo A REDE NO SENTINDO CONTRARIO

		input = InputNeuron(self.NUMBER_INPUT_NEURONS)
		self.input_layer = input

		hidden = HiddenNeuron(self.NUMBER_HIDDEN_NEURONS, input)
		self.hiddens_layers = []
		self.hiddens_layers.append(hidden)

		if(self.NUMBER_OF_LAYERS > 1):
			for i in range(0, self.NUMBER_OF_LAYERS - 1):
				hidden = HiddenNeuron(self.NUMBER_HIDDEN_NEURONS, self.hiddens_layers[-1])
				self.hiddens_layers.append(hidden)


		output = OutputNeuron(self.NUMBER_OUTPUT_NEURONS, self.hiddens_layers[-1])
		self.output_layer = output


		print("Created Input Layer:\t", self.NUMBER_INPUT_NEURONS)
		print("Created Hidden Layer:\t", self.NUMBER_HIDDEN_NEURONS)
		print("Created Output Layer:\t", self.NUMBER_OUTPUT_NEURONS)


	def propagation(self, data, prediction=False):
		self.input_layer.z = np.array(data)
		self.input_layer.synapse()
		return np.argmax(self.output_layer.z)


	def back_propagation(self,neuron_expected):
		a = np.zeros(self.NUMBER_OUTPUT_NEURONS)
		a[neuron_expected] = 1
		self.output_layer.calculate_error(a, self.TX_LEARNING, self.MOMENTUM)
