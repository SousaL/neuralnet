#Author	: Leonardo Pedro da Silva de Sousa
#Subject: AI - 2
#NET MODULE
from neuron import InputNeuron
from neuron import OutputNeuron
from neuron import HiddenNeuron



class NeuralNet:

	def __init__(self, input_n, hidden_n, output_n, layers_n):
		#SETA AS CONSTANTES DA REDE NEURAL
		self.NUMBER_INPUT_NEURONS = input_n
		self.NUMBER_HIDDEN_NEURONS = hidden_n
		self.NUMBER_OUTPUT_NEURONS = output_n
		self.NUMBER_OF_LAYERS = layers_n
		self._start_net();


	def _start_net(self):
		#INICIA A REDE NEURAL COM OS PARAMETROS
		#MONTANDo A REDE NO SENTINDO CONTRARIO

		self.output_layer = []
		for i in range(0,self.NUMBER_OUTPUT_NEURONS):
			output_neuron = OutputNeuron()
			self.output_layer.append(output_neuron)


		self.hidden_layer = []
		for i in range(0,self.NUMBER_HIDDEN_NEURONS):
			hidden_neuron = HiddenNeuron()
			hidden_neuron.add_neurons(self.output_layer)
			self.hidden_layer.append(hidden_neuron)


		self.input_layer = []
		for i in range(0, self.NUMBER_INPUT_NEURONS):
			input_neuron = InputNeuron()
			input_neuron.add_neurons(self.hidden_layer)
			self.input_layer.append(self.hidden_layer)



	def propagation(self, data):
		i = 0
		for input_neuron in self.input_layer:
			input_neuron.synapse(data[i])
			i+=1

		for hidden_neuron in self.hidden_layer:
			hidden_neuron.transfer_unit()
			hidden_neuron.synapse()
