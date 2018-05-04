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
		self.TX_LEARNING = 0.5
		self.MOMENTUM = 0.9
		self._start_net();
		print("Neural Start!...")

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
			self.input_layer.append(input_neuron)

		print("Created Input Layer:\t", self.NUMBER_INPUT_NEURONS)
		print("Created Hidden Layer:\t", self.NUMBER_HIDDEN_NEURONS)
		print("Created Output Layer:\t", self.NUMBER_OUTPUT_NEURONS)


	def propagation(self, data):
		i = 0
		print("========= INPUT LAYER =========")
		for input_neuron in self.input_layer:
			print("n(",i,")=",data[i]," ", end="")
			input_neuron.synapse(data[i])
			i+=1

		i = 0
		#print("========= HIDDEN LAYER =========")
		for hidden_neuron in self.hidden_layer:
			hidden_neuron.transfer_unit()
			#print("n(",i,")=",hidden_neuron.result," ", end="")
			hidden_neuron.synapse()
			i+=1

		i = 0
		print("========= OUTPUT LAYER =========")
		for output_neuron in self.output_layer:
			output_neuron.transfer_unit()
			print("n(",i,")=",round(output_neuron.result,3)," ")
			i+=1

	def back_propagation(self,neuron_expected):

		for i in range(0, self.NUMBER_OUTPUT_NEURONS):
			if(i==neuron_expected):
				self.output_layer[i].calculate_error(1)
			else:
				self.output_layer[i].calculate_error(0)

		for hidden_neuron in self.hidden_layer:
			hidden_neuron.calculate_error(self.MOMENTUM, self.TX_LEARNING)

		for input_neuron in self.input_layer:
			input_neuron.calculate_error(self.MOMENTUM, self.TX_LEARNING)
