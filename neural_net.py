#Author	: Leonardo Pedro da Silva de Sousa
#Subject: AI - 2
#NET MODULE
from neuron import InputNeuron
from neuron import OutputNeuron
from neuron import HiddenNeuron



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

		self.output_layer = []
		for i in range(0,self.NUMBER_OUTPUT_NEURONS):
			output_neuron = OutputNeuron()
			self.output_layer.append(output_neuron)


		self.hidden_layer = []
		for i in range(0,self.NUMBER_HIDDEN_NEURONS):
			hidden_neuron = HiddenNeuron()
			hidden_neuron.add_neurons(self.output_layer)
			self.hidden_layer.append(hidden_neuron)

		self.hiddens_layers = [self.hidden_layer]
		if self.NUMBER_OF_LAYERS > 1:
			for j in range(0, self.NUMBER_OF_LAYERS - 1):
				new_layer = []
				for i in range(0, self.NUMBER_HIDDEN_NEURONS):
					hidden_neuron = HiddenNeuron()
					hidden_neuron.add_neurons(self.hiddens_layers[-1])
					new_layer.append(hidden_neuron)

				self.hiddens_layers.append(new_layer)



		self.input_layer = []
		for i in range(0, self.NUMBER_INPUT_NEURONS):
			input_neuron = InputNeuron()
			input_neuron.add_neurons(self.hiddens_layers[-1])
			self.input_layer.append(input_neuron)

		print("Created Input Layer:\t", self.NUMBER_INPUT_NEURONS)
		print("Created Hidden Layer:\t", self.NUMBER_HIDDEN_NEURONS)
		print("Created Output Layer:\t", self.NUMBER_OUTPUT_NEURONS)


	def propagation(self, data, prediction=False):
		i = 0
		#print(data)
		#print("synapse input layer")
		for input_neuron in self.input_layer:
			#print("n(",i,")=",data[i],"\t", end="")
			input_neuron.synapse(data[i])
			#print(input_neuron.result," ", end = "")
			i+=1
		#print("")
		i = 0
		#print("hidden layer")
		for hidden_layer in reversed(self.hiddens_layers):
		#	print("========= HIDDEN LAYER ========= ",i)
		#	print("-------------")
			for hidden_neuron in hidden_layer:
		#		print("SUM=",hidden_neuron.sum)
				hidden_neuron.transfer_unit()
		#		print("n=",hidden_neuron.result,"\t", end="")
				hidden_neuron.synapse()
			i+=1
		#	print("")
		i = 0
		g = 0
		#print("========= OUTPUT LAYER =========")
		#print("ouput layer")
		for output_neuron in self.output_layer:
			output_neuron.transfer_unit()
			print("n(",i,")=",output_neuron.result,"\t", end="")
			if(output_neuron.result > self.output_layer[g].result):
				g = i
			i+=1
		print("")

		if(prediction == True):
			return g


	def back_propagation(self,neuron_expected):
		#print("Output error -------")
		for i in range(0, self.NUMBER_OUTPUT_NEURONS):
			if(i!=neuron_expected):
				self.output_layer[i].calculate_error(0)
			else:
				self.output_layer[i].calculate_error(1)

		#print("Hiden error -------")
		for hidden_layer in self.hiddens_layers:
		#	print("h -------")
			for hidden_neuron in hidden_layer:
				hidden_neuron.calculate_error()

		#print("Input error -------")
		for input_neuron in self.input_layer:
			input_neuron.calculate_error()

		for input_neuron in self.input_layer:
			input_neuron.update_weights(self.MOMENTUM, self.TX_LEARNING)

		for hidden_layer in reversed(self.hiddens_layers):
			for hidden_neuron in hidden_layer:
				hidden_neuron.update_weights(self.MOMENTUM, self.TX_LEARNING)
