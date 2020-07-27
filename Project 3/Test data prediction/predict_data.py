class NeuralNetwork(NeuralNetworkBase):

    def predict(self, x1, x2):

        input_values = np.matrix([[x1],[x2]])

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input = self.input_to_hidden_weights.dot(input_values) + self.biases
        ReLU_vec = np.vectorize(rectified_linear_unit)
        hidden_layer_activation = ReLU_vec(hidden_layer_weighted_input)
        output = self.hidden_to_output_weights.dot(hidden_layer_activation)
        activated_output = output_layer_activation(output)

        return activated_output.item()
#pragma: coderesponse end

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)
