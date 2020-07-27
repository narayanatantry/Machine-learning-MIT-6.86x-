class NeuralNetwork(NeuralNetworkBase):

    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = self.input_to_hidden_weights.dot(input_values) + self.biases #(3 by 1 matrix)
        ReLU_vec = np.vectorize(rectified_linear_unit)  # Vectorize ReLU function
        hidden_layer_activation = ReLU_vec(hidden_layer_weighted_input) #(3 by 1 matrix)

        output = self.hidden_to_output_weights.dot(hidden_layer_activation)
        activated_output = output_layer_activation(output)

        ### Backpropagation ###

        # Compute gradients
        output_layer_error = -(y - activated_output) #dC/df(u1)
        
        output_derivative_vec = np.vectorize(output_layer_activation_derivative)    # Vectorize derivative of output activation
        hidden_layer_error = np.multiply(output_derivative_vec(activated_output),self.hidden_to_output_weights.transpose())*output_layer_error #(3 by 1 matrix)
        
        ReLU_derivative_vec = np.vectorize(rectified_linear_unit_derivative) # Vectorize ReLU derivative
        bias_gradients = np.multiply(hidden_layer_error, ReLU_derivative_vec(hidden_layer_weighted_input)) #dC/db

        hidden_to_output_weight_gradients = np.multiply(hidden_layer_activation, output_layer_error).transpose() #dC/dV
        input_to_hidden_weight_gradients = bias_gradients.dot(input_values.transpose()) #dC/dW

        # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases - self.learning_rate*bias_gradients
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate*input_to_hidden_weight_gradients
        self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate*hidden_to_output_weight_gradients
