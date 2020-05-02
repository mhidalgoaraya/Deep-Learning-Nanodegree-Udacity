import numpy as np


def MSE(y, Y):
    return np.mean((y-Y)**2)


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes,
                                                         self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes,
                                                          self.output_nodes))
        self.lr = learning_rate

        # Activation function
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        """ Train the network on batch of features and targets.

        :param features: 2D array, each row is one data record, each column is a feature
        :param targets: 1D array of target values
        :return:
        """

        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):

            # Forward propagation
            final_outputs, hidden_outputs = self.forward_pass_train(X)
            # Backward propagation
            delta_weights_i_h, delta_weights_h_o = \
                self.backpropagation(final_outputs, hidden_outputs, X, y,
                                     delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    def forward_pass_train(self, X):
        """ Implement forward pass here

        :param X: features batch
        :return:
        """

        # Hidden layer # signals into hidden layer
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        # signals from final output layer
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h,
                        delta_weights_h_o):
        """

        :param final_outputs: output from forward pass
        :param hidden_outputs:
        :param X:
        :param y:
        :param delta_weights_i_h: change in weights from input to hidden layers
        :param delta_weights_h_o: change in weights from hidden to output layers
        :return:
       """

        #Backward pass

        # Output error
        error = y - final_outputs

        # hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output, error)

        # Backpropagated error terms
        output_error_term = error
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        """

        :param delta_weights_i_h: change in weights from input to hidden layers
        :param delta_weights_h_o: change in weights from hidden to output layers
        :param n_records: number of records
        :return:
        """
        # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        """ Run a forward pass through the network with input features

        :param features: 1D array of feature values
        :return:
        """
        # Hidden layer
        # signals into hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        # signals from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Output layer
        # signals into final output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        # signals from final output layer
        final_outputs = final_inputs

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 5000
learning_rate = 0.5
hidden_nodes = 20
output_nodes = 1
