// Include necessary header files
#include <stdio.h>  // For input/output operations 
#include <stdlib.h> // For rand() function (random number generation)
#include <math.h>   // For exp() function used in sigmoid function

// Define the structure of the neural network
#define INPUT_NEURONS 2     // Number of input neurons (2 for XOR problem)
#define HIDDEN_NEURONS 2    // Number of neurons in the hidden layer
#define OUTPUT_NEURONS 1    // Number of output neurons (1 for binary classification)
#define LEARNING_RATE 0.1   // Learning rate for adjusting weights during training

// Activation function (sigmoid)
// This function maps any input to a value between 0 and 1
// It introduces non-linearity into the network, allowing it to learn complex patterns
double sigmoid(double x) {
    return 1 / (1 + exp(-x)); // Sigmoid function formula
    // exp(-x) computes e^(-x), where e is Euler's number
    // The output approaches 1 as x approaches infinity and 0 as x approaches negative infinity
}

// Derivative of sigmoid function
// Used in backpropagation to calculate gradients
// This represents the slope of the sigmoid function at a given point
double sigmoid_derivative(double x) {
    return x * (1 - x); // Simplified derivative based on the sigmoid output
    // This simplification works because we're passing in the sigmoid output, not the input
}

// Initialize weights with random values between -1 and 1
void init_weights(double weights[], int size) {
    for (int i = 0; i < size; i++) {
        // rand() generates a random integer between 0 and RAND_MAX
        // Dividing by RAND_MAX normalizes it to a value between 0 and 1
        // Multiplying by 2 and subtracting 1 shifts the range to between -1 and 1
        weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Initialize each weight randomly
    }
}

int main() {
    // Declare weight matrices and bias vectors
    // Using 1D arrays to represent 2D matrices for simplicity in C
    double hidden_weights[INPUT_NEURONS * HIDDEN_NEURONS]; // Weights between input and hidden layers
    double output_weights[HIDDEN_NEURONS * OUTPUT_NEURONS]; // Weights between hidden and output layers
    double hidden_bias[HIDDEN_NEURONS]; // Biases for the hidden layer neurons
    double output_bias[OUTPUT_NEURONS]; // Biases for the output layer neurons

    // Initialize weights and biases with random values
    init_weights(hidden_weights, INPUT_NEURONS * HIDDEN_NEURONS); // Initialize hidden layer weights
    init_weights(output_weights, HIDDEN_NEURONS * OUTPUT_NEURONS); // Initialize output layer weights
    init_weights(hidden_bias, HIDDEN_NEURONS); // Initialize hidden layer biases
    init_weights(output_bias, OUTPUT_NEURONS); // Initialize output layer biases

    // Training data for XOR problem, a classic nonlinear function
    // XOR truth table: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    double training_inputs[4][INPUT_NEURONS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double training_outputs[4][OUTPUT_NEURONS] = {{0}, {1}, {1}, {0}}; // Corresponding expected outputs

    // Training loop: repeat the training process for a specified number of epochs
    for (int epoch = 0; epoch < 10000; epoch++) {
        // Iterate through each training example
        for (int i = 0; i < 4; i++) {
            // Forward propagation

            // Calculate hidden layer activations
            double hidden_layer[HIDDEN_NEURONS]; // Array to hold the output of hidden layer neurons
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                double sum = hidden_bias[j]; // Start with the bias for this neuron
                for (int k = 0; k < INPUT_NEURONS; k++) {
                    // Multiply input by the corresponding weight and add to sum
                    sum += training_inputs[i][k] * hidden_weights[k * HIDDEN_NEURONS + j];
                }
                // Apply the sigmoid activation function to the sum to get the output of the hidden neuron
                hidden_layer[j] = sigmoid(sum);
            }

            // Calculate output layer activations
            double output_layer[OUTPUT_NEURONS]; // Array to hold the output of output layer neurons
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                double sum = output_bias[j]; // Start with the bias for this output neuron
                for (int k = 0; k < HIDDEN_NEURONS; k++) {
                    // Multiply hidden layer output by the corresponding weight and add to sum
                    sum += hidden_layer[k] * output_weights[k * OUTPUT_NEURONS + j];
                }
                // Apply the sigmoid activation function to the sum to get the output of the output neuron
                output_layer[j] = sigmoid(sum);
            }

            // Backpropagation

            // Calculate output layer error
            double output_error[OUTPUT_NEURONS]; // Array to hold the error for output neurons
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                // Calculate error as the difference between target and actual output
                // This error is then scaled by the derivative of the activation function
                output_error[j] = (training_outputs[i][j] - output_layer[j]) * sigmoid_derivative(output_layer[j]);
            }

            // Calculate hidden layer error
            double hidden_error[HIDDEN_NEURONS]; // Array to hold the error for hidden neurons
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                hidden_error[j] = 0; // Initialize hidden error to zero
                for (int k = 0; k < OUTPUT_NEURONS; k++) {
                    // Propagate the error back from the output layer to the hidden layer
                    hidden_error[j] += output_error[k] * output_weights[j * OUTPUT_NEURONS + k];
                }
                // Scale the hidden error by the derivative of the activation function
                hidden_error[j] *= sigmoid_derivative(hidden_layer[j]);
            }

            // Update weights and biases for the output layer
            // Adjust weights and biases based on the calculated errors
            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                // Update bias for the output neuron
                output_bias[j] += LEARNING_RATE * output_error[j]; // Adjust bias based on the output error
                for (int k = 0; k < HIDDEN_NEURONS; k++) {
                    // Update weight connecting hidden neuron to output neuron
                    output_weights[k * OUTPUT_NEURONS + j] += LEARNING_RATE * output_error[j] * hidden_layer[k];
                }
            }

            // Update weights and biases for the hidden layer
            for (int j = 0; j < HIDDEN_NEURONS; j++) {
                // Update bias for the hidden neuron
                hidden_bias[j] += LEARNING_RATE * hidden_error[j]; // Adjust bias based on the hidden error
                for (int k = 0; k < INPUT_NEURONS; k++) {
                    // Update weight connecting input neuron to hidden neuron
                    hidden_weights[k * HIDDEN_NEURONS + j] += LEARNING_RATE * hidden_error[j] * training_inputs[i][k];
                }
            }
        }
    }

    // Test the trained network after training
    printf("Testing the neural network:\n");
    for (int i = 0; i < 4; i++) {
        // Forward propagation for testing to get the predicted outputs
        double hidden_layer[HIDDEN_NEURONS]; // Array to hold hidden layer outputs during testing
        for (int j = 0; j < HIDDEN_NEURONS; j++) {
            double sum = hidden_bias[j]; // Start with the bias for the hidden neuron
            for (int k = 0; k < INPUT_NEURONS; k++) {
                // Multiply input by the corresponding weight and add to sum
                sum += training_inputs[i][k] * hidden_weights[k * HIDDEN_NEURONS + j];
            }
            hidden_layer[j] = sigmoid(sum); // Apply the activation function to get hidden neuron output
        }

        double output_layer[OUTPUT_NEURONS]; // Array to hold output layer results during testing
        for (int j = 0; j < OUTPUT_NEURONS; j++) {
            double sum = output_bias[j]; // Start with the bias for the output neuron
            for (int k = 0; k < HIDDEN_NEURONS; k++) {
                // Multiply hidden layer output by the corresponding weight and add to sum
                sum += hidden_layer[k] * output_weights[k * OUTPUT_NEURONS + j];
            }
            output_layer[j] = sigmoid(sum); // Apply the activation function to get output neuron output
        }

        // Print inputs with their corresponding outputs, truth-table style
        printf("Input: %.0f %.0f, Output: %.4f\n", training_inputs[i][0], training_inputs[i][1], output_layer[0]);
    }

    // Return 0 to indicate successful program execution
    // In C, returning 0 from main() is a convention that shows the program ran successfully
    return 0;
}
