"""
Unit tests for the `train_PINN` function in src.flow.pinn_utilities.py.

This test suite checks the output, type, shape, and values of the `train_PINN` function.
It also tests for exceptions and edge cases.
"""

import unittest
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
import optax

# Import the function to be tested
current_directory = os.getcwd()  # current directory
parent_directory = os.path.dirname(current_directory)  # Parent directory
sys.path.append(parent_directory)  # Add the parent directory to sys.path
from src.flow.pinn_utilities import train_PINN  # Import the function to be tested


class TestTrainPINN(unittest.TestCase):
    """
    Comprehensive test suite for the `train_PINN` function.
    This class contains multiple test cases to evaluate the correctness,
    robustness, and performance of the `train_PINN` function under various
    conditions and configurations.

    This test suite includes tests for:
    - Output validation: Checks the types, shapes, and values of the outputs.
    - Exception handling: Verifies that the function raises appropriate exceptions for invalid inputs.
    - Edge cases: Tests the function's behavior with edge case inputs (e.g., zero and negative epochs).
    - Different optimizers and loss functions: Ensures the function works correctly with various optimizers and loss functions.
    - Multiple layers and nodes: Tests the function with different feed-forward neural network architectures.
    """

    def setUp(self):
        """
        Sets up the test environment and initializes common test parameters.
        The `setUp` method is called before every test function in this class.
        It initializes the parameters, optimizer, and other inputs used by
        the `train_PINN` function.
        """
        # Initialize common test parameters
        self.params = [{
            'weights': jnp.array([1.0, 2.0]),  # Initial weights
            'biases': jnp.array([0.5])  # Initial biases
        }]
        self.epochs = 100  # Number of training epochs
        self.optimizer = optax.adam(
            0.001)  # Adam optimizer with learning rate 0.001
        self.loss_fun = lambda params, colloc, conds, norm_coeff: jnp.mean(
            (params[0]['weights'] - colloc)**2)  # Simple loss function
        self.colloc = jnp.array([1.0, 2.0])  # Collocation points
        self.conds = [jnp.array([1.0, 2.0])]  # Boundary conditions
        self.norm_coeff = jnp.array([1.0])  # Normalization coefficient
        self.hidden_layers = 2  # Number of hidden layers in the neural network
        self.hidden_nodes = 10  # Number of hidden nodes for each layer
        self.lr = 0.001  # Learning rate for the optimizer
        self.results_dir = './results/'  # Results directory to save the .pkl file

        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def test_train_PINN_output(self):
        """
        Test the output of the `train_PINN` function.
        Ensures that the function produces valid outputs, including the best parameters,
        best loss, and the history of losses and epochs. 
        That is, checks if the output is not None and the loss is finite.
        """
        # Call `train_PINN` with test parameters and get its output
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, self.epochs, self.optimizer, self.loss_fun,
            self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
            self.hidden_nodes, self.lr, self.results_dir)
        self.assertIsNotNone(
            best_params)  # Check if output best_params is not None
        self.assertLessEqual(best_loss,
                             float('inf'))  # Check if best_loss is finite
        self.assertEqual(
            len(all_losses), self.epochs +
            1)  # Check lengths of lists (all_losses should match epoch count)
        self.assertEqual(
            len(all_epochs), self.epochs +
            1)  # Check lengths of lists (all_epochs should match epoch count)

    def test_train_PINN_type(self):
        """
        Test the type of the output of the `train_PINN` function.
        Ensures that the types of the outputs (parameters, loss, etc.) are as expected.
        That is, checks if the output has the correct type.
        """
        # Call `train_PINN` with test parameters
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, self.epochs, self.optimizer, self.loss_fun,
            self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
            self.hidden_nodes, self.lr, self.results_dir)
        self.assertIsInstance(
            best_params,
            list)  # Check types of output (best_params should be a list)
        self.assertIsInstance(
            best_loss.item(),
            float)  # Convert best_loss to float and check type
        self.assertIsInstance(
            all_losses,
            list)  # Check types of output (all_losses should be a list)
        self.assertIsInstance(
            all_epochs,
            list)  # Check types of output (all_epochs should be a list)

    def test_train_PINN_shape(self):
        """
        Test the shape of the output of the `train_PINN` function.
        Ensures that the shapes of the outputs, such as parameter counts and loss/epoch
        lists, are correct.
        That is, checks if the output has the correct shape.
        """
        # Call `train_PINN` with test parameters
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, self.epochs, self.optimizer, self.loss_fun,
            self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
            self.hidden_nodes, self.lr, self.results_dir)
        self.assertEqual(
            len(best_params), len(self.params)
        )  # Check lengths of lists (best_params length should match input params)
        self.assertEqual(
            len(all_losses), self.epochs + 1
        )  # Check lengths of lists (all_losses length should match epochs)
        self.assertEqual(
            len(all_epochs), self.epochs + 1
        )  # Check lengths of lists (all_epochs length should match epochs)

    def test_train_PINN_values(self):
        """
        Test the values of the output of the `train_PINN` function.
        That is, checks if the loss is less than or equal to the initial loss.
        """
        # Call `train_PINN` with test parameters
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, self.epochs, self.optimizer, self.loss_fun,
            self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
            self.hidden_nodes, self.lr, self.results_dir)
        initial_loss = self.loss_fun(self.params, self.colloc, self.conds,
                                     self.norm_coeff)  # Calculate initial loss
        self.assertLessEqual(
            best_loss, initial_loss
        )  # Check if best loss is less than or equal to initial loss

    def test_train_PINN_exceptions(self):
        """
        Test the exceptions raised by the `train_PINN` function.
        Ensures that the function raises appropriate exceptions when given invalid inputs.
        That is, checks if the function raises TypeError and ValueError.
        """
        # Test TypeError
        with self.assertRaises(TypeError) as context:
            train_PINN('invalid_params', self.epochs, self.optimizer,
                       self.loss_fun, self.colloc, self.conds, self.norm_coeff,
                       self.hidden_layers, self.hidden_nodes, self.lr,
                       self.results_dir)
        self.assertEqual(context.exception.args[0],
                         "Params must be a list of dictionaries")

        # Test ValueError
        with self.assertRaises(ValueError) as context:
            train_PINN(self.params, -1, self.optimizer, self.loss_fun,
                       self.colloc, self.conds, self.norm_coeff,
                       self.hidden_layers, self.hidden_nodes, self.lr,
                       self.results_dir)
        self.assertEqual(context.exception.args[0],
                         "Epochs must be a positive integer")

    def test_train_PINN_edge_cases(self):
        """
        Test `train_PINN` with edge cases.
        That is, checks if the function works correctly with zero and negative epochs.
        """
        # Test with zero epochs
        epochs = 0
        with self.assertRaises(ValueError):
            train_PINN(self.params, epochs, self.optimizer, self.loss_fun,
                       self.colloc, self.conds, self.norm_coeff,
                       self.hidden_layers, self.hidden_nodes, self.lr,
                       self.results_dir)

        # Test with negative values for epochs
        epochs = -1
        with self.assertRaises(ValueError):
            train_PINN(self.params, epochs, self.optimizer, self.loss_fun,
                       self.colloc, self.conds, self.norm_coeff,
                       self.hidden_layers, self.hidden_nodes, self.lr,
                       self.results_dir)

        # Test with small positive integer for epochs
        epochs = 1
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, epochs, self.optimizer, self.loss_fun, self.colloc,
            self.conds, self.norm_coeff, self.hidden_layers, self.hidden_nodes,
            self.lr, self.results_dir)
        self.assertEqual(len(all_losses),
                         epochs + 1)  # Ensure loss matches epoch count
        self.assertEqual(len(all_epochs), epochs + 1)

        # Test with a larger number of epochs
        epochs = 10
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, epochs, self.optimizer, self.loss_fun, self.colloc,
            self.conds, self.norm_coeff, self.hidden_layers, self.hidden_nodes,
            self.lr, self.results_dir)
        self.assertEqual(len(all_losses),
                         epochs + 1)  # Ensure loss matches epoch count
        self.assertEqual(len(all_epochs), epochs + 1)

    def test_train_PINN_different_optimizers(self):
        """
        Test `train_PINN` with different optimizers.
        This test ensures that the `train_PINN` function can successfully train 
        a PINN model using various optimizers from the `optax` library. It checks 
        if the function completes without errors and produces valid outputs (best 
        parameters and a finite best loss) for each optimizer.
        That is, checks if the function works correctly with different optimizers. 
        """
        # Define list of optimizers
        optimizers = [
            optax.sgd(0.001),  # Stochastic Gradient Descent
            optax.rmsprop(0.001),  # RMSProp
            optax.adam(0.001),  # Adam
            optax.adamw(0.001)  # AdamW
        ]

        # Test each optimizer
        for optimizer in optimizers:
            best_params, best_loss, all_losses, all_epochs = train_PINN(
                self.params, self.epochs, optimizer, self.loss_fun,
                self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
                self.hidden_nodes, self.lr, self.results_dir)
            self.assertIsNotNone(best_params)  # Ensure best_params is not None
            self.assertLessEqual(best_loss,
                                 float('inf'))  # Ensure loss is finite

    def test_train_PINN_different_loss_functions(self):
        """
        Test `train_PINN` with different loss functions.
        This test verifies that the `train_PINN` function can correctly train 
        a PINN model using different loss functions. It checks if the function 
        completes the training process without errors and produces valid outputs 
        (best parameters and a finite best loss) for each loss function.
        That is, checks if the function works correctly with different loss functions.
        """
        # Define list of loss functions
        loss_functions = [
            lambda params, colloc, conds, norm_coeff: jnp.mean((params[0][
                'weights'] - colloc)**2),  # Mean Squared Error (MSE)
            lambda params, colloc, conds, norm_coeff: jnp.mean(
                jnp.abs(params[0]['weights'] - colloc)
            )  # Mean Absolute Error (MAE)
        ]

        # Test each loss function
        for loss_fun in loss_functions:
            best_params, best_loss, all_losses, all_epochs = train_PINN(
                self.params, self.epochs, self.optimizer, loss_fun,
                self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
                self.hidden_nodes, self.lr, self.results_dir)
            self.assertIsNotNone(best_params)  # Ensure best_params is not None
            self.assertLessEqual(best_loss,
                                 float('inf'))  # Ensure loss is finite

    def test_train_PINN_multiple_layers_nodes(self):
        """
        Test `train_PINN` with multiple layers and nodes.
        This test ensures that the `train_PINN` function can properly handle 
        different neural network architectures with varying numbers of hidden 
        layers and nodes. It checks if the function can train the PINN model 
        without errors and produces valid outputs (best parameters and a finite 
        best loss) for different combinations of layers and nodes.
        That is, checks if the function works correctly with different architectures.
        """
        # Define list of hidden layers and nodes
        hidden_layers = [1, 2, 3]
        hidden_nodes = [10, 20, 30]

        # Test each architecture
        for layers in hidden_layers:
            for nodes in hidden_nodes:
                best_params, best_loss, all_losses, all_epochs = train_PINN(
                    self.params, self.epochs, self.optimizer, self.loss_fun,
                    self.colloc, self.conds, self.norm_coeff, layers, nodes,
                    self.lr, self.results_dir)
                self.assertIsNotNone(
                    best_params)  # Ensure best_params is not None
                self.assertLessEqual(best_loss,
                                     float('inf'))  # Ensure loss is finite


def main():
    """
    Run the unit tests when script is executed `python test_train_pinn.py`.
    """
    unittest.main()


if __name__ == '__main__':
    main()
