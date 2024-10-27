import unittest
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random, jit, grad, vmap
import optax

# import functions
current_directory = os.getcwd()  # current directory
parent_directory = os.path.dirname(current_directory)  # parent directory
sys.path.append(parent_directory)  # Add the parent directory to sys.path
from src.flow.pinn_utilities import train_PINN  # Import the function to be tested


class TestTrainPINN(unittest.TestCase):

    def setUp(self):
        # Setup test parameters
        self.params = [{
            'weights': jnp.array([1.0, 2.0]),
            'biases': jnp.array([0.5])
        }]
        self.epochs = 100
        self.optimizer = optax.adam(0.001)
        self.loss_fun = lambda params, colloc, conds, norm_coeff: jnp.mean(
            (params[0]['weights'] - colloc)**2)
        self.colloc = jnp.array([1.0, 2.0])
        self.conds = [jnp.array([1.0, 2.0])]
        self.norm_coeff = jnp.array([1.0])
        self.hidden_layers = 2
        self.hidden_nodes = 10
        self.lr = 0.001
        self.results_dir = './results'

    def test_train_PINN_output(self):
        # Test the output of the train_PINN function
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, self.epochs, self.optimizer, self.loss_fun,
            self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
            self.hidden_nodes, self.lr, self.results_dir)
        self.assertIsNotNone(best_params)
        self.assertLessEqual(best_loss, float('inf'))
        self.assertEqual(len(all_losses), self.epochs + 1)
        self.assertEqual(len(all_epochs), self.epochs + 1)

    def test_train_PINN_type(self):
        # Test the type of the output of the train_PINN function
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, self.epochs, self.optimizer, self.loss_fun,
            self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
            self.hidden_nodes, self.lr, self.results_dir)
        self.assertIsInstance(best_params, list)
        self.assertIsInstance(best_loss.item(),
                              float)  #Convert to python float
        self.assertIsInstance(all_losses, list)
        self.assertIsInstance(all_epochs, list)

    def test_train_PINN_shape(self):
        # Test the shape of the output of the train_PINN function
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, self.epochs, self.optimizer, self.loss_fun,
            self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
            self.hidden_nodes, self.lr, self.results_dir)
        self.assertEqual(len(best_params), len(self.params))
        self.assertEqual(len(all_losses), self.epochs + 1)
        self.assertEqual(len(all_epochs), self.epochs + 1)

    def test_train_PINN_values(self):
        # Test the values of the output of the train_PINN function
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, self.epochs, self.optimizer, self.loss_fun,
            self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
            self.hidden_nodes, self.lr, self.results_dir)
        self.assertLessEqual(
            best_loss,
            self.loss_fun(self.params, self.colloc, self.conds,
                          self.norm_coeff))

    def test_train_PINN_exceptions(self):
        # Test the exceptions raised by the train_PINN function
        with self.assertRaises(TypeError) as context:
            train_PINN('invalid_params', self.epochs, self.optimizer,
                       self.loss_fun, self.colloc, self.conds, self.norm_coeff,
                       self.hidden_layers, self.hidden_nodes, self.lr,
                       self.results_dir)
        self.assertEqual(context.exception.args[0],
                         "Params must be a list of dictionaries")

        with self.assertRaises(ValueError) as context:
            train_PINN(self.params, -1, self.optimizer, self.loss_fun,
                       self.colloc, self.conds, self.norm_coeff,
                       self.hidden_layers, self.hidden_nodes, self.lr,
                       self.results_dir)
        self.assertEqual(context.exception.args[0],
                         "Epochs must be a positive integer")

    def test_train_PINN_edge_cases(self):
        # Test train_PINN with edge cases
        epochs = 0
        with self.assertRaises(ValueError):
            train_PINN(self.params, epochs, self.optimizer, self.loss_fun,
                       self.colloc, self.conds, self.norm_coeff,
                       self.hidden_layers, self.hidden_nodes, self.lr,
                       self.results_dir)

        epochs = -1
        with self.assertRaises(ValueError):
            train_PINN(self.params, epochs, self.optimizer, self.loss_fun,
                       self.colloc, self.conds, self.norm_coeff,
                       self.hidden_layers, self.hidden_nodes, self.lr,
                       self.results_dir)

        epochs = 1  # Test with a positive integer
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, epochs, self.optimizer, self.loss_fun, self.colloc,
            self.conds, self.norm_coeff, self.hidden_layers, self.hidden_nodes,
            self.lr, self.results_dir)
        self.assertEqual(len(all_losses), epochs + 1)
        self.assertEqual(len(all_epochs), epochs + 1)

        epochs = 10  # Test with a larger number of epochs
        best_params, best_loss, all_losses, all_epochs = train_PINN(
            self.params, epochs, self.optimizer, self.loss_fun, self.colloc,
            self.conds, self.norm_coeff, self.hidden_layers, self.hidden_nodes,
            self.lr, self.results_dir)
        self.assertEqual(len(all_losses), epochs + 1)
        self.assertEqual(len(all_epochs), epochs + 1)

    def test_train_PINN_different_optimizers(self):
        # Test train_PINN with different optimizers
        optimizers = [
            optax.sgd(0.001),
            optax.rmsprop(0.001),
            optax.adam(0.001),
            optax.adamw(0.001)
        ]
        for optimizer in optimizers:
            best_params, best_loss, all_losses, all_epochs = train_PINN(
                self.params, self.epochs, optimizer, self.loss_fun,
                self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
                self.hidden_nodes, self.lr, self.results_dir)
            self.assertIsNotNone(best_params)
            self.assertLessEqual(best_loss, float('inf'))

    def test_train_PINN_different_loss_functions(self):
        # Test train_PINN with different loss functions
        loss_functions = [
            lambda params, colloc, conds, norm_coeff: jnp.mean(
                (params[0]['weights'] - colloc)**2),
            lambda params, colloc, conds, norm_coeff: jnp.mean(
                jnp.abs(params[0]['weights'] - colloc))
        ]
        for loss_fun in loss_functions:
            best_params, best_loss, all_losses, all_epochs = train_PINN(
                self.params, self.epochs, self.optimizer, loss_fun,
                self.colloc, self.conds, self.norm_coeff, self.hidden_layers,
                self.hidden_nodes, self.lr, self.results_dir)
            self.assertIsNotNone(best_params)
            self.assertLessEqual(best_loss, float('inf'))

    def test_train_PINN_multiple_layers_nodes(self):
        # Test train_PINN with multiple layers and nodes
        hidden_layers = [1, 2, 3]
        hidden_nodes = [10, 20, 30]
        for layers in hidden_layers:
            for nodes in hidden_nodes:
                best_params, best_loss, all_losses, all_epochs = train_PINN(
                    self.params, self.epochs, self.optimizer, self.loss_fun,
                    self.colloc, self.conds, self.norm_coeff, layers, nodes,
                    self.lr, self.results_dir)
                self.assertIsNotNone(best_params)
                self.assertLessEqual(best_loss, float('inf'))


if __name__ == '__main__':
    unittest.main()
