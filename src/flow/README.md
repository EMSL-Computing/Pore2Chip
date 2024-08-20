# Description of the files in the `flow` folder

This readme summarizes and focuses on the flow modeling functionalities using physics-informed neural network (PINN) within the Pore2Chip repository.

## Purpose and functionality of `pinn_utilities.py`

The Python script provides a set of functions that plays a crucial role in setting up the PINN's training, initializing the model, and evaluating the loss during training.

- The `generate_BCs_and_colloc_xct` function generates boundary conditions (Dirichlet and Neumann) and collocation points for a 2D domain. BCs are defined along the boundaries of the grid, while collocation points are used inside the domain to enforce the PDE. It returns a tuple containing the boundary points and the collocation points.
- The `pde_flow_2d_hetero_resiual` function computes the residual of a 2D flow PDE for a heterogeneous medium. It calculates the partial derivatives with respect to x and y using `jax.grad` and applies the normalization coefficients to account for heterogeneity. It returns the sum of second derivatives, representing the residual of the PDE.
- The `init_params` function initializes the parameters (weights and biases) of a neural network using Xavier initialization. The initialization ensures that the initial variance of the outputs is consistent across layers.
- The `neural_net` function performs a forward pass through a feed-forward neural network. It uses the `tanh` activation function for the hidden layers and then returns the output of the network.
- The `MSE` function calculates the Mean Squared Error between predicted and true values. It is decorated with `@jax.jit` to compile the function for faster execution, especially when the function is called repeatedly.
- The `loss_fun` function computes the loss function for training the PINN. The loss includes the PDE residuals at the collocation points and the boundary conditions. This function is decorated with `@jax.jit` for optimized performance.
- The `save_training_data_to_file` function saves the training data of a neural network model to a file. The data includes the best model parameters, the best epoch, the best loss value, and the history of all losses and epochs during training. The data is saved in a binary file using Python's pickle module, which allows for easy serialization and deserialization of Python objects. This function is useful in our PIML workflows, especially for tracking and saving the state of a model after training. It allows you to later retrieve the best parameters, review the training history, and even resume training or perform further analysis (for testing and debugging PINNs; essential part of a reproducible research pipeline).
- The `train_PINN` function is designed to train a PINN model to solve the PDE by minimizing the residual of the PDE and ensuring boundary conditions are satisfied. The function handles the entire training process, from parameter updates to saving the results, and is optimized using JIT compilation to improve performance during training.

## Purpose and functionality of `plotting_results.py`

The Python script .

- The