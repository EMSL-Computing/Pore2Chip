# Description of the files in the `flow` folder

This readme summarizes and focuses on the flow modeling functionalities using physics-informed neural networks within the Pore2Chip repository.

## Purpose and functionality of `pinn_utilities.py`

The Python script .

- The `generate_BCs_and_colloc_xct` function generates boundary conditions (Dirichlet and Neumann) and collocation points for a 2D domain. BCs are defined along the boundaries of the grid, while collocation points are used inside the domain to enforce the PDE. It returns a tuple containing the boundary points and the collocation points.
- The `pde_flow_2d_hetero_resiual` function computes the residual of a 2D flow PDE for a heterogeneous medium. It calculates the partial derivatives with respect to x and y using `jax.grad` and applies the normalization coefficients to account for heterogeneity. It returns the sum of second derivatives, representing the residual of the PDE.
- The `init_params` function initializes the parameters (weights and biases) of a neural network using Xavier initialization. The initialization ensures that the initial variance of the outputs is consistent across layers.
- The `neural_net` function performs a forward pass through a feed-forward neural network. It uses the `tanh` activation function for the hidden layers and then returns the output of the network.
- The `MSE` function calculates the Mean Squared Error between predicted and true values. It is decorated with `@jax.jit` to compile the function for faster execution.
- The `loss_fun` function computes the loss function for training the PINN. The loss includes the PDE residuals at the collocation points and the boundary conditions. This function is decorated with `@jax.jit` for optimized performance.

## Purpose and functionality of `plotting_results.py`

The Python script .

- The