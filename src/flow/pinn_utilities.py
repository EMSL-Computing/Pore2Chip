"""
Developed by: 
    Lal Mamud, 
    Postdoc - Subsurface Modeler, 
    Environmental Subsurface Science Group, 
    Energy & Environment Division, 
    Pacific Northwest National Laboratory, 
    Richland, WA, USA.
Mentors: Maruti K. Mudunuru and Satish Karra
"""

# `jax` and `jax.numpy` are used for automatic differentiation and working with numerical arrays.
# `optax` is a library that provides optimization algorithms.
import jax
import jax.numpy as jnp
import optax
import pickle as pk  #Serialization library for saving and loading Python objects


def generate_BCs_and_colloc_xct(xx, yy, P1, P2, dP_dy1, dP_dy2):
    """
    Generates boundary conditions and collocation points for a micromodel.

    Args:
        xx (jnp.ndarray): A 2D array representing the x-coordinates of the grid points..
        yy (jnp.ndarray): A 2D array representing the y-coordinates of the grid points.
        P1 (float): Pressure value at the left boundary (x=0).
        P2 (float): Pressure value at the right boundary (x=1).
        dP_dy1 (float): Derivative of pressure with respect to y at the bottom boundary (y=0).
        dP_dy2 (float): Derivative of pressure with respect to y at the top boundary (y=1).

    Returns:
        tuple: Contains boundary condition points (x_b1, y_b1, bc_1, ...), collocation points, and boundary conditions.
            - x_b1, y_b1, bc_1: Arrays representing x-coordinates, y-coordinates, and pressure values at the left boundary.
            - x_b2, y_b2, bc_2: Arrays representing x-coordinates, y-coordinates, and pressure values at the right boundary.
            - x_b3, y_b3, bc_3: Arrays representing x-coordinates, y-coordinates, and derivative values at the bottom boundary.
            - x_b4, y_b4, bc_4: Arrays representing x-coordinates, y-coordinates, and derivative values at the top boundary.
            - x_c, y_c: Arrays representing x-coordinates and y-coordinates of the collocation points.
            - conds: A list of arrays for each boundary condition, where each array has three columns (x, y, value/derivative).
            - colloc: An array of collocation points, where each row represents a point (x, y).
    """

    # Left BC: P[0,y] = left_bc (which is P1)
    # Extracting the x and y coordinates for the left boundary (x = 0)
    x_b1 = xx[:, 0]
    y_b1 = yy[:, 0]
    left_bc = P1  # The pressure value at the left boundary is set to P1

    # Create a boundary condition array for the left side
    bc_1 = jnp.ones_like(y_b1) * left_bc

    # Stack the x, y, and boundary condition values into a single array
    BC_1 = jnp.column_stack([x_b1, y_b1, bc_1])

    # Right BC: P[1,y] = right_bc (which is P2)
    # Extracting the x and y coordinates for the right boundary (x = 1)
    x_b2 = xx[:, -1]
    y_b2 = yy[:, -1]
    right_bc = P2  # The pressure value at the right boundary is set to P2

    # Create a boundary condition array for the right side
    bc_2 = jnp.ones_like(y_b2) * right_bc

    # Stack the x, y, and boundary condition values into a single array
    BC_2 = jnp.column_stack([x_b2, y_b2, bc_2])

    # Bottom BC: ∂P/∂y[x,0] = 0 (which is dP_dy1)
    # Extracting the x and y coordinates for the bottom boundary (y = 0), excluding the corners
    x_b3 = xx[0, 1:-1]
    y_b3 = yy[0, 1:-1]

    # The derivative of pressure with respect to y at the bottom boundary is set to dP_dy1
    bc_3 = jnp.ones_like(x_b3) * dP_dy1

    # Stack the x, y, and boundary condition values into a single array
    BC_3 = jnp.column_stack([x_b3, y_b3, bc_3])

    # Top BC: ∂P/∂y[x,1] = 0 (which is dP_dy2)
    # Extracting the x and y coordinates for the top boundary (y = 1), excluding the corners
    x_b4 = xx[-1, 1:-1]
    y_b4 = yy[-1, 1:-1]

    # The derivative of pressure with respect to y at the top boundary is set to dP_dy2
    bc_4 = jnp.ones_like(x_b4) * dP_dy1

    # Stack the x, y, and boundary condition values into a single array
    BC_4 = jnp.column_stack([x_b4, y_b4, bc_4])

    # Combine all boundary condition arrays into a list for easy access
    conds = [BC_1, BC_2, BC_3, BC_4]

    # Collocation points inside the domain (excluding boundaries)
    # Flatten the x and y coordinates for the interior points
    x_c = xx[1:-1, 1:-1].flatten()
    y_c = yy[1:-1, 1:-1].flatten()

    # Stack the x and y coordinates into a single array for collocation points
    colloc = jnp.column_stack([x_c, y_c])

    # Return the boundary conditions and collocation points
    return x_b1, y_b1, bc_1, x_b2, y_b2, bc_2, x_b3, y_b3, bc_3, x_b4, y_b4, bc_4, x_c, y_c, conds, colloc


def pde_flow_2d_hetero_resiual(x, y, P, norm_coeff):
    """
    Computes the residual of the 2D heterogeneous flow PDE.
    ∂/∂x(norm_coeff * ∂P/∂x) + ∂/∂y(norm_coeff * ∂P/∂y) = 0

    Args:
        x (jnp.ndarray): A JAX array representing the x coordinates of collocation points.
        y (jnp.ndarray): A JAX array representing the y coordinates of collocation points.
        P (function): A callable function representing the neural network approximation of the pressure field..
        norm_coeff (jnp.ndarray): A 2D JAX array representing the heterogeneous normalization coefficients.

    Returns:
        jnp.ndarray: A JAX array representing the residuals of the PDE at the given collocation points.
    """

    # Compute the gradient of pressure with respect to x
    P_x = lambda x, y: jax.grad(lambda x, y: jnp.sum(P(x, y)), 0)(x, y)

    # Compute the gradient of pressure with respect to y
    P_y = lambda x, y: jax.grad(lambda x, y: jnp.sum(P(x, y)), 1)(x, y)
    #print("P_x = ", P_x(x,y))

    # Extract k_x and k_y from the given field k_hetero
    # Normalization coefficients for the interior points
    norm_coeff_x = jnp.array(norm_coeff[1:-1, 1:-1])  # Convert to jnp.array
    norm_coeff_x = norm_coeff_x.flatten()
    norm_coeff_x = norm_coeff_x.reshape(-1, 1)
    #print("K_x = ",k_x)

    norm_coeff_y = jnp.array(norm_coeff[1:-1, 1:-1])  # Convert to jnp.array
    norm_coeff_y = norm_coeff_y.flatten()
    norm_coeff_y = norm_coeff_y.reshape(-1, 1)

    # Define callable functions for term_x and term_y
    # Define the terms involving the normalization coefficients and gradients
    term_x = lambda x, y: (norm_coeff_x * P_x(x, y))
    term_y = lambda x, y: (norm_coeff_y * P_y(x, y))
    #print("term_x = ", term_x(x, y))

    # Compute the second derivatives of term_x and term_y with respect to x and y
    P_xx = lambda x, y: jax.grad(lambda x, y: jnp.sum(term_x(x, y)), 0)(x, y)
    P_yy = lambda x, y: jax.grad(lambda x, y: jnp.sum(term_y(x, y)), 1)(x, y)

    # Return the residual of the PDE
    return P_xx(x, y) + P_yy(x, y)


def init_params(layers):
    """
    Initializes parameters for a neural network using Xavier initialization.

    Args:
        layers (list[int]): A list specifying the number of neurons in each layer of the neural network.
                             For example, [2, 4, 1] represents a network with 2 input neurons, 
                             one hidden layer with 4 neurons, and 1 output neuron.

    Returns:
        params (list[dict]): A list of dictionaries, each containing weight ('W') and 
                    bias ('B') matrices for a layer. Each dictionary has two keys:
                     - 'W': A JAX array representing the weight matrix for the layer.
                     - 'B': A JAX array representing the bias vector for the layer.
    """

    # Split the random key for initializing each layer's weights and biases
    keys = jax.random.split(jax.random.PRNGKey(0), len(layers) - 1)
    params = list()

    # Initialize weights and biases for each layer using Xavier initialization
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        lb, ub = -(1 / jnp.sqrt(n_in)), (
            1 / jnp.sqrt(n_in))  # xavier initialization lower and upper bound
        W = lb + (ub - lb) * jax.random.uniform(key, shape=(n_in, n_out))
        B = jax.random.uniform(key, shape=(n_out, ))
        params.append({'W': W, 'B': B})

    return params


def neural_net(params, x, y):
    """
    Forward pass through a neural network.

    Args:
        params (list[dict]): List of parameters for the network (weights and biases).
        x (jnp.ndarray): Input x values.
        y (jnp.ndarray): Input y values.

    Returns:
        jnp.ndarray: Output of the neural network.
    """

    # Concatenate x and y into a single input matrix
    X = jnp.concatenate([x, y], axis=1)
    *hidden, last = params

    # Apply each layer's weights and biases using tanh activation for hidden layers
    for layer in hidden:
        X = jax.nn.tanh(X @ layer['W'] + layer['B'])

    # Compute the final output
    return X @ last['W'] + last['B']


@jax.jit
def MSE(true, pred):
    """
    Computes the Mean Squared Error (MSE) between the true and predicted values for the loss function.

    Args:
        true (jnp.ndarray): Ground truth values.
        pred (jnp.ndarray): Predicted values.

    Returns:
        jnp.ndarray: The mean squared error.
    """
    return jnp.mean((true - pred)**2)


@jax.jit
def loss_fun(params, colloc, conds, norm_coeff):
    """
    Computes the total loss for training the PINN model, including PDE residuals and boundary conditions.

    Args:
        params (list[dict]): List of parameters for the neural network.
        colloc (jnp.ndarray): Collocation points within the domain.
        conds (list[jnp.ndarray]): Boundary conditions for the domain.
        norm_coeff (jnp.ndarray): Normalization coefficients for heterogeneity.

    Returns:
        jnp.ndarray: The total loss.
    """

    # Extract collocation points for x and y
    x_c, y_c = colloc[:, [0]], colloc[:, [1]]
    P_nn = lambda x, y: neural_net(params, x, y)

    # Compute PDE residual loss
    loss = jnp.mean(pde_flow_2d_hetero_resiual(x_c, y_c, P_nn, norm_coeff)**2)

    # Compute Dirichlet BC loss (left and right boundaries)
    for cond in conds[0:2]:
        x_b, y_b, u_b = cond[:, [0]], cond[:, [1]], cond[:, [2]]
        loss += MSE(P_nn(x_b, y_b), u_b)

    # Compute Neumann BC loss (bottom and top boundaries)
    P_nn_y = lambda x, y: jax.grad(lambda x, y: jnp.sum(P_nn(x, y)), 1)(
        x, y)  # single derivative of P for Neumann BCs
    for cond in conds[2:4]:
        x_b, y_b, u_b = cond[:, [0]], cond[:, [1]], cond[:, [2]]
        loss += MSE(P_nn_y(x_b, y_b), u_b)
    return loss


def save_training_data_to_file(sim_name, best_params, best_epoch, best_loss,
                               all_losses, all_epochs, results_dir):
    """
    Saves the training data, including the best model parameters and the training history, to a file.

    Args:
        sim_name (str): The name of the simulation or model, which will be used as the filename.
        best_params (list): A list of dictionaries containing the best parameters (weights and biases) of the neural network.
        best_epoch (int): The epoch number at which the best model (with the lowest loss) was found.
        best_loss (float): The minimum loss value achieved during training.
        all_losses (list): A list containing the loss values for each epoch during training.
        all_epochs (list): A list containing the epoch numbers corresponding to the `all_losses`.
        results_dir (str): The directory where the results file will be saved.

    Returns:
        None: The function does not return anything. It saves the data to a file.
    """

    # Organize all the data into a dictionary.
    # This dictionary will contain the best parameters, the epoch at which the best parameters were found,
    # the minimum loss achieved, and the history of losses and epochs during training.
    data = {
        'best_params':
        best_params,  # The best parameters (weights and biases) of the neural network.
        'best_epoch':
        best_epoch,  # The epoch number corresponding to the best parameters.
        'min_loss':
        best_loss,  # The minimum loss value achieved during training.
        'all_losses': all_losses,  # A list of loss values for each epoch.
        'all_epochs':
        all_epochs  # A list of epoch numbers corresponding to the losses.
    }

    # Create the full file path where the data will be saved.
    # The file path is constructed by concatenating the results directory, the simulation name, and the file extension '.pkl'
    # which allows for storing Python objects in a binary format.
    file_path = results_dir + sim_name + '.pkl'
    with open(file_path, 'wb') as f:
        pk.dump(data, f)


def train_PINN(params, epochs, optimizer, loss_fun, colloc, conds, norm_coeff,
               hidden_layers, hidden_nodes, lr, results_dir):

    @jax.jit
    def update(opt_state, params, colloc, conds, norm_coeff):
        # Get the gradient w.r.t to MLP params
        grads = jax.grad(loss_fun, argnums=0)(params, colloc, conds,
                                              norm_coeff)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params

    # Initialize optimizer state
    opt_state = optimizer.init(params)

    # Training loop
    best_params = params
    best_loss = float('inf')
    best_epoch = 0
    all_losses = []
    all_epochs = []

    print('PINN training started...')
    for epoch in range(epochs + 1):
        opt_state, params = update(opt_state, params, colloc, conds,
                                   norm_coeff)
        current_loss = loss_fun(params, colloc, conds, norm_coeff)

        # Save the loss and epoch
        all_losses.append(current_loss)
        all_epochs.append(epoch)

        # Update best parameters and loss if the current loss is lower
        if current_loss < best_loss:
            best_loss = current_loss
            best_params = params
            best_epoch = epoch

        # Print loss and epoch info
        if epoch % 100 == 0:
            print(f'   Epoch={epoch}\t loss={current_loss:.3e}')

    # After training, print the best epoch, and loss
    print('PINN training done!')
    print(f'   Best Epoch = {best_epoch}\tBest Loss = {best_loss:.3e}')

    # Save data
    sim_name = f'pinn_model_hl-{hidden_layers}_nn-{hidden_nodes}_lr-{lr}_epoch-{epoch}'
    save_training_data_to_file(sim_name, best_params, best_epoch, best_loss,
                               all_losses, all_epochs, results_dir)

    return best_params, best_loss, all_losses, all_epochs
