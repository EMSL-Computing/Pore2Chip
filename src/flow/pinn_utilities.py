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

import jax
import jax.numpy as jnp
import optax
import pickle as pk
import optax


def generate_BCs_and_colloc_xct(xx, yy, P1, P2, dP_dy1, dP_dy2):
    """
    Generates boundary conditions and collocation points.

    Parameters:
    - ymin, ymax: Minimum and maximum values for y.
    - xmin, xmax: Minimum and maximum values for x.
    - left_bc: Boundary condition value at x=0.
    - right_bc: Boundary condition value at x=1.

    Returns:
    - conds: A list of arrays for each boundary condition.
    - colloc: An array of collocation points.
    """

    # Left BC: P[0,y] = left_bc
    x_b1 = xx[:, 0]
    y_b1 = yy[:, 0]
    left_bc = P1
    bc_1 = jnp.ones_like(y_b1) * left_bc
    BC_1 = jnp.column_stack([x_b1, y_b1, bc_1])

    # Right BC: P[1,y] = right_bc
    x_b2 = xx[:, -1]
    y_b2 = yy[:, -1]
    right_bc = P2
    bc_2 = jnp.ones_like(y_b2) * right_bc
    BC_2 = jnp.column_stack([x_b2, y_b2, bc_2])

    # Bottom BC: P_y[x,0] = 0
    x_b3 = xx[0, 1:-1]
    y_b3 = yy[0, 1:-1]
    bc_3 = jnp.ones_like(x_b3) * dP_dy1
    BC_3 = jnp.column_stack([x_b3, y_b3, bc_3])

    # Top BC: P_y[x,1] = 0
    x_b4 = xx[-1, 1:-1]
    y_b4 = yy[-1, 1:-1]
    bc_4 = jnp.ones_like(x_b4) * dP_dy1
    BC_4 = jnp.column_stack([x_b4, y_b4, bc_4])

    conds = [BC_1, BC_2, BC_3, BC_4]

    # Collocation points
    x_c = xx[1:-1, 1:-1].flatten()
    y_c = yy[1:-1, 1:-1].flatten()
    colloc = jnp.column_stack([x_c, y_c])

    return x_b1, y_b1, bc_1, x_b2, y_b2, bc_2, x_b3, y_b3, bc_3, x_b4, y_b4, bc_4, x_c, y_c, conds, colloc


#  ∂/∂x(norm_coeff * ∂P/∂x) + ∂/∂y(norm_coeff * ∂P/∂y) = 0
def pde_flow_2d_hetero_resiual(x, y, P, norm_coeff):
    # Define the gradients of P with respect to x and y
    P_x = lambda x, y: jax.grad(lambda x, y: jnp.sum(P(x, y)), 0)(x, y)
    P_y = lambda x, y: jax.grad(lambda x, y: jnp.sum(P(x, y)), 1)(x, y)
    #print("P_x = ", P_x(x,y))

    # Extract k_x and k_y from the given field k_hetero
    norm_coeff_x = jnp.array(norm_coeff[1:-1, 1:-1])  # Convert to jnp.array
    norm_coeff_x = norm_coeff_x.flatten()
    norm_coeff_x = norm_coeff_x.reshape(-1, 1)
    #print("K_x = ",k_x)

    norm_coeff_y = jnp.array(norm_coeff[1:-1, 1:-1])  # Convert to jnp.array
    norm_coeff_y = norm_coeff_y.flatten()
    norm_coeff_y = norm_coeff_y.reshape(-1, 1)

    # Define callable functions for term_x and term_y
    term_x = lambda x, y: (norm_coeff_x * P_x(x, y))
    term_y = lambda x, y: (norm_coeff_y * P_y(x, y))
    #print("term_x = ", term_x(x, y))

    # Compute the second derivatives of term_x and term_y with respect to x and y
    P_xx = lambda x, y: jax.grad(lambda x, y: jnp.sum(term_x(x, y)), 0)(x, y)
    P_yy = lambda x, y: jax.grad(lambda x, y: jnp.sum(term_y(x, y)), 1)(x, y)

    return P_xx(x, y) + P_yy(x, y)


def init_params(layers):
    keys = jax.random.split(jax.random.PRNGKey(0), len(layers) - 1)
    params = list()
    for key, n_in, n_out in zip(keys, layers[:-1], layers[1:]):
        lb, ub = -(1 / jnp.sqrt(n_in)), (
            1 / jnp.sqrt(n_in))  # xavier initialization lower and upper bound
        W = lb + (ub - lb) * jax.random.uniform(key, shape=(n_in, n_out))
        B = jax.random.uniform(key, shape=(n_out, ))
        params.append({'W': W, 'B': B})
    return params


def neural_net(params, x, y):
    X = jnp.concatenate([x, y], axis=1)
    *hidden, last = params
    for layer in hidden:
        X = jax.nn.tanh(X @ layer['W'] + layer['B'])
    return X @ last['W'] + last['B']


# MSE function to calculates loss
@jax.jit
def MSE(true, pred):
    return jnp.mean((true - pred)**2)


@jax.jit
def loss_fun(params, colloc, conds, norm_coeff):
    x_c, y_c = colloc[:, [0]], colloc[:, [1]]
    P_nn = lambda x, y: neural_net(params, x, y)
    loss = jnp.mean(pde_flow_2d_hetero_resiual(x_c, y_c, P_nn, norm_coeff)**2)

    # loss at the left and right Dirichlet BCs
    for cond in conds[0:2]:
        x_b, y_b, u_b = cond[:, [0]], cond[:, [1]], cond[:, [2]]
        loss += MSE(P_nn(x_b, y_b), u_b)

    # loss at the bottom and top Neumann BCs
    P_nn_y = lambda x, y: jax.grad(lambda x, y: jnp.sum(P_nn(x, y)), 1)(
        x, y)  # single derivative of P for Neumann BCs
    for cond in conds[2:4]:
        x_b, y_b, u_b = cond[:, [0]], cond[:, [1]], cond[:, [2]]
        loss += MSE(P_nn_y(x_b, y_b), u_b)
    return loss


def save_training_data_to_file(sim_name, best_params, best_epoch, best_loss,
                               all_losses, all_epochs, results_dir):
    data = {
        'best_params': best_params,
        'best_epoch': best_epoch,
        'min_loss': best_loss,
        'all_losses': all_losses,
        'all_epochs': all_epochs
    }

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
