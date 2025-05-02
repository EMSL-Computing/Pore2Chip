"""
Developed by: 
    Lal Mamud, 
    Postdoc - Subsurface Modeler, 
    Subsurface Science Group, 
    Earth System Science Division, 
    Pacific Northwest National Laboratory, 
    Richland, WA, USA.
Mentors: Maruti K. Mudunuru and Satish Karra
"""

import sys
import os, shutil, glob
import os.path
import numpy as np
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import FormatStrFormatter
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

#% Figure parameters
dpi = 600
width = 4.3
height = 3.2
fontsize = 12
fontsize_legend = 9
figure_format = '.png'  # e.g .png, .svg, etc.
plt.rcParams['font.family'] = 'serif'  # Set the font family
plt.rcParams['font.size'] = 12  # Set the font size


def plot_2d_xct_data_seg(xct_xy_slice, results_dir):
    plt.figure(figsize=(width, height), dpi=dpi)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['grey', 'white'],
                                             N=2)
    inmshow_plot = plt.imshow(xct_xy_slice, cmap=cmap)
    cbar = plt.colorbar(inmshow_plot, format=FormatStrFormatter('%.0e'))
    cbar.set_label('Normalized Intensity')
    bins = np.linspace(0, 1,
                       3)  # Define three bins including start and end values
    midpoints = 0.5 * (bins[:-1] + bins[1:])
    cbar.set_ticks(midpoints)
    cbar.ax.set_yticklabels(['0', '1'], rotation=90)
    plt.xlabel('Data Points')
    plt.ylabel('Data Points')
    plt.title("Normalized XCT data")
    plt.tight_layout()
    filename = 'XCT_image.png'
    plt.savefig(results_dir + filename, dpi=dpi)


def plot_2d_permeability_seg(X, Y, perm_field, results_dir):
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['grey', 'blue'],
                                             N=2)
    plt.figure(figsize=(width, height), dpi=dpi)
    kmin = np.min(perm_field)
    kmax = np.max(perm_field)
    pcolor_plot = plt.pcolor(X, Y, perm_field, cmap=cmap, vmin=kmin, vmax=kmax)
    cbar = plt.colorbar(pcolor_plot, format=FormatStrFormatter('%.0e'))
    cbar.set_label('Permeability ($m^2$)')
    max_perm = np.max(perm_field)
    min_perm = np.min(perm_field)
    bins = np.linspace(min_perm, max_perm,
                       3)  # Define three bins including start and end values
    midpoints = 0.5 * (bins[:-1] + bins[1:])
    cbar.set_ticks(midpoints)
    cbar.ax.set_yticklabels([f'{kmin:.0E}', f'{kmax:.0E}'], rotation=90)
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.xlim([np.min(X), np.max(X)])
    plt.ylim([np.min(Y), np.max(Y)])
    plt.tight_layout()
    filename = 'permeability.png'
    plt.savefig(results_dir + filename, dpi=dpi)


def plot_boundary_conditions(x_b1, y_b1, bc_1, x_b2, y_b2, bc_2,
                              x_b3, y_b3, bc_3, x_b4, y_b4, bc_4,
                              x_c, y_c, results_dir):
    plt.figure(figsize=(width, height), dpi=dpi)

    # Compute vmin and vmax across all boundary condition values
    all_bc_values = np.concatenate([bc_1, bc_2, bc_3, bc_4])
    vmin = np.min(all_bc_values)
    vmax = np.max(all_bc_values)

    s = 3

    # Plot the first boundary condition and link it to the colorbar
    sc = plt.scatter(x_b1, y_b1, c=bc_1, marker='x', vmin=vmin, vmax=vmax,
                     label='P[0,y]', cmap=cm.jet, s=s)

    # Remaining boundary conditions
    plt.scatter(x_b2, y_b2, c=bc_2, marker='^', vmin=vmin, vmax=vmax,
                label='P[1,y]', cmap=cm.jet, s=s)
    plt.scatter(x_b3, y_b3, c=bc_3, marker='*', vmin=vmin, vmax=vmax,
                label='$\\partial P/\\partial y[x,0]$', cmap=cm.jet, s=s)
    plt.scatter(x_b4, y_b4, c=bc_4, marker='o', vmin=vmin, vmax=vmax,
                label='$\\partial P/\\partial y[x,1]$', cmap=cm.jet, s=s)

    # Collocation points
    plt.scatter(x_c, y_c, c='k', marker='.', alpha=0.5,
                label='Collocation points', s=1)

    # Labels and colorbar
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    cbar = plt.colorbar(sc, aspect=30)
    cbar.set_label('Pressure (kPa)')

    #plt.legend(fontsize=6) 
    # plt.legend(fontsize=6, loc='best', frameon=True,
    #        title='Legend', title_fontsize=6)
    plt.legend(fontsize=6, loc='upper left', bbox_to_anchor=(0.50, 0.95))
    plt.tight_layout()

    figure_name = 'bcs_collocs'
    plt.savefig(results_dir + figure_name + figure_format,
                dpi=dpi,
                format=figure_format.strip('.'),
                bbox_inches='tight')
    plt.show()




#%% plot for pinn loss
def plot_pinn_training(all_losses, all_epochs, title, fig_name, results_dir):
    plt.figure(figsize=(width, height), dpi=dpi)
    plt.semilogy(all_epochs,
                 all_losses,
                 '-r',
                 markersize=4,
                 linewidth=1.0,
                 label="FV")
    plt.xlabel("Epochs", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    #plt.legend(prop={"size": fontsize_legend}, loc="best")
    plt.xlim(min(all_epochs), max(all_epochs))
    plt.tight_layout()
    filename = fig_name + '.png'
    plt.savefig(results_dir + filename, dpi=dpi)


def plot_2d_pressure_distribution(X, Y, pressure, cbar_lebel, title, fig_name,
                                  results_dir):
    plt.figure(figsize=(width, height), dpi=dpi)
    cmap = 'jet'
    plt.pcolor(X, Y, pressure, cmap=cmap)
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    vmin = np.round(np.min(pressure), 1)
    vmax = np.round(np.max(pressure), 1)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(
        vmin=vmin, vmax=vmax),
                                              cmap=cm.jet),
                        aspect=30)
    #cbar.set_label('Pressure (MPa)')
    cbar.set_label(f'{cbar_lebel}')
    plt.title(title)
    plt.tight_layout()
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    filename = fig_name + '.png'
    plt.savefig(results_dir + filename, dpi=dpi)
    plt.show()


def plot_pressure_along_x(X, Y, pressure, title, fig_name, results_dir):
    plt.figure(figsize=(width * 0.9, height), dpi=dpi)
    # Determine the middle index of the Y domain
    mid_y_index = Y.shape[0] // 2
    # Extract the pressure values along the middle of the Y domain
    pressure_along_x = pressure[mid_y_index, :]
    # Corresponding X values
    x_values = X[mid_y_index, :]
    plt.plot(x_values, pressure_along_x, color="blue")
    plt.xlabel('$X$ ')
    plt.ylabel('$Pressure (kPa)$')
    plt.title(title)
    plt.xlim([min(x_values), max(x_values)])
    plt.ylim([min(pressure_along_x) - 0.1, max(pressure_along_x) + 0.1])
    plt.grid(True)
    plt.tight_layout()
    filename = fig_name + '.png'
    plt.savefig(results_dir + filename, dpi=dpi)
    plt.show()


def plot_2d_pressure_distribution_masked(X, Y, pressure, perm_field,
                                         cbar_label, title, fig_name,
                                         results_dir):
    plt.figure(figsize=(width, height), dpi=dpi)
    cmap = mpl.cm.jet.copy()  # Use a copy of the jet colormap
    cmap.set_bad('grey')  # Set color for masked values

    # Create a mask where the permeability field is less than 1E-15
    masked_pressure = np.ma.masked_where(perm_field < 1E-15, pressure)

    # Determine vmin and vmax to ensure proper scaling
    vmin_value = np.min(pressure[perm_field >= 1E-15]) if np.any(
        perm_field >= 1E-15) else 0
    vmax_value = np.max(pressure)

    # Plotting the pressure field with masking using imshow
    cax = plt.imshow(masked_pressure,
                     cmap=cmap,
                     vmin=vmin_value,
                     vmax=vmax_value,
                     extent=[X.min(), X.max(),
                             Y.min(), Y.max()],
                     origin='lower',
                     aspect='auto')

    # Adding color bar
    norm = mpl.colors.Normalize(vmin=vmin_value, vmax=vmax_value)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, aspect=30)
    cbar.set_label(cbar_label)

    # Set additional plot attributes
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.title(title)
    plt.tight_layout()

    # Save the figure
    filename = fig_name + '.png'
    plt.savefig(results_dir + filename, dpi=dpi)
    # Set masked values to zero
    pressure_with_zeros = masked_pressure.filled(0)
    plt.show()
    return pressure_with_zeros

