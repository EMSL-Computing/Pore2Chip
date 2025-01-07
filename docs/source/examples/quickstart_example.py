# Step 1 Start
import pore2chip as p2c
import openpnm as op # For network visualization and for any manual topology changes
from matplotlib import pyplot as plt # For showing loaded images
# Step 1 End

# Step 2 Start
image_path = r'data/' # Location of the image data

# Filter and crop to 100 x 100 x 100
filtered_img_stack = p2c.filter_im.read_and_filter_list(
                                    image_path, [0, 100], [0, 100], 
                                    100, invert=True
                                )

# Show the first layer using Matplotlib
fig, ax = plt.subplots()
ax.imshow(filtered_img_stack[0,:,:], cmap='gray')
# Step 2 End

# Step 3.1 Start
pore_diameters, throat_diameters = p2c.metrics.extract_diameters(filtered_img_stack)
# Step 3.1 End

# Step 3.2 Start
coordination_numbers = p2c.coordination.coordination_nums_3D(filtered_images)
# Step 3.2 End

# Step 3.3 Start
fig, ax = plt.subplots(1, 3, figsize=(14, 6))

ret = ax[0].hist(pore_diameters, density = True)
ret2 = ax[1].hist(throat_diameters, density = True)
ret3 = ax[2].hist(coordination_numbers, density = True)
ax[0].set_xlabel("Pore Diameter (pixels)")
ax[0].set_ylabel("Probability Density")
ax[1].set_xlabel("Pore Throat Diameter (pixels)")
ax[1].set_ylabel("Probability Density")
ax[2].set_xlabel("Pore Coordination Numbers")
ax[2].set_ylabel("Probability Density")
# Step 3.3 End

# Step 4 Start
network = p2c.generate.generate_network(
                                        6, 18, 
                                        pore_diameters, 
                                        throat_diameters, 
                                        coordination_numbers, 
                                        center_channel=3
                                       )

h = op.visualization.plot_connections(network)
op.visualization.plot_coordinates(network, ax=h)
# Step 4 End

# Step 5 Start
design = export.network2svg(network, 6, 18, 200, 600)

save_path = r'micromodel.svg' # Relative path to save file

design.save_svg(save_path) # Saves SVG file using drawsvg
# Step 5 End