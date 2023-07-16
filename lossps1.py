import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from noise import pnoise2
from queue import PriorityQueue


def get_mountain_surface_z(x, y, scale=10.0, octaves=6, persistence=0.5):
    return pnoise2(x / scale, y / scale, octaves=octaves, persistence=persistence)


def draw_sphere(ax, x, y, z, radius, color):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x_sphere = radius * np.outer(np.cos(u), np.sin(v)) + x
    y_sphere = radius * np.outer(np.sin(u), np.sin(v)) + y
    z_sphere = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color)


def generate_node_cloud(x, y, z, spacing):
    node_cloud = []
    color_dict = {}
    min_z = np.min(z)
    max_z = np.max(z)
    for xi in np.arange(np.min(x), np.max(x), spacing):
        for yi in np.arange(np.min(y), np.max(y), spacing):
            zi = get_mountain_surface_z(xi, yi)
            if zi >= min_z and zi <= max_z:
                node = (xi, yi, zi)
                node_cloud.append(node)
                color_dict[(xi, yi)] = 'red'
            else:
                color_dict[(xi, yi)] = 'gray'
    return node_cloud, color_dict


def heuristic(node, goal):
    return np.linalg.norm(np.array(node[:2]) - np.array(goal[:2]))


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return path[::-1]


def a_star(node_cloud, color_dict, start, goal):
    open_set = PriorityQueue()
    open_set.put((0, start))

    came_from = {}
    g_score = {node: float('inf') for node in node_cloud}
    g_score[start] = 0

    f_score = {node: float('inf') for node in node_cloud}
    f_score[start] = heuristic(start, goal)

    while not open_set.empty():
        _, current = open_set.get()

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in node_cloud:
            if neighbor == current:
                continue

            dist = np.linalg.norm(np.array(current) - np.array(neighbor))
            if dist > spacing:
                continue

            tentative_g_score = g_score[current] + dist
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + \
                    heuristic(neighbor, goal)
                open_set.put((f_score[neighbor], neighbor))

    return None


# Generate data
x = np.linspace(-5, 5, 100)  # X coordinates
y = np.linspace(-5, 5, 100)  # Y coordinates
X, Y = np.meshgrid(x, y)  # Create a grid of X, Y coordinates

# Define the mountain-like surface using Perlin noise
Z = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        Z[i, j] = get_mountain_surface_z(X[i, j], Y[i, j])

# Define the colormap for the terrain
colors = [
    (0.0, 'green'),   # Bottoms (green)
    (1.0, 'brown')    # Mountains (brown)
]
cmap = LinearSegmentedColormap.from_list('terrain', colors)

# Plot the surface with custom colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cmap)

# Customize the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Mountain-like Surface with Node Cloud')
ax.set_zlim(-4, 4)  # Adjusted z-axis limit

# Example usage of draw_sphere function
x_coord = 2.5
y_coord = -3.8
z_coord = get_mountain_surface_z(x_coord, y_coord)
radius = 0.2
draw_sphere(ax, x_coord, y_coord, z_coord, radius, 'red')

# Generate the node cloud
spacing = 0.1
node_cloud, color_dict = generate_node_cloud(x, y, Z, spacing)

# Plot the node cloud
# for node in node_cloud:
#     x_coord, y_coord, z_coord = node
#     color = color_dict[(x_coord, y_coord)]
#     radius = 0.03
#     draw_sphere(ax, x_coord, y_coord, z_coord, radius, color)

# Example usage of A* algorithm
start_node = (2, 2, 2)
goal_node = (1, 1, 2)
path = a_star(node_cloud, color_dict, start_node, goal_node)

# Plot the path
if path is not None:
    path_x = [node[0] for node in path]
    path_y = [node[1] for node in path]
    path_z = [node[2] for node in path]
    ax.plot(path_x, path_y, path_z, color='blue')
else:
    print("no path found")

# # Show the plot
plt.show()
