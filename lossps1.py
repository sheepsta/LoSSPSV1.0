import numpy as np
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
from noise import pnoise2
from queue import PriorityQueue
import math
from astar import astar
import time

z_limit = 1
node_spacing = 1
maximum_distance = 4
minimum_distance = 2

# Helper function to calculate Euclidean distance between two nodes


def distance(node1, node2):
    return np.linalg.norm(np.array(node1) - np.array(node2))


def line_of_sight(node1, node2, X, Y, Z):
    x1, y1, z1 = node1
    x2, y2, z2 = node2

    # Calculate the direction vector of the line
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # Calculate the number of steps based on the longest axis
    steps = int(max(abs(dx), abs(dy), abs(dz))) + 1

    # Calculate the step sizes for each axis
    step_x = dx / steps
    step_y = dy / steps
    step_z = dz / steps

    # Perform ray tracing
    for i in range(steps):
        x = x1 + i * step_x
        y = y1 + i * step_y
        z = z1 + i * step_z

        # Interpolate the Z coordinate based on the current (x, y) position
        z_interp = np.interp([x], X[0], Z[:, 0])[0]

        # Check if the interpolated Z coordinate is higher than the current Z coordinate
        if z_interp > z:
            return False
    if abs(distance(node1, node2)) <= minimum_distance:
        return False
    if abs(distance(node1, node2)) >= maximum_distance:
        return False

    return True


def get_nodes_within_line_of_sight(node, nodes, X, Y, Z, original_path=None):
    visible_nodes = []

    for other_node in nodes:
        if other_node == node:
            continue

        if original_path is not None and other_node in original_path:
            continue

        los = line_of_sight(node, other_node, X, Y, Z)
        if los:
            visible_nodes.append(other_node)

    return visible_nodes


def get_mountain_surface_z(x, y, scale=6, octaves=100, persistence=0.5):
    return pnoise2(x / scale, y / scale, octaves=octaves, persistence=persistence)


# Generate data
x = np.arange(-10, 10, node_spacing)  # X coordinates
y = np.arange(-10, 10, node_spacing)  # Y coordinates
X, Y = np.meshgrid(x, y)  # Create a grid of X, Y coordinates

# Define the mountain-like surface using Perlin noise
Z = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        Z[i, j] = get_mountain_surface_z(X[i, j], Y[i, j]) * 8

# Define the colormap for the terrain
colors = [
    (0.0, 'green'),   # Bottoms (green)
    (1.0, 'black')    # Mountains (brown)
]
cmap = LinearSegmentedColormap.from_list('terrain', colors)

# Generate node cloud
nodes = [(x, y, z) for x in np.arange(-11, 11, node_spacing)
         for y in np.arange(-11, 11, node_spacing)
         for z in np.arange(-11, 11, node_spacing)
         if z >= get_mountain_surface_z(x, y)]
nodes = [node for node in nodes if node[2] <= z_limit]

# Check if there are any nodes available
if not nodes:
    raise ValueError("No nodes available with the specified spacing.")

# Define start and goal nodes
start_node = (-8, 8, 1)
goal_node = (7, -6, 1)

# Check if start node falls below the mountain surface
if start_node[2] < get_mountain_surface_z(start_node[0], start_node[1]):
    raise ValueError("Start node falls below the mountain surface.")

# Check if goal node falls below the mountain surface
if goal_node[2] < get_mountain_surface_z(goal_node[0], goal_node[1]):
    raise ValueError("Goal node falls below the mountain surface.")

# Check if start node is above Z limit
if start_node[2] > z_limit:
    raise ValueError("Start node is above the Z limit.")

# Check if goal node is above Z limit
if goal_node[2] > z_limit:
    raise ValueError("Goal node is above the Z limit.")

# Check line of sight between start and goal nodes
los = line_of_sight(start_node, goal_node, X, Y, Z)
if los:
    print("There is line of sight between the start and goal nodes.")
else:
    print("The mountain obstructs the line of sight between the start and goal nodes.")

# A* algorithm for path finding


def astar(start, goal, input_nodes):
    frontier = [(0, start)]  # Priority queue of nodes to explore
    came_from = {}  # Dictionary to store the path
    g_score = {node: float('inf')
               for node in input_nodes}  # Cost from start to each node
    g_score[start] = 0
    # Cost from start to goal through each node
    f_score = {node: float('inf') for node in input_nodes}
    f_score[start] = distance(start, goal)

    while frontier:
        current = heappop(frontier)[1]

        if current == goal:
            break

        for neighbor in get_neighbors(current, input_nodes):
            tentative_g_score = g_score[current] + distance(current, neighbor)

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + \
                    distance(neighbor, goal)
                heappush(frontier, (f_score[neighbor], neighbor))

    # If no valid path is found
    if goal not in came_from:
        raise ValueError("No valid path exists.")

    # Retrieve the path
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()

    # Remove nodes above and below the path
    path_z_coordinates = [node[2] for node in path]
    max_path_z = max(path_z_coordinates)
    min_path_z = min(path_z_coordinates)
    input_nodes = [node for node in input_nodes if min_path_z <=
                   node[2] <= max_path_z]

    return path


# Helper function to get valid neighbors for a node
def get_neighbors(node, input_nodes):
    x, y, z = node
    neighbors = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx != 0 or dy != 0 or dz != 0:
                    new_node = (x + dx * node_spacing, y + dy *
                                node_spacing, z + dz * node_spacing)
                    if new_node in input_nodes:
                        neighbors.append(new_node)
    return neighbors


try:
    # Run A* algorithm
    path = astar(start_node, goal_node, nodes)
    # Get nodes within line of sight of the path
    line_of_sight_nodes = []
    for node in path:
        line_of_sight_nodes.extend(
            get_nodes_within_line_of_sight(node, nodes, X, Y, Z))

    # list of tuples after getting all line of sight
    filtered_los_nodes = []
    for node in line_of_sight_nodes:
        if node not in path:
            filtered_los_nodes.append(node)

    filtered_los_nodes.append(start_node)
    filtered_los_nodes.append(goal_node)

    line_of_sight_nodes = filtered_los_nodes

    # for node in line_of_sight_nodes:
    #     if node in path:
    #         index = np.where(any(line_of_sight_nodes) ==  node)
    #         print(index)
    #         time.sleep(5)
    #         print(np.delete(line_of_sight_nodes, index))
    #         # time.sleep(5)
    #         line_of_sight_nodes = np.delete(line_of_sight_nodes, index)

    # Remove duplicates and sort the line of sight nodes
    line_of_sight_nodes = sorted(list(set(line_of_sight_nodes)))
    # Run A* algorithm through the line of sight nodes
    # Use nodes as input instead of line_of_sight_nodes

    path2 = astar(start_node, goal_node, line_of_sight_nodes)

    # Extract coordinates for plotting
    x_nodes, y_nodes, z_nodes = zip(*nodes)
    x_path, y_path, z_path = zip(*path)
    x_los_nodes, y_los_nodes, z_los_nodes = zip(*line_of_sight_nodes)
    try:
        x_final_path, y_final_path, z_final_path = zip(*path2)
    except:
        pass

    # Plot the node cloud and the path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_nodes, y_nodes, z_nodes, color='b', label='Nodes', s=.3)
    # Plot the line of sight nodes
    ax.scatter(x_los_nodes, y_los_nodes, z_los_nodes, color='r',
               label='Line of Sight Nodes', s=3)
    ax.plot(x_path, y_path, z_path, color='black', linewidth=2, label='Path')
    try:
        ax.plot(x_final_path, y_final_path, z_final_path,
                color='orange', linewidth=2, label='Final Path')
    except:
        pass
    ax.scatter(*start_node, color='g', label='Start Node')
    ax.scatter(*goal_node, color='y', label='Goal Node')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.plot_surface(X, Y, Z, cmap=cmap)

    # Show the plot
    plt.show()

    # Print the path
    print("Path:")
    for node in path:
        print(node)
    print("Second Path:")
    for node in path2:
        print(node)
except ValueError as e:
    print("Error:", str(e))
