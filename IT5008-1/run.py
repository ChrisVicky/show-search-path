from collections import deque
import heapq
import math

####################### PLOTS #######################
# Define the class for creating the grid and plotting the elements
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


class Problem:
    """The abstract class for a formal problem. A new domain subclasses this,
    overriding `actions` and `results`, and perhaps other methods.
    The default heuristic is 0 and the default action cost is 1 for all states.
    When you create an instance of a subclass, specify `initial`, and `goal` states
    (or give an `is_goal` method) and perhaps other keyword args for the subclass."""

    def __init__(self, initial=None, goal=None, **kwds):
        self.__dict__.update(initial=initial, goal=goal, **kwds)

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def is_goal(self, state):
        return state == self.goal

    def action_cost(self, s, a, s1):
        return 1

    def h(self, node):
        return 0

    def __str__(self):
        return "{}({!r}, {!r})".format(type(self).__name__, self.initial, self.goal)


class Node:
    "A Node in a search tree."

    def __init__(self, state, parent=None, action=None, path_cost: float = 0):
        self.__dict__.update(
            state=state, parent=parent, action=action, path_cost=path_cost
        )

    def __repr__(self):
        return "<{}>".format(self.state)

    def __len__(self):
        return 0 if self.parent is None else (1 + len(self.parent))

    def __lt__(self, other):
        return self.path_cost < other.path_cost


failure = Node(
    "failure", path_cost=math.inf
)  # Indicates an algorithm couldn't find a solution.
cutoff = Node(
    "cutoff", path_cost=math.inf
)  # Indicates iterative deepening search was cut off.


def expand(problem, node):
    "Expand a node, generating the children nodes."
    s = node.state
    for action in problem.actions(s):
        s1 = problem.result(s, action)
        cost = node.path_cost + problem.action_cost(s, action, s1)
        yield Node(s1, node, action, cost)


def path_actions(node: Node):
    "The sequence of actions to get to this node."
    if node.parent is None:
        return []
    return path_actions(node.parent) + [node.action]


def path_states(node):
    "The sequence of states to get to this node."
    if node in (cutoff, failure, None):
        return []
    return path_states(node.parent) + [node.state]


FIFOQueue = deque

LIFOQueue = list


class PriorityQueue:
    """A queue in which the item with minimum f(item) is always popped first."""

    def __init__(self, items=(), key=lambda x: x):
        self.key = key
        self.items = []  # a heap of (score, item) pairs
        for item in items:
            self.add(item)

    def add(self, item):
        """Add item to the queuez."""
        pair = (self.key(item), item)
        heapq.heappush(self.items, pair)

    def pop(self):
        """Pop and return the item with min f(item) value."""
        return heapq.heappop(self.items)[1]

    def top(self):
        return self.items[0][1]

    def __len__(self):
        return len(self.items)


MAZE_ROWS = 16
MAZE_COLUMNS = 24
ACTIONS = {"UP", "LEFT", "RIGHT", "DOWN"}
# ACTIONS = {"RIGHT", "LEFT", "UP", "DOWN"}
xya = {
    "UP": [1, 0],
    "LEFT": [0, -1],
    "RIGHT": [0, 1],
    "DOWN": [-1, 0],
}
_costs_ = {
    "UP": 1,
    "LEFT": 10,
    "RIGHT": 10,
    "DOWN": 1,
}
_BLACK_ = {
    (14, 9),
    (13, 10),
    (12, 10),
    (11, 10),
    (11, 12),
    (11, 13),
    (10, 9),
    (10, 10),
    (10, 12),
    (10, 13),
    (9, 9),
    (8, 9),
    (7, 9),
    (6, 9),
}


def nxt_xy(state: tuple[int, int], _action_: str) -> tuple[int, int]:
    # assert _action_ in ACTIONS
    x, y = state
    ax, ay = xya[_action_]
    return (x + ax, y + ay)


def xy_ok(state: tuple[int, int]) -> bool:
    nx, ny = state
    ok = (
        (nx >= 0)
        and (nx < MAZE_ROWS)
        and (ny >= 0)
        and (ny < MAZE_COLUMNS)
        and (state not in _BLACK_)
    )
    return ok


def action_cost(_action_: str) -> int:
    # assert _action_ in ACTIONS
    return _costs_[_action_]


class Maze(Problem):
    def __init__(self, initial=None, goal=None):
        Problem.__init__(self, initial=initial, goal=goal)

    def h1(self, node):
        """Heuristic 1: Weighted Manhattan distance considering action costs."""
        x1, y1 = node.state
        x2, y2 = self.goal
        vertical_distance = abs(x2 - x1)  # Vertical moves cost 1
        horizontal_distance = abs(y2 - y1)  # Horizontal moves cost 10
        return 1 * vertical_distance + 10 * horizontal_distance

    def h2(self, node):
        """Heuristic 2: Simplified Manhattan distance."""
        x1, y1 = node.state
        x2, y2 = self.goal
        return abs(x2 - x1) + abs(y2 - y1)

    def h3(self, node):
        return node.path_cost

    def action_cost(self, s, a, s1):
        nx, ny = nxt_xy(s, a)
        assert (nx, ny) == s1
        return action_cost(a)
        # raise NotImplementedError                   #Your Code goes here

    def result(self, state, action):
        nx, ny = nxt_xy(state, action)
        return (nx, ny)
        # raise NotImplementedError                   #Your Code goes here

    def actions(self, state):
        ret = []
        for a in ACTIONS:
            ns = nxt_xy(state, a)
            if xy_ok(ns):
                ret.append(a)
        return set(ret)
        # raise NotImplementedError                   #Your Code goes here


search_pathes = []


goal = (11, 9)


def record_path(node):
    global search_pathes
    ps = path_states(node)
    idx = len(search_pathes)
    print(f"{idx}\tLen: {len(ps)}, {node.state} vs {goal}", end="\r")
    search_pathes.append(ps)


search_nodes = []


def record_node(node):
    global search_nodes
    idx = len(search_nodes)
    print(f"{idx}", end="\r")
    search_nodes.append(node)


class GridPlotter:
    def __init__(self, grid_size, initial, goal, blocks):
        self.grid_size = grid_size
        self.initial = initial
        self.goal = goal
        self.blocks = blocks
        self.path = []

    def set_path(self, path):
        self.path = path

    def plot_grid(self):
        fig, ax = plt.subplots()

        # Manually drawing the gridlines
        for x in range(self.grid_size[1] + 1):
            ax.plot([x, x], [0, self.grid_size[0]], color="black", linewidth=1)
        for y in range(self.grid_size[0] + 1):
            ax.plot([0, self.grid_size[1]], [y, y], color="black", linewidth=1)

        # Draw the black obstacles inside the grid
        for obstacle in self.blocks:
            rect = plt.Rectangle((obstacle[1], obstacle[0]), 1, 1, facecolor="black")
            ax.add_patch(rect)

        # Mark the start and goal positions within the grid
        ax.text(
            self.initial[1] + 0.5,
            self.initial[0] + 0.5,
            "S",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )
        ax.text(
            self.goal[1] + 0.5,
            self.goal[0] + 0.5,
            "G",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            fontweight="bold",
        )

        # Plot the path with a gradient of colors
        if self.path:
            colors = cm.rainbow(np.linspace(0, 1, len(self.path)))
            for i, (x, y) in enumerate(self.path):
                ax.add_patch(plt.Rectangle((y, x), 1, 1, facecolor=colors[i]))
                ax.text(
                    y + 0.5,
                    x + 0.5,
                    f"{i+1}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

        # Set labels for axes
        ax.set_xlabel("Y axis", fontsize=12, fontweight="bold")
        ax.set_ylabel("X axis", fontsize=12, fontweight="bold")

        # Set the ticks to align with the centers of squares
        ax.set_xticks(np.arange(0.5, self.grid_size[1], 1))
        ax.set_yticks(np.arange(0.5, self.grid_size[0], 1))
        ax.set_xticklabels(np.arange(0, self.grid_size[1], 1))
        ax.set_yticklabels(np.arange(0, self.grid_size[0], 1))

        # Adjust the limits so that the ticks correspond to the little squares
        ax.set_xlim(0, self.grid_size[1])
        ax.set_ylim(0, self.grid_size[0])

        # Show the plot
        plt.savefig()


import matplotlib.animation as animation


class GridPlotterWithAnimation(GridPlotter):
    def __init__(self, grid_size, initial, goal, blocks):
        super().__init__(grid_size, initial, goal, blocks)

    def animate_paths(self, path_series, output_video="output.mp4"):
        # Set up the figure and axis
        fig, ax = plt.subplots()

        # Initialize the grid and obstacles
        def init():
            # Manually drawing the gridlines
            for x in range(self.grid_size[1] + 1):
                ax.plot([x, x], [0, self.grid_size[0]], color="black", linewidth=1)
            for y in range(self.grid_size[0] + 1):
                ax.plot([0, self.grid_size[1]], [y, y], color="black", linewidth=1)

            # Draw the black obstacles inside the grid
            for obstacle in self.blocks:
                rect = plt.Rectangle(
                    (obstacle[1], obstacle[0]), 1, 1, facecolor="black"
                )
                ax.add_patch(rect)

            # Mark the start and goal positions within the grid
            ax.text(
                self.initial[1] + 0.5,
                self.initial[0] + 0.5,
                "S",
                ha="center",
                va="center",
                color="violet",
                fontsize=12,
                fontweight="bold",
            )
            ax.text(
                self.goal[1] + 0.5,
                self.goal[0] + 0.5,
                "G",
                ha="center",
                va="center",
                color="violet",
                fontsize=12,
                fontweight="bold",
            )

            # Set labels for axes
            ax.set_xlabel("Y axis", fontsize=12, fontweight="bold")
            ax.set_ylabel("X axis", fontsize=12, fontweight="bold")

            # Set the ticks to align with the centers of squares
            ax.set_xticks(np.arange(0.5, self.grid_size[1], 1))
            ax.set_yticks(np.arange(0.5, self.grid_size[0], 1))
            ax.set_xticklabels(np.arange(0, self.grid_size[1], 1))
            ax.set_yticklabels(np.arange(0, self.grid_size[0], 1))

            # Adjust the limits so that the ticks correspond to the little squares
            ax.set_xlim(0, self.grid_size[1])
            ax.set_ylim(0, self.grid_size[0])

            return []

        # Function to update the path frame by frame
        def update(step):
            # Remove previous patches (paths) but keep obstacles
            for patch in ax.patches[len(self.blocks) :]:
                patch.remove()

            path = path_states(path_series[step])
            # path = path_series[step]
            colors = cm.rainbow(np.linspace(0, 1, len(path)))
            for i, (x, y) in enumerate(path):
                ax.add_patch(plt.Rectangle((y, x), 1, 1, facecolor=colors[i]))
                ax.text(
                    y + 0.5,
                    x + 0.5,
                    f"{i+1}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

            return ax.patches

        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=len(path_series), init_func=init, repeat=False
        )

        # Save animation as a video
        print(f"Saving to {output_video}")
        FFMpegWriter = animation.writers["ffmpeg"]
        writer = FFMpegWriter(fps=24)  # 1 frame per second for each path step
        ani.save(output_video, writer=writer)
        print(f"Saved to {output_video}")

        plt.close()


# Example usage
grid_size = (16, 24)
initial = (8, 10)
goal = (11, 9)
blocks = [
    (14, 9),
    (13, 10),
    (12, 10),
    (11, 10),
    (11, 12),
    (11, 13),
    (10, 9),
    (10, 10),
    (10, 12),
    (10, 13),
    (9, 9),
    (8, 9),
    (7, 9),
    (6, 9),
]
# [(9, 7), (10, 7), (11, 7), (12, 7), (12, 8), (12, 9)]
# Create the plotter instance
# plotter = GridPlotter(grid_size, initial, goal, blocks)
# plotter.set_path(path)
# plotter.plot_grid()


# Your code for A* Search
# def astar_search(problem, h=None):
#     global pathes
#     h = h or problem.h
#     frontier = PriorityQueue(
#         [Node(problem.initial)], key=lambda node: node.path_cost + h(node)
#     )
#     explored = set()
#     while frontier:
#         node = frontier.pop()
# #         record_path(node)
#         record_node(node)
#         if problem.is_goal(node.state):
#             return node
#         explored.add(node.state)
#         for child in expand(problem, node):
#             if child.state not in explored and child.state not in [
#                 n.state for _, n in frontier.items
#             ]:
#                 frontier.add(child)
#             elif child.state in [n.state for _, n in frontier.items]:
#                 index = next(
#                     (
#                         i
#                         for i, (_, n) in enumerate(frontier.items)
#                         if n.state == child.state
#                     ),
#                     None,
#                 )
#                 if index is not None:
#                     incumbent = frontier.items[index][1]
#                     if child.path_cost < incumbent.path_cost:
#                         frontier.items[index] = (child.path_cost + h(child), child)
#                         heapq.heapify(frontier.items)
#     return failure
#
#
# # use A* search and heuristic 1 to solve
# m = Maze(initial=(8, 10), goal=(11, 9))
# result_astar = astar_search(m, m.h3)
# print(result_astar, result_astar.path_cost)


def is_cycle(node, k=1000):
    "Does this node form a cycle of length k or less?"

    def find_cycle(ancestor, k):
        return (
            ancestor is not None
            and k > 0
            and (ancestor.state == node.state or find_cycle(ancestor.parent, k - 1))
        )

    return find_cycle(node.parent, k)


def depth_first_search(problem):
    "Search deepest nodes in the search tree first."
    frontier = LIFOQueue([Node(problem.initial)])
    result = failure
    while frontier:
        node = frontier.pop()
        # record_path(node)
        if problem.is_goal(node.state):
            return node
        elif not is_cycle(node):
            for child in expand(problem, node):
                frontier.append(child)
            record_node(node)
    return result


m = Maze(initial=(8, 10), goal=(11, 9))
result_dfs = depth_first_search(m)
print(result_dfs.path_cost)
print(path_states(result_dfs))

# Example usage
plotter_with_video = GridPlotterWithAnimation(grid_size, initial, goal, blocks)
plotter_with_video.animate_paths(
    search_nodes
)  # This will create a video showing the pathfinding steps
