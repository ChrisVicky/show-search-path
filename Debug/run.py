from debugger import GridPlotterWithMultiprocessing
from background import *


# print(ACTIONS)
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


plotter = GridPlotterWithMultiprocessing()


IDX_SIZE = 1000
num = 0


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
    global num
    idx = 0
    "Search deepest nodes in the search tree first."
    frontier = LIFOQueue([Node(problem.initial)])
    result = failure
    # print(f"Actions {problem.actions(problem.initial)}")
    while frontier:
        node = frontier.pop()
        if problem.is_goal(node.state):
            plotter.add_node(node)
            print(f"\nNum: {num}")
            return node
        elif not is_cycle(node):
            plotter.add_node(node)
            idx += 1
            print(f"{idx}\tnode: {node}", end="\r")
            for child in expand(problem, node):
                frontier.append(child)
        if idx >= IDX_SIZE:
            return node
    return result


goal = (11, 9)
m = Maze(initial=(8, 10), goal=goal)
result_dfs = depth_first_search(m)
# Current Time stamp
plotter.create_video(output_video=f"{ORDER}-{result_dfs.state==goal}.mp4")
