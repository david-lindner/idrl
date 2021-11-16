"""Adapted from https://github.com/rll/rllab."""

import numpy as np


class Move(object):
    X = 11
    Y = 12
    Z = 13
    XY = 14
    XZ = 15
    YZ = 16
    XYZ = 17
    SpinXY = 18


def can_move_x(movable):
    return movable in [Move.X, Move.XY, Move.XZ, Move.XYZ, Move.SpinXY]


def can_move_y(movable):
    return movable in [Move.Y, Move.XY, Move.YZ, Move.XYZ, Move.SpinXY]


def can_move_z(movable):
    return movable in [Move.Z, Move.XZ, Move.YZ, Move.XYZ]


def can_spin(movable):
    return movable in [Move.SpinXY]


def can_move(movable):
    return can_move_x(movable) or can_move_y(movable) or can_move_z(movable)


def get_random_free_cell(structure, n_samples):
    free_tiles = (0, "r", "g")
    samples = []
    len_y = len(structure)
    len_x = len(structure[0])
    while len(samples) < n_samples:
        y = np.random.choice(len_y)
        x = np.random.choice(len_x)
        if structure[y][x] in free_tiles:
            samples.append((x, y))
    return samples


def construct_maze(maze_id="Maze"):
    if maze_id == "Maze":
        structure = [
            [1, 1, 1, 1, 1],
            [1, "r", 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, "g", 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id in ("Maze1D-Tiny", "Maze1D-Small", "Maze1D-Big"):
        if maze_id == "Maze1D-Tiny":
            n_before_goal = 1
            n_after_goal = 1
        elif maze_id == "Maze1D-Small":
            n_before_goal = 2
            n_after_goal = 2
        elif maze_id == "Maze1D-Big":
            n_before_goal = 5
            n_after_goal = 5

        wall = [1] * (5 + n_before_goal + n_after_goal)
        empty = [1] + [0] * (3 + n_before_goal + n_after_goal) + [1]
        structure = [
            wall,
            [1, "r"] + [0] * n_before_goal + ["g"] + [0] * n_after_goal + ["r", 1],
            wall,
        ]
    elif maze_id == "MazeT":
        structure = [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, "r", 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, "g", 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    elif maze_id == "Square":
        structure = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, "r", 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    elif maze_id == "Push":
        structure = [
            [1, 1, 1, 1, 1],
            [1, 0, "r", 1, 1],
            [1, 0, Move.XY, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == "Fall":
        structure = [
            [1, 1, 1, 1],
            [1, "r", 0, 1],
            [1, 0, Move.YZ, 1],
            [1, -1, -1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
    elif maze_id == "Block":
        O = "r"
        structure = [
            [1, 1, 1, 1, 1],
            [1, O, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    elif maze_id == "BlockMaze":
        O = "r"
        structure = [
            [1, 1, 1, 1],
            [1, O, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ]
    else:
        raise NotImplementedError("The provided MazeId %s is not recognized" % maze_id)

    return structure
