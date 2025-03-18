import math
from collections import defaultdict
import numpy as np
from constants import maze_layout

# --- Maze Class ---
class Maze:
    def __init__(self, layout):
        self.layout = [row[:] for row in layout]
        self.pellets = [(x, y) for y, row in enumerate(self.layout) for x, val in enumerate(row) if val == 2]
        self.powerups = [(x, y) for y, row in enumerate(self.layout) for x, val in enumerate(row) if val == 3]

    def is_wall(self, x, y):
        if 0 <= x < len(self.layout[0]) and 0 <= y < len(self.layout):
            return self.layout[y][x] == 1
        return True  # Out of bounds is considered a wall

    def is_pellet(self, x, y):
        return (x, y) in self.pellets

    def is_powerup(self, x, y):
        return (x, y) in self.powerups

    def remove_pellet(self, x, y):
        if (x, y) in self.pellets:
            self.pellets.remove((x, y))

    def remove_powerup(self, x, y):
        if (x, y) in self.powerups:
            self.powerups.remove((x, y))

    def get_valid_moves(self, x, y):
        """Return list of valid moves from position (x, y)"""
        valid_moves = []
        for action, (dx, dy) in [("UP", (0, -1)), ("DOWN", (0, 1)), ("LEFT", (-1, 0)), ("RIGHT", (1, 0))]:
            new_x, new_y = x + dx, y + dy
            if not self.is_wall(new_x, new_y):
                valid_moves.append(action)
        return valid_moves

    def get_state(self, pacman_pos, ghost1_pos, ghost2_pos):
        return (pacman_pos, ghost1_pos, ghost2_pos, tuple(self.pellets), tuple(self.powerups))

# --- Helper: Convert State to Hashable Key ---
def state_to_key(state):
    # Positions only - include direction to pellet for better learning
    pacman_pos, ghost1_pos, ghost2_pos, pellets, powerups = state

    # Include relative position of nearest pellet (if any)
    pellet_direction = (0, 0)
    if pellets:
        px, py = pacman_pos
        nearest_pellet = min(pellets, key=lambda p: math.hypot(px - p[0], py - p[1]))
        px_dir = 0 if nearest_pellet[0] == px else (1 if nearest_pellet[0] > px else -1)
        py_dir = 0 if nearest_pellet[1] == py else (1 if nearest_pellet[1] > py else -1)
        pellet_direction = (px_dir, py_dir)

    # Include relative positions of ghosts (within a range)
    ghost1_dir = get_direction_vector(pacman_pos, ghost1_pos, max_distance=3)
    ghost2_dir = get_direction_vector(pacman_pos, ghost2_pos, max_distance=3)

    return (pacman_pos, ghost1_dir, ghost2_dir, pellet_direction)

def get_direction_vector(from_pos, to_pos, max_distance=3):
    """Get direction vector from from_pos to to_pos with limited distance info"""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    distance = math.hypot(dx, dy)

    # If too far, don't consider
    if distance > max_distance:
        return (0, 0)

    # Normalize to -1, 0, 1
    dx_dir = 0 if dx == 0 else (1 if dx > 0 else -1)
    dy_dir = 0 if dy == 0 else (1 if dy > 0 else -1)

    return (dx_dir, dy_dir)