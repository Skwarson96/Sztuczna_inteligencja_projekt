import random
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

from gridutil import *

best_turn = {
    ("N", "E"): "turnright",
    ("N", "S"): "turnright",
    ("N", "W"): "turnleft",
    ("E", "S"): "turnright",
    ("E", "W"): "turnright",
    ("E", "N"): "turnleft",
    ("S", "W"): "turnright",
    ("S", "N"): "turnright",
    ("S", "E"): "turnleft",
    ("W", "N"): "turnright",
    ("W", "E"): "turnright",
    ("W", "S"): "turnleft",
}


class LocAgent:
    def __init__(self, size, walls, eps_perc, eps_move):
        self.size = size
        self.walls = walls

        self.directions = ["N", "E", "S", "W"]
        self.locations = list({*locations(self.size)}.difference(self.walls))

        self.eps_perc = eps_perc
        self.eps_move = eps_move

        self.prev_action = None

        # P - macierz z prawdopodobienstwami dla kazdego pola
        self.P = (1.0 / (len(self.locations) * 4)) * np.ones(
            (4, self.size, self.size), dtype=np.float
        )

    def __call__(self, percept):
        transition_matrix = self.get_transition_matrix(percept)
        sensor_matrix = self.get_sensor_matrix(percept)

        self.P = transition_matrix * sensor_matrix

        self.P /= np.sum(self.P)

        action = self.heuristic(percept)

        return action

    def get_transition_matrix(self, percept):
        transition_matrix = np.ones([4, self.size, self.size], dtype=np.float)

        if self.prev_action == "turnright":
            transition_matrix = np.zeros([4, self.size, self.size], dtype=np.float)
            for dir_idx, direction in enumerate(self.directions):
                for loc_idx, location in enumerate(self.locations):
                    transition_matrix[dir_idx, location[0], location[1]] = self.P[
                        dir_idx, location[0], location[1]
                    ] * self.eps_move + self.P[
                        dir_idx - 1, location[0], location[1]
                    ] * (
                        1 - self.eps_move
                    )

        elif self.prev_action == "turnleft":
            transition_matrix = np.zeros([4, self.size, self.size], dtype=np.float)
            for dir_idx, direction in enumerate(self.directions):
                for loc_idx, location in enumerate(self.locations):
                    transition_matrix[dir_idx, location[0], location[1]] = self.P[
                        dir_idx, location[0], location[1]
                    ] * self.eps_move + self.P[
                        dir_idx - 3, location[0], location[1]
                    ] * (
                        1 - self.eps_move
                    )

        elif self.prev_action == "forward":
            transition_matrix = np.zeros([4, self.size, self.size], dtype=np.float)
            for dir_idx, direction in enumerate(self.directions):
                for loc_idx, location in enumerate(self.locations):
                    fwd_location = nextLoc(location, self.directions[dir_idx])
                    if (
                        legalLoc(fwd_location, self.size)
                        and fwd_location not in self.walls
                    ):
                        transition_matrix[
                            dir_idx, fwd_location[0], fwd_location[1]
                        ] = self.P[
                            dir_idx, location[0], location[1]
                        ] * self.eps_move + self.P[
                            dir_idx, fwd_location[0], fwd_location[1]
                        ] * (
                            1 - self.eps_move
                        )

        elif "bump" in percept:
            transition_matrix = np.ones([4, self.size, self.size], dtype=np.float)

        return transition_matrix

    def get_sensor_matrix(self, percept):
        sensor_matrix = np.zeros([4, self.size, self.size], dtype=np.float)
        for dir_idx, direction in enumerate(self.directions):
            for loc_idx, location in enumerate(self.locations):
                probability = 1.0

                fwd_location = nextLoc(location, self.directions[dir_idx])
                right_location = nextLoc(location, self.directions[dir_idx - 3])
                bckwd_location = nextLoc(location, self.directions[dir_idx - 2])
                left_location = nextLoc(location, self.directions[dir_idx - 1])

                probability = (
                    probability
                    * self.get_sensor_probability("fwd", percept, fwd_location)
                    * self.get_sensor_probability("right", percept, right_location)
                    * self.get_sensor_probability("bckwd", percept, bckwd_location)
                    * self.get_sensor_probability("left", percept, left_location)
                )

                sensor_matrix[dir_idx, location[0], location[1]] = round(probability, 5)

        return sensor_matrix

    def get_sensor_probability(self, site, percept, next_location):
        if site in percept:
            if not legalLoc(next_location, self.size) or next_location in self.walls:
                probability = 1.0 - self.eps_perc
            else:
                probability = self.eps_perc

            return probability

        if site not in percept:
            if legalLoc(next_location, self.size) and next_location not in self.walls:
                probability = 1.0 - self.eps_perc
            else:
                probability = self.eps_perc

            return probability

    def heuristic(self, percept):
        action = "forward"
        # losowe poruszanie sie
        if "fwd" in percept:
            # skret w prawo lub w lewo z prawdopodobienstwe 50%
            action = np.random.choice(["forward", "turnleft", "turnright"], 1, p=[0.2, 0.4, 0.4])
        else:
            # Ruch do przodu z malym prawdopodobienstwem skretu
            action = np.random.choice(
                ["forward", "turnleft", "turnright"], 1, p=[0.8, 0.1, 0.1]
            )

        self.prev_action = action

        return action

    # -----------------------------------------

    def getPosterior(self):
        # directions in order 'N', 'E', 'S', 'W'
        p_arr = np.transpose(self.P, (1, 2, 0))
        # -----------------------
        return p_arr

    def forward(self, cur_loc, cur_dir):
        if cur_dir == "N":
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        elif cur_dir == "E":
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == "W":
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == "S":
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        ret_loc = (
            min(max(ret_loc[0], 0), self.size - 1),
            min(max(ret_loc[1], 0), self.size - 1),
        )
        return ret_loc, cur_dir

    def backward(self, cur_loc, cur_dir):
        if cur_dir == "N":
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        elif cur_dir == "E":
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == "W":
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == "S":
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        ret_loc = (
            min(max(ret_loc[0], 0), self.size - 1),
            min(max(ret_loc[1], 0), self.size - 1),
        )
        return ret_loc, cur_dir

    @staticmethod
    def turnright(cur_loc, cur_dir):
        dir_to_idx = {"N": 0, "E": 1, "S": 2, "W": 3}
        dirs = ["N", "E", "S", "W"]
        idx = (dir_to_idx[cur_dir] + 1) % 4
        return cur_loc, dirs[idx]

    @staticmethod
    def turnleft(cur_loc, cur_dir):
        dir_to_idx = {"N": 0, "E": 1, "S": 2, "W": 3}
        dirs = ["N", "E", "S", "W"]
        idx = (dir_to_idx[cur_dir] + 4 - 1) % 4
        return cur_loc, dirs[idx]
