import random
import numpy as np
import sys
import networkx as nx

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
        self.visited_locations = {}
        self.target_location = None

        # P - macierz z prawdopodobienstwami dla kazdego pola
        self.P = (1.0 / (len(self.locations) * 4)) * np.ones(
            (4, self.size, self.size), dtype=np.float
        )

    def __call__(self, percept):
        transition_matrix = self.get_transition_matrix(percept)
        sensor_matrix = self.get_sensor_matrix(percept)

        self.P = transition_matrix * sensor_matrix

        self.P /= np.sum(self.P)

        action, path = self.heuristic(percept)

        return action, path

    def get_transition_matrix(self, percept):
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

        else:
            transition_matrix = np.zeros([4, self.size, self.size], dtype=np.float)

            for dir_idx, direction in enumerate(self.directions):
                for loc_idx, location in enumerate(self.locations):
                    fwd_location = nextLoc(location, self.directions[dir_idx])
                    if fwd_location not in self.locations:
                        transition_matrix[dir_idx, location[0], location[1]] = (
                            1.0 * self.P[dir_idx, location[0], location[1]]
                        )

            for dir_idx, direction in enumerate(self.directions):
                for loc_idx, location in enumerate(self.locations):
                    fwd_location = nextLoc(location, self.directions[dir_idx])
                    if fwd_location in self.locations:
                        transition_matrix[
                            dir_idx, location[0], location[1]
                        ] = transition_matrix[dir_idx, location[0], location[1]] + (
                            self.eps_move * self.P[dir_idx, location[0], location[1]]
                        )
                        transition_matrix[
                            dir_idx, fwd_location[0], fwd_location[1]
                        ] = transition_matrix[
                            dir_idx, fwd_location[0], fwd_location[1]
                        ] + (
                            (1 - self.eps_move)
                            * self.P[dir_idx, location[0], location[1]]
                        )

        # if "bump" in percept:
        #     transition_matrix = np.ones([4, self.size, self.size], dtype=np.float)

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

                sensor_matrix[
                    dir_idx, location[0], location[1]
                ] = probability

        return sensor_matrix

    def get_sensor_probability(self, site, percept, next_location):
        if site in percept:
            if site == "fwd" and "bump" in percept:
                probability = 1.0
                return probability

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
        max_value, max_value_pos = np.max(self.P), np.argmax(self.P)
        max_value_pos_3d = np.unravel_index(max_value_pos, self.P.shape)
        current_location = max_value_pos_3d[1:]
        current_direction = self.directions[max_value_pos_3d[0]]

        print("max_value:", max_value)
        print("current_direction:", current_direction, "location:", current_location)
        print("self.target_location", self.target_location)

        path = []

        if max_value > 0.8:
            self.visited_locations[current_location] = max_value

            if self.target_location is None or self.target_location == current_location:
                self.target_location = self.calculate_fahrest_point(current_location)
                print("self.target_location", self.target_location)

            path = self.calculate_path(current_location)
            print(path)

            next_location = path[0]

            action = np.array([self.calculate_next_action(
                current_location, current_direction, next_location
            )])

        else:

            if "fwd" in percept:
                action = np.random.choice(
                    ["forward", "turnleft", "turnright"], 1, p=[0.2, 0.4, 0.4]
                )
            else:
                action = np.random.choice(
                    ["forward", "turnleft", "turnright"], 1, p=[0.8, 0.1, 0.1]
                )

        self.prev_action = action

        return action, path

    def calculate_fahrest_point(self, max_value_location):
        farthest_location = random.choice(self.locations)
        max_manhattan_distance = 0

        for location in self.locations:
            manhattan_distance = np.sum(
                np.abs(np.array(location) - np.array(max_value_location))
            )
            if manhattan_distance > max_manhattan_distance:
                max_manhattan_distance = manhattan_distance
                farthest_location = location

        return farthest_location

    def calculate_path(self, current_location):
        graph = nx.DiGraph()

        for location in self.locations:
            graph.add_node(location)

        for location1 in self.locations:
            for location2 in self.locations:
                if (
                    (
                        location1[0] + 1 == location2[0]
                        or location1[0] - 1 == location2[0]
                    )
                    and location1[1] == location2[1]
                ) or (
                    (
                        location1[1] + 1 == location2[1]
                        or location1[1] - 1 == location2[1]
                    )
                    and location1[0] == location2[0]
                ):
                    graph.add_edge(location1, location2, weight=1)

        path = nx.astar_path(graph, current_location, self.target_location)[1:]

        return path

    def calculate_next_action(self, current_location, current_direction, next_location):
        action = None
        if current_direction == "N":
            # next_location u góry
            if (
                current_location[0] == next_location[0]
                and current_location[1] < next_location[1]
            ):
                action = "forward"
            # next_location po prawej
            if (
                current_location[0] < next_location[0]
                and current_location[1] == next_location[1]
            ):
                action = "turnright"
            # next_location z tylu
            if (
                current_location[0] == next_location[0]
                and current_location[1] > next_location[1]
            ):
                action = "turnright"
            # next_location po lewej
            if (
                current_location[0] > next_location[0]
                and current_location[1] == next_location[1]
            ):
                action = "turnleft"

        if current_direction == "E":
            # next_location u góry
            if (
                current_location[0] == next_location[0]
                and current_location[1] < next_location[1]
            ):
                action = "turnleft"
            # next_location po prawej
            if (
                current_location[0] < next_location[0]
                and current_location[1] == next_location[1]
            ):
                action = "forward"
            # next_location z tylu
            if (
                current_location[0] == next_location[0]
                and current_location[1] > next_location[1]
            ):
                action = "turnright"
            # next_location po lewej
            if (
                current_location[0] > next_location[0]
                and current_location[1] == next_location[1]
            ):
                action = "turnleft"

        if current_direction == "S":
            # next_location u góry
            if (
                current_location[0] == next_location[0]
                and current_location[1] < next_location[1]
            ):
                action = "turnleft"
            # next_location po prawej
            if (
                current_location[0] < next_location[0]
                and current_location[1] == next_location[1]
            ):
                action = "turnleft"
            # next_location z tylu
            if (
                current_location[0] == next_location[0]
                and current_location[1] > next_location[1]
            ):
                action = "forward"
            # next_location po lewej
            if (
                current_location[0] > next_location[0]
                and current_location[1] == next_location[1]
            ):
                action = "turnright"

        if current_direction == "W":
            # next_location u góry
            if (
                current_location[0] == next_location[0]
                and current_location[1] < next_location[1]
            ):
                action = "turnright"
            # next_location po prawej
            if (
                current_location[0] < next_location[0]
                and current_location[1] == next_location[1]
            ):
                action = "turnleft"
            # next_location z tylu
            if (
                current_location[0] == next_location[0]
                and current_location[1] > next_location[1]
            ):
                action = "turnleft"
            # next_location po lewej
            if (
                current_location[0] > next_location[0]
                and current_location[1] == next_location[1]
            ):
                action = "forward"

        return action

    def getPosterior(self):
        p_arr = np.transpose(self.P, (1, 2, 0))
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
