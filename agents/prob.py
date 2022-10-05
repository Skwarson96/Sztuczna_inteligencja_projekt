# prob.py
# This is

import random
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

from gridutil import *

best_turn = {('N', 'E'): 'turnright',
             ('N', 'S'): 'turnright',
             ('N', 'W'): 'turnleft',
             ('E', 'S'): 'turnright',
             ('E', 'W'): 'turnright',
             ('E', 'N'): 'turnleft',
             ('S', 'W'): 'turnright',
             ('S', 'N'): 'turnright',
             ('S', 'E'): 'turnleft',
             ('W', 'N'): 'turnright',
             ('W', 'E'): 'turnright',
             ('W', 'S'): 'turnleft'}


class LocAgent:

    def __init__(self, size, walls, eps_perc, eps_move):
        self.size = size
        self.walls = walls

        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        self.locations = list({*locations(self.size)}.difference(self.walls))
        print(self.locations)

        self.loc_with_orientation = []
        for i in self.locations:
            for idx2 in dir_to_idx.values():
                new = (i[0], i[1], idx2)
                self.loc_with_orientation.append(new)

        # dictionary from location to its index in the list
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}
        print(self.loc_to_idx)
        self.eps_perc = eps_perc
        self.eps_move = eps_move

        self.prev_action = None

        self.t = 0
        prob = 1.0 / (len(self.locations) * 4)
        # P - macierz z prawdopodobienstwami dla kazdego pola
        self.P = prob * np.ones((4, self.size, self.size), dtype=np.float)
        print(self.P)
        print('shape self.P', self.P.shape)

        self.P_prev = None


    def __call__(self, percept):
        # MACIERZ TRANZYCJI
        transition_matrix = np.zeros([4, self.size, self.size], dtype=np.float)

        if self.prev_action == 'forward':
            transition_matrix = np.ones([4, self.size, self.size], dtype=np.float) * (1.0 - self.eps_move)
        elif self.prev_action == 'turnright' or self.prev_action == 'turnleft':
            transition_matrix = np.ones([4, self.size, self.size], dtype=np.float)
        else:
            transition_matrix = np.ones([4, self.size, self.size], dtype=np.float)

        # print('transition_matrix', transition_matrix)


        sensor_matrix = np.zeros([4, self.size, self.size], dtype=np.float)
        print('percept', percept)
        directions = ['N', 'E', 'S', 'W']
        for dir_idx, direction in enumerate(directions):
            probability = 1.0
            for loc_idx, location in enumerate(self.locations):

                fwd_location = nextLoc(location, directions[dir_idx])
                right_location = nextLoc(location, directions[dir_idx-3])
                bckwd_location = nextLoc(location, directions[dir_idx-2])
                left_location = nextLoc(location, directions[dir_idx-1])

                if 'forward' in percept:
                    if legalLoc(fwd_location, self.size) or fwd_location in self.walls:
                        probability *= 1.0 - self.eps_perc
                    else:
                        probability *= self.eps_perc

                if 'right' in percept:
                    if legalLoc(right_location, self.size) or right_location in self.walls:
                        probability *= 1.0 - self.eps_perc
                    else:
                        probability *= self.eps_perc

                if 'bckwd' in percept:
                    if legalLoc(bckwd_location, self.size) or bckwd_location in self.walls:
                        probability *= 1.0 - self.eps_perc
                    else:
                        probability *= self.eps_perc

                if 'left' in percept:
                    if legalLoc(left_location, self.size) or left_location in self.walls:
                        probability *= 1.0 - self.eps_perc
                    else:
                        probability *= self.eps_perc



                print(direction, location, probability)
                probability = round(probability, 5)
                sensor_matrix[dir_idx, location[0], location[1]] = probability




        # locations -> (kolumna, wiersz), (0, 0) -> lewy dolny rog
        print('sensor_matrix.shape', sensor_matrix.shape)
        print('transition_matrix.shape', transition_matrix.shape)
        self.P = transition_matrix * sensor_matrix
        if 'bump' in percept:
            self.P = np.array([np.eye(self.size), np.eye(self.size), np.eye(self.size), np.eye(self.size)])
        #
        # print(self.P.shape)

        action = self.heuristic(percept)

        return action

# ------------------------------------------
    def heuristic(self, percept):

        action = "forward"
        # losowe poruszanie sie
        if 'fwd' in percept:
            # skret w prawo lub w lewo z prawdopodobienstwe 50%
            action = np.random.choice(['turnleft', 'turnright'], 1, p=[0.5, 0.5])
        else:
            # Ruch do przodu z malym prawdopodobienstwem skretu
            action = np.random.choice(['forward', 'turnleft', 'turnright'], 1, p=[0.8, 0.1, 0.1])

        self.prev_action = action

        return action

# -----------------------------------------

    def getPosterior(self):
        # directions in order 'N', 'E', 'S', 'W'
        p_arr = np.zeros([4, self.size, self.size], dtype=np.float)
        p_arr = self.P


        p_arr = np.transpose(p_arr, (1, 2, 0))
        # -----------------------
        return p_arr

    def forward(self, cur_loc, cur_dir):
        if cur_dir == 'N':
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        elif cur_dir == 'E':
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == 'W':
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == 'S':
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        ret_loc = (min(max(ret_loc[0], 0), self.size - 1), min(max(ret_loc[1], 0), self.size - 1))
        return ret_loc, cur_dir

    def backward(self, cur_loc, cur_dir):
        if cur_dir == 'N':
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        elif cur_dir == 'E':
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == 'W':
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == 'S':
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        ret_loc = (min(max(ret_loc[0], 0), self.size - 1), min(max(ret_loc[1], 0), self.size - 1))
        return ret_loc, cur_dir

    @staticmethod
    def turnright(cur_loc, cur_dir):
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        dirs = ['N', 'E', 'S', 'W']
        idx = (dir_to_idx[cur_dir] + 1) % 4
        return cur_loc, dirs[idx]

    @staticmethod
    def turnleft(cur_loc, cur_dir):
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        dirs = ['N', 'E', 'S', 'W']
        idx = (dir_to_idx[cur_dir] + 4 - 1) % 4
        return cur_loc, dirs[idx]
