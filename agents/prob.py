# prob.py
# This is

import random
import numpy as np

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
        # list of valid locations

        # tutaj trzeba wrzucic orientacje
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        #
        self.locations = list({*locations(self.size)}.difference(self.walls))
        loc_with_orientation = []
        for i in self.locations:
            for idx2 in dir_to_idx.values():
                new = (i[0], i[1],idx2)
                loc_with_orientation.append(new)
                # print(new)

        # print(len(loc_with_orientation))
        # print(loc_with_orientation)
        # print("self.locations ",type(self.locations), np.shape(self.locations))
        # print(len(self.locations))
        # print(self.locations)

        # dictionary from location to its index in the list
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}


        self.eps_perc = eps_perc
        self.eps_move = eps_move

        # previous action
        self.prev_action = None


        self.t = 0
        prob = 1.0 / len(self.locations)
        # P - macierz z prawdopodobienstwami dla kazdego pola
        self.P = prob * np.ones([len(self.locations)], dtype=np.float)
        self.P = np.array([[self.P],
                          [self.P],
                          [self.P],
                          [self.P]])


        self.P = np.transpose(self.P, (0, 2, 1))
        # (4, 42, 1)
        print("START", type(self.P), np.shape(self.P))

        # print(self.P)
    def __call__(self, percept):
        # update posterior
        # TODO PUT YOUR CODE HERE
        # T = np.zeros([len(self.locations), len(self.locations)], dtype=np.float)
        T = np.zeros([len(self.locations), len(self.locations), 4], dtype=np.float)
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        # jezeli poprzednia akcja byl krok PROSTO
        if self.prev_action == 'forward':
            for index2, direction in enumerate(dir_to_idx.keys()):
                # print(index2, direction)
                for index, loc in enumerate(self.locations):
                    #next_loc = nextLoc(loc, self.dir)
                    # {'N': 0, 'E': 1, 'S': 2, 'W': 3}
                    next_loc = nextLoc(loc, direction)
                    # print("next_loc ", next_loc)
                    if legalLoc(next_loc, self.size) and (next_loc not in self.walls):
                        next_index = self.loc_to_idx[next_loc]
                        T[index, next_index, index2] = 1.0 - self.eps_move
                        T[index,index, index2] = self.eps_move

                    else:
                        T[index,index, index2] = 1.0
        # jezeli poprzednia akcja byl skred w LEWO lub PRAWO
        else:
            for index2 in range(4):
                for index, loc in enumerate(self.locations):
                    # macierz jednostkowa
                    T[index,index, index2] = 1.0
            # print(np.shape(T))

        # print(np.shape(T))
        # print(T)
        O = np.zeros([len(self.locations)], dtype=np.float)
        for index, loc in enumerate(self.locations):
            prob = 1.0
            for d in ['N', 'E', 'S', 'W']:
                nh_loc = nextLoc(loc, d)

                obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)

                if obstale == (d in percept):
                    prob *= (1 - self.eps_perc)
                else:
                    prob *= self.eps_perc
            O[index] = prob


        self.t += 1

        # print(type(T), np.shape(T.transpose()))
        # print(type(self.P), np.shape(self.P))
        # print(type(O), np.shape(O))

        self.P = T.transpose() @ self.P
        self.P = O * self.P
        self.P /= np.sum(self.P)

        # -----------------------

        action = 'forward'
        # TODO CHANGE THIS HEURISTICS TO SPEED UP CONVERGENCE
        # if there is a wall ahead then lets turn
        if 'fwd' in percept:
            # higher chance of turning left to avoid getting stuck in one location
            action = np.random.choice(['turnleft', 'turnright'], 1, p=[0.8, 0.2])
        else:
            # prefer moving forward to explore
            action = np.random.choice(['forward', 'turnleft', 'turnright'], 1, p=[0.8, 0.1, 0.1])

        self.prev_action = action

        return action

    def getPosterior(self):
        # directions in order 'N', 'E', 'S', 'W'
        P_arr = np.zeros([self.size, self.size, 4], dtype=np.float)
        print("P_arr ",type(P_arr), np.shape(P_arr))

        # put probabilities in the array
        # metoda ma zwracac macierz z wartosciami rozkladu o wymiarach: [size, size, 4]
        # 4 wartosci, po jednej dla kazdego kierunku

        # TODO PUT YOUR CODE HERE
        print("self.locations ",type(self.locations), np.shape(self.locations))
        print("self.P ",type(self.P), np.shape(self.P))
        self.P = np.transpose(self.P, (1, 2, 0))
        print("self.P ",type(self.P), np.shape(self.P))

        
        for idx, loc in enumerate(self.locations):
            print(idx, loc,np.shape(self.P[idx] ))
            # P_arr[loc[0], loc[1]] = self.P[idx]
        print("P_arr ",type(P_arr), np.shape(P_arr))



        self.P = np.transpose(self.P, (2, 0, 1))
        print("self.P ",type(self.P), np.shape(self.P))

        #
        # dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        # for idx2 in dir_to_idx.values():
        #     for idx, loc in enumerate(self.locations):
        #     # print(np.shape(self.P[idx]))
        #         print(loc[0], loc[1], idx2)
        #         P_arr[loc[0], loc[1], idx2 ] = self.P[idx]
        #         # print(self.P[idx])

        # print(P_arr)
        # -----------------------
        return P_arr

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
