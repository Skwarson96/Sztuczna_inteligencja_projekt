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
        # list of valid locations

        # tutaj trzeba wrzucic orientacje
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        #
        self.locations = list({*locations(self.size)}.difference(self.walls))
        self.loc_with_orientation = []
        for i in self.locations:
            for idx2 in dir_to_idx.values():
                new = (i[0], i[1],idx2)
                self.loc_with_orientation.append(new)
                # print(new)

        # print(len(loc_with_orientation))
        # print(self.loc_with_orientation)
        # print("self.locations ",type(self.locations), np.shape(self.locations))
        # print(len(self.locations))
        # print(self.locations)

        # dictionary from location to its index in the list
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}
        self.loc_with_orientation_to_idx = {loc: idx for idx, loc in enumerate(self.loc_with_orientation)}
        # print(self.loc_to_idx)
        # print(self.loc_with_orientation_to_idx)

        self.eps_perc = eps_perc
        self.eps_move = eps_move

        # previous action
        self.prev_action = None

        self.t = 0
        # prob = 1.0 / len(self.loc_with_orientation)
        prob = 1.0 / len(self.locations)
        # P - macierz z prawdopodobienstwami dla kazdego pola
        self.P = prob * np.ones([len(self.locations)], dtype=np.float)
        # print(self.P, np.shape(self.P))
        self.P = np.array([[self.P],
                          [self.P],
                          [self.P],
                          [self.P]])

        print(np.shape(self.P))

        # self.P = np.transpose(self.P, (0, 2, 1))
        # (4, 42, 1)
        print("START", type(self.P), np.shape(self.P))
        # print(self.P)


    def __call__(self, percept):
        # update posterior
        # TODO PUT YOUR CODE HERE
        # T = np.zeros([len(self.locations), len(self.locations)], dtype=np.float)
        T = np.zeros([len(self.locations), len(self.locations), 4], dtype=np.float)
        # print("  macierz T   ", type(T), np.shape(T))
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        # jezeli poprzednia akcja byl krok PROSTO
        if self.prev_action == 'forward':
            for index2, direction in enumerate(dir_to_idx.keys()):
                # print(index2, direction)
                for index, loc in enumerate(self.locations):
                    #next_loc = nextLoc(loc, self.dir)
                    # {'N': 0, 'E': 1, 'S': 2, 'W': 3}
                    # print(((loc[0], loc[1]), loc[2]))

                    next_loc = nextLoc((loc[0], loc[1]), direction)

                    # next_loc = nextLoc((loc[0], loc[1]), 'E')
                    #
                    #
                    # next_loc = nextLoc((loc[0], loc[1]), 'S')
                    #
                    #
                    # next_loc = nextLoc((loc[0], loc[1]), 'W')

                    # print("next_loc ", next_loc, type(next_loc))
                    # next_loc = (next_loc[0], next_loc[1], 1)
                    # print(next_loc[0], next_loc[1], 1)
                    # print("next_loc ", next_loc, type(next_loc))
                    if legalLoc((next_loc[0], next_loc[1]), self.size) and ((next_loc[0], next_loc[1]) not in self.walls):
                        # print("test 1")
                        next_index = self.loc_to_idx[next_loc]
                        # print("index", index, "next_indeks ", next_index)
                        T[index, next_index, index2] = 1.0 - self.eps_move
                        # print(T[index, next_index, index2])
                        T[index,index, index2] = self.eps_move
                        # print(T[index,index, index2])
                        # print("macierz T ", type(T), np.shape(T))
                    else:
                        # print("test 2")
                        T[index,index, index2] = 1.0
                        # print("macierz T ", type(T), np.shape(T))


        # jezeli poprzednia akcja byl skret w LEWO lub PRAWO
        else:
            for index2 in range(4):
                for index, loc in enumerate(self.locations):
                    # macierz jednostkowa
                    T[index,index, index2] = 1.0
            #
            #
        # print(np.shape(T))
        T = np.transpose(T, (2, 0, 1))
        print(np.shape(T))
        # print(T)
        O = np.zeros([len(self.locations)], dtype=np.float)
        for index, loc in enumerate(self.locations):
            # print(loc)
            prob = 1.0
            for d in ['N', 'E', 'S', 'W']:
                nh_loc = nextLoc((loc[0], loc[1]), d)

                obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)

                if obstale == (d in percept):
                    prob *= (1 - self.eps_perc)
                else:
                    prob *= self.eps_perc
            O[index] = prob
        # print(np.shape(O))
        # T = np.transpose(T, (2, 0, 1))
        # print(np.shape(T))
        # print(O)
        self.t += 1

        # print("macierz T ", type(T), np.shape(T))
        # print("macierz T.transpose() ", type(T), np.shape(T.transpose()))
        # print("macierz O ", type(O), np.shape(O))
        print("self.P", type(self.P), np.shape(self.P))
        # print(self.P)
        print(T)
        print(O)
        # print(type(O), np.shape(O))
        self.P = np.transpose(self.P, (0,2 , 1))
        print("self.P", type(self.P), np.shape(self.P))
        # self.P = T.transpose() @ self.P
        self.P = T @ self.P
        # self.P = np.transpose(T, (1, 2, 0)) @ self.P
        self.P = O * self.P
        self.P /= np.sum(self.P)
        # print(np.max(self.P))
        # print(type(self.P), np.shape(self.P))
        # print(self.P)
        # -----------------------



        action = 'forward'
        # TODO CHANGE THIS HEURISTICS TO SPEED UP CONVERGENCE

        # if there is a wall ahead then lets turn
        #         rel_dirs = {'fwd': 0, 'right': 1, 'bckwd': 2, 'left': 3}
        if 'fwd' in percept:
            # higher chance of turning left to avoid getting stuck in one location
            action = np.random.choice(['turnleft', 'turnright'], 1, p=[0.8, 0.2])
        else:
            # prefer moving forward to explore
            action = np.random.choice(['forward', 'turnleft', 'turnright'], 1, p=[0.8, 0.1, 0.1])

        ilosc = len(percept)
        # for per in percept:
        #     if per == 'left':
        #         action = 'forward'
        #     if per == 'right':
        #         pass
        #     if per == 'fwd':
        #         pass
        #     if per == 'bckwd':
        #         action = 'turnleft'
        #     if per == 'bump':
        #         action = 'turnleft'
        # if 'fwd' in percept:
        #     action = 'turnright'
        # if 'right' in percept and not 'fwd' in percept:
        #     action = 'forward'
        #
        # if 'bump' in percept:
        #     action = np.random.choice(['turnleft', 'turnright'], 1, p=[0.5, 0.5])


        # if self.plan_next_move:
        #     # randomly choose from not obstacles
        #     dirs = [d for d in ['N', 'E', 'S', 'W'] if d not in percept]
        #     self.next_dir = random.choice(dirs)
        #     self.plan_next_move = False
        #
        # action = 'forward'
        # if self.dir != self.next_dir:
        #     action = best_turn[(self.dir, self.next_dir)]
        # else:
        #     self.plan_next_move = True
        #
        # if action == 'turnleft':
        #     self.dir = leftTurn(self.dir)
        # elif action == 'turnright':
        #     self.dir = rightTurn(self.dir)


        self.prev_action = action

        return action





    def getPosterior(self):
        # directions in order 'N', 'E', 'S', 'W'
        P_arr = np.zeros([self.size, self.size, 4], dtype=np.float)
        # print("P_arr ",type(P_arr), np.shape(P_arr))

        # put probabilities in the array
        # metoda ma zwracac macierz z wartosciami rozkladu o wymiarach: [size, size, 4]
        # 4 wartosci, po jednej dla kazdego kierunku

        # TODO PUT YOUR CODE HERE
        # print("self.locations ",type(self.locations), np.shape(self.locations))
        # print("self.P ",type(self.P), np.shape(self.P))
        # self.P = np.transpose(self.P, (1, 2, 0))
        # print("self.P ",type(self.P), np.shape(self.P))
        # print(self.P)
        #
        # for idx, loc in enumerate(self.locations):
        #     # print(idx, loc, self.P[loc[2], idx])
        #     # for index2 in range(4):
        #     #     # print(idx, loc,np.shape(self.P[idx, index2] ))
        #     #     # print(loc[0], loc[1], index2, self.P[idx, index2, index2])
        #     #     # P_arr[loc[0], loc[1], index2] = 0.1
        #     P_arr[loc[0], loc[1], loc[2]] = self.P[loc[2],idx]
                #          prob = 1.0 / len(self.locations)
                # print("test")
                # print(P_arr[loc[0], loc[1], index2])
        # print(P_arr)
        # print(np.max(P_arr))
        for index2 in range(4):
            for idx, loc in enumerate(self.locations):
                # print(type(self.locations), np.shape(self.locations))
                # print(type(self.P[idx]), np.shape(self.P[idx, index2]))
                # print(self.P[idx, index2])
                # print(idx, loc)
                # print("self.P ", type(self.P), np.shape(self.P))
                # print(self.P[index2, idx, idx])
                P_arr[loc[0], loc[1], index2] = self.P[index2, idx, idx]
                # P_arr[loc[0], loc[1], index2] = 1.0
        # print("P_arr ",type(P_arr), np.shape(P_arr))



        # self.P = np.transpose(self.P, (2, 0, 1))
        # print("self.P ",type(self.P), np.shape(self.P))

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
