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
        self.loc_with_orientation = []
        for i in self.locations:
            for idx2 in dir_to_idx.values():
                new = (i[0], i[1],idx2)
                self.loc_with_orientation.append(new)

        # dictionary from location to its index in the list
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}

        self.eps_perc = eps_perc
        self.eps_move = eps_move

        self.prev_action = None

        self.t = 0
        prob = 1.0 / len(self.loc_with_orientation)
        # P - macierz z prawdopodobienstwami dla kazdego pola
        self.P = prob * np.ones([len(self.locations)], dtype=np.float)
        self.P = np.array([[self.P],
                          [self.P],
                          [self.P],
                          [self.P]])
        self.P = np.transpose(self.P, (0, 2, 1))
        self.next_action = 'fwd'
        self.nawrotka = False



    def __call__(self, percept):
        # TODO PUT YOUR CODE HERE

        # MACIERZ TRANZYCJI
        T = np.zeros([4, len(self.locations), len(self.locations)], dtype=np.float)

        # jezeli poprzednia akcja byl krok PROSTO
        if self.prev_action == 'forward':
            for dir_index, direction in enumerate(['N', 'E', 'S', 'W']):
                for index, loc in enumerate(self.locations):
                    next_loc = nextLoc((loc[0], loc[1]), direction)
                    # print(loc, direction, next_loc)
                    if legalLoc((next_loc[0], next_loc[1]), self.size) and ((next_loc[0], next_loc[1]) not in self.walls):
                        next_index = self.loc_to_idx[next_loc]
                        # prawdopodobienstwo ze sie ruszy z miejsca
                        T[dir_index, index, next_index] = 1.0 - self.eps_move
                        # prawdopodobienstwo ze zostanie na tym samym miejscu
                        T[dir_index, index,index] = self.eps_move
                    else:
                        T[dir_index, index,index] = 1.0

        # jezeli poprzednia akcja byl skret w PRAWO lub LEWO
        # robot nie przesuwa sie, zostaje w tym samym miejscu, macierz jednostkowa
        if self.prev_action == 'turnright' or self.prev_action == 'turnleft':
            for dir_index, direction in enumerate(['N', 'E', 'S', 'W']):
                for index, loc in enumerate(self.locations):
                        T[dir_index, index, index] = 1.0

        # zmiana kolejnosci w macierzy self.P w zaleznosci w ktora strone nastapil obrot
        if self.prev_action == 'turnright':
            # [N, E, S, W]
            # [E, S, W, N]
            i = [3, 0, 1, 2]
            self.P = self.P[i, :, :]

        if self.prev_action == 'turnleft':
            # [N, E, S, W]
            # [W, N, E, S]
            i = [1, 2, 3, 0]
            self.P = self.P[i, :, :]

        # JEZELI W PERCEPT JEST BUMP TO MACIERZ TRANZYCJI TO SAME ZERA Z JEDNA JEDYNKA


        # jezeli poprzednia akcja jest rowna None (poczatek!)
        if self.prev_action == None:
            for dir_index, direction in enumerate(['N', 'E', 'S', 'W']):
                for index, loc in enumerate(self.locations):
                    T[dir_index, index, index, ] = 1.0


        # MACIERZ SENSORA
        O = np.zeros([4, 1, len(self.locations)], dtype=np.float)
        world_dir = ['N', 'E', 'S', 'W']
        # dla kazdej lokacji
        for index, loc in enumerate(self.locations):
            # dla kazdego kierunku swiata
            for dir_index, dir in enumerate(['N', 'E', 'S', 'W']):
                prob = 1.0

                nh_loc = nextLoc((loc[0], loc[1]), world_dir[dir_index])
                obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                if obstale == ('fwd' in percept):
                    prob *= (1 - self.eps_perc)
                else:
                    prob *= self.eps_perc

                nh_loc = nextLoc((loc[0], loc[1]), world_dir[dir_index-3])
                obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                if obstale == ('right' in percept):
                    prob *= (1 - self.eps_perc)
                else:
                    prob *= self.eps_perc

                nh_loc = nextLoc((loc[0], loc[1]), world_dir[dir_index-2])
                obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                if obstale == ('bckwd' in percept):
                    prob *= (1 - self.eps_perc)
                else:
                    prob *= self.eps_perc

                nh_loc = nextLoc((loc[0], loc[1]), world_dir[dir_index-1])
                obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                if obstale == ('left' in percept):
                    prob *= (1 - self.eps_perc)
                else:
                    prob *= self.eps_perc

                prob = round(prob, 5)
                O[dir_index, 0, index] = prob

        # Gdy czujnik nic nie wykryje
        if len(percept) == 0:
            O = self.O_prev
        else:
            self.O_prev = O


        T = np.transpose(T, (0, 2, 1))
        self.P = T @ self.P

        O = np.transpose(O, (0, 2, 1))
        self.P = O * self.P

        self.P /= np.sum(self.P)

# ------------------------------------------
        # HEURISTICS
        action = 'forward'
        if percept == ['fwd', 'right', 'left']:
            # print('nawrotka 1')
            self.nawrotka = True
            action = 'turnleft'
            self.prev_action = action
            return action

        if self.nawrotka == True:
            # print('nawrotka 2')
            self.nawrotka = False
            action = 'turnleft'
            self.prev_action = action
            return action

        if percept == [ 'right', 'bckwd', 'left']:
            action = 'forward'
            self.prev_action = action
            return action

        # if 'fwd' in percept:
        #     action = np.random.choice(['turnleft', 'turnright'], 1, p=[0.5, 0.5])
        #     if self.prev_action == 'turnleft':
        #         action = 'turnleft'
        #     if self.prev_action == 'turnright':
        #         action = 'turnright'
        #     self.prev_action = action
        #     return action

        if 'bump' in percept:
            action = np.random.choice(['turnleft', 'turnright'], 1, p=[0.5, 0.5])
            if self.prev_action == 'turnleft':
                action = 'turnleft'
            if self.prev_action == 'turnright':
                action = 'turnright'
            self.prev_action = action
            return action

        if percept == [ 'right']  or percept == ['fwd', 'right'] :
            action = np.random.choice(['turnleft', 'forward'], 1, p=[0.5, 0.5])
            self.prev_action = action
            return action

        if percept == ['left'] or percept == ['fwd', 'left']:
            action = np.random.choice(['turnright', 'forward'], 1, p=[0.5, 0.5])
            self.prev_action = action
            return action

# -----------------------------------------

        self.prev_action = action
        return action

    def getPosterior(self):
        # directions in order 'N', 'E', 'S', 'W'
        P_arr = np.zeros([self.size, self.size, 4], dtype=np.float)

        # TODO PUT YOUR CODE HERE
        for index2 in range(4):
            for idx, loc in enumerate(self.locations):
                P_arr[loc[0], loc[1], index2] = self.P[index2, idx, 0]
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
