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
        self.loc_with_orientation_to_idx = {loc: idx for idx, loc in enumerate(self.loc_with_orientation)}

        self.eps_perc = eps_perc
        self.eps_move = eps_move

        self.prev_action = None

        self.t = 0
        prob = 1.0 / len(self.loc_with_orientation)
        # P - macierz z prawdopodobienstwami dla kazdego pola
        self.P = prob * np.ones([len(self.locations)], dtype=np.float)
        # print(self.P, np.shape(self.P))
        self.P = np.array([[self.P],
                          [self.P],
                          [self.P],
                          [self.P]])

        self.P = np.transpose(self.P, (0, 2, 1))
        self.O_prev = np.zeros([len(self.locations), 4, 4], dtype=np.float)
        self.next_action = 'fwd'
        self.nawrotka = False
        self.prev_percept = []

    def __call__(self, percept):
        # update posterior
        # TODO PUT YOUR CODE HERE

        T = np.zeros([len(self.locations), len(self.locations), 4], dtype=np.float)
        # print("  macierz T   ", type(T), np.shape(T))

        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        # jezeli poprzednia akcja byl krok PROSTO
        if self.prev_action == 'forward':
            for index2, direction in enumerate(dir_to_idx.keys()):
                for index, loc in enumerate(self.locations):
                    next_loc = nextLoc((loc[0], loc[1]), direction)
                    if legalLoc((next_loc[0], next_loc[1]), self.size) and ((next_loc[0], next_loc[1]) not in self.walls):
                        next_index = self.loc_to_idx[next_loc]
                        # prawdopodobienstwo ze sie ruszy z miejsca
                        T[index, next_index, index2] = 1.0 - self.eps_move
                        # prawdopodobienstwo ze zostanie na tym samym miejscu
                        T[index,index, index2] = self.eps_move
                    # jezeli trafi na przeszkode to zostaje w tym samym miejscu
                    else:
                        T[index,index, index2] = 1.0

        # jezeli poprzednia akcja byl skret w PRAWO lub LEWO
        if self.prev_action == 'turnright' or self.prev_action == 'turnleft':
            for index2, direction in enumerate(dir_to_idx.keys()):
                for index, loc in enumerate(self.locations):
                    # prawdopodobienstwo ze sie obroci
                    # jezeli robot sie obrocil to zmienia sie wartosci w percept
                    if self.prev_percept != percept:
                        # T[index,index, index2] = 1.0 - self.eps_move
                        T[index, index, index2] = 1.0
                        # prawdopodobienstwo ze sie nie obroci
                    else:
                        # T[index,index, index2] = self.eps_move
                        T[index, index, index2] = 1.0

        # JEZELI W PERCEPT JEST BUMP TO MACIERZ TRANZYCJI TO SAME ZERA Z JEDNA JEDYNKA


        # jezeli poprzednia akcja jest rowna None (poczatek!)
        if self.prev_action == None:
            for index2, direction in enumerate(dir_to_idx.keys()):
                for index, loc in enumerate(self.locations):
                    T[index, index, index2] = 1.0




        # macierz sensora
        O = np.zeros([4, 1, len(self.locations)], dtype=np.float)
        sensor_dir = ['fwd', 'right', 'bckwd', 'left']
        # dla kazdej lokacji
        for index, loc in enumerate(self.locations):
            # dla kazdego kierunku swiata
            for dir_index, dir in enumerate(['N', 'E', 'S', 'W']):
                prob = 1.0
                for dir_index2, dir2 in enumerate(['N', 'E', 'S', 'W']):
                    # sprawdz czy rozwazana lokalizacja w tym kierunku jest przeszkoda
                    nh_loc = nextLoc((loc[0], loc[1]), dir2)
                    obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                    # obstale False - nie ma przeszkody
                    # obstale True - jest przeszkoda
                    # print( loc, dir, dir2, nh_loc, obstale)
                    # print(percept)
                    if dir == 'N':
                        if dir2 == 'N':
                            # print("N", dir2, obstale , 'fwd' in percept)
                            if obstale == ('fwd' in percept):
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        if dir2 == 'E':
                            if obstale == ('right' in percept):
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        if dir2 == 'S':
                            if obstale == ('bckwd' in percept):
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        if dir2 == 'W':
                            if obstale == ('left' in percept):
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                    else:
                        prob *= 0

                prob = round(prob, 5)
                O[dir_index,  0, index] = prob



                    # for idx, per in enumerate(['fwd', 'right', 'bckwd', 'left']):
                    #     # print(idx, per)
                    #     # print(sensor_dir[dir_index])
                    #     # jezeli (jest/nie ma) przeszkody i w percept (znajduje/nie znajduje) sie dany kierunek to:
                    #     print(loc, dir, nh_loc, obstale, (per in percept), per, percept)
                    #     if obstale == (per in percept):
                    #         # print("test")
                    #         # print(per, percept, 1 - self.eps_perc)
                    #         prob *= (1 - self.eps_perc)
                    #     else:
                    #         # print(per, percept, self.eps_perc)
                    #         prob *= self.eps_perc




                        # if per in percept:
                        #     # jezeli jest przeszkoda, to sie wszystko zgadza i pomnoz razy 0.9
                        #     if obstale == True:
                        #         # print('1 w kierunku',per,  dir,'Jest przeszkoda')
                        #         prob *= (1 - self.eps_perc)
                        #         print(1 - self.eps_perc)
                        #     # jezeli przeszkody nie ma to sensor klamie i pomnoz razy 0.1
                        #     else:
                        #         # print('2 w kierunku',per,  dir,"Nie ma przeszkody")
                        #         prob *= self.eps_perc
                        #         print(self.eps_perc)
                        # # jezeli w percept nie ma danego kierunku to:
                        # else:
                        #     # jezeli jest przeszkoda, to znaczy ze sensor klamie, bo nie wykryl przeszkode
                        #     if obstale == True:
                        #         # print('3 w kierunku',per,  dir,'Jest przeszkoda')
                        #         prob *=  self.eps_perc
                        #         print(self.eps_perc)
                        #     # jezeli nie ma przeszkody w tymi miejscu to wszystko sie zgadza
                        #     else:
                        #         # print('4 w kierunku',per,  dir, "Nie ma przeszkody")
                        #         prob *= (1 -self.eps_perc)
                        #         print(1 -self.eps_perc)
                        # print(prob)


#----------------------------------------------------------------------------------------



            '''
                # grot gora
                if i == 0:
                    # fwd = N
                    prob = 1.0
                    for idx, p in enumerate(per):
                        if p == 'fwd':
                            d = 'N'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            # obstale False - nie ma przeszkody
                            # obstale True - jest przeszkoda
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'N'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                        if p == 'right':
                            d = 'E'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'E'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                        if p == 'bckwd':
                            d = 'S'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'S'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                        if p == 'left':
                            d = 'W'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'W'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                    prob = round(prob, 5)
                    O[index, 0, i] = prob

                # grot prawo
                if i == 1:
                    # fwd = E
                    prob = 1.0
                    for idx, p in enumerate(per):
                        if p == 'fwd':
                            d = 'E'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'E'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                        if p == 'right':
                            d = 'S'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                # print('test 2')
                                prob *= self.eps_perc
                        else:
                            d = 'S'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                        if p == 'bckwd':
                            d = 'W'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                # print('test 2')
                                prob *= self.eps_perc
                        else:
                            d = 'W'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                        if p == 'left':
                            d = 'N'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                # print('test 2')
                                prob *= self.eps_perc
                        else:
                            d = 'N'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                    prob = round(prob, 5)
                    O[index, 0, i] = prob

                # grot dol
                if i == 2:
                    # fwd = S
                    prob = 1.0
                    for idx, p in enumerate(per):
                        if p == 'fwd':
                            d = 'S'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'S'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                        if p == 'right':
                            d = 'W'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'W'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                        if p == 'bckwd':
                            d = 'N'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'N'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                        if p == 'left':
                            d = 'E'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'E'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc

                    prob = round(prob, 5)
                    O[index, 0, i] = prob

                # grot lewo
                if i == 3:
                    # fwd = N
                    prob = 1.0
                    for idx, p in enumerate(per):
                        if p == 'fwd':
                            d = 'W'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'W'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc



                        if p == 'right':
                            d = 'N'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'N'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc


                        if p == 'bckwd':
                            d = 'E'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'E'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc


                        if p == 'left':
                            d = 'S'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == True:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc
                        else:
                            d = 'S'
                            nh_loc = nextLoc((loc[0], loc[1]), d)
                            obstale = (not legalLoc(nh_loc, self.size)) or (nh_loc in self.walls)
                            if obstale == False:
                                prob *= (1 - self.eps_perc)
                            else:
                                prob *= self.eps_perc


                    prob = round(prob, 5)
                    O[index, 0, i] = prob
        '''

        # data = np.load('O_01.npy')
        # print(data)
        # print(np.shape(data))

        # print(O)
        # Gdy czujnik nic nie wykryje
        if len(percept) == 0:
            O = self.O_prev
        else:
            self.O_prev = O



        # print("  macierz T   ", type(T), np.shape(T))
        # print(T)

        # T = np.transpose(T, (2, 1, 0))

        # print("  macierz T   ", type(T), np.shape(T))
        # print(T)

        # print("  macierz O   ", type(O), np.shape(O))

        # print("  macierz O   ", type(O), np.shape(O))
        # for x in range(np.shape(T)[0]):
        #     for y in range(np.shape(T)[2]):
        #         for z in range(np.shape(T)[1]):
        #             print(T[x,z,y], end=' ')
        #         print('\n')
        #     break
        #     print('\n')

        # print("self.P 1", type(self.P), np.shape(self.P))

        #

        # print("O", type(O), np.shape(O))
        print(O)
        #
        # print(T)
        # print("T", np.shape(T), "self.P",np.shape(self.P))

        T = np.transpose(T, (2, 0, 1))

        # print(T)

        # print("T", np.shape(T), "self.P",np.shape(self.P))
        self.P = T @ self.P
        # print(T @ self.P)
        # print("self.P 2 ", type(self.P), np.shape(self.P))

        # for i in range(4):
        #     for idx in range(42):
        #         print(idx, O[i,idx, 0], "*", self.P[i, idx, 0], '=', round(O[i,idx, 0]*self.P[i, idx, 0], 5))
        # print("O", np.shape(O), "self.P",np.shape(self.P))


        # O_test = O
        # O_test = np.transpose(O_test, (2, 1, 0))
        # print(O_test)
        # print(np.shape(O_test))

        # print(np.shape(O))
        # print(O)
        # O = np.transpose(O, (2, 0, 1))
        # print(np.shape(O))
        # print('self.P', np.shape(self.P))
        # print(T)
        # print("O", np.shape(O), "self.P",np.shape(self.P))

        self.P = O * self.P

        # print(self.P)

        # print("self.P 3", type(self.P), np.shape(self.P))
        # self.P = np.transpose(self.P, (0, 2, 1))
        # print(self.P)
        # print("self.P 4", type(self.P), np.shape(self.P))
        # self.P = np.transpose(self.P, (0, 2, 1))

        self.P /= np.sum(self.P)
        # print(self.P)

        # print("self.P 5 ", type(self.P), np.shape(self.P))


        # ------------------------------------------
        action = 'forward'

        # HEURISTICS
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

        if 'fwd' in percept:
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


        self.prev_percept = percept
        self.prev_action = action
        return action

    def getPosterior(self):
        # directions in order 'N', 'E', 'S', 'W'
        P_arr = np.zeros([self.size, self.size, 4], dtype=np.float)
        # P_arr = np.transpose(P_arr, (2, 0, 1))
        # TODO PUT YOUR CODE HERE

        for index2 in range(4):
            for idx, loc in enumerate(self.locations):

                P_arr[loc[0], loc[1], index2] = self.P[index2, idx, 0]

        # -----------------------

        # print(P_arr)
        # print(np.shape(P_arr))
        # P_arr = np.transpose(P_arr, (2, 1, 0))
        # print(np.shape(P_arr))
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
