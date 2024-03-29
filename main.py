#!/usr/bin/env python

"""code template"""

import random
import numpy as np

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from graphics import *
from gridutil import *

import agents

from PIL import Image as NewImage

class LocWorldEnv:
    actions = "turnleft turnright forward".split()

    def __init__(self, size, walls, eps_perc, eps_move):
        self.size = size
        self.walls = walls
        self.action_sensors = []
        self.locations = {*locations(self.size)}.difference(self.walls)
        self.eps_perc = eps_perc
        self.eps_move = eps_move
        self.reset()

    def reset(self):
        self.agentLoc = random.choice(list(self.locations))
        self.agentDir = random.choice(["N", "E", "S", "W"])

    def getPercept(self):
        p = self.action_sensors
        self.action_sensors = []
        rel_dirs = {"fwd": 0, "right": 1, "bckwd": 2, "left": 3}
        for rel_dir, incr in rel_dirs.items():
            nh = nextLoc(self.agentLoc, nextDirection(self.agentDir, incr))
            prob = 0.0 + self.eps_perc
            if (not legalLoc(nh, self.size)) or nh in self.walls:
                prob = 1.0 - self.eps_perc
            if random.random() < prob:
                p.append(rel_dir)

        return p

    def doAction(self, action):
        points = -1
        if action == "turnleft":
            if random.random() < self.eps_move:
                # small chance that the agent will not turn
                print("Robot did not turn")
            else:
                self.agentDir = leftTurn(self.agentDir)
        elif action == "turnright":
            if random.random() < self.eps_move:
                # small chance that the agent will not turn
                print("Robot did not turn")
            else:
                self.agentDir = rightTurn(self.agentDir)
        elif action == "forward":
            if random.random() < self.eps_move:
                # small chance that the agent will not move
                print("Robot did not move")
                loc = self.agentLoc
            else:
                # normal forward move
                loc = nextLoc(self.agentLoc, self.agentDir)
            if legalLoc(loc, self.size) and loc not in self.walls:
                self.agentLoc = loc
            else:
                self.action_sensors.append("bump")
        return points  # cost/benefit of action

    def finished(self):
        return False


class LocView:
    # LocView shows a view of a LocWorldEnv. Just hand it an env, and
    #   a window will pop up.

    Size = 0.2
    Points = {
        "N": (0, -Size, 0, Size),
        "E": (-Size, 0, Size, 0),
        "S": (0, Size, 0, -Size),
        "W": (Size, 0, -Size, 0),
    }

    color = "black"

    def __init__(self, state, height=800, title="Loc World"):
        xySize = state.size
        win = self.win = GraphWin(title, 1.33 * height, height, autoflush=False)
        win.setBackground("gray99")
        win.setCoords(-0.5, -0.5, 1.33 * xySize - 0.5, xySize - 0.5)
        cells = self.cells = {}
        self.dir_cells = {}
        for x in range(xySize):
            for y in range(xySize):
                cells[(x, y)] = Rectangle(
                    Point(x - 0.5, y - 0.5), Point(x + 0.5, y + 0.5)
                )
                cells[(x, y)].setWidth(2)
                cells[(x, y)].draw(win)
                for dir in DIRECTIONS:
                    if dir == "N":
                        self.dir_cells[(x, y, dir)] = Circle(Point(x, y + 0.25), 0.15)
                    elif dir == "E":
                        self.dir_cells[(x, y, dir)] = Circle(Point(x + 0.25, y), 0.15)
                    elif dir == "S":
                        self.dir_cells[(x, y, dir)] = Circle(Point(x, y - 0.25), 0.15)
                    elif dir == "W":
                        self.dir_cells[(x, y, dir)] = Circle(Point(x - 0.25, y), 0.15)
                    self.dir_cells[(x, y, dir)].setWidth(1)
                    self.dir_cells[(x, y, dir)].draw(win)
        self.agt = None
        self.arrow = None
        ccenter = 1.167 * (xySize - 0.5)

        self.time = Text(Point(ccenter, (xySize - 1) * 0.75), "Time").draw(win)
        self.time.setSize(36)
        self.setTimeColor("black")

        self.info = Text(Point(ccenter, (xySize - 1) * 0.5), "info").draw(win)
        self.info.setSize(20)
        self.info.setFace("courier")
        self.target_point = None

        self.neighbours = {(0, -1): "N", (-1, 0): "E", (0, 1): "S", (1, 0): "W"}
        self.path_visualization = []

        self.update(state)

    def setTime(self, seconds):
        text = f"Step: {str(seconds)}"
        self.time.setText(text)

    def setInfo(self, percept_info, action_info):
        info = (
            f"Percept:\n"
            + "\n".join("-" + item for item in percept_info)
            + f" \n Action: \n {action_info[0]}"
        )
        self.info.setText(info)

    def update(self, state, path=[], P=None, step=None, save_img=False):
        # View state in exiting window
        for loc, cell in self.cells.items():
            if loc in state.walls:
                cell.setFill("black")
            else:
                cell.setFill("white")
                if P is not None:
                    for i, dir in enumerate(DIRECTIONS):
                        c = int(round(P[loc[0], loc[1], i] * 255))
                        self.dir_cells[(loc[0], loc[1], dir)].setFill(
                            "#ff%02x%02x" % (255 - c, 255 - c)
                        )
        if self.agt:
            self.agt.undraw()
            if len(self.path_visualization) > 0:
                for path_line in self.path_visualization:
                    path_line.undraw()

        if state.agentLoc:
            self.agt = self.drawArrow(state.agentLoc, state.agentDir, 5, self.color)

        if len(path) > 0:
            if self.target_point is None or state.agentLoc == self.target_point:
                self.target_point = path[-1]
            self.cells[self.target_point].setFill("green")

            self.draw_path(path)

        if save_img:
            self.win.postscript(file="image.eps", colormode='color')
            img = NewImage.open("image.eps")
            img.save("images/image_"+str(step)+".jpg")

    def drawArrow(self, loc, heading, width, color):
        x, y = loc
        dx0, dy0, dx1, dy1 = self.Points[heading]
        p1 = Point(x + dx0, y + dy0)
        p2 = Point(x + dx1, y + dy1)
        a = Line(p1, p2)
        a.setWidth(width)
        a.setArrow("last")
        a.setFill(color)
        a.draw(self.win)
        return a

    def draw_path(self, path):
        for index, point in enumerate(path):
            x, y = point

            if point != path[len(path) - 1] and point != path[0]:
                next_point = path[index + 1]
                prev_point = path[index - 1]

                prev_direction = self.neighbours[
                    (prev_point[0] - point[0], prev_point[1] - point[1])
                ]
                next_direction = self.neighbours[
                    (point[0] - next_point[0], point[1] - next_point[1])
                ]

                p1 = Point(x, y)
                p2 = Point(x, y)
                p3 = Point(x, y)

                if next_direction == "N" and prev_direction == "N":
                    p1 = Point(x, y - 0.4)
                    p2 = Point(x, y)
                    p3 = Point(x, y + 0.4)

                if next_direction == "N" and prev_direction == "E":
                    p1 = Point(x - 0.4, y)
                    p2 = Point(x, y)
                    p3 = Point(x, y + 0.4)

                if next_direction == "N" and prev_direction == "W":
                    p1 = Point(x + 0.4, y)
                    p2 = Point(x, y)
                    p3 = Point(x, y + 0.4)

                if next_direction == "E" and prev_direction == "N":
                    p1 = Point(x, y - 0.4)
                    p2 = Point(x, y)
                    p3 = Point(x + 0.4, y)

                if next_direction == "E" and prev_direction == "E":
                    p1 = Point(x - 0.4, y)
                    p2 = Point(x, y)
                    p3 = Point(x + 0.4, y)

                if next_direction == "E" and prev_direction == "S":
                    p1 = Point(x, y + 0.4)
                    p2 = Point(x, y)
                    p3 = Point(x + 0.4, y)

                if next_direction == "S" and prev_direction == "W":
                    p1 = Point(x + 0.4, y)
                    p2 = Point(x, y)
                    p3 = Point(x, y - 0.4)

                if next_direction == "S" and prev_direction == "E":
                    p1 = Point(x - 0.4, y)
                    p2 = Point(x, y)
                    p3 = Point(x, y - 0.4)

                if next_direction == "S" and prev_direction == "S":
                    p1 = Point(x, y + 0.4)
                    p2 = Point(x, y)
                    p3 = Point(x, y - 0.4)

                if next_direction == "W" and prev_direction == "W":
                    p1 = Point(x + 0.4, y)
                    p2 = Point(x, y)
                    p3 = Point(x - 0.4, y)

                if next_direction == "W" and prev_direction == "N":
                    p1 = Point(x, y - 0.4)
                    p2 = Point(x, y)
                    p3 = Point(x - 0.4, y)

                if next_direction == "W" and prev_direction == "S":
                    p1 = Point(x, y + 0.4)
                    p2 = Point(x, y)
                    p3 = Point(x - 0.4, y)

                pth = Polygon(p1, p2, p3, p2, p1)
                self.path_visualization.append(pth)

                pth.setOutline("green")
                pth.setWidth(3)

                pth.draw(self.win)

    def pause(self):
        self.win.getMouse()

    def setTimeColor(self, c):
        self.time.setTextColor(c)

    def close(self):
        self.win.close()


def main():
    random.seed(16)
    # rate of executing actions
    rate = 1
    # chance that perception will be wrong
    eps_perc = 0.1
    # chance that the agent will not move forward despite the command
    eps_move = 0.05
    # eps_move = 0.2
    # number of actions to execute
    n_steps = 50
    # map of the environment: 1 - wall, 0 - free
    map = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    # (0, 0)(X, Y)

    # size of the environment
    env_size = map.shape[0]

    # build the list of walls locations
    walls = []
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j] == 1:
                walls.append((j, env_size - i - 1))

    # create the environment and viewer
    env = LocWorldEnv(env_size, walls, eps_perc, eps_move)
    view = LocView(env)

    # create the agent
    agent = agents.prob.LocAgent(env.size, env.walls, eps_perc, eps_move)
    for t in range(n_steps):
        print("step %d" % t)
        view.setTime(t)

        percept = env.getPercept()

        action, path = agent(percept)

        view.setInfo(percept, action)
        # get what the agent thinks of the environment
        prob = agent.getPosterior()

        print("Percept: ", percept)
        print("Action ", action)

        view.update(state=env, path=path, P=prob, step=t, save_img=True)
        update(rate)

        # uncomment to pause before action
        # view.pause()

        env.doAction(action)

    # pause until mouse clicked
    view.pause()


if __name__ == "__main__":
    main()
