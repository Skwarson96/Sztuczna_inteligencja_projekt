# Robot localisation

<p align="center">
<img src="https://github.com/Skwarson96/Sztuczna_inteligencja_projekt/blob/master/img/gif.gif" />
</p>

The purpose of the project:

- the robot moves through the maze. He knows about its size and location of obstacles, but he doesn't know where it is. The aim of the project is to locate the robot based on its movements and sensor readings.


Main project rules:

- the robot can make 3 moves: forward, turn right and turn left. Probability of move succes is 95%.
- robot posiada sensor mogący wykrywać przeszkody obok niego. Jest 5 rodzajów odczytu sensora: forward (fwd), right, left, backward (bckw) and bump (when the robot would hit an obstacle). Sensor is not perfect and returns signals with a probability of 90% success.


Rules of movement:
- at the begining robot makes random movements
- when the probability of being in a specific place will be greater than 0.8, the robot finds the farthest point in the maze from itself (manhattan distance) and then calculates the path and follows it to that point. After reaching the destination, he repeats the procedure
- in case of getting lost, the robot starts making random movements again
