Adam Carcione
Intro to Machine Learning
12/10/18

Files Included:

GA_TSP.py
10Cit.txt
17Cit.txt
26Cit.txt
48Cit.txt
readme.txt
Report.pdf

How to run:

--GA_TSP.py--
To run GA_TSP.py, execute:

"python3 GA_TSP.py <InFile> <popSize> <numGen> <probMutate> <probCrossover> <percentElite>"
Example: GA_TSP.py 10Cit.txt 300 200 0.4 0.8 0.4

In order to run, my program needs a predetermined population size, number of generations to run, probability of an individual mutating,
probability of two individuals crossing their genes, and percentage of the population that will be taken in by Elitism every generation.
In addition to this, my program also requires an input file which must be either 10Cit.txt, 17Cit.txt, 26Cit.txt, and 48Cit.txt.
The first thing that my code does is preprocess the input file and create a distance table of all of the cities to be used for 
fitness evaluation. I then encode an individual from 0 to N-1 (Ex: 10 City - [0,1,2,3,4,5,6,7,8,9]) and randomly shuffle them <popSize> 
separate times to create an initial population. 

At this point, I start the for loop where all of the functionality of the GA comes from. The first thing I do in this loop is evaluate 
fitness of my population by summing the distances between all the cities in a certain path. Once I have evaluated the fitness of my 
population I link the individuals to their fitness score and then use inverse proportional selection to select the <popSize-numElite>/2 
sets of parents. This method works by taking the inverse of every individuals fitness value, so that the smallest distances have the 
largest fitness value. Next I will perform crossover on my set of parents, running a probability on each set of parents based on 
<probCross> to determine whether or not to actually crossover the genes. If the probability fails, the two parents get copied into the 
next generation, but if it passes the parents cross their genes and create two new children who get added to the next generation. 
Finally, I perform one of 3 mutation methods on my population, running the same type of probability on each individual as I did in 
crossover. Inverse mutation can be selected as the mutation choice, which simply selects a sub-array of the individual and inverts it. 
Insert mutation can also be selected as the mutation choice, which simply selects two random items in the individual and inserts
the second one next to the first one, shifting everything afterwards over one. Swap mutation is the final mutation method that can be 
used, which works by picking two random items in the individual, and swapping only those two values. 

Once I have mutated (or not mutated) all of my individuals the loop restarts all over again, evaluating the fitness, taking the best 
of the population and copying them into the next generation (Elitism), selecting the parents for the next generation based on their
fitness, performing crossover on all of the parents, and performing mutation on all of the individuals of the new generation.

If the best fitness does not improve after 20 generations, then my program will think that the problem has converged and will terminate 
early before the total number of generations can run. 

At the end, my program outputs some useful data such as what generation the algorithm converged on, the first gen average fitness 
compared to the last gen average fitness, the distance value of the best fit individual of the population, the encoded path to take,
the percent improvement of my average fitness score from the first generation to the last generation, and the difference between the 
best fit individual's distance value and the average fitness distance values of the last generation. It will also print a graph which
plots the max, average, and best distances of every generation.




