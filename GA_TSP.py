import sys
import math
import random
import getopt
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches

def preprocess(fileIn, dist):
    data = open(fileIn, "r")
    line = data.readline()
    cities = line
    cities = cities.split()

    while line:
        line = data.readline()
        if (line != ""):
            line = line.split()
            dist.append(line)
    return cities

def fitness(dist, cities, pop, fit):
    for i in range(len(pop)):
        sum = 0
        #pop[i].append(pop[i][0])
        for j in range(len(pop[0])-1):
            c1 = pop[i][j]
            c2 = pop[i][j+1]
            sum += dist[c1][c2]

        fC = pop[i][0] #add distance to return to origin city
        lC = pop[i][-1]
        sum += dist[lC][fC]
        
        fit.append(sum)

def generatePop(pop, popSize, cityOrder):
    for i in range(popSize):
        temp = cityOrder.copy()
        random.shuffle(temp)
        pop.append(temp)
    
    return pop

def selection(popFit, popSize, numElite):
    totalFit = 0

    for i in range(len(popFit)):    #Find Iverse proportional slices
        fitVal = 1/popFit[i][1]
        newTup = (popFit[i][0], fitVal)
        popFit[i] = newTup

    totalFit = 0
    for i in range(len(popFit)):    #Final Total Fitness Value or normalized values
        totalFit += popFit[i][1]

    for i in range(len(popFit)):
        fitVal = popFit[i][1]
        newTup = (popFit[i][0], (fitVal/totalFit))
        popFit[i] = newTup   

    popFit = sorted(popFit, key = lambda x : x[1])  #Sort Population by Fitness

    accNorm = []
    normSum = 0
    for i in range(len(popFit)):    #Create list of the accumulate normalization values
        accNorm.append(normSum + popFit[i][1])
        normSum += popFit[i][1]

    parents = []
    indexArr = []
    intPop = int(popSize)
    for i in range(int(popSize-numElite/2)): 
        parentss = []
        for j in range(2):
            randNum = random.randint(0,100)/100
            for j in range(len(accNorm)):
                if (j == 0):
                    if (randNum < accNorm[j]):
                        parentss.append(popFit[j][0])
                else:
                    if (randNum < accNorm[j] and randNum > accNorm[j-1]):
                        parentss.append(popFit[j][0])
        parents.append(parentss)

    #Ensure every value in parents is a couple and not a single
    for i in range(len(parents)-1):
        if (len(parents[i]) == 0):
            parents.pop(i)
        if (len(parents[i]) != 2):
            parents[i].append(parents[i][0])

    return parents

def crossover(parents, popSize, pCross, numElite):
    newPop = []
    for i in range(int(len(parents)-numElite)):
        prob = random.randint(0,100)/100
        if (prob > pCross): #Dont Crossover
            if (len(parents[i]) == 1):
                parents[i].append(parents[i][0])
            newPop.append(parents[i][0])
            newPop.append(parents[i][1])
            pass
        elif (prob <= pCross):  #Perform Crossover
            portionArr = []
            child1 = []
            child2 = []
            childArr = [child1, child2]
            cut = []
            cut2 = []
            cutArr = [cut, cut2]

            acc = 0

            if (len(parents[i]) == 1):
                parents[i].append(parents[i][0])

            #print(len(parents[i]))
            parentsArr = [parents[i][0],parents[i][1]]   #Parents
            for j in range(2):  #One Iteration per parent
                portionInd = random.sample(range(0, len(parents[j][0])+1), 2)  #Indexes used for isolation crossover portion
                portionInd = sorted(portionInd) 
                portion = parentsArr[j][portionInd[0]:portionInd[1]]
                portionArr.append(portion)

                for k in range(len(parentsArr[j])):     #Check if every value from pX is in the portion, if it is:ignore, if not:append to cutArr
                    if (j == 0):   
                        if (parentsArr[1][k] not in portion):
                            cutArr[0].append(parentsArr[1][k])
                    if (j == 1):
                        if (parentsArr[0][k] not in portion):
                            cutArr[1].append(parentsArr[0][k])

                tempInd = 0
                tempInd2 = 0

                for l in range(len(parentsArr[j])):
                    if (l < portionInd[0]):
                        childArr[j].append(cutArr[j][l])
                        tempInd +=1
                    if (l >= portionInd[0] and l < portionInd[1]):
                        childArr[j].append(portionArr[j][tempInd2])
                        tempInd2 +=1
                    if (l >= portionInd[1]):
                        childArr[j].append(cutArr[j][tempInd])
                        tempInd +=1
            newPop.append(childArr[0])
            newPop.append(childArr[1])

    return newPop

def insMut(pop, pMut, numElite): #Takes a child after crossover
    for i in range(int(len(pop)-numElite)):
        prob = random.randint(0,100)/100
        if (prob <= pMut):  #Mutate
            chrom = pop[i]
            ind = sorted(random.sample(range(0,len(pop[i])), 2))

            val1 = pop[i][ind[0]]
            val2 = pop[i][ind[1]]
            indAcc = val2
            while (pop[i][ind[0]+1] != val2): #While the number after the first index isnt the desired value
                temp = pop[i][ind[1]-1]
                pop[i][ind[1]-1] = pop[i][ind[1]]
                pop[i][ind[1]] = temp
                ind[1] -= 1

    return pop

def invMut(pop, pMut, numElite):
    for i in range(int(len(pop)-numElite)):
        prob = random.randint(0,100)/100
        if (prob <= pMut):  #Mutate
            #print("Mutate")
            chrom = pop[i]
            #print(pop[i])
            ind = sorted(random.sample(range(0,len(pop[i])), 2))

            s1 = chrom[0:ind[0]]
            seg = chrom[ind[0]:ind[1]]
            s2 = chrom[ind[1]:]

            seg.reverse()
            newChrom = s1 + seg + s2
            pop[i] = newChrom
            #print(pop[i])
    return pop

def swapMut(pop, pMut, numElite): #Takes a child after crossover
    for i in range(int(len(pop)-numElite)):
        prob = random.randint(0,100)/100
        if (prob <= pMut):  #Mutate
            chrom = pop[i]
            ind = sorted(random.sample(range(0,len(pop[i])), 2))


            val1 = pop[i][ind[0]]
            val2 = pop[i][ind[1]]
            temp = val1
            pop[i][ind[0]] = val2
            pop[i][ind[1]] = temp

    return pop

def main():
    fileIn = sys.argv[1]
    popSize = int(sys.argv[2])
    numGen = int(sys.argv[3])
    #numCities = int(sys.argv[4])
    pMut = float(sys.argv[4])
    pCross = float(sys.argv[5]) 
    pElite = float(sys.argv[6])   

    Title = ""
    tempArr = []
    distArr = []
    cities = preprocess(fileIn, tempArr) # Find distance array and list of cities
    numElite = popSize*pElite

    for i in range(len(tempArr)):   # Convert data into ints for convenience
        y = []
        for j in range(len(tempArr[0])):
            val = tempArr[i][j]
            y.append(int(val))
        distArr.append(y)
    distArr = np.asarray(distArr)

    if (fileIn == "10Cit.txt"):
        Title = "10 Cities"
        cityOrder = [0,1,2,3,4,5,6,7,8,9]
    elif(fileIn == "17Cit.txt"):
        Title = "17 Cities"
        cityOrder = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    elif(fileIn == "26Cit.txt"):
        Title = "26 Cities"
        cityOrder = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    elif(fileIn == "48Cit.txt"):
        Title = "48 Cities"
        cityOrder = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47]
    
    population = []
    population = generatePop(population, popSize, cityOrder)    # Create Inital Population

    firstItFit = []
    gen = 0

    allFit = []
    avgFit = []
    minFit = []
    xaxis = []
    convergeCount = 0
    minFitness = float('inf')

    ### START OF GENETIC ALGORITHM LOOP ###
    for i in range(numGen): 
        print("Generation:", gen+1)
        fitScores = []
        elite = []

        fitness(distArr,cities, population, fitScores)          # Determine fitness of population
        
        if (i == 0):    #Record fitness of first iteration
            firstItFit = fitScores
            firstItFit = np.asarray(firstItFit)

        #print(np.min(fitScores))
        if(np.min(fitScores) < minFitness): # checking for max fitness score
            minFitness = np.min(fitScores)
            convergeCount = 0 #reset convergeCount
        elif(minFitness == np.min(fitScores)):
            convergeCount += 1
            
        if(convergeCount > 20): # checking for convergence
            #gen+=1
            print("\nConverged at Generation", gen-20)
            break
        

        popFitDict = {} 
        popFitArr = []
        finPopFitArr = []
        for i in range(len(fitScores)):     # Link the fitness scores to the index of their population
            fitVal = fitScores[i]
            popFitDict[i] = fitVal
            tup = (population[i],fitScores[i])
            popFitArr.append(tup)
   
        temporary = sorted(popFitArr, key = lambda x : x[1])  #Sort Population by Fitness
        for i in range(int(numElite)):
            elite.append(temporary[i][0])

        
        ####### Used for Testing and Plotting ######
        allFit.append(max(fitScores))
        minFit.append(np.min(fitScores))
        a = fitScores
        a = np.asarray(a)
        avgFit.append(np.average(a))
        ############################################

        for i in range(len(popFitArr)): #Append before normalization so we know final best chromosome
            finPopFitArr.append(popFitArr[i])

        ''' Selection '''
        parents = selection(popFitArr, popSize, numElite)

        ''' Crossover '''
        population = crossover(parents, popSize, pCross, numElite)

        ''' Mutation '''
        #population = insMut(population, pMut, numElite)
        population = invMut(population, pMut, numElite)
        #population = swapMut(population, pMut, numElite)

        population += elite
        gen+=1
        xaxis.append(gen)


    fitScores = np.asarray(fitScores)
    print()
    print("First Generation Average Fitness:",  np.average(firstItFit))
    #print()
    print("Last Generation Average Fitness:", np.average(fitScores))

    v1 = 1/np.average(fitScores)
    v2 = 1/np.average(firstItFit)
    Impr = (v1/v2)*100

    theInd = 0
    min = float('inf')
    bestRoute = []
    for i in range(len(finPopFitArr)):
        if (finPopFitArr[i][1] < min):
            min = finPopFitArr[i][1]
            bestRoute = finPopFitArr[i][0]

    print("\nBest Fit:", min)
    print("Best Route:", bestRoute, "\n")


    print("######################## STATS ########################")
    if (min == 1127 or min == 2085 or min == 937 or min == 10628):
        print("Optimal Convergence: Yes")
    else:
        print("Optimal Convergence: No")
    print("Fitness Improvement from First Gen to Final Gen: " + str(int(Impr)) + "%" )
    #print("Last Generation Average Fitness adj w/ StD:", "[" + str(int(np.average(fitScores)-np.std(fitScores)))+","+str(int(np.average(fitScores)+np.std(fitScores)))+"]")
    print("Average Fitness V. Best Fitness Difference:", int(np.average(fitScores))-min)
    print("#######################################################")

 

    ################### PLOT ####################
    allFit = np.asarray(allFit)
    avgFit = np.asarray(avgFit)
    minFit = np.asarray(minFit)

    plot.plot(xaxis, allFit, "b") #plot max fitness of each generation
    plot.plot(xaxis, avgFit, "g") #plot average fitness of each generation
    plot.plot(xaxis, minFit, "r") #plot min fitness of each generation
    plot.xlabel("Generation")
    plot.ylabel("Fitness (Distance)")

    blue_patch = mpatches.Patch(color = 'blue', label = "Worst Fitness")
    green_patch = mpatches.Patch(color = 'green', label = "Average Fitness")
    red_patch = mpatches.Patch(color = 'red', label = "Best Fitness")

    plot.legend(handles = [blue_patch, red_patch, green_patch])
    plot.suptitle(Title, fontsize = 16)
    plot.axis([1,gen,0, max(allFit) + 100])
    plot.show()
    #############################################
    
main()
