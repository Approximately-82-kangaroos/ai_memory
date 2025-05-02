import neat
import os
import matplotlib.pyplot as plt
from game import Game

def best_fit(X, Y):
    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    if (denum == 0):
        denum = 1

    b = numer / denum
    a = ybar - b * xbar

    return a, b

def displayCurrentData():
    global generationList
    xList = list()
    yList = list()
    for i in range(len(generationList)):
        for j in range(len(generationList[i])):
            xList.append(i)
            yList.append(generationList[i][j])
    plt.title("15 Memory Neurons")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.xlim((0, 10000))
    plt.scatter(xList, yList)
    a, b = best_fit(xList, yList)
    yfit = [a + b * xi for xi in xList]
    plt.plot(xList, yfit, color="red")
    try:
        plt.savefig("15-mem-graph4-10000gen-100pop.png", dpi=300)
    except PermissionError:
        print("Unable to plot most recent graph.")
    plt.close()

def fitness(genomes, config):
    global generationList
    global generationListIndex
    aiGame = Game()
    generationList.append(list())
    for (genomeID, genome) in genomes:
        genome.fitness = aiGame.trainAI(genome, config)
        generationList[generationListIndex].append(genome.fitness)
    generationListIndex = generationListIndex + 1
    displayCurrentData()

def runNEAT(config, loadingFromCheckpoint = False, checkpointName = ""):
    global generationList
    generationList = list()
    global generationListIndex
    generationListIndex = 0
    if (loadingFromCheckpoint):
        pop = neat.Checkpointer.restore_checkpoint(checkpointName)
    else:
        pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    #pop.add_reporter(neat.Checkpointer(1))
    pop.run(fitness, 10000)

if (__name__ == "__main__"):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    runNEAT(config)


"""
if (__name__ == "__main__"):
    newGame = Game()
    newGame.sub.xPos = 0
    newGame.sub.yPos = 1
    print(newGame.sub.checkIfHitByBeam(0, 1))
"""