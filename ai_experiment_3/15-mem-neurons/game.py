import random
import math
import neat

class EnemySub:
    xPos = 0
    yPos = 0
    xVel = 0
    yVel = 0
    radius = 0

    def __init__(self, enemyRadius):
        angle = random.random() * 360
        self.xPos = math.cos(angle * math.pi / 180)
        self.yPos = math.sin(angle * math.pi / 180)
        self.radius = enemyRadius
    
    def __del__(self):
        pass
    
    def generateNewVelocity(self):
        angle = random.random() * 360
        self.xVel = math.cos(angle * math.pi / 180)
        self.yVel = math.sin(angle * math.pi / 180)
    
    def updatePosition(self, deltaTime):
        self.xPos = self.xPos + (self.xVel / deltaTime)
        self.yPos = self.yPos + (self.yVel / deltaTime)
    
    def checkIfHitByBeam(self, dx, dy):
        b = -2 * (self.xPos * dx + self.yPos * dy)
        c = self.xPos**2 + self.yPos**2 - self.radius**2
        discriminant = b**2 - 4 * c
        if (discriminant < 0):
            return False
        sqrtDisc = math.sqrt(discriminant)
        t1 = (-b - sqrtDisc) / 2
        t2 = (-b + sqrtDisc) / 2
        if (t1 >= 0):
            return True
        if (t2 >= 0):
            return True
        return False

class Player:
    angle = 0
    currentAngularVelocity = 0

    def __init__(self):
        self.currentAngularVelocity = 0
        self.angle = 0

    def changeVelocity(self, newVelocity):
        self.currentAngularVelocity = newVelocity
    
    def updateAngle(self, deltaTime):
        self.angle = self.angle + (self.currentAngularVelocity * deltaTime)
        isNegative = self.angle < 0
        self.angle = self.angle % 360
        if (isNegative):
            self.angle = -self.angle

class Game:
    sub = EnemySub
    player = Player
    deltaTime = 0
    radius = 0
    currentSteps = 0
    maxNumSteps = 0
    fitness = 0
    lastKnownPositionX = 0
    lastKnownPositionY = 0
    lastKnownVelocityX = 0
    lastKnownVelocityY = 0
    stepsSinceLastPing = 0
    memoryNeuronIn = [0 for i in range(15)]

    def __init__(self, delta = 1/60, enemyRadius = 0.5, maxTime = 100):
        self.radius = enemyRadius
        self.sub = EnemySub(enemyRadius)
        self.player = Player()
        self.maxNumSteps = maxTime / delta
        self.deltaTime = delta

    def updateGame(self, playerInput):
        self.player.changeVelocity(playerInput[0])
        self.player.updateAngle(self.deltaTime)
        self.sub.updatePosition(self.deltaTime)
        currentUnitX = math.cos(self.player.angle * math.pi/180)
        currentUnitY = math.sin(self.player.angle * math.pi/180)
        if (self.sub.checkIfHitByBeam(currentUnitX, currentUnitY)):
            self.fitness = self.fitness + 10
        if (playerInput[1] >= 0.5):
            resultTuple = self.sub.checkIfHitByBeam(math.cos(self.player.angle * math.pi/180), math.sin(self.player.angle * math.pi/180))
            if (resultTuple):
                # print("Hit!")
                self.fitness = self.fitness + 100
            else:
                self.fitness = self.fitness - 0.1
        self.stepsSinceLastPing = self.stepsSinceLastPing + 1
        if ((self.currentSteps * self.deltaTime) % 1 == 0):
            self.lastKnownPositionX = self.sub.xPos
            self.lastKnownPositionY = self.sub.yPos
            self.sub.generateNewVelocity()
            self.lastKnownVelocityX = self.sub.xVel
            self.lastKnownVelocityY = self.sub.yVel
            self.stepsSinceLastPing = 0
        self.currentSteps = self.currentSteps + 1
        if (self.currentSteps > self.maxNumSteps):
            return self.fitness
        else:
            return math.inf

    def trainAI(self, genome, config):
        neuralNet = neat.nn.FeedForwardNetwork.create(genome, config)
        currentStatus = self.updateGame((0, 0))
        actions = neuralNet.activate((self.lastKnownPositionX, self.lastKnownPositionY, self.lastKnownVelocityX,
                                      self.lastKnownVelocityY, self.player.angle, self.player.currentAngularVelocity,
                                      self.stepsSinceLastPing * self.deltaTime,
                                      self.memoryNeuronIn[0], self.memoryNeuronIn[1], self.memoryNeuronIn[2],
                                      self.memoryNeuronIn[3], self.memoryNeuronIn[4], self.memoryNeuronIn[5],
                                      self.memoryNeuronIn[6], self.memoryNeuronIn[7], self.memoryNeuronIn[8],
                                      self.memoryNeuronIn[9], self.memoryNeuronIn[10], self.memoryNeuronIn[11],
                                      self.memoryNeuronIn[12], self.memoryNeuronIn[13], self.memoryNeuronIn[14]))
        actionsToTake = [0, 0]
        actionsToTake[0] = (actions[0] % 90) - 45 / self.deltaTime
        actionsToTake[1] = actions[1] % 1
        # print(actions)
        while(currentStatus == math.inf):
            currentStatus = self.updateGame(actions)
            actions = neuralNet.activate((self.lastKnownPositionX, self.lastKnownPositionY, self.lastKnownVelocityX,
                                          self.lastKnownVelocityY, self.player.angle, self.player.currentAngularVelocity,
                                          self.stepsSinceLastPing * self.deltaTime,
                                          self.memoryNeuronIn[0], self.memoryNeuronIn[1], self.memoryNeuronIn[2],
                                          self.memoryNeuronIn[3], self.memoryNeuronIn[4], self.memoryNeuronIn[5],
                                          self.memoryNeuronIn[6], self.memoryNeuronIn[7], self.memoryNeuronIn[8],
                                          self.memoryNeuronIn[9], self.memoryNeuronIn[10], self.memoryNeuronIn[11],
                                          self.memoryNeuronIn[12], self.memoryNeuronIn[13], self.memoryNeuronIn[14]))
            actionsToTake[0] = (actions[0] % 90) - 45 / self.deltaTime
            actionsToTake[1] = actions[1] % 1
            for _ in range(2):
                actions.pop(0)
            for i in range(int(len(actions) / 2)):
                if (actions[i] % 1 >= 0.5):
                    self.memoryNeuronIn[i] = actions[i + 15]
            # print(actions)
            self.updateGame(actionsToTake)
        return currentStatus