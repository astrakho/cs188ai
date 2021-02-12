# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood().asList()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPos = []
        if successorGameState.isWin():
            return 100
        if successorGameState.isLose():
            return -100
        if action == "Stop":
            return -1

        i = 0
        #for each ghost on the board, get the next position of that ghost
        for s in newGhostStates:
            i += 1
            if (s.getDirection() in currentGameState.getLegalActions(i)):
                newGhostPos.append(successorGameState.generateSuccessor(i, s.getDirection()).getGhostStates()[0].getPosition())
            else:
                newGhostPos.append(s.getPosition())

        # sharing space with a ghost is very bad
        if newPos in newGhostPos:
            return -100

        if currentGameState.hasFood(newPos[0], newPos[1]):
            return 10

        distance_to_food = 10000
        for f in newFood:
            distance_to_food = min(distance_to_food, manhattanDistance(f, newPos))
        distance_to_food = (1 / distance_to_food)
        "*** YOUR CODE HERE ***"
        return distance_to_food

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

def minNode(sa, gameState, depth, agentindex):
    minvalue = 1000
    num_agents = gameState.getNumAgents()
    if gameState.isWin() or gameState.isLose():
        return sa.evaluationFunction(gameState)

    actions = gameState.getLegalActions(agentindex)
    for action in actions:
        newvalue = 0
        successor = gameState.generateSuccessor(agentindex, action)
        if agentindex == num_agents - 1:
            newvalue = maxNode(sa, successor, depth - 1)
        else:
            newvalue = minNode(sa, successor, depth, agentindex + 1)

        minvalue = min(newvalue, minvalue)

    return minvalue

def maxNode(sa, gameState, depth):
    if gameState.isWin() or gameState.isLose() or depth == 0:
        return sa.evaluationFunction(gameState)
    maxvalue = -10000
    pacmanLegalActions = gameState.getLegalActions(0)
    for action in pacmanLegalActions:
        successor = gameState.generateSuccessor(0, action)
        newvalue = minNode(sa, successor, depth, 1)
        maxvalue = max(maxvalue, newvalue)

    return maxvalue

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        eval_func = self.evaluationFunction
        isMinAgent = lambda agent_index: agent_index > 1

        pacmanLegalActions = gameState.getLegalActions(0)
        newvalue = 0
        best_action = 0
        maxvalue = -10000
        for action in pacmanLegalActions:
            successor = gameState.generateSuccessor(0, action)
            newvalue = minNode(self, successor, depth, 1)
            if newvalue >= maxvalue:
                best_action = action
                maxvalue = newvalue
        return best_action

def minNodeAB(sa, gameState, depth, agentindex, a, b):
    #print("min ", gameState.state)
    minvalue = 1000
    num_agents = gameState.getNumAgents()
    if gameState.isWin() or gameState.isLose():
        return sa.evaluationFunction(gameState)

    actions = gameState.getLegalActions(agentindex)
    for action in actions:
        newvalue = 0
        successor = gameState.generateSuccessor(agentindex, action)
        if agentindex == num_agents - 1:
            newvalue, dum = maxNodeAB(sa, successor, depth - 1, a, b)
        else:
            newvalue = minNodeAB(sa, successor, depth, agentindex + 1, a, b)

        minvalue = min(newvalue, minvalue)
        if minvalue < a:
            return minvalue
        b = min(b, minvalue)

    return minvalue

def maxNodeAB(sa, gameState, depth, a, b):
    #print("max ", gameState.state)
    if gameState.isWin() or gameState.isLose() or depth == 0:
        #print("eval ", gameState.state)
        return sa.evaluationFunction(gameState), None
    maxvalue = -10000
    pacmanLegalActions = gameState.getLegalActions(0)
    bestAction = None
    for action in pacmanLegalActions:
        successor = gameState.generateSuccessor(0, action)
        newvalue = minNodeAB(sa, successor, depth, 1, a, b)
        if newvalue > maxvalue:
            maxvalue = newvalue
            bestAction = action

        if maxvalue > b:
            return maxvalue, bestAction
        a = max(a, maxvalue)

    return maxvalue, bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth
        a = -100000
        b = 100000
        return maxNodeAB(self, gameState, depth, a, b)[1]

def minNodeExp(sa, gameState, depth, agentindex):
    #print("min ", gameState.state)
    minvalue = 1000
    num_agents = gameState.getNumAgents()
    if gameState.isWin() or gameState.isLose():
        return sa.evaluationFunction(gameState)

    actions = gameState.getLegalActions(agentindex)
    values = []
    for action in actions:
        newvalue = 0
        successor = gameState.generateSuccessor(agentindex, action)
        if agentindex == num_agents - 1:
            newvalue, dum = maxNodeExp(sa, successor, depth - 1)
        else:
            newvalue = minNodeExp(sa, successor, depth, agentindex + 1)

        values.append(newvalue)

    return sum(values) / len(values)

def maxNodeExp(sa, gameState, depth):
    #print("max ", gameState.state)
    if gameState.isWin() or gameState.isLose() or depth == 0:
        #print("eval ", gameState.state)
        return sa.evaluationFunction(gameState), None
    maxvalue = -10000
    pacmanLegalActions = gameState.getLegalActions(0)
    bestAction = None
    for action in pacmanLegalActions:
        successor = gameState.generateSuccessor(0, action)
        newvalue = minNodeExp(sa, successor, depth, 1)
        if newvalue > maxvalue:
            maxvalue = newvalue
            bestAction = action

    return maxvalue, bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return maxNodeExp(self, gameState, self.depth)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return 10000
    elif currentGameState.isLose():
        return -10000

    # distance to ghost
    # distance to food
    # distance to pellet
    # whether food pellet is active
    pacmanPos = currentGameState.getPacmanPosition()

    foodGrid = currentGameState.getFood()
    food = foodGrid.asList()
    minDistToFood = 10000
    for f in food:
        minDistToFood = min(minDistToFood, manhattanDistance(pacmanPos, f))
    foodValue = -currentGameState.getNumFood() + (len(foodGrid.asList(True)) + len(foodGrid.asList(False)) - minDistToFood)

    ghostValue = 0

    newGhostStates = currentGameState.getGhostStates()
    for gs in newGhostStates:
        ghostPos = gs.getPosition()
        ghostTimer = gs.scaredTimer
        ghostValue += manhattanDistance(pacmanPos, ghostPos) * (1 - bool(ghostTimer))

    #if ghost is not scared compute a negative value for ghost distance

    return foodValue * 10 + ghostValue + currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction
