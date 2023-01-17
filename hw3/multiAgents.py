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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        return childGameState.getScore()

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

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (par1-1)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        def MinMaxLevel(gameState, depth, agentIndex):
            if agentIndex == 0:
                if gameState.isWin() or gameState.isLose() or depth + 1 == self.depth:
                    return self.evaluationFunction(gameState)
                val = []
                actions = gameState.getLegalActions(0)            
                val = [MinMaxLevel(gameState.getNextState(0,action), depth + 1, 1) for action in actions]                
                return max(val)
            else:
                if gameState.isWin() or gameState.isLose():
                    return self.evaluationFunction(gameState)
                val = []
                actions = gameState.getLegalActions(agentIndex)
                if agentIndex < gameState.getNumAgents() - 1:
                    val = [MinMaxLevel(gameState.getNextState(agentIndex, action), depth, agentIndex+1) for action in actions]
                else:
                    val = [MinMaxLevel(gameState.getNextState(agentIndex,action), depth, 0) for action in actions]
                return min(val)
                
        score = []
        actions = gameState.getLegalActions(0)
        score = [MinMaxLevel(gameState.getNextState(0, action), 0, 1) for action in actions]
        return actions[score.index(max(score))]
                
        # End your code

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (part1-2)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Begin your code
        def ExpectMaxLevel(gameState, depth, agentIndex):
            if agentIndex == 0:
                if gameState.isWin() or gameState.isLose() or depth + 1 == self.depth:
                    return self.evaluationFunction(gameState)

                val = []
                actions = gameState.getLegalActions(0)            
                val = [ExpectMaxLevel(gameState.getNextState(0, action), depth + 1, 1) for action in actions]
                return max(val)
            else:
                if gameState.isWin() or gameState.isLose():
                    return self.evaluationFunction(gameState)
                actions = gameState.getLegalActions(agentIndex)
                if len(actions) == 0:
                    return 0
                ExpectiVal = []
                if agentIndex < gameState.getNumAgents() - 1:
                    ExpectiVal = [ExpectMaxLevel(gameState.getNextState(agentIndex, action), depth, agentIndex + 1) for action in actions]
                else:
                    ExpectiVal = [ExpectMaxLevel(gameState.getNextState(agentIndex, action), depth, 0) for action in actions]
                return float(sum(ExpectiVal) / len(actions))
                          
        actions = gameState.getLegalActions(0)
        score = []
        score = [ExpectMaxLevel(gameState.getNextState(0, action), 0, 1) for action in actions]                
        return actions[score.index(max(score))]
    
            
        # End your code

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (part1-3).

    DESCRIPTION: 
        1. calculate the manhattan diastance between pacman and foods
        3. the closer the food not eaten is, the greater the score is
        4. general score = currentScore + eaten food + food not eaten
        5. calculate the manhattan diastance between pacman and ghosts
        6. if normal ghost, the closer the ghost is, the lower the score is
        7. Otherwise, the closer the ghost is, the greater the score is. and the scaredTime will also considered
        
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    food2pacPos =[manhattanDistance(pacPos, food) for food in foodPos if manhattanDistance(pacPos, food) > 0]
    foodEaten = len(currentGameState.getFood().asList(False))
    
    if len(food2pacPos) != 0:
        score = currentGameState.getScore() + (1.0 /min(food2pacPos)) + foodEaten
    else:
        score = currentGameState.getScore() + foodEaten
    
    ghosts = currentGameState.getGhostStates()
    ghost2pacPos =[manhattanDistance(pacPos, ghost.getPosition()) for ghost in ghosts]
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

    
    for scaredTime, ghost in zip(scaredTimes, ghost2pacPos):
        if scaredTime == 0 :
            if ghost == 0:
                return 0
            else:
                score += ghost
        else:
            score += scaredTime + (-1 * ghost)

    return score
    # End your code

# Abbreviation
"""
If you complete this part, please replace scoreEvaluationFunction with betterEvaluationFunction ! !
"""
better = betterEvaluationFunction # betterEvaluationFunction or scoreEvaluationFunction
