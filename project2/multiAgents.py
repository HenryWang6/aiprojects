# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util, sys

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFood_dis = [manhattanDistance(newPos, xy) for xy in newFood.asList()]
        newGhosts_pos = [newGhostStates[i].getPosition() for i in range(len(newGhostStates))]
        # ghost distance seperate store
        newGhost_dis = [manhattanDistance(newPos, ghostxy) for ghostxy in newGhosts_pos]
        # encourage for eating food
        encourage = 0
        # penalty for avoiding ghost
        penalty = [i * 0 for i in range(len(newGhost_dis))]

        # dynamic_ghost_dis = manhattanDistance(newPos, newGhostPos) - manhattanDistance(currentPos, currentGhostPos)
        import sys
        # eliminate penalty if ghost doesn't exist.
        for i in range(len(newScaredTimes)):
            if newScaredTimes[i] > 0:
                newGhost_dis[i] = sys.maxint
        # calculate penalty according to distance to ghosts
        for i in range(len(newGhost_dis)):
            if newGhost_dis[i] > 0:
                penalty[i] = - 25.0 / ((newGhost_dis[i] / 2.0) ** 2)

            # if dynamic_ghost_dis != 0:
            # score += 20/dynamic_ghost_dis
        # calculate encourage according to min food distance.
        if len(newFood_dis):
            encourage = 10.0 / min(newFood_dis)

        penalty = sum(penalty)

        return successorGameState.getScore() + encourage + penalty


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


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

    Directions.STOP:
      The stop direction, which is always legal

    gameState.generateSuccessor(agentIndex, action):
      Returns the successor game state after an agent takes an action

    gameState.getNumAgents():
      Returns the total number of agents in the game
    """
        '*** YOUR CODE HERE ***'
        # pacman's index
        agentIndex = 0
        # print self.depth
        # cur_depth = 1
        # all possible actions for pacman
        actions = gameState.getLegalActions(agentIndex)
        # It is root action
        final_action = actions[0]

        import sys
        pacman_value = -sys.maxint

        # Loop the value for root.
        for action in actions:
            # remove action "stop" from actions
            if action == "Stop":
                continue

            successorGameState = gameState.generateSuccessor(0, action)

            # calculate value of ghosts, start from depth 0 and ghost index 1
            min_value = self.min_value(successorGameState, 1, 0)

            # get the maximum value and corresponding action
            if min_value > pacman_value:
                pacman_value = min_value
                final_action = action

        return final_action

    # textbook p166

    # agentIndex for function max_value must be 0. we only have one pacman
    def max_value(self, gameState, depth):
        import sys
        # terminal test: win or lose or depth limit
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        v = -sys.maxint
        # get the pacman's possible action
        actions = gameState.getLegalActions(0)
        for action in actions:
            # remove action "stop" from actions
            if action == "Stop":
                continue
            successorGameState = gameState.generateSuccessor(0, action)
            min_value = self.min_value(successorGameState, 1, depth)
            # get the maximum value and corresponding action
            v = max(v, min_value)
        return v

    def min_value(self, gameState, agentIndex, depth):
        import sys
        # terminal test: win or lose or depth limit
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        v = sys.maxint
        # get the ghost's possible action
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex + 1 == gameState.getNumAgents():
                # the next agent is pacman
                min_value = self.max_value(successorGameState, depth + 1)
            else:
                # the next agent is next ghost
                min_value = self.min_value(successorGameState, agentIndex + 1, depth)
            v = min(v, min_value)
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
  """

    def getAction(self, gameState):
        """
            Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # pacman's index
        agentIndex = 0
        # print self.depth
        # cur_depth = 1
        # all possible actions for pacman
        actions = gameState.getLegalActions(agentIndex)
        final_action = actions[0]

        import sys
        pacman_value = -sys.maxint

        for action in actions:
            # remove action "stop" from actions
            if action == "Stop":
                continue

            successorGameState = gameState.generateSuccessor(0, action)

            # calculate value of ghosts, start from depth 0 and ghost index 1
            min_value = self.min_value(successorGameState, 1, 0, -sys.maxint, sys.maxint)

            # get the maximum value and corresponding action
            if min_value > pacman_value:
                pacman_value = min_value
                final_action = action
        return final_action

    # textbook p166

    # agentIndex for function max_value must be 0. we only have one pacman
    def max_value(self, gameState, depth, alpha, beta):
        import sys
        # terminal test: win or lose or depth limit
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        v = -sys.maxint
        # get the pacman's possible action
        actions = gameState.getLegalActions(0)
        for action in actions:
            # remove action "stop" from actions
            if action == "Stop":
                continue
            successorGameState = gameState.generateSuccessor(0, action)
            min_value = self.min_value(successorGameState, 1, depth, alpha, beta)
            # get the maximum value and corresponding action
            v = max(v, min_value)
            # see pdf
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        import sys
        # terminal test: win or lose or depth limit
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        v = sys.maxint
        # get the ghost's possible action
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex + 1 == gameState.getNumAgents():
                # the next agent is pacman
                min_value = self.max_value(successorGameState, depth + 1, alpha, beta)
            else:
                # the next agent is next ghost
                min_value = self.min_value(successorGameState, agentIndex + 1, depth, alpha, beta)
            v = min(v, min_value)
            # see pdf
            if v <= alpha:
                return v
            beta = min(v, beta)
        return v


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
        # pacman's index
        agentIndex = 0
        # print self.depth
        # cur_depth = 1
        # all possible actions for pacman
        actions = gameState.getLegalActions(agentIndex)
        final_action = actions[0]

        import sys
        pacman_value = -sys.maxint

        for action in actions:
            # remove action "stop" from actions
            if action == "Stop":
                continue

            successorGameState = gameState.generateSuccessor(0, action)

            # calculate value of ghosts, start from depth 0 and ghost index 1
            min_value = self.exp_value(successorGameState, 1, 0)

            # get the maximum value and corresponding action
            if min_value > pacman_value:
                pacman_value = min_value
                final_action = action
        return final_action

    # textbook p166

    # agentIndex for function max_value must be 0. we only have one pacman
    def max_value(self, gameState, depth):
        import sys
        # terminal test: win or lose or depth limit
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        v = -sys.maxint
        # get the pacman's possible action
        actions = gameState.getLegalActions(0)
        for action in actions:
            # remove action "stop" from actions
            if action == "Stop":
                continue
            successorGameState = gameState.generateSuccessor(0, action)
            min_value = self.exp_value(successorGameState, 1, depth)
            # get the maximum value and corresponding action
            v = max(v, min_value)
        return v

    def exp_value(self, gameState, agentIndex, depth):
        # terminal test: win or lose or depth limit
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        v = 0
        # get the ghost's possible action
        actions = gameState.getLegalActions(agentIndex)
        for action in actions:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            if agentIndex + 1 == gameState.getNumAgents():
                # the next agent is pacman
                min_value = self.max_value(successorGameState, depth + 1)
            else:
                # the next agent is next ghost
                min_value = self.exp_value(successorGameState, agentIndex + 1, depth)
            v += min_value * (1.0 / float(len(actions)))
        return v


'''def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # TODO: optimize the evaluation function
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    newFood_dis = [manhattanDistance(newPos, xy) for xy in newFood.asList()]
    newGhosts_pos = [newGhostStates[i].getPosition() for i in range(len(newGhostStates))]
    # ghost distance seperate store
    newGhost_dis = [manhattanDistance(newPos, ghostxy) for ghostxy in newGhosts_pos]
    # encourage for eating food
    encourage = 0
    # penalty for avoiding ghost
    penalty = [i * 0 for i in range(len(newGhost_dis))]

    import sys
    # eliminate penalty if ghost doesn't exist.
    for i in range(len(newScaredTimes)):
        if newScaredTimes[i] > 0:
            newGhost_dis[i] = sys.maxint
    # calculate penalty according to distance to ghosts
    for i in range(len(newGhost_dis)):
        if newGhost_dis[i] > 0:
            penalty[i] = - 25.0 / ((newGhost_dis[i] / 2.0) ** 2)

        # if dynamic_ghost_dis != 0:
        # score += 20/dynamic_ghost_dis
    # calculate encourage according to min food distance.
    if len(newFood_dis):
        encourage = 10.0 / min(newFood_dis)
    return currentGameState.getScore() + encourage + sum(penalty)'''


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # TODO: optimize the evaluation function
    "*** YOUR CODE HERE ***"
    import sys, searchAgents

    pacmanPos = currentGameState.getPacmanPosition()
    foodPos = currentGameState.getFood().asList()
    foodDis = [manhattanDistance(pacmanPos, xy) for xy in foodPos]
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostsPos_float = [ghostStates[i].getPosition() for i in range(len(ghostStates))]
    ghostsPos = []
    for each in ghostsPos_float:
        xy = []
        for num in each:
            xy.append(int(num))
        ghostsPos.append(xy)
    ghostsMazeDis = [searchAgents.mazeDistance(pacmanPos, ghostxy, currentGameState) for ghostxy in ghostsPos]
    # ghostsDis = [manhattanDistance(pacmanPos, ghostxy) for ghostxy in ghostsPos]
    capsulesPos = currentGameState.getCapsules()
    capsulesMazeDis = [searchAgents.mazeDistance(pacmanPos, capsulePos, currentGameState) for capsulePos in capsulesPos]

    # capsulesDis = [manhattanDistance(pacmanPos, xy) for xy in capsulesPos]
    # minCapsuleDis = min(capsulesDis)

    encourage = 0
    ghostPenalty = [i * 0 for i in range(len(ghostStates))]
    capsulesPenalty = -50.0

    # when Ghosts are scared
    for i in range(len(scaredTimes)):
        if scaredTimes[i] > 0:
            ghostsMazeDis[i] = sys.maxint

    # penalty from ghost
    for i in range(len(ghostStates)):
        if ghostsMazeDis[i] > 0 and ghostsMazeDis[i] <= 1:
            ghostPenalty[i] = sys.maxint
        elif ghostsMazeDis[i] > 1:
            ghostPenalty[i] = - 25.0 / ((ghostsMazeDis[i] / 2.0) ** 5)

    # encourage from food
    if len(foodDis):
        minFoodDis = min(foodDis)
        # meanFoodDis = float(sum(foodDis))/len(foodDis)
        encourage = 10.0 / minFoodDis
        if foodDis.count(minFoodDis) > 1 and minFoodDis >= 2:
            indexes = [i for i, v in enumerate(foodDis) if v == minFoodDis]
            minDisfoodsPos = [foodPos[index] for index in indexes]
            minFoodMazeDis = min(
                [searchAgents.mazeDistance(pacmanPos, minDisfoodPos, currentGameState) for minDisfoodPos in
                 minDisfoodsPos])
            encourage = (5.0 / minFoodDis + 15.0 / minFoodMazeDis) / 2.0

        # minFoodMazeDis = min([searchAgents.mazeDistance(pacmanPos, foodxy, currentGameState) for foodxy in foodPos])
        # meanFoodMazeDis = float(sum([searchAgents.mazeDistance(pacmanPos, foodxy, currentGameState) for foodxy in foodPos])) / len(foodPos)
        # encourage = 10.0 / minFoodDis + 5.0 / minFoodMazeDis

    # penalty from not eating capsule
    if len(capsulesMazeDis):
        minCapsuleMazeDis = min(capsulesMazeDis)
        if minCapsuleMazeDis <= 2 * ghostsMazeDis[0] and minCapsuleMazeDis <= 2 * ghostsMazeDis[
            1] and minCapsuleMazeDis <= 3 * minFoodDis:
            capsulesPenalty = - minCapsuleMazeDis**2 - len(capsulesPos)*40.0

    return currentGameState.getScore() + encourage + sum(ghostPenalty) + capsulesPenalty


# Abbreviation
better = betterEvaluationFunction

'''def directDistance(xy1, xy2):
    # return the direct distance
    import math
    return math.sqrt((xy1[0]-xy2[0])**2 + (xy2[1]-xy2[1])**2)

def mazeDistancetoDenseFood(currentPos, foodPos, currentGameState):
    x = 0
    y = 0

    for food in foodPos:
        #print food
        x += food[0]
        y += food[0]
    xy = (x//len(foodPos), y//len(foodPos))

    import searchAgents
    return searchAgents.mazeDistance(currentPos, xy, currentGameState)'''


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest
  """

    def getAction(self, gameState):
        """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
