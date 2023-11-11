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
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # If moving there causes you to lose
        if successorGameState.isLose():
            return float('-inf')
        # If moving there causes you to win
        if successorGameState.isWin():
            return float('inf')
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        currentScarredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

        newFood = newFood.asList()
        closest_food = min([manhattanDistance(newPos, f) for f in newFood]) if len(newFood) != 0 else 0
        closest_ghost = min([manhattanDistance(newPos, g.getPosition()) for g in newGhostStates])
        timer_total = sum(newScaredTimes)
        current_timer_total = sum(currentScarredTimes)

        # Base game score
        score = successorGameState.getScore()
        # Food incentive (the farther the food, the more the score is lowered)
        score -= closest_food * 2
        # Ghost incentive (the farther the ghost, the bigger the score)
        score += closest_ghost * 2
        # Incentive for eating food (useful actions)
        score += (successorGameState.getNumFood() < currentGameState.getNumFood()) * 10
        # Incentive for eating power pellets (useful actions)
        if newPos in currentGameState.getCapsules():
            score += 10000
        if len(currentGameState.getCapsules()) > len(successorGameState.getCapsules()):
            score += 10000
            score -= current_timer_total * 100
        # Incentive if there are ghost timers
        score += timer_total * 5
        # Penalty for stopping
        if action == Directions.STOP:
            score -= 10000
        return score


def scoreEvaluationFunction(currentGameState: GameState):
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


# Tree node class
class Node:
    def __init__(self, state, action, parent, depth, agent):
        self.state = state
        # self._position = state.getPacmanPosition() if agent == 0 else state.getGhostPosition(agent)
        self.action = action
        self.parent = parent
        self.depth = depth
        self.agent = agent
        self.children = []
        self.eval = None

    def addChild(self, child):
        self.children.append(child)

    def getChildren(self):
        return self.children

    def getState(self):
        return self.state

    def getAction(self):
        return self.action

    def getParent(self):
        return self.parent

    def getDepth(self):
        return self.depth

    def getAgent(self):
        return self.agent

    def isLeaf(self):
        return len(self.children) == 0

    def isRoot(self):
        return self.parent is None

    def __str__(self):
        return (f"Action: {self.action}, Depth: {self.depth}, Eval: {self.eval}, "
                f"Agent: {'Pacman' if self.agent == 0 else f'Ghost{self.agent}'}")

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)


def printTree(tree):
    printTreeHelper(tree, 0)


def printTreeHelper(node: Node, depth):
    print("Depth: " + str(depth) + " Node: " + str(node))
    for child in node.getChildren():
        printTreeHelper(child, depth + 1)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def createMinimaxTreeHelper(self, node: Node, depth, agent, gameState: GameState):
        if depth == self.depth or node.getState().isWin() or node.getState().isLose():
            return
        actions = node.getState().getLegalActions(agent)
        for action in actions:
            child = Node(node.getState().generateSuccessor(agent, action), action, node, node.getDepth() + 1,
                         (agent + 1) % gameState.getNumAgents())
            node.addChild(child)
            self.createMinimaxTreeHelper(child, depth + 1 if (agent + 1) % gameState.getNumAgents() == 0 else depth,
                                         (agent + 1) % gameState.getNumAgents(), gameState)

    def createMinimaxTree(self, gameState: GameState, depth, agent):
        root = Node(gameState, None, None, 0, agent)
        self.createMinimaxTreeHelper(root, depth, agent, gameState)
        return root

    def minVal(self, node: Node):
        if node.isLeaf():
            node.eval = self.evaluationFunction(node.getState())
            return node.eval
        v = float('inf')
        for child in node.getChildren():
            if child.getAgent() == 0:
                v = min(v, self.maxVal(child))
            else:
                v = min(v, self.minVal(child))
        node.eval = v
        return v

    def maxVal(self, node: Node):
        if node.isLeaf():
            node.eval = self.evaluationFunction(node.getState())
            return node.eval
        v = float('-inf')
        for child in node.getChildren():
            if child.getAgent() == 0:
                v = max(v, self.maxVal(child))
            else:
                v = max(v, self.minVal(child))
        node.eval = v
        return v

    def getAction(self, gameState: GameState):
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
        minimax_tree = self.createMinimaxTree(gameState, 0, 0)

        max_val = self.maxVal(minimax_tree)
        # printTree(minimax_tree)

        return minimax_tree.getChildren()[minimax_tree.getChildren().index(
            next(filter(lambda x: x.eval == max_val, minimax_tree.getChildren())))].getAction()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def createAlphaBetaTreeHelper(self, node: Node, depth, agent, gameState: GameState, alpha, beta):
        if depth == self.depth or node.getState().isWin() or node.getState().isLose():
            return
        actions = node.getState().getLegalActions(agent)
        for action in actions:
            child = Node(node.getState().generateSuccessor(agent, action), action, node, node.getDepth() + 1,
                         (agent + 1) % gameState.getNumAgents())
            node.addChild(child)
            self.createAlphaBetaTreeHelper(child, depth + 1 if (agent + 1) % gameState.getNumAgents() == 0 else depth,
                                           (agent + 1) % gameState.getNumAgents(), gameState, alpha, beta)
            # EVALUATE
            if agent == 0:
                self.maxVal(child.parent, alpha, beta)
                if child.eval > beta:
                    return
                alpha = max(alpha, child.eval)
            else:
                self.minVal(child.parent, alpha, beta)
                if child.eval < alpha:
                    return
                beta = min(beta, child.eval)

    def createAlphaBetaTree(self, gameState: GameState, depth, agent):
        root = Node(gameState, None, None, 0, agent)
        self.createAlphaBetaTreeHelper(root, depth, agent, gameState, float('-inf'), float('inf'))
        return root

    def minVal(self, node: Node, alpha, beta):
        if node.isLeaf():
            node.eval = self.evaluationFunction(node.getState())
            return node.eval
        v = float('inf')
        for child in node.getChildren():
            if child.getAgent() == 0:
                v = min(v, self.maxVal(child, alpha, beta))
            else:
                v = min(v, self.minVal(child, alpha, beta))
            if v < alpha:
                node.eval = v
                return v
            beta = min(beta, v)
        node.eval = v
        return v

    def maxVal(self, node: Node, alpha, beta):
        if node.isLeaf():
            node.eval = self.evaluationFunction(node.getState())
            return node.eval
        v = float('-inf')
        for child in node.getChildren():
            if child.getAgent() == 0:
                v = max(v, self.maxVal(child, alpha, beta))
            else:
                v = max(v, self.minVal(child, alpha, beta))
            if v > beta:
                node.eval = v
                return v
            alpha = max(alpha, v)
        node.eval = v
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        tree = self.createAlphaBetaTree(gameState, 0, 0)
        max_val = self.maxVal(tree, float('-inf'), float('inf'))
        return tree.getChildren()[tree.getChildren().index(
            next(filter(lambda x: x.eval == max_val, tree.getChildren())))].getAction()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def createExpectimaxTreeHelper(self, node: Node, depth, agent, gameState: GameState):
        if depth == self.depth or node.getState().isWin() or node.getState().isLose():
            return
        actions = node.getState().getLegalActions(agent)
        for action in actions:
            child = Node(node.getState().generateSuccessor(agent, action), action, node, node.getDepth() + 1,
                         (agent + 1) % gameState.getNumAgents())
            node.addChild(child)
            self.createExpectimaxTreeHelper(child, depth + 1 if (agent + 1) % gameState.getNumAgents() == 0 else depth,
                                            (agent + 1) % gameState.getNumAgents(), gameState)

    def createExpectimaxTree(self, gameState: GameState, depth, agent):
        root = Node(gameState, None, None, 0, agent)
        self.createExpectimaxTreeHelper(root, depth, agent, gameState)
        return root

    def expVal(self, node: Node):
        if node.isLeaf():
            node.eval = self.evaluationFunction(node.getState())
            return node.eval
        v = 0
        for child in node.getChildren():
            if child.getAgent() == 0:
                v += self.maxVal(child)
            else:
                v += self.expVal(child)
        node.eval = v / len(node.getChildren())
        return node.eval

    def maxVal(self, node: Node):
        if node.isLeaf():
            node.eval = self.evaluationFunction(node.getState())
            return node.eval
        v = float('-inf')
        for child in node.getChildren():
            v = max(v, self.expVal(child))
        node.eval = v
        return v

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        tree = self.createExpectimaxTree(gameState, 0, 0)
        max_val = self.maxVal(tree)
        return tree.getChildren()[tree.getChildren().index(
            next(filter(lambda x: x.eval == max_val, tree.getChildren())))].getAction()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    if currentGameState.isLose():
        return float("-inf")
    if currentGameState.isWin():
        return float("inf")

    score = currentGameState.getScore()

    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    total_timer = sum([ghostState.scaredTimer for ghostState in ghosts])
    closest_food = min([manhattanDistance(currentGameState.getPacmanPosition(), f) for f in food]) if len(
        food) != 0 else 0
    closest_ghost_distance = min([manhattanDistance(currentGameState.getPacmanPosition(),
                                                    g.getPosition()) for g in ghosts])
    closest_ghost = [g for g in ghosts if manhattanDistance(currentGameState.getPacmanPosition(),
                                                            g.getPosition()) == closest_ghost_distance][0]

    closest_capsule = min([manhattanDistance(currentGameState.getPacmanPosition(), c) for c in capsules]) if len(
        capsules) != 0 else 0

    # Quantitative values
    score -= 1 / len(food) if len(food) != 0 else 0
    score -= 1 / len(ghosts) if len(ghosts) != 0 else 0
    score -= 1 / len(capsules) if len(capsules) != 0 else 0

    # Distance values
    score -= closest_food
    score -= closest_capsule / 10
    score += closest_ghost_distance * 0.6

    # Incentives for scaring ghosts
    score += total_timer * 5
    score += closest_ghost.scaredTimer > 0

    return score


# Abbreviation
better = betterEvaluationFunction
