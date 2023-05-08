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


from functools import partial
from math import inf
import random
import util
import imp
from numpy import sqrt

from pacman import Directions
from pacman import PacmanRules
from game import Agent
from mcts import MCTS, Node 


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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print(successorGameState)
        # print(newPos)
        # print(newFood.asList())
        # print(newGhostStates)
        # print(newScaredTimes) # Larger the number, the further away the ghost is from pacman
        
        def dist_ghost(ghost):
            # Distance between Pacman and ghost
            distGhost = util.manhattanDistance(newPos, ghost.getPosition())
            # If the number of moves when the ghost is scared greater than the Manhattan Distance away from Pacman 
            if ghost.scaredTimer > distGhost:
                return inf
            # If the number of moves when the ghost is scared less than or equal to 1 Manhattan Distance away from Pacman 
            elif distGhost <= 1:
                return -inf
            # Anything else
            else :
                return 0
        
        # Min the distance between Pacman and ghost
        dist2ghost = min([dist_ghost(i) for i in newGhostStates])
        
        # Mininmize the distance between Pacman and food 
        # dist2food = inf if there are no more foods left 
        dist2food = 1 / min([util.manhattanDistance(newPos, j) for j in newFood.asList()], default=inf)

        # The closer the food is and the further away the ghost is, the higher the score will be
        return successorGameState.getScore() + dist2food + dist2ghost

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
        #print(gameState.isWin())
        #print(gameState.isLose())
        #print(gameState.getNumAgents())
        #print(self.depth)

        # agentIndex=0 means Pacman
        pacmanIndex = 0
        # ghosts are >= 1 for agentIndex
        ghostIndex = list(range(1, gameState.getNumAgents()))

        # Function for terminal states to return 
        def terminal_test(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth
        
        #function MINIMAX-DECISION(state) returns an action
        def minimax_decision(state, depth, index):
            #     return arg max a ∈ ACTIONS(s) MIN-VALUE(RESULT(state, a))
            md = [(min_value(state.generateSuccessor(pacmanIndex, a), depth, index), a) for a in state.getLegalActions(pacmanIndex)]
            return str(max(md)[1])

        # function MAX-VALUE(state) returns a utility value
        def max_value(state, depth):
            #if TERMINAL-TEST(state) then return UTILITY(state)
            if terminal_test(state, depth):
                return self.evaluationFunction(state)
            #v ← −∞
            max_v = -inf

            # for each a in ACTIONS(state) do
            for a in state.getLegalActions(pacmanIndex):
                # v ← MAX(v, MIN-VALUE(RESULT(s, a)))
                max_v = max(max_v, min_value(state.generateSuccessor(pacmanIndex, a), depth, pacmanIndex + 1))
            # return v
            return max_v
        
        # function MIN-VALUE(state) returns a utility value
        def min_value(state, depth, index):
        #     if TERMINAL-TEST(state) then return UTILITY(state)
            if terminal_test(state, depth):
                return self.evaluationFunction(state)
        #     v ← ∞
            min_v = inf
        #     for each a in ACTIONS(state) do
            for a in state.getLegalActions(index):
                # Check if this is the last ghost agent 
                if index == ghostIndex[-1]:
                    # Pacman is nextState
                    min_v = min(min_v, max_value(state.generateSuccessor(index, a), depth + 1))
                else:
            #         v ← MIN(v, MAX-VALUE(RESULT(s, a)))
                    min_v = min(min_v, min_value(state.generateSuccessor(index, a), depth, index + 1))
        #     return v
            return min_v
        
        return minimax_decision(gameState, 0, ghostIndex[0])


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"
        #def minimax_ab(state, depth, index, alpha, beta):
            
        pacmanIndex = 0 

        ghostIndex = list(range(1, gameState.getNumAgents()))

        def terminal_test(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth
        
        # alpha - MAX's best option on path to root s
        # beta - MIN's best option on path to root

        def alpha_beta_search(state):
            current_v = -inf
            alpha = -inf
            beta = inf
            for a in state.getLegalActions(pacmanIndex):
                # Get next state 
                next_state = state.generateSuccessor(pacmanIndex, a)

                ab_v = max_valueAB(next_state, 0, alpha, beta)

                if ab_v > current_v:
                    current_v = ab_v 
                    action = a

                alpha = max(alpha, current_v)

            return action

        def max_valueAB(state, depth, alpha, beta):
            if terminal_test(state, depth):
                return self.evaluationFunction(state)
            #v ← −∞
            max_v = -inf

            # for each a in ACTIONS(state) do
            for a in state.getLegalActions(pacmanIndex):
                # v ← MAX(v, MIN-VALUE(RESULT(s, a), alpha, beta))
                max_v = max(max_v, min_valueAB(state.generateSuccessor(pacmanIndex, a), depth, pacmanIndex + 1, alpha, beta))
            
                if max_v >= beta:
                    return max_v
                alpha = max(alpha, max_v)
            return max_v
        
        def min_valueAB(state, depth, index, alpha, beta):
            if terminal_test(state, depth):
                return self.evaluationFunction(state)
            min_v = inf 

            for a in state.getLegalActions(index):
                # Check if this is the last ghost agent 
                if index == ghostIndex[-1]:  
                    min_v = min(min_v, max_valueAB(state.generateSuccessor(index, a), depth + 1, alpha, beta))

                else:
            #         v ← MIN(v, MAX-VALUE(RESULT(s, a), alpha, beta))
                    min_v = min(min_v, min_valueAB(state.generateSuccessor(index, a), depth, index + 1, alpha, beta))

                if min_v <= alpha:
                    return min_v
                beta = min(beta, min_v)
            return min_v

        return alpha_beta_search(gameState)
        
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
        #Define constants
        pacmanIndex = 0

        ghostIndex = list(range(1, gameState.getNumAgents()))
        
        def terminal_test(state, depth):
            return state.isWin() or state.isLose() or depth == self.depth

        def expectimax(state, depth, index):
            if terminal_test(state, depth):
                return self.evaluationFunction(state)

            # If Pacman's turn, choose max value action
            if index == pacmanIndex:
                return max(expectimax(state.generateSuccessor(index, a), depth, ghostIndex[0]) for a in state.getLegalActions(index))
                
            # If Ghosts' turn, choose expected value action based on random move
            else:
                actions = state.getLegalActions(index)
                num_actions = len(actions)
                p = 1 / num_actions
                return sum(p * expectimax(state.generateSuccessor(index, a), depth, index + 1) for a in actions)

        # Choose best action based on expectimax values
        actions = gameState.getLegalActions(pacmanIndex)
        values = [expectimax(gameState.generateSuccessor(pacmanIndex, a), 0, ghostIndex[0]) for a in actions]
        bestIndex = values.index(max(values))

        return actions[bestIndex]
        #util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """

    "*** YOUR CODE HERE ***"
    #Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    
    def dist_ghost(ghost):
            # Distance between Pacman and ghost
            distGhost = util.manhattanDistance(newPos, ghost.getPosition())
            # If the number of moves when the ghost is scared greater than the Manhattan Distance away from Pacman 
            if ghost.scaredTimer > distGhost:
                return inf
            # If the number of moves when the ghost is scared less than or equal to 1 Manhattan Distance away from Pacman 
            elif distGhost <= 1:
                return -inf
            # Anything else
            else :
                return 0
        
    # Minimize the distance between Pacman and ghost
    dist2ghost = 100*min([dist_ghost(i) for i in newGhostStates])
    
    # Mininmize the distance between Pacman and food 
    # dist2food = inf if there are no more foods left 
    dist2food = 100 / min([util.manhattanDistance(newPos, j) for j in newFood.asList()], default=inf)    

    # The closer the food is and the further away the ghost is, the higher the score will be
    return currentGameState.getScore() + dist2food + dist2ghost 
    
    #util.raiseNotDefined()

    
# Abbreviation
better = betterEvaluationFunction

class MCTSAgent(MultiAgentSearchAgent):
    def __init__(self, max_iterations=50, exploration_parameter=sqrt(2)):
        self.max_iterations = max_iterations
        self.mcts = MCTS(exploration_parameter=exploration_parameter)
        self.PacmanRules = PacmanRules
    
    # def getAction(self, gameState):
    #     root = Node(state=gameState)
    #     for i in range(self.max_iterations):
    #         state = gameState.deepCopy()
    #         selected_node = self.mcts.search(state, self.max_iterations)
    #         print("Selected Node : ", selected_node)
    #         gameState = state.generateSuccessor(0, selected_node.state.get_action())
    #     return selected_node.state.get_action()

    def getAction(self, gameState):
        root = Node(state=gameState)
        for i in range(self.max_iterations):
            state = gameState.deepCopy()
            selected_node = self.mcts.search(state, self.max_iterations)
            print("Selected Node : ", selected_node)
            action = selected_node.state.get_action()
            legal_actions = self.PacmanRules.getLegalActions(state)
            if action not in legal_actions:
                continue
            gameState = state.generateSuccessor(0, action)
        return selected_node.state.get_action()

