import random
import math

class Node:
    def __init__(self, state, parent = None):
        self.state = state
        self.parent = parent
        self.children = []
        self.num_visits = 0
        self.total_reward = 0

    def expand(self):
        actions = self.state.getLegalActions()
        for action in actions:
            child_state = self.state.generateSuccessor(0, action)
            child_node = Node(child_state, self)
            self.children.append(child_node)

    def select_child(self, exploration_parameter):
        # use the UCB formula to select the child with the highest UCB value
        best_child = None
        best_ucb_value = -float('inf')
        for child in self.children:
            if child.num_visits == 0:
                return child
            else:
                ucb_value = (child.total_reward / child.num_visits) + exploration_parameter * math.sqrt(math.log(self.num_visits) / child.num_visits)
                if ucb_value > best_ucb_value:
                    best_ucb_value = ucb_value
                    best_child = child
        return best_child

    def simulate(self):
        # simulate the game from the current state to the end
        current_state = self.state
        while not current_state.is_terminal():
            current_state = current_state.get_random_next_state()
        # return the reward
        return current_state.get_reward()

    def update(self, reward):
        # update the statistics of the current node and its ancestors
        self.num_visits += 1
        self.total_reward += reward
        if self.parent is not None:
            self.parent.update(reward)

class MCTS:
    def __init__(self, exploration_parameter):
        self.exploration_parameter = exploration_parameter
        
    def search(self, initial_state, num_iterations):
        # create the root node
        root_node = Node(initial_state)
        
        # run the search for the specified number of iterations
        for i in range(num_iterations):
            # select a child node to expand
            current_node = root_node
            while not current_node.state.is_terminal():
                if len(current_node.children) == 0:
                    current_node.expand()
                    break
                else:
                    current_node = current_node.select_child(self.exploration_parameter)
            
            # simulate the game from the selected child node
            reward = current_node.simulate()
            
            # update the statistics of the selected child node and its ancestors
            current_node.update(reward)
        
        # return the action with the highest expected reward
        best_child = None
        best_reward = -float('inf')
        for child in root_node.children:
            if child.total_reward > best_reward:
                best_reward = child.total_reward
                best_child = child
        return best_child.state.get_action()