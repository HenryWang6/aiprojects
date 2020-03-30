# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    # explored_set: record all visited nodes
    # solution: a map, key is state, value is actions to move to this state from start state
    # frontier: fringe (state)
    if problem.isGoalState(problem.getStartState()):
        return []
    explored_set = []
    solution = {problem.getStartState(): []}
    frontier = util.Stack()
    frontier.push(problem.getStartState())
    while not frontier.isEmpty():
        # leaf_node: (state)
        leaf_node = frontier.pop()
        if problem.isGoalState(leaf_node):
            return solution[leaf_node]
        # add this node to the list of visited nodes
        explored_set.append(leaf_node)
        # for each child nodes
        expand_nodes = problem.getSuccessors(leaf_node)
        for result_node in expand_nodes:
            # result_node: state, direction, cost
            # check if this state in frontier
            for state,actions in frontier.list:
                if result_node[0] == state:
                    continue
            # check if this state in list of visited nodes
            if explored_set.count(result_node[0]) == 0:
                frontier.push(result_node[0])
                solution[result_node[0]] = list(solution[leaf_node])
                solution[result_node[0]].append(result_node[1])
          	print solution
    return []


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    # pseudocode: book p82

    # solution: a map, key is state, value is actions to move to this state from start state.
    # solution only applies to q1, for remaining question, we only need to use frontier
    # solution = {problem.getStartState(): []}

    # start position is goal state
    if problem.isGoalState(problem.getStartState()):
        return []

    # explored_set: record all visited nodes, graph search version
    explored_set = []

    # frontier: fringe (state, actions)
    frontier = util.Queue()
    frontier.push((problem.getStartState(), []))
    while not frontier.isEmpty():
        # leaf_node: (state, actions from start to this state)
        leaf_node = frontier.pop()
        if problem.isGoalState(leaf_node[0]):
            return leaf_node[1]

        # this version can also solve problem, but expands more search nodes
        # explored_set.append(leaf_node[0])
        #
        #
        # expand_nodes = problem.getSuccessors(leaf_node[0])
        #
        # for result_node in expand_nodes:
        #     # check if this state in frontier
        #     for state,actions in frontier.list:
        #         if result_node[0] == state:
        #             continue
        #     # check if this state in list of visited nodes
        #     if explored_set.count(result_node[0]) == 0:
        #         actions = list(leaf_node[1])
        #         actions.append(result_node[1])
        #         frontier.push((result_node[0], actions))
        # #

        # check if this state in list of visited nodes
        if leaf_node[0] not in explored_set:
            # check if this state in frontier
            for state,actions in frontier.list:
                if leaf_node[0] == state:
                    continue
            # add this node to the list of visited nodes
            explored_set.append(leaf_node[0])
            expand_nodes = problem.getSuccessors(leaf_node[0])
            # result_node (state,action)
            for result_node in expand_nodes:
                # for each child nodes
                actions = list(leaf_node[1])
                actions.append(result_node[1])
                frontier.push((result_node[0], actions))
        #
    return []


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    # pseudocode: book p84
    # the definition of explored_set and solution is same as q1.
    if problem.isGoalState(problem.getStartState()):
        return []
    frontier = util.PriorityQueue()
    # frontier: a priority queue ordered by PATH-COST, with node as the only element
    frontier.push(problem.getStartState(), 0)
    explored_set = []
    solution = {problem.getStartState(): []}
    while not frontier.isEmpty():
        leaf_node = frontier.pop()
        if problem.isGoalState(leaf_node):
            return solution[leaf_node]

    #
        explored_set.append(leaf_node)
        expand_nodes = problem.getSuccessors(leaf_node)
        for result_node in expand_nodes:
            # I check three commands and this state will never be in frontier
            # check if this state in list of visited nodes
            if explored_set.count(result_node[0]) == 0:
                solution[result_node[0]] = list(solution[leaf_node])
                solution[result_node[0]].append(result_node[1])
                frontier.push(result_node[0], problem.getCostOfActions(solution[result_node[0]]))
    #


        #
        # if explored_set.count(leaf_node) == 0:
        #     explored_set.append(leaf_node)
        #     expand_nodes = problem.getSuccessors(leaf_node)
        #     for result_node in expand_nodes:
        #         solution[result_node[0]] = list(solution[leaf_node])
        #         solution[result_node[0]].append(result_node[1])
        #         frontier.push(result_node[0], problem.getCostOfActions(solution[result_node[0]]))
        #

    return []


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    # pseudocode: book p99. its a recursive method. but we only need to change part of q1, q2 q3 to achieve astar
    # explored_set: record all visited nodes
    # solution: a map, key is state, value is (actions to move to this state from start state, heuristic_cost = g(n)+h(n))
    # frontier: fringe ((state, actions, cost), heuristic_cost = g(n)+h(n))
    if problem.isGoalState(problem.getStartState()):
        return []
    frontier = util.PriorityQueue()
    frontier.push((problem.getStartState(), [], 0), 0)
    explored_set = []
    while not frontier.isEmpty():
        # leaf_node: state, actions, cost
        leaf_node = frontier.pop()
        if problem.isGoalState(leaf_node[0]):
            return leaf_node[1]
        if explored_set.count(leaf_node[0]) == 0:
            for state in frontier.heap:
                if leaf_node[0] == state[1][0]:
                    continue
            explored_set.append(leaf_node[0])
            expand_nodes = problem.getSuccessors(leaf_node[0])
            for result_node in expand_nodes:
                # result_node: state, actions, cost
                # print result_node
                # calculate g(n)+h(n) g(n) = result_node[2] + leaf_node[2] h(n)=heuristic(result_node[0], problem)
                heuristic_cost = result_node[2] + heuristic(result_node[0], problem) + leaf_node[2]
                #actions
                actions = list(leaf_node[1])
                actions.append(result_node[1])
                # push (state, actions, g(n)) Priority:g(n)+h(n)
                frontier.push((result_node[0], actions, leaf_node[2] + result_node[2]), heuristic_cost)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
