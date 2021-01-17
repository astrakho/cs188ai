# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
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
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def genSearch(problem, fringe_data_structure, heuristic=nullHeuristic):
    """
    Takes a search problem, a Data Structure to use for the fringe, and an optional heuristic and returns a list
    of actions.
    """
    actions = []
    visited = []
    total_cost = 0
    fringe = fringe_data_structure()

    currentState = problem.getStartState()
    visited.append(currentState)
    while True:
        successors = problem.getSuccessors(currentState)
        for s in successors:
            # According to the assignment instructions, only expand nodes we have not visited. Don't put known nodes in
            # the fringe.
            if s[0] not in visited:
                # Fringe cost it real cost to get to the node (total cost to parent + cost of action) and heuristic cost
                fringe_cost = total_cost + s[2] + heuristic(s[0], problem)
                s = [actions, s, fringe_cost]
                # the update method only exists for a priority queue data structure
                if getattr(fringe, "update", None):
                    fringe.update(s, fringe_cost)
                else:
                    fringe.push(s)

        # Keep popping fringe nodes until we get something we haven't seen before. Even though we don't add nodes
        # to the fringe that are in visited it is possible that there are multiple paths to an unvisited node and we
        # have added each path to the fringe.
        fringe_just_popped = fringe.pop()
        while fringe_just_popped[1][0] in visited:
            fringe_just_popped = fringe.pop()

        # Item coming off the fringe looks like [[<list of actions for the parent node>, <successor of parent>, cost]
        actions = fringe_just_popped[0].copy()
        currentState = fringe_just_popped[1][0]

        # Total cost to get to this node does not include the heuristic cost used for prioritizing on the fringe
        total_cost = fringe_just_popped[2] - heuristic(currentState, problem)
        # Now fringe_just_popped is [<state>, direction, cost]
        fringe_just_popped = fringe_just_popped[1]
        actions.append(fringe_just_popped[1])
        visited.append(currentState)
        if problem.isGoalState(currentState):
            break

    return actions

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    from util import Stack

    return genSearch(problem, Stack)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue

    return genSearch(problem, Queue)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    return genSearch(problem, PriorityQueue)

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue

    return genSearch(problem, PriorityQueue, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
