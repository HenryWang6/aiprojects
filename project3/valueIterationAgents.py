# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0

    "*** YOUR CODE HERE ***"
    # pseudocode textbook p653
    states = self.mdp.getStates()
    k = 0
    # k iterations
    while k < iterations:
      k+=1
      # a copy of all old values: if this is the (k+1)th iteration, newValues is Vk(s).
      newValues = self.values.copy()
      for state in states:
        actions = self.mdp.getPossibleActions(state)
        values = []
        # not terminal state
        if len(actions)>0:
          for action in actions:
            probability_nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
            value = 0
            for item in probability_nextStates:
              nextState, probability = item
              # formula of value iteration
              value +=probability *(self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))
            values.append(value)
          newValues[state] = max(values)
      # if this is the (k+1)th iteration, self.values now is V(k+1)(s)
      self.values = newValues.copy()


  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"
    q_values = []
    probability_nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
    value = 0
    for item in probability_nextStates:
      nextState, probability = item
      # formula of value iteration
      value += probability * (self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState))
    q_values.append(value)
    return max(q_values)

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"
    actions = self.mdp.getPossibleActions(state)
    final_action = None
    import sys
    maximum = -sys.maxint
    for action in actions:
      value = self.getQValue(state,action)
      if value>maximum:
        maximum = value
        final_action = action

    return final_action



  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)

