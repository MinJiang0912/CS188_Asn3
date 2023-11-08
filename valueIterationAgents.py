# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for iteration in range(self.iterations):
            states=self.mdp.getStates()
            values=util.Counter()
            for state in states:
                if self.mdp.isTerminal(state):
                    continue
                
                actionsQvalue=[]
                for action in self.mdp.getPossibleActions(state):
                    actionsQvalue.append(self.computeQValueFromValues(state,action))    
                values[state]=max(actionsQvalue)
        
            self.values=values
 


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        return sum(prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action))

        
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state according to the values currently
        stored in self.values.
        """
        if self.mdp.isTerminal(state):
            return None

        return max(self.mdp.getPossibleActions(state),
                key=lambda action: self.computeQValueFromValues(state, action),
                default=None)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        num_states = len(states)

        for i in range(self.iterations):
            # This will ensure we cycle through the states
            state = states[i % num_states]
            
            # Skip the update if the state is terminal
            if self.mdp.isTerminal(state):
                continue

            # Perform the value iteration update for non-terminal states
            # V[k+1](s) <- max_a Σ_s' T(s, a, s')[R(s, a, s') + γV[k](s')]
            actions = self.mdp.getPossibleActions(state)
            if actions:  # Make sure there are actions available from the state
                # Compute Q-values for all actions in the state
                q_values = [self.computeQValueFromValues(state, action) for action in actions]
                # Update the value of the state to the max Q-value
                self.values[state] = max(q_values)
            else:
                # If there are no available actions, set the value to 0
                self.values[state] = 0

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Step 1: Initialize a priority queue.
        pq = util.PriorityQueue()
        
        # Step 2: Fill the priority queue with states from the MDP.
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                currentValue = self.getValue(state)
                highestQValue = max(self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))
                diff = abs(currentValue - highestQValue)
                pq.push(state, -diff)
        
        # Step 3: Iteratively update the states' values.
        for _ in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                # Update the state's value.
                highestQValue = max(self.computeQValueFromValues(state, action) for action in self.mdp.getPossibleActions(state))
                self.values[state] = highestQValue
                
                # Update priorities of predecessors.
                for pred in self.getPredecessors(state):
                    if not self.mdp.isTerminal(pred):
                        currentValue = self.getValue(pred)
                        highestQValue = max(self.computeQValueFromValues(pred, action) for action in self.mdp.getPossibleActions(pred))
                        diff = abs(currentValue - highestQValue)
                        if diff > self.theta:
                            pq.update(pred, -diff)

    def getPredecessors(self, state):
        """ Return a set of states that have a probability > 0 of reaching the state. """
        predecessors = set()
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                    if nextState == state and prob > 0:
                        predecessors.add(s)
        return predecessors




