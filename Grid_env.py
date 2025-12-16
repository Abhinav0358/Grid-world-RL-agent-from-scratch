import numpy as np
class gridenv:
    def __init__(self):
        self.observation_space = [[0,0],[0,1],[0,2],[0,3],
                                  [1,0],[1,1],[1,2],[1,3],
                                  [2,0],[2,1],[2,2],[2,3],
                                  [3,0],[3,1],[3,2],[3,3]]
        self.action_space = [0,1,2,3]
        self.terminated=False
        self.truncated=False
        self.terminal_state = [3,3]
        self.reward = {}
        self.current_state = [0,0]
        self.max_steps = 1000
        self.step_count = 0
        #initialize q values for all state action pairs
        self.q_values={}
        for obs in self.observation_space:
            for action in self.action_space:
                self.q_values[(tuple(obs), action)] = 0.0
        for obs in self.observation_space:
            if obs == self.terminal_state:
                self.reward[tuple(obs)] = 1
            else:
                self.reward[tuple(obs)] = 0
        
    def reset(self):
            self.current_state = [0,0]
            self.step_count = 0
            self.terminated = False
            self.truncated = False
            return self.current_state
        
    def step(self,action):
            self.step_count += 1
            if action == 0: #left
                if self.current_state[0] > 0:
                    self.current_state[0] -= 1
            elif action == 1: #right
                if self.current_state[0] < 3:
                    self.current_state[0] += 1
            elif action == 2: #down
                if self.current_state[1] > 0:
                    self.current_state[1] -= 1
            elif action == 3: #up
                if self.current_state[1] < 3:
                    self.current_state[1] += 1
            
            reward = self.reward[tuple(self.current_state)]
            self.terminated = (self.current_state == self.terminal_state)
            self.truncated = (self.step_count >= self.max_steps)
            return self.current_state, reward, self.terminated, self.truncated
        
    def close(self):
            self.current_state = [0,0]
            self.step_count = 0
            self.terminated = False
            self.truncated = False
            pass            