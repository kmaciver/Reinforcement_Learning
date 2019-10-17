import numpy as np
import gym

env = gym.make("FrozenLake-v0", is_slippery=True)
'''
Actions:
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3'''

'''
MAPS = {
"4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"]'''

        
V = np.zeros(env.nS)
S = np.arange(0,16)
threshold = 1e-3
gamma = 0.9
actions = {'LEFT':0,'DOWN':1,'RIGHT':2,'UP':3}

def optimal_state_value(env,S,V):
    loop = True
    i = 0
    while loop == True:
        delta = 0
        for s in S:
            v_old = V[s]
            best_v = float('-inf')
            for a in actions.values():
                expected_v = 0
                expected_r = 0
                transitions = env.P[s][a]
                for (probs, state_prime, r, done) in transitions:
                    expected_r += probs * r
                    expected_v += probs * V[state_prime] 
                v_new = expected_r + (gamma * expected_v)
                if v_new > best_v:
                    best_v = v_new
            V[s] = best_v
            delta = max(delta, abs(v_old - best_v))
        if delta <= threshold:
            loop = False
        i+=1
    return (V)

def optimal_policy(env,S,V):
    policy = np.zeros(env.nS)
    V = optimal_state_value(env,S,V)
    for s in S:
        best_a = None
        best_v = float('-inf')
        for k,a in actions.items():
            expected_v = 0
            expected_r = 0
            transitions = env.P[s][a]
            for (probs, state_prime, r, done) in transitions:
                expected_r += probs * r
                expected_v += probs * V[state_prime] 
            v_new = expected_r + (gamma * expected_v)
            if v_new > best_v:
                best_v = v_new
                best_a = a
        policy[s] = best_a
    return(policy)
        
env.render()
print(optimal_state_value(env,S,V).reshape(4,4), optimal_policy(env,S,V).reshape(4,4), actions, sep="\n")
    