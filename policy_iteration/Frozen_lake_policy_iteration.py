import numpy as np
import gym
import random
import time

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


def policy_evaluation(env, V, S, policy):
    while True:
        delta = 0
        for s in S:
            v_old = V[s]
            expected_r = 0
            expected_v = 0
            transition = env.P[s][policy[s]]
            for (probs, state_prime, r, done) in transition:
                expected_r += probs * r
                expected_v += probs * V[state_prime]
            V[s] = expected_r + (gamma*expected_v)
            delta = max(delta, abs(v_old - V[s]))
        if delta <= threshold:
            break
    return(V)

def random_policy(env, S, actions):
    policy_random = {}
    for s in S:
        s_a = random.choice(list(actions.values()))
        policy_random[s] = s_a
    return(policy_random)


def policy_improvement_iteration(env, V, S):
    policy = random_policy(env, S, actions)
    while True:
        V = policy_evaluation(env, V, S, policy)
        new_policy = dict()
        for s in S:
            best_a = None
            best_p = float('-inf')
            for a in actions.values():
                expected_r = 0
                expected_v = 0
                transition = env.P[s][a]
                for (probs, state_prime, r, done) in transition:
                    expected_r += probs * r
                    expected_v += probs * V[state_prime]
                V_new = expected_r + (gamma*expected_v)
                if V_new >= best_p:
                    best_p = V_new
                    best_a = a
            new_policy[s] = best_a
        if new_policy == policy:
            break
        else:
            policy = new_policy
    return(policy)
            
start_time = time.time()
optimal_policy = policy_improvement_iteration(env, V, S)
print("--- %s seconds ---" % (time.time() - start_time))
optimal_actions = np.asarray(list(optimal_policy.values()))
print(optimal_actions.reshape(4,4), actions, sep='\n')

