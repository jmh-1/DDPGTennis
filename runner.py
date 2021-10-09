from collections import deque
import numpy as np
from IPython.core.debugger import set_trace
import timeit
import matplotlib.pyplot as plt
import torch

def run(env, agent, n_episodes=20000, max_t=5000, brain_name="", breakWhenSolved=True, state_file_name="checkpoint.pth"):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    solved = False
    start = timeit.default_timer()
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] 
        state = env_info.vector_observations           
        score = np.array([[0.],[0.]])
        agent.reset()
        steps = 0
        for t in range(max_t):
            action = agent.act(state, add_noise=True)

            env_info = env.step(action)[brain_name]        
            next_state = env_info.vector_observations   
            reward = np.array(env_info.rewards).reshape(2,1)
            done =  np.array(env_info.local_done).reshape(2,1)

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if np.any(done):
                steps = t
                break 

        max_score = np.max(score)
        scores_window.append(max_score)    
        scores.append(max_score)          
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}\tTime: {:.2f}\tMax Score: {:.2f}\tMem size: {:.2f}\tSteps:{}'.format(i_episode, np.mean(scores_window), max_score, timeit.default_timer() - start, np.max(scores), len(agent.memory), steps), end="")
        if i_episode % 100 == 0:
            print('\nEpisode {}\tAverage Score: {:.2f}\tTime: {:.2f}'.format(i_episode, np.mean(scores_window), timeit.default_timer() - start))
            plot_scores(scores)
        if np.mean(scores_window)>=.5 and not solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tTime: {:.2f}'.format(i_episode-100, np.mean(scores_window), timeit.default_timer() - start))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            if breakWhenSolved:
                break
            solved = True
    return scores

def plot_scores(scores):
    n=100
    running_average = np.correlate(scores, np.ones(n)/n, mode='valid')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores, label='Score')
    plt.plot(np.arange(len(running_average)) + n - 1 , running_average, label=f'Average Score over prior {n} episodes')
    plt.xlabel('Episode #')
    ax.legend()
    plt.show()
