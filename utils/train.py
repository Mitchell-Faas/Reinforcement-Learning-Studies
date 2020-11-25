import os
import gym
import numpy as np
from utils.plotting import plot_all, moving_average


def train_agent(agent_class, num_steps=1e4, num_experiments=25, env='CartPole-v1'):
    """
    Function that contains the main logic required to run experiments and generate
    the corresponding plots.
    """
    if type(num_steps) != int:
        num_steps = int(num_steps)

    # Set up the agent and environment
    env = gym.make(env)
    agent = agent_class(env)

    # Ensure data directories are appropriately set up
    base_path = os.path.join(os.getcwd(), 'data', env.spec.id, agent.name)
    verify_dict_structure(base_path)

    # Initialize variance and mean arrays which will be updated iteratively
    mean_moving_avg = np.zeros(num_steps)
    variance_moving_avg = np.zeros(num_steps)
    # Start running experiments
    for experiment in range(num_experiments):
        # Prepare agent
        agent = agent_class(env)

        # Run the experiment
        scores = agent.train(num_steps,
                             progress_prefix=f'Experiment {experiment+1}/{num_experiments}'
                             )
        moving_avg = moving_average(scores)

        # Iteratively update variance
        # https://math.stackexchange.com/questions/102978/incremental-computation-of-standard-deviation
        if experiment > 0:
            variance_moving_avg = ((experiment - 1) / experiment) * variance_moving_avg \
                                  + (1 / (experiment + 1)) * (moving_avg - mean_moving_avg) ** 2

        # Iteratively update mean
        mean_moving_avg = (moving_avg + experiment * mean_moving_avg) / (experiment + 1)

        # Save the weights of this agent
        weights_path = os.path.join(base_path, 'weights', f'experiment-{experiment}.weights')
        agent.save(weights_path)

    # Plot the experimental results
    plot_path = os.path.join(base_path, f'experiment-avg.jpg')
    plot_all(mean=mean_moving_avg,
             std=np.sqrt(variance_moving_avg),
             n=num_experiments,
             path=plot_path
             )


def verify_dict_structure(base_path):
    required_dirs = [os.path.join(base_path, 'weights')]

    for path in required_dirs:
        if not os.path.isdir(path):
            os.makedirs(path)
