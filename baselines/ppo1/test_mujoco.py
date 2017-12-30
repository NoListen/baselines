#!/usr/bin/env python3
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
from ppo_utils import saveFromFlat
from tqdm import tqdm
import numpy as np

def test(env_id, num_episodes, model_path, seed):
    from baselines.ppo1 import mlp_policy
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)


    pi = mlp_policy.MlpPolicy(name='pi', ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=64, num_hid_layers=2)
    pi_vars = pi.get_variables()
    for v in pi_vars:
        print(v.name)
    
    saveFromFlat(pi.get_variables(), model_path)
    env = bench.Monitor(env, logger.get_dir())
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    ep_rews = []
    ob = env.reset()
    for _ in tqdm(range(num_episodes)):
        ep_rew = 0
        new = False
        while not new:
            env.render()
            ac, vpred = pi.act(stochastic=False,ob=ob)
            ob, rew, new, _ = env.step(ac)
            ep_rew += rew
        ob = env.reset()
        ep_rews.append(ep_rew)
    print("----------- Summary ------------")
    print("episode mean %.3f" % np.mean(ep_rews))

    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Humanoid-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--model-path', default="checkpoints-l/model-21.p")

    args = parser.parse_args()
    logger.configure()
    test(args.env, num_episodes=args.num_episodes, model_path = args.model_path, seed=args.seed)


if __name__ == '__main__':
    main()
