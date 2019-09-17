import gym
from gym import wrappers
import numpy as np
from matplotlib.pyplot import imsave
import os
# from scipy.stats import linregress
#
#
# def l2_norm_dist_2d(self):
#     data_point = np.asarray(self.sim.data.geom_xpos[1:])
#     _, _, _, _, error = linregress(data_point[:, 0], data_point[:, 1])
#     return error
# def action(env, action):
#     if not isinstance(env.action_space, spaces.Box):
#         return action
#
#     # rescale the action
#     low, high = env.action_space.low, env.action_space.high
#     scaled_action = low + (action + 1.0) * (high - low) / 2.0
#     scaled_action = np.clip(scaled_action, low, high)
#
#     return scaled_action
#
# def reverse_action(self, action):
#     raise NotImplementedError

# env=gym.make('FetchPickAndPlace-v1')
env=gym.make("Rope-v1")
# env=wrappers.Monitor(env, "/tmp/Rope-v1/random",force=True)
# normalize(env)

for episode in range(20):
    observation = env.reset()
    step = 0
    total_reward = 0
    pos = env.sim.data.geom_xpos
    # print(1000*np.linalg.norm(pos[11] - pos[21]))
    while True:
        step += 1
        env.render()
        # print(image)
        # image=env.unwrapped.render(mode='rgb_array',width=64,height=64)
        # print(image.shape)
        # break
        # print(image)
        # video_save_dir = os.path.expanduser('~/softlearning/rope')
        # video_save_path = os.path.join(video_save_dir, f'episode_{episode}_step_{step}.png')
        # imsave \
        #     (video_save_path, image)
        action = env.action_space.sample()
        # pos = env.sim.data.geom_xpos
        # dist = np.linalg.norm(pos[11] - pos[21])
        # print(pos[11])
        # print(pos[21])
        # print(1000 * dist)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        # if step>5:
        #     break
        if done:
            print("Episode: {0},\tSteps: {1},\tscore: {2}"
                  .format(episode, step, total_reward)
            )
            break
env.close()
# import gym
# from gym import wrappers
#
# env = gym.make("HalfCheetah-v2")
# env = wrappers.Monitor(env, "/tmp/gym-results",force='True')
# observation = env.reset()
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample()  # your agent here (this takes random actions)
#     observation, reward, done, info = env.step(action)
#     if done:
#         env.reset()
#
# env.close()
# import gym
#
# env = gym.make('Humanoid-v2')
#
# for i_episode in range(100):
#     env.reset()
#     for t in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()


#
# observation=env.reset()
# for i in range(100000):
#     env.render()
#     # print(env.sim.data.xfrc_applied)
#     # env.sim.data.xfrc_applied[12,:2] = 0.1
#     pos=env.sim.data.geom_xpos
#     dist=np.linalg.norm(pos[11]-pos[21])
#     print(1000*dist)
#     # env.step(env.action_space.sample())
#     # _,reward,_,dict=env.step(0)
#     _,reward,_,dict=env.step(env.action_space.sample())
#     #
#     # if dict['reward_dist']>-820:
#     #     break
#     # print(dict['reward_dist'])
# env.close()

#
# env=gym.make('Rope-v0')
# # normalize(env)
# env.reset()
# for i in range(1000):
#     env.render()
#     # env.step(env.action_space.sample())
# env.close()

# env=gym.make('Ant-v2')
# # normalize(env)
# env.reset()
# for i in range(100000):
#     env.render()
#     # print(env.sim.data.xfrc_applied)
#     # env.sim.data.xfrc_applied[21,:2] = 1
#     # env.step(env.action_space.sample())
#     # _,reward,_,dict=env.step(0)
#     _,reward,_,dict=env.step(env.action_space.sample())
#     #
#     # if dict['reward_dist']>-820:
#     #     break
#     print(reward)
# env.close()