import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from scipy.stats import linregress
from gym import spaces

class RopeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'rope_v1.xml', 4)
        utils.EzPickle.__init__(self)
        # self.action_space=spaces.Box(low=np.array([0.0, 0.0]), high=np.array([2, 2]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-2.0, -2.0]), high=np.array([2.0, 2.0]), dtype=np.float32)
        # self.action_space=self.sim.data.xfrc_applied[21,:2]




    def l2_norm_dist_2d(self,xpos):
        data_point=np.asarray(xpos[1:])
        _,_,_,_,error=linregress(data_point[:,0],data_point[:,1])
        return error



    # def step(self, a):
    #     ctrl_cost_coeff = 0.0001
    #     xposbefore = self.sim.data.qpos[0]
    #     self.do_simulation(a, self.frame_skip)
    #     xposafter = self.sim.data.qpos[0]
    #     reward_fwd = (xposafter - xposbefore) / self.dt
    #     reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
    #     reward = reward_fwd + reward_ctrl
    #     ob = self._get_obs()
    #     return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)
    def step(self,action):
        # ctrl_cost_coeff = 0.0001
        # dist_before=self.l2_norm_dist_2d()
        # notdone =-dist_before/self.dt*100 < 0
        self.do_simulation_external(action,self.frame_skip)
        # dist_after=self.l2_norm_dist_2d()
        # reward_dist=(-dist_after)*1000
        pos=self.sim.data.geom_xpos
        B0_pos=self.get_body_com('B0')
        B20_pos=self.get_body_com('B20')
        reward_dist_goal = np.linalg.norm(B0_pos - B20_pos)
        # reward_ctrl=ctrl_cost_coeff*np.square(action).sum()
        com=np.sum(pos[1:21],axis=0)/21
        reward_dist_penalty=-0.25*np.linalg.norm(com)
        reward=reward_dist_goal+reward_dist_penalty
        # print(dist_after)
        # print(reward_dist)
        # notdone =(-dist_after/self.dt*100 )<-1
        # done=not notdone
        done=False
        # reward=reward_dist+reward_ctrl
        ob=self._get_obs()
        return ob,reward,done,dict(reward_dist_goal=reward_dist_goal,reward_dist_penalty=reward_dist_penalty)
        # return ob,reward, False, dict(reward_dist=reward_dist,reward_ctrl=reward_ctrl)

    # def viewer_setup(self):
    #         self.viewer.cam.trackbodyid = 12
    #         self.viewer.cam.distance = 0


    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])
    # def _get_obs(self):
    #     qpos_head=self.sim.data.qpos('B0')
    #     qpos_tail=self.sim.data.qpos('B20')
    #     return np.concatenate([qpos_head,qpos_tail])

    def reset_model(self):
        # self.init_qpos[1:10]=self.init_qpos[11:20]
        qpos=self.init_qpos + self.np_random.uniform(low=-.2, high=.2, size=self.model.nq)
        qvel=self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        self.set_state(
            qpos,
            qvel
        )
        for _ in range(200):
            self.sim.step()
        # self.sim.data.geom_xpos[1:]=self.sim.data.geom_xpos[1:]+np.random.uniform(-0.5,0.5,size=self.sim.data.geom_xpos[1:].shape)
        return self._get_obs()
