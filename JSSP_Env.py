import gym
import numpy as np
from gym.utils import EzPickle
from uniform_instance_gen import override
from updateEntTimeLB import calEndTimeLB
from Params import configs
from permissibleLS import permissibleLeftShift
from updateAdjMat import getActionNbghs
import torch
import os
from model import LocalWLNet

PATH = "/../2WL_link_pred/checkpoint/"
DIR = "20221213_111636/"

class SJSSP(gym.Env, EzPickle):
    def __init__(self,
                 n_j,
                 n_m):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs
        
        # NOTE: generate gnn for each SJSSP instance to use for link prediction
        model = torch.load(os.getcwd() + PATH + DIR + '2WL_dict_9.pt', map_location=configs.device)  
        model.load_state_dict(torch.load(os.getcwd() + PATH + DIR + '2WL_state_dict_9.pt'))
        # device = torch.device(configs.device)
        self.gnn = model

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    @override
    def step(self, action):
        # action is a int 0 - 224 for 15x15 for example
        # redundant action makes no effects
        if action not in self.partial_sol_sequeence:

            # UPDATE BASIC INFO:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            self.partial_sol_sequeence.append(action)

            # UPDATE STATE:
            # permissible left shift
            # NOTE: here, permissibleLeftShift
            # NOTE: startTime_a: int, flag: bool
            # TODO: replace permissibleLeftShift function with ours
            # startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            startTime_a, flag = self.predictLinkByGNN(action=action)
            
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                self.mask[action // self.number_of_machines] = 1

            self.temp1[row, col] = startTime_a + dur_a

            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)

            # adj matrix
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:  # Remove the old arc when a new operation inserts between two operations
                self.adj[succd, precd] = 0

        # prepare for return
        fea = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        reward = - (self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()

        return self.adj, fea, reward, self.done(), self.omega, self.mask

    @override
    def reset(self, data):

        self.step_count = 0
        self.m = data[-1]
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        # initialize adj matrix
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream

        # initialize features
        # NOTE: LB = Lower Bound
        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max() if not configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        # NOTE: finished_mark = machine used mark
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)

        fea = np.concatenate((self.LBs.reshape(-1, 1)/configs.et_normalize_coef,
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        # initialize feasible omega
        self.omega = self.first_col.astype(np.int64)

        # initialize mask
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        # start time of operations on machines
        self.mchsStartTimes = -configs.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)

        self.temp1 = np.zeros_like(self.dur, dtype=np.single)

        return self.adj, fea, self.omega, self.mask

    def predictLinkByGNN(self, action):
        durMat=self.dur
        mchMat=self.m
        mchsStartTimes=self.mchsStartTimes
        opIDsOnMchs=self.opIDsOnMchs
        mod = self.gnn
        # x, na, ei, ea, pos1, y, ei2
        
        # TODO: Create dataset wtih above info
        # dataset = Data()
        dataset = 0
        
        mod.eval()
        if isinstance(mod, LocalWLNet):
            pred = mod(
                dataset.x,
                dataset.ei,
                dataset.pos1,
                dataset.ei.shape[1]
                + torch.arange(dataset.y.shape[0], device=dataset.x.device),
                dataset.ei2,
                True,
            )
        else:
            pred_links = dataset.pos1[
                dataset.ei.shape[1]
                + torch.arange(dataset.y.shape[0], device=dataset.x.device)
            ][:, 0].reshape(-1, 2)
            pred = mod(dataset.x, dataset.ei, pred_links, dataset.ei2, True)
        
        maxindex = np.argmax(pred)
        
        startTime = 0
        flag = True
        
        statTime, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
        
        return startTime, flag