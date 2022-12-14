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
from torch_geometric.data import Data
import itertools

PATH = "/2WL_link_pred/checkpoint/"
DIR = "20221214_15_15/"

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
        model = torch.load(os.getcwd() + PATH + DIR + '2WL_dict_15_15_9.pt').to(configs.device)
        model.load_state_dict(torch.load(os.getcwd() + PATH + DIR + '2WL_state_dict_15_15_9.pt'))
        model.to(configs.device)
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
            # print(self.mchsStartTimes)
            startTime_a, flag = self.predictLinkByGNN(action=action)
            # print(startTime_a)
            # print(flag)
            # print(self.mchsStartTimes)
            
            
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
        
        machineNumber = mchMat[action//self.number_of_machines][action%self.number_of_machines] - 1
        if opIDsOnMchs[machineNumber][1] < 0:
            startTime, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            return startTime, flag
        
        # print("FLAG2")
        
        # TODO: Create dataset wtih above info
        jNum = self.number_of_jobs
        mNum = self.number_of_machines
        opList = opIDsOnMchs.flatten()
        opList = opList[opList >= 0]
        # completionTime = torch.tensor(mchsStartTimes + durMat, dtype=torch.long)
        # completionTime.to(configs.device)
        completionTime = torch.zeros([durMat.shape[0], durMat.shape[1]], dtype=torch.long)
        for rowIdx, row in enumerate(opIDsOnMchs):
            for colIdx, opId in enumerate(row):
                if opId >=0:
                    completionTime[rowIdx][colIdx] = mchsStartTimes[rowIdx][colIdx] + durMat[opId//15][opId%15]
        completionTime.to(configs.device)
        
        node = torch.stack([completionTime, torch.tensor(durMat), torch.arange(0, jNum).expand(mNum, jNum)], dim=2).reshape(jNum * mNum, 3)
        node.to(configs.device)
        
        edge = []
        for row in opIDsOnMchs:
            for idx, element in enumerate(row):
                if idx+1 is not len(row) and element>=0 and row[idx+1]>=0:
                    edge.append([row[idx], row[idx+1]])
        for jobNum, job in enumerate(mchMat):
            for idx, operation in enumerate(job):
                if idx+1 is not len(job):
                    op1 = jobNum*mchMat.shape[1] + idx
                    op2 = jobNum*mchMat.shape[1] + idx+1 
                    if op1 in opList and op2 in opList:
                        edge.append([op1, op2])
        # print(edge)
        edge = np.transpose(edge)
        edge = torch.tensor(edge, dtype=torch.long)
        edge.to(configs.device)
        
        edgeToSearch = []
        machineNumber = mchMat[action // 15][action%15] -1
        for element in opIDsOnMchs[machineNumber]:
            if element >= 0:
                edgeToSearch.append([element, action])
                edgeToSearch.append([action, element])
        # edgeToSearch = np.transpose(edgeToSearch)
        edgeToSearch = torch.tensor(edgeToSearch, dtype=torch.long)
        edgeToSearch.to(configs.device)
        # print(edgeToSearch)
        # print(action)
        # print(machineOfAction)
        # print(opIDsOnMchs[machineOfAction])
        # print(opIDsOnMchs)
        
        # dataset = Data(x=node, ei=edgeToSearch, pos=edge)
        dataset = Data(x=node, ei=edge, pos=edgeToSearch)
        dataset.to(configs.device)
        # print(len(dataset.pos))
        # print(dataset.ei)
        
        mod.eval()
        if isinstance(mod, LocalWLNet):
            # print("FLAG")
            pred = mod(
                dataset.x,
                dataset.ei,
                dataset.pos,
                True,
            )
        else:
            pred = mod(dataset.x, dataset.ei, dataset.pos, test=True)
        
        maxIndex = np.argmax(pred.unsqueeze(-1).tolist())
        selectedEdgeIndex = maxIndex
        # op -> action, action -> op
        edgeToAdd = [dataset.pos[maxIndex], dataset.pos[maxIndex+1]]
        # print(edgeToAdd)
        
        startTime = 0
        flag = True
        
        # TODO: algorithm
        # startTime: new operatioin start time
        # flag: true if other opartion start time changed
        startTime, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
        
        return startTime, flag