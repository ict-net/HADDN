import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.cluster import KMeans
from tqdm import tqdm

import os
directory = os.path.dirname(os.path.abspath(__file__))


class Environment(object):
    def __init__(self, contextualBandits, labels, odp, n_trials, contextual_New, labels_New, odp_New):
        self.contextualBandits = contextualBandits
        self.n_trials = n_trials
        self.total_reward = 0
        self.n_k = len(contextualBandits)
        self.labels = labels
        self.shape = np.shape(self.contextualBandits)
        self.rewards = []
        self.selected_arms = []  #ot
        self.plot_data = pd.DataFrame()
        self.save_data = pd.DataFrame()
        self.odp = odp
        self.save_data_test = pd.DataFrame()

        self.contextual_New = contextual_New
        self.n_new = len(contextual_New)
        self.labels_new = labels_New
        self.plot_data_new = pd.DataFrame()
        self.save_data_new = pd.DataFrame()
        self.odp_New = odp_New

    def run(self, agent):
        assert isinstance(agent, Agent)
        self.rewards = []
        self.selected_arms = [] #ot
        for i in tqdm(range(self.n_trials)):
            x_chosen = agent.select_arm(self.n_k, 0)
            if x_chosen != '':
                if self.labels[x_chosen] == "BENIGN":
                    reward = 0
                else:
                    reward = 1
                self.selected_arms.append(x_chosen)
                self.rewards.append(reward)
                agent.reward = reward
                agent.selected_arms = self.selected_arms
                agent.update()
            else:
                break
        self.plot_data[agent.name] = np.cumsum(self.rewards)
        self.save_data[agent.name] = np.cumsum(self.rewards)
        self.save_data[agent.name + "_arms"] = self.selected_arms
        self.save_data[agent.name + "_labels"] = self.labels[self.selected_arms]

    def test(self, agent):
        assert isinstance(agent, Agent)
        self.rewards = []
        self.selected_arms = []  # ot
        agent.selected_arms = self.selected_arms
        agent.odp = self.odp_New
        if isinstance(agent, HADDN):
            agent.label_pred = agent.estimator.predict(agent.contextualBandits)
            self.labels = self.labels_new
            agent.contextualBandits = self.contextual_New
            agent.oc = [[] for _ in range(agent.K_cluster)]
            if isinstance(agent, HADDN):
                agent.Z = agent.construct_z()
        self.n_trials = 300
        for i in tqdm(range(self.n_trials)):
            x_chosen = agent.select_arm(self.n_new, 1)
            if x_chosen != '':
                if self.labels_new[x_chosen] == "BENIGN":
                    reward = 0
                else:
                    reward = 1
                self.selected_arms.append(x_chosen)
                self.rewards.append(reward)
                agent.reward = reward
                agent.selected_arms = self.selected_arms
            else:
                break
        self.save_data_test[agent.name + "test"] = np.cumsum(self.rewards)
        self.save_data_test[agent.name + "_labels_test"] = self.labels_new[self.selected_arms]

    def run_new(self,agent):
        assert isinstance(agent, Agent)
        self.rewards = []
        self.selected_arms = [] #ot
        agent.selected_arms = self.selected_arms
        agent.contextualBandits = self.contextual_New
        if isinstance(agent, HADDN):
            agent.label_pred = agent.estimator.predict(agent.contextualBandits)
            if isinstance(agent, HADDN):
                agent.odp = self.odp_New
                agent.Z = agent.construct_z()
            agent.oc = [[] for _ in range(agent.K_cluster)]
        for i in tqdm(range(self.n_trials)):
            x_chosen = agent.select_arm(self.n_new, 0)
            if x_chosen != '':
                if self.labels_new[x_chosen] == "BENIGN":
                    reward = 0
                else:
                    reward = 1
                self.selected_arms.append(x_chosen)
                self.rewards.append(reward)
                agent.reward = reward
                agent.selected_arms = self.selected_arms
                agent.update()
            else:
                break
        self.plot_data_new[agent.name] = np.cumsum(self.rewards)
        self.save_data_new[agent.name + "new"] = np.cumsum(self.rewards)
        self.save_data_new[agent.name + "_arms_new"] = self.selected_arms
        self.save_data_new[agent.name + "_labels_new"] = self.labels_new[self.selected_arms]

class Agent(object):
    def __init__(self, env, lamb):
        self.contextualBandits = env.contextualBandits
        self.n_trials = env.n_trials
        self.n_k = env.n_k
        self.n_new = env.n_new
        self.theta = np.zeros(self.n_trials)
        self.shape = env.shape
        self.lamb = lamb
        self.selected_arms = []
        self.reward = 0
        self.odp = env.odp
        self.odp_new = env.odp_New

    def select_arm(self, n, test):
        pass

    def update(self):
        pass

class HADDN(Agent):
    def __init__(self, env, lamb=1.0, n = 10000, K_cluster=3):
        super(HADDN, self).__init__(env, lamb)
        self.K_cluster = K_cluster
        self.n = n
        self.name = "HADDN"
        self.d = self.shape[1]
        self.p = self.lamb * np.eye(self.K_cluster)
        self.q = np.zeros(self.K_cluster)
        self.delta = 1
        self.Rb = 1#np.sqrt(0.5*np.log(2*self.n_trials*self.K_cluster/self.delta))
        self.Rr = 1#np.sqrt(0.5*np.log(2*self.n_trials*self.n/self.delta))
        self.R = 1

        self.alpha = self.R*np.sqrt(-2*np.log(self.delta/2))
        self.beta = self.alpha
        self.gamma = self.alpha

        self.parameters_acb = [[self.lamb * np.eye(self.d), np.zeros((self.d,self.K_cluster)),np.zeros(self.d)]
                             for _ in range(self.K_cluster)]
        self.estimator = KMeans(n_clusters=K_cluster)
        self.estimator.fit(self.contextualBandits)
        self.label_pred = self.estimator.labels_

        self.cluster_set = {}
        self.Z = self.construct_z()
        self.oc = [[] for _ in range(self.K_cluster)]

        self.delta_A = np.eye(self.d)
        self.delta_B = np.eye(self.K_cluster)

        self.betalin = [self.alpha for _ in range(self.K_cluster)]
        self.gammalin = [self.alpha for _ in range(self.K_cluster)]

        self.f_lsum = [[0] for _ in range(self.K_cluster)]
        self.g_lsum = [[0] for _ in range(self.K_cluster)]
        self.r_lsum = [[0] for _ in range(self.K_cluster)]

    def construct_z(self):
        Z = []
         #ip->[c1,c2,...C]
        for i in range(len(self.contextualBandits)):
            origin, dest = self.odp[i]
            if origin not in self.cluster_set:
                self.cluster_set[origin] = np.zeros(self.K_cluster)
            self.cluster_set[origin][self.label_pred[i]] += 1
            if dest not in self.cluster_set:
                self.cluster_set[dest] = np.zeros(self.K_cluster)
            self.cluster_set[dest][self.label_pred[i]] += 1
        for i in range(len(self.contextualBandits)):
            Z.append(np.array(self.cluster_set[self.odp[i][0]])+np.array(self.cluster_set[self.odp[i][0]]))
        return Z

    def select_arm(self, n, test):
        self.rou = np.dot(np.linalg.inv(self.p), self.q)
        self.thetas = [[]  for _ in range(self.K_cluster)]
        self.b = np.array([0.0  for _ in range(self.K_cluster)])
        self.meanx = np.array([0.0  for _ in range(self.K_cluster)])
        self.meanz = np.array([0.0  for _ in range(self.K_cluster)])
        for c in range(self.K_cluster):
            A = self.parameters_acb[c][0]
            C = self.parameters_acb[c][1]
            B = self.parameters_acb[c][2]
            self.thetas[c] = np.dot(np.linalg.inv(A), B-np.dot(C,self.rou))
            self.b[c] = 0
            if len(self.oc[c]) != 0:
                r_l = [(1 if env.labels[a] != 'BENIGN' else 0) for a in self.oc[c]] + self.r_lsum[c]
                f_l = [np.dot(self.contextualBandits[a], self.thetas[c]) for a in self.oc[c]] + self.f_lsum[c]
                g_l = [np.dot(self.Z[a],self.rou) for a in self.oc[c]] + self.g_lsum[c]
                self.b[c] = np.mean(r_l) - np.mean(f_l) -np.mean(g_l)
            self.meanx[c] = np.mean([self.contextualBandits[a] for a in self.oc[c]])
            self.meanz[c] = np.mean([self.Z[a] for a in self.oc[c]])
        estimates = np.array([float('-inf')  for _ in range(n)])

        for i in set(range(n)) - set(self.selected_arms):
            if test == 0:
                c = self.label_pred[i]
                delta_x = self.contextualBandits[i] - self.meanx[c]
                delta_z = self.Z[i] - self.meanz[c]
                estimates[i] = np.dot(self.contextualBandits[i], self.thetas[c]) + \
                            np.dot(self.Z[i],self.rou) + self.b[c]+self.alpha+ \
                            (self.Rb + 1) * np.sqrt(delta_x.T.dot(self.betalin[c]).dot(delta_x)) +\
                            (1+self.Rr)*delta_z.T.dot(self.gammalin[c]).dot(delta_z)  # self.beta+self.gamma
            else:
                estimates[i] = np.dot(self.contextualBandits[i], self.thetas[self.label_pred[i]]) + \
                            np.dot(self.Z[i], self.rou) + self.b[self.label_pred[i]]
            if math.isnan(estimates[i]):
                estimates[i] = 0
        if len(set(range(n)) - set(self.selected_arms)) != 0:
            arm_to_select = np.random.choice(np.where(estimates == estimates.max())[0])
        else:
            arm_to_select = ''
        return arm_to_select

    def update(self):
        arm = self.selected_arms[-1]
        cluster = self.label_pred[arm]
        A,C,B = self.parameters_acb[cluster]
        self.p += C.T.dot(np.linalg.inv(A)).dot(C)
        self.q += C.T.dot(np.linalg.inv(A)).dot(B)

        self.parameters_acb[cluster][0] += np.outer(self.contextualBandits[arm], self.contextualBandits[arm])
        self.parameters_acb[cluster][1] += np.outer(self.contextualBandits[arm], self.Z[arm])
        self.parameters_acb[cluster][2] += self.contextualBandits[arm] * self.reward

        if np.linalg.det(A) == 0:
            A = self.lamb * np.eye(self.d)
        self.p += np.outer(self.Z[arm], self.Z[arm]) - C.T.dot(np.linalg.inv(A)).dot(C)
        self.q += self.Z[arm] * self.reward - C.T.dot(np.linalg.inv(A)).dot(B)

        self.oc[cluster].append(arm)
        self.alpha = self.R*np.sqrt(-2*np.log(self.delta/2)/len(self.oc[cluster]))

        #
        delta_x = self.contextualBandits[arm]-np.mean([self.contextualBandits[a] for a in self.oc[cluster]])
        self.delta_A += np.outer(delta_x,delta_x)
        if np.linalg.det(self.delta_A) == 0:
            self.delta_A = self.lamb * np.eye(self.d)
        self.betalin[cluster] = np.linalg.inv(self.delta_A)

        #
        delta_z = self.Z[arm]-np.mean([self.Z[a] for a in self.oc[cluster]])
        self.delta_B += np.outer(delta_z,delta_z)
        self.gammalin[cluster] = np.linalg.inv(self.delta_B)

n_trials = 300
features = [10]
clusters = [10]
for d in features:
    columnNames = ['Label']
    for i in range(d):
        columnNames.append('f' + str(i))
    columnNames.append('Source IP')
    columnNames.append('Destination IP')
    df_old = pd.read_csv('data/old'+str(d)+'.csv', index_col=None, header=None, names=columnNames)
    df_new = pd.read_csv('data/new'+str(d)+'.csv', index_col=None, header=None, names=columnNames)

    odp = [((df_old['Source IP'].values)[i], (df_old['Destination IP'].values)[i]) for i in range(df_old.shape[0])]
    odp_new = [((df_new['Source IP'].values)[i], (df_new['Destination IP'].values)[i]) for i in
               range(df_new.shape[0])]
    labels = df_old["Label"].values
    labels_new = df_new["Label"].values

    data = df_old.drop(['Label', 'Source IP', 'Destination IP'], inplace=False, axis=1)
    data = data.astype(float)
    dataArr = data.values

    data_new = df_new.drop(['Label', 'Source IP', 'Destination IP'], inplace=False, axis=1)
    data_new = data_new.astype(float)
    dataArr_new = data_new.values
    for c in clusters:
        env = Environment(dataArr, labels, odp, n_trials, dataArr_new, labels_new, odp_new)
        haddnAgent = HADDN(env, lamb=10, n = len(dataArr), K_cluster=c)
        env.run(haddnAgent)

        df_results = env.save_data
        df_results.to_csv(
            "ucb_old.csv",
            index=False)

        env.test(haddnAgent)

        df_results = env.save_data_test
        df_results.to_csv("ucb_test.csv", index=False)

        env.run_new(haddnAgent)

        df_results = env.save_data_new
        df_results.to_csv(
            "ucb_new.csv",
            index=False)

        sns.lineplot(data=env.plot_data_new)
        plt.xlabel("Query budget T")
        plt.ylabel("Accumulated Reward")
        plt.xlim(0, env.n_trials)
        plt.ylim(0, None)
        plt.savefig("ucb_sign.png")
        plt.show()

