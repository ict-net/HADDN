import numpy as np
import pandas as pd
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
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
        agent.label_pred = agent.estimator.predict(agent.contextualBandits)
        self.labels = self.labels_new
        agent.contextualBandits = self.contextual_New
        agent.oc = [[] for _ in range(agent.K_cluster)]
        if isinstance(agent, HADDN):
            agent.Z = agent.construct_z()
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
        self.save_data_test[agent.name + "_arms_test"] = self.selected_arms
        self.save_data_test[agent.name + "_labels_test"] = self.labels_new[self.selected_arms]

    def run_new(self,agent):
        assert isinstance(agent, Agent)
        self.rewards = []
        self.selected_arms = [] #ot
        self.labels = self.labels_new
        agent.selected_arms = self.selected_arms
        agent.contextualBandits = self.contextual_New
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
    def __init__(self, env, K_cluster, sigma1=0.3, sigma2=0.01, sigma3=0.3, sigma4=0.03):
        super(HADDN, self).__init__(env, K_cluster)
        self.name = "TS_HADDN"
        self.n, self.d = np.shape(self.contextualBandits)
        self.K_cluster = K_cluster
        self.sigma1 = sigma1
        self.sigma2 = sigma2  # *sigma2
        self.sigma3 = sigma3  # *sigma3
        self.sigma4 = sigma4
        self.theta_rand = np.random.RandomState()
        self.rou_rand = np.random.RandomState()
        self.gamma_rand = np.random.RandomState()

        self.D = [np.eye(self.d, self.d) / self.sigma3 for _ in range(self.K_cluster)]
        self.E = [np.zeros(self.d) / self.sigma2 for _ in range(self.K_cluster)]
        self.invD = [np.linalg.inv(self.D[i]) for i in range(self.K_cluster)]
        self.theta_mean = [self.invD[i].dot(self.E[i]) for i in range(self.K_cluster)]
        self.theta = [self.theta_rand.multivariate_normal(self.theta_mean[i], self.invD[i]) for i in range(self.K_cluster)]

        self.F = np.zeros([self.K_cluster, self.K_cluster]) / self.sigma4
        self.G = np.zeros(self.K_cluster) / self.sigma2
        self.invF = self.F
        self.rou_mean = self.invF.dot(self.G)
        self.rou = self.rou_rand.multivariate_normal(self.rou_mean, self.invF)

        self.r_l = [[] for _ in range(self.K_cluster)]
        self.e_l = [[] for _ in range(self.K_cluster)]
        self.g_l = [[] for _ in range(self.K_cluster)]

        self.estimator = KMeans(n_clusters=K_cluster)
        self.estimator.fit(self.contextualBandits)
        self.label_pred = self.estimator.labels_

        self.cluster_set = {}
        self.Z = self.construct_z()
        self.oc = [[] for _ in range(self.K_cluster)]

    def construct_z(self):
        Z = []
        # ip->[c1,c2,...C]
        for i in range(len(self.contextualBandits)):
            origin, dest = self.odp[i]
            if origin not in self.cluster_set:
                self.cluster_set[origin] = np.zeros(self.K_cluster)
            self.cluster_set[origin][self.label_pred[i]] += 1
            if dest not in self.cluster_set:
                self.cluster_set[dest] = np.zeros(self.K_cluster)
            self.cluster_set[dest][self.label_pred[i]] += 1
        for i in range(len(self.contextualBandits)):
            Z.append(np.array(self.cluster_set[self.odp[i][0]]) + np.array(self.cluster_set[self.odp[i][0]]))
        return Z

    def select_arm(self, n, test):
        estimates = np.array([float('-inf') for _ in range(n)])
        for i in set(range(n)) - set(self.selected_arms):
            c = self.label_pred[i]
            pri_gamma = np.dot(self.theta[c], self.contextualBandits[i]) + np.dot(self.rou, self.Z[i])
            mean_gamma = (self.sigma2 * np.sum(self.r_l[c]) + self.sigma1 * pri_gamma) / (
                        len(self.r_l[c]) * self.sigma2 + self.sigma1)
            gamma_cov = self.sigma1 * self.sigma2 / (self.sigma1 + len(self.r_l[c]) * self.sigma2)
            if test == 0:
                estimates[i] = self.gamma_rand.normal(mean_gamma, gamma_cov)
                if estimates[i] > 1:
                    estimates[i] = 1
            else:
                estimates[i] = self.gamma_rand.normal(mean_gamma, gamma_cov)
                if estimates[i] > 1:
                    estimates[i] = 1
            if math.isnan(estimates[i]):
                estimates[i] = 0
        if len(set(range(n)) - set(self.selected_arms)) != 0:
            arm_to_select = np.random.choice(np.where(estimates == estimates.max())[0])
        else:
            arm_to_select = ''
        return arm_to_select

    def update(self):
        arm = self.selected_arms[-1]
        c = self.label_pred[arm]
        self.oc[c].append(arm)

        pre_E = (np.sum(self.r_l[c]) - len(self.r_l[c]) * np.dot(self.rou, self.Z[arm])) / self.sigma2
        pre_G = (np.sum(self.r_l[c]) - len(self.r_l[c]) * np.dot(self.theta[c],
                                                                 self.contextualBandits[arm])) / self.sigma2

        self.r_l[c].append(self.reward)

        cur = 1 / self.sigma2
        cur_E = (np.sum(self.r_l[c]) - len(self.r_l[c]) * np.dot(self.rou, self.Z[arm])) / self.sigma2
        cur_G = (np.sum(self.r_l[c]) - len(self.r_l[c]) * np.dot(self.theta[c],
                                                                 self.contextualBandits[arm])) / self.sigma2
        delta_E = cur_E - pre_E
        delta_D = cur
        self.D[c] += delta_D * np.outer(self.contextualBandits[arm], self.contextualBandits[arm])
        self.E[c] += self.contextualBandits[arm].T.dot(delta_E)
        if np.linalg.det(self.D[c]) == 0:
            self.D[c] = np.eye(self.d, self.d)
        self.invD[c] = np.linalg.inv(self.D[c])
        self.theta_mean[c] = self.invD[c].dot(self.E[c])

        delta_G = cur_G - pre_G
        self.F += delta_D * np.outer(self.Z[arm], self.Z[arm])
        self.G += self.Z[arm].T.dot(delta_G)
        if np.linalg.det(self.F) != 0:
           self.invF = np.linalg.inv(self.F)
        else:
            self.F = np.zeros([self.K_cluster, self.K_cluster]) / self.sigma4
            self.invF = self.F
        self.rou_mean = self.invF.dot(self.G)

        self.theta[c] = self.theta_rand.multivariate_normal(self.theta_mean[c], self.invD[c])
        self.rou = self.rou_rand.multivariate_normal(self.rou_mean, self.invF)

n_trials = 300
sig1 = 0.01
sig2 = 0.3
sig3 = 0.01
sig4 = 0.01
d = 10
c = 10
columnNames = ['Label']
for i in range(d):
    columnNames.append('f' + str(i))
columnNames.append('Source IP')
columnNames.append('Destination IP')
df_old = pd.read_csv('data/old'+str(d)+'.csv', index_col=None, header=None, names=columnNames)
df_new = pd.read_csv('data/new'+str(d)+'.csv', index_col=None, header=None, names=columnNames)

odp = [((df_old['Source IP'].values)[i], (df_old['Destination IP'].values)[i]) for i in range(df_old.shape[0])]
odp_new = [((df_new['Source IP'].values)[i], (df_new['Destination IP'].values)[i]) for i in range(df_new.shape[0])]
labels = df_old["Label"].values
labels_new = df_new["Label"].values

data = df_old.drop(['Label', 'Source IP', 'Destination IP'], inplace=False, axis=1)
data = data.astype(float)
dataArr = data.values

data_new = df_new.drop(['Label', 'Source IP', 'Destination IP'], inplace=False, axis=1)
data_new = data_new.astype(float)
dataArr_new = data_new.values
env = Environment(dataArr, labels, odp, n_trials, dataArr_new, labels_new, odp_new)
haddnAgent = HADDN(env, K_cluster=c,sigma1=sig1,sigma2=sig2,sigma3=sig3,sigma4=sig4)

env.run(haddnAgent)
df_results = env.save_data
df_results.to_csv(
    "ts_old.csv",
    index=False)

env.test(haddnAgent)
df_results = env.save_data_test
df_results.to_csv("ts_test.csv", index=False)

env.run_new(haddnAgent)
df_results = env.save_data_new
df_results.to_csv(
    "ts_new.csv",
    index=False)

sns.lineplot(data=env.plot_data_new)
plt.xlabel("Query budget T")
plt.ylabel("Accumulated Reward")
plt.xlim(0, env.n_trials)
plt.ylim(0, None)
plt.savefig("ts_sign.png")
plt.show()
