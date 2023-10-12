# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:30:49 2023

@author: DELL
"""


import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer

from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# barchart 1
x1 = np.array([93.2, 21.7])
y1 = np.array([100-93.2, 100-21.7])
x2 = np.array([94.3, 14.0])
y2 = np.array([100-94.3, 100-14.0])

plt.figure(figsize=(8,3.5), dpi = 300)
plt.subplot(121)
p1 = plt.bar(np.arange(2), y1, 0.7, color = '#CDCDCD', alpha = 0.5, bottom = x1, edgecolor = 'black')
p2 = plt.bar(np.arange(2), x1, 0.7, color = '#ED0000', alpha = 0.5, edgecolor = 'black')
plt.tick_params(labelsize = 13)
plt.xticks(np.arange(2), ('DeepMASS', 'Tracefinder'))
plt.yticks(np.arange(0, 105, 20))
plt.ylabel('Annotated Pct (%)', fontsize = 13)

plt.subplot(122)
p1 = plt.bar(np.arange(2), y2, 0.7, color = '#CDCDCD', alpha = 0.5, bottom = x2, edgecolor = 'black')
p2 = plt.bar(np.arange(2), x2, 0.7, color = '#ED0000', alpha = 0.5, edgecolor = 'black')
plt.tick_params(labelsize = 13)
plt.xticks(np.arange(2), ('DeepMASS', 'Tracefinder'))
plt.yticks(np.arange(0, 105, 20))
plt.ylabel('Annotated Pct (%)', fontsize = 13)
plt.subplots_adjust(wspace=0.3)
plt.legend((p1[0], p2[0]), ('Unknown', 'Annotated'), fontsize=13, loc='upper center', bbox_to_anchor=(-0.2, -0.1), ncol=2)
plt.show()


# barchart 2
x1 = np.array([8704, 1531])
x2 = np.array([8824, 970])

plt.figure(figsize=(8,3.5), dpi = 300)
plt.subplot(121)
p2 = plt.bar(np.arange(2), x1, 0.7, color = ['#ED0000','#00468B'], edgecolor = 'black', alpha = 0.5)
plt.tick_params(labelsize = 13)
plt.xticks(np.arange(2), ('DeepMASS', 'Tracefinder'))
plt.ylabel('Annotated Nums', fontsize = 13)

plt.subplot(122)
p2 = plt.bar(np.arange(2), x2, 0.7, color = ['#ED0000','#00468B'], edgecolor = 'black', alpha = 0.5)
plt.tick_params(labelsize = 13)
plt.xticks(np.arange(2), ('DeepMASS', 'Tracefinder'))
plt.ylabel('Annotated Nums', fontsize = 13)
plt.subplots_adjust(wspace=0.3)
plt.show()


# PCA plot
def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    # Define a function to sort the eigenvalues and eigenvectors in descending order
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    
    if ax is None:
        ax = plt.gca()
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_artist(ellip)
    return ellip


class Dimensional_Reduction:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.x_scl = x
        self.model = None
   
    def scale_data(self, with_mean = True, with_std = True):
        scl = StandardScaler(with_mean = with_mean, with_std = with_std)
        self.x_scl = scl.fit_transform(self.x)
         
    def perform_PCA(self, n_components = 2):
        pca = PCA(n_components = n_components).fit(self.x_scl)
        self.model = pca

    def perform_tSNE(self, n_components = 2):
        tsne = TSNE(n_components=n_components).fit(self.x_scl)
        self.model = tsne

    def plot_2D(self):
        y = np.array(self.y)
        x_map = self.model.fit_transform(self.x_scl)
        colors = ["#00468B", "#ED0000", "#42B540", "#0099B4", "#925E9F", "#FDAF91", "#AD002A"]
        lbs = np.unique(y)
        for i, l in enumerate(lbs):
            k = np.where(y == l)[0]
            pts = x_map[k,:]
            x1, x2 = pts[:,:2].T
            plt.plot(x1, x2, '.', color=colors[i], label = l)
            plot_point_cov(pts[:,:2], nstd=3, alpha=0.2, color=colors[i])
        if type(self.model) == sklearn.decomposition._pca.PCA:
            plt.xlabel('PC1 ({} %)'.format(round(self.model.explained_variance_ratio_[0] * 100, 2)))
            plt.ylabel('PC2 ({} %)'.format(round(self.model.explained_variance_ratio_[1] * 100, 2)))
        elif type(self.model) == sklearn.manifold._t_sne.TSNE:
            plt.xlabel('tSNE 1')
            plt.ylabel('tSNE 2')
        else:
            raise


plt.figure(dpi = 300, figsize = (7.3,6))
imputer = KNNImputer(missing_values=np.nan)

plt.subplot(221)
path = 'Example/Tomato/pca/POS col1-m without QCdata_original.csv'
data = pd.read_csv(path)
x = data.iloc[1:,1:].apply(pd.to_numeric, errors='coerce').T
x = imputer.fit_transform(x)
y = list(data.iloc[0,1:].values)

model = Dimensional_Reduction(x,y)
model.scale_data()
model.perform_PCA()
model.plot_2D()

plt.subplot(222)
path = 'Example/Tomato/pca/Pos_col1_DM_without_QCdata_original.csv'
data = pd.read_csv(path)
x = data.iloc[1:,1:].apply(pd.to_numeric, errors='coerce').T
x = imputer.fit_transform(x)
y = list(data.iloc[0,1:].values)

model = Dimensional_Reduction(x,y)
model.scale_data()
model.perform_PCA()
model.plot_2D()

plt.subplot(223)
path = 'Example/Tomato/pca/Neg col1-m without QCdata_original.csv'
data = pd.read_csv(path)
x = data.iloc[1:,1:].apply(pd.to_numeric, errors='coerce').T
x = imputer.fit_transform(x)
y = list(data.iloc[0,1:].values)

model = Dimensional_Reduction(x,y)
model.scale_data()
model.perform_PCA()
model.plot_2D()

plt.subplot(224)
path = 'Example/Tomato/pca/Neg_col1_DM_without_QCdata_original.csv'
data = pd.read_csv(path)
x = data.iloc[1:,1:].apply(pd.to_numeric, errors='coerce').T
x = imputer.fit_transform(x)
y = list(data.iloc[0,1:].values)

model = Dimensional_Reduction(x,y)
model.scale_data()
model.perform_PCA()
model.plot_2D()

plt.subplots_adjust(wspace=0.3)
plt.legend(fontsize=13, loc='upper center', bbox_to_anchor=(-0.2, -0.15), ncol=6)
plt.show()


# biomarker
def plot_biomarker(data):
    pltdata = []
    for i in range(1, data.shape[0]):
        compound = data.iloc[i, 0]
        for j in range(1, data.shape[1]):
            group = data.iloc[0, j]
            value = float(data.iloc[i, j]) / np.max( data.iloc[i, 1:].values.astype(float) )
            pltdata.append([compound, group, value])
    pltdata = pd.DataFrame(pltdata, columns=['compound', 'group', 'value'])
    ax = sns.barplot(y="compound", x="value", hue="group", data=pltdata, orient="h", errwidth=0.5, capsize=0.1)
    ax.get_legend().remove()
    plt.xlabel("Relative abundance")
    plt.ylabel("")
    plt.grid(False)
    
plt.figure(dpi = 300, figsize = (12,7))
plt.subplot(121)
data = pd.read_csv('Example/Tomato/biomarker/fig4a.csv', header=None)
plot_biomarker(data)

plt.subplot(122)
data = pd.read_csv('Example/Tomato/biomarker/fig4b.csv', header=None)
plot_biomarker(data)
plt.subplots_adjust(wspace=0.95)
plt.legend(fontsize=13, loc='upper center', bbox_to_anchor=(-0.7, -0.1), ncol=6)
plt.show()


# heatmap
def plot_heatmap(data):
    data = data.loc[data['cluster'] >= 0, :]
    data = data.iloc[:, :-1]
    data = data.reset_index(drop=True)
    cmap = sns.light_palette("#79C")
    ax = sns.clustermap(data, row_cluster=True, col_cluster=False, cmap=cmap, col_linkage=False, figsize=(5,8))
    ax.ax_heatmap.set_yticks([])
    for tick in ax.ax_heatmap.get_xticklabels():
        tick.set_fontsize(20)
    ax.savefig("cluster_heatmap.png", dpi=300)

data = pd.read_excel('Example/Tomato/kmeans/Pos_data_results_0.9_10.xlsx', index_col = 0)
plot_heatmap(data)

data = pd.read_excel('Example/Tomato/kmeans/Neg_data_results_0.9_10.xlsx', index_col = 0)
plot_heatmap(data)


# line plot
def plot_line_plot(data, clusters = []):
    pltdata = []
    for i in range(0, data.shape[0]):
        if data.iloc[i, -1] not in clusters:
            continue
        cluster = 'cluster {}'.format(data.iloc[i, -1])
        for j in range(0, data.shape[1] - 1):
            group = data.columns[j]
            value = data.iloc[i,j]
            pltdata.append([cluster, group, value])
    pltdata = pd.DataFrame(pltdata, columns=['cluster', 'group', 'value'])
    plt.figure(dpi = 300, figsize = (5,2.5))
    sns.lineplot(x="group", y="value",hue="cluster", data=pltdata)
    plt.xlabel('')
    plt.ylabel("Relative abundance")
    plt.grid(False)
    plt.legend(bbox_to_anchor=(1, 1.05))

data = pd.read_excel('Example/Tomato/kmeans/Pos_data_results_0.9_10.xlsx', index_col = 0)
plot_line_plot(data, clusters=[32,24,20,14,3])

data = pd.read_excel('Example/Tomato/kmeans/Neg_data_results_0.9_10.xlsx', index_col = 0)
plot_line_plot(data, clusters=[2,6,14,18,33])