import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.decomposition import PCA
import matplotlib
import sklearn
import copy
from seaborn import clustermap
from scipy.cluster.hierarchy import linkage
import random
from random import randint
from sklearn.manifold import MDS
# beta_diversity = 1
# # import skbio
# from skbio.diversity import beta_diversity

import scipy
from helper import *
from matplotlib import cm
from dataLoader import *
import scipy.stats as st
# from statsmodels.graphics.gofplots import qqplot

def rank_sum(x, targets, method = 'ranksum', cutoff=.05):
    if isinstance(targets[0], str):
        targets = (np.array(targets) == 'Recurrer').astype('float')
    else:
        targets = np.array(targets)
    pval = []
    teststat = []
    for i in range(x.shape[1]):
        xin = np.array(x)[:, i]
        X = xin[targets == 1]
        Y = xin[targets == 0]
        # xin1 = (xin - np.min(xin,0))/(np.max(xin,0)-np.min(xin,0))
        if method == 'ranksum':
            s, p = st.ranksums(X, Y)
        elif method == 'kruskal':
            try:
                s, p = st.kruskal(X,Y)
            except:
                p = 1
        elif method == 'ttest':
            s, p = st.ttest_ind(X,Y)
        pval.append(p)
        teststat.append(s)

    # corrected, alpha = bh_corr(np.array(pval), .05)
    from statsmodels.stats.multitest import multipletests
    reject, corrected, a1, a2 = multipletests(pval, alpha=.05, method='fdr_bh')
    df = pd.DataFrame(np.vstack((pval, corrected)).T, columns=[
        'P_Val', 'BH corrected'], index=x.columns.values)

    return df.sort_values('P_Val', ascending=True)


def plot_PCA(finalDf, variance, targets, path = 'pca', target_labels=None, fig_title='', colors=None, markers=None, fig = None, ax = None):
    if fig == None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        save_fig = True
    else:
        save_fig = False
    ax.set_xlabel('PC 1, variance explaned= ' +
                  str(np.round(variance[0]*100, 3)) + '%', fontsize=20)
    ax.set_ylabel('PC 2, variance explaned= ' +
                  str(np.round(variance[1]*100, 3)) + '%', fontsize=20)
    ax.set_title(fig_title, fontsize=20)

    if len(targets)==2:
        tlabs = [np.unique(targs) for targs in targets]
    else:
        tlabs = np.unique(targets)

    if markers is None and len(tlabs)>1:
        pos_markers = ['o','+','v','s','x','D','X','1','^','<','>','p','*','P','|','-']
        markers = {}
        for i,t in enumerate(tlabs[1]):
            markers[t] = pos_markers[i]

    if colors is None:
        colors = {}
        cmap = matplotlib.cm.get_cmap('hsv')

        vec = len(tlabs[0])
        for i,t in enumerate(tlabs[0]):
            cv = np.expand_dims(np.array(cmap(i / len(tlabs[0]))), 0)
            colors[t] = cv

    sizes = {'Recurrer':70, 'Non-recurrer':30}
    alphas = {'Recurrer':1, 'Non-recurrer':0.65}

    if len(targets) == 2:
        if len(tlabs[0])==2:
            a,b = (0,1)
        else:
            a, b = (1,0)

        all_targs = np.concatenate([list(zip(tlabs[a],x)) for x in itertools.permutations(tlabs[b],len(tlabs[a]))])
        lines = {}
        lines[a] = {}
        lines[b] = {}
        for targ in all_targs:
            indicesToKeep1 = np.where(targets[0] == targ[a])[0]
            # if targ[0] not in lines.keys():
            #     lines[targ[0]] = {}
            indicesToKeep2 = np.where(targets[1] == targ[b])[0]
            ix_combo = list(set(indicesToKeep1).intersection(indicesToKeep2))
            # recur/cleared will always be targ[0]
            # if recur/cleared are first, want recur/cleared to be colors
            # otherwise, want weeks to be colors & recur/cleared to be markers

            # if Recur/cleared is the second label OR if weeks is the second label
            # if targ[b] in tlabs[1]:
                # Condition met for whichever labels we want to be markers

            if targ[b] not in lines[b].keys():
                # make line for legend for MARKERS
                line = ax.scatter(finalDf.iloc[ix_combo]['PC1'],
                                  finalDf.iloc[ix_combo]['PC2'], facecolors='none', c='k', s=30,
                                  marker=markers[targ[b]],
                                  alpha=alphas[targ[0]])
                lines[b][targ[b]] = line
            # if targ[a] in tlabs[0]:
            if targ[a] not in lines[a].keys():
                line = ax.scatter(finalDf.iloc[ix_combo]['PC1'],
                                  finalDf.iloc[ix_combo]['PC2'], c=colors[targ[a]], s=30,
                                  marker=list(markers.values())[1],
                                  alpha=alphas[targ[0]])
                lines[a][targ[a]] = line
            line_temp = ax.scatter(finalDf.iloc[ix_combo]['PC1'],
                                   finalDf.iloc[ix_combo]['PC2'], c=colors[targ[a]], s=sizes[targ[0]],
                                   marker=markers[targ[b]],
                                   alpha=alphas[targ[0]])

        k2, v2 = list(lines[a].keys()), list(lines[a].values())
        if target_labels[b]:
            k2 = [target_labels[b][k] for k in k2]
        legend1 = ax.legend(v2, ['Week ' + str(tt) for tt in k2], loc = 1)
        k1, v1 = list(lines[b].keys()), list(lines[b].values())
        if target_labels[a]:
            k1 = [target_labels[a][k] for k in k1]
        legend2 = ax.legend(v1, k1, loc = 2)
        ax.add_artist(legend1)
    else:
        for target, color in zip(tlabs, colors):
            indicesToKeep = np.array(targets == target)
            ax.scatter(finalDf.loc[indicesToKeep, 'PC1'],
                       finalDf.loc[indicesToKeep, 'PC2'], c=colors[color], s=50)
        if target_labels is None:
            ax.legend(tlabs, prop={'size': 15})
        else:
            ax.legend(target_labels, prop={'size': 15})
    if save_fig:
        plt.savefig(path + '.pdf')
    else:
        return fig, ax

def pcoa_custom(x, metric = 'braycurtis', metric_mds = True):
    if metric == 'euclidean':
        dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(x, metric = 'euclidean'))
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents, columns=[
                                'PC1', 'PC2'])
        variance = pca.explained_variance_ratio_  # calculate variance ratios
    else:
        if metric == 'spearman':
            corr, p = st.spearmanr(x.T)
            dist_mat = (1 - corr)/2
            dist_mat = (dist_mat + dist_mat.T) / 2
            np.fill_diagonal(dist_mat,0)
        else:
            dist_mat = scipy.spatial.distance.pdist(x, metric = 'braycurtis')
            dist_mat = scipy.spatial.distance.squareform(dist_mat)

        print(dist_mat.shape)
        mds = MDS(n_components = 2, dissimilarity = 'precomputed', metric = metric_mds)
        Y = mds.fit_transform(dist_mat)

        C = np.eye(dist_mat.shape[0]) - (1/dist_mat.shape[0])*np.ones(dist_mat.shape)
        D = dist_mat**2
        B = -0.5*(C@D@C)
        eigen_values, eigen_vectors = np.linalg.eig(B)
        sorted_ixs = np.argsort(eigen_values)[::-1]
        sorted_eig = eigen_values[sorted_ixs]
        E = eigen_vectors[:,sorted_ixs]

        # Y = (E@x)

        variance = [(se/np.sum(abs(sorted_eig))) for se in sorted_eig]
        # variance_tot = np.var(Y,0)
        # variance = (variance_tot/ sum(abs(variance_tot)))*100
        principalDf = pd.DataFrame(np.array(Y)[:,:2], columns = ['PC1', 'PC2'])

    return principalDf, variance, dist_mat

def normalized_entropy(counts):
    '''Calculate the normailized entropy
    Entropy is defined as
        E = - \sum_i (b_i * \log_n(b_i))
    where
        b_i is the relative abundance of the ith ASV
    Parameters
    ----------
    counts (array_like)
        - Vector of counts
    Returns
    -------
    double
    '''
#     counts = _validate_counts(counts)
    rel = counts[counts>0]
    rel = rel / np.sum(rel)
    a = rel * np.log(rel)
    a = -np.sum(a) / np.log(len(rel))
    return a

def plot_heatmap(data, name = ''):
    pts = [c.split('-')[0] for c in data.index.values]

    ix_pts = []
    colnames_ixs = []
    for i, pt in enumerate(np.unique(pts)):
        ix_pts.append(np.where(np.array(pts) == pt)[0])
        #     colnames_ixs.append(1+i+np.floor(np.mean(np.where(np.array(pts) == pt)[0])))
        colnames_ixs.append(np.floor(np.mean(np.where(np.array(pts) == pt)[0])))
    ixpt = [x[0] for x in ix_pts]

    metabs = data.columns.values

    fig, ax = plt.subplots(figsize=(150, 150));
    pos = ax.imshow(np.array(data).T, cmap=plt.get_cmap('bwr'), vmin = -4, vmax = 4);
    plt.yticks(np.arange(len(metabs)), metabs, fontsize=8);
    plt.xticks(colnames_ixs, np.unique(pts), fontsize=20, rotation=45);

    cbar = fig.colorbar(pos, fraction=0.15, shrink=0.5, pad=0.01, ticks = np.arange(-4,4))
    cbar.ax.tick_params(labelsize=200)
    plt.savefig(name + '.pdf')

if __name__ == "__main__":
    dl_1 = dataLoader(pt_perc=.25, meas_thresh=0, var_perc=15, pt_tmpts=1)
    dl_2 = dataLoader(pt_perc=.05, meas_thresh=10, var_perc=5, pt_tmpts=1)
    metabs = pd.DataFrame(np.vstack([dl_1.week['metabs'][i]['x'] for i in [0, 1, 2, 3, 4]]), index= \
        np.concatenate([dl_1.week['metabs'][i]['x'].index.values for i in [0, 1, 2, 3, 4]]), \
                          columns=dl_1.week['metabs'][0]['x'].columns.values)
    met_y = np.concatenate([dl_1.week['metabs'][i]['y'] for i in [0, 1, 2, 3, 4]])
    met_y_tmpts = np.array([x.split('-')[1] for x in metabs.index.values])

    bile_acids = pd.DataFrame(np.vstack([dl_1.week['bile_acids'][i]['x'] for i in [0, 1, 2, 3, 4]]), index= \
        np.concatenate([dl_1.week['bile_acids'][i]['x'].index.values for i in [0, 1, 2, 3, 4]]), \
                              columns=dl_1.week['bile_acids'][0]['x'].columns.values)
    ba_y = np.concatenate([dl_1.week['bile_acids'][i]['y'] for i in [0, 1, 2, 3, 4]])
    ba_y_tmpts = np.array([x.split('-')[1] for x in bile_acids.index.values])

    counts = pd.DataFrame(np.vstack([dl_2.week_raw['16s'][i]['x'] for i in [0, 1, 2, 3, 4]]), index= \
        np.concatenate([dl_2.week_raw['16s'][i]['x'].index.values for i in [0, 1, 2, 3, 4]]), \
                          columns=dl_2.week_raw['16s'][0]['x'].columns.values)
    proportions = pd.DataFrame(np.divide(counts.T, np.sum(counts, 1)).T, index= \
        np.concatenate([dl_2.week_raw['16s'][i]['x'].index.values for i in [0, 1, 2, 3, 4]]), \
                               columns=dl_2.week_raw['16s'][0]['x'].columns.values)
    prop_y = np.concatenate([dl_2.week_raw['16s'][i]['y'] for i in [0, 1, 2, 3, 4]])
    prop_y_tmpts = np.array([x.split('-')[1] for x in proportions.index.values])

    fig = plt.figure(figsize=(20, 15))
    ax_otus = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax_metabs = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    pca_dF, variances, d1 = pcoa_custom(metabs, metric='spearman')
    fig, ax_metabs = plot_PCA(pca_dF, variances, [met_y_tmpts, met_y], path='paper_figs/pca/metabs_sp',
                              fig_title='Metabolic Levels, PCoA', target_labels=
                              [{}, {'Non-recurrer': 'Will Recur (R)', 'Recurrer': 'Will not recur (NR)'}],
                              markers=None, fig=fig, ax=ax_metabs)

    # Univariate analysis
    for key in dl.week.keys():
        for week in dl.week[key].keys():
            if len(np.unique(dl.week[key][week]['y_eventually']))<2:
                continue
            df = rank_sum(dl.week[key][week]['x'], dl.week[key][week]['y_eventually'])
            df.to_csv('paper_figs/univariate_analysis/' + key + '_' + str(week) + '.csv')
            plt.figure()
            plt.hist(df['P_Val'], bins = np.arange(0,1.05,.05))
            plt.xlabel('p-values')
            plt.ylabel('Frequency')
            plt.title(key + ', week ' + str(week))
            plt.savefig('paper_figs/univariate_analysis/p_vals_' + key + str(week) + '.pdf')
            plt.close()

    key = '16s'
    for week in dl.week[key].keys():
        if len(np.unique(dl.week[key][week]['y_eventually'])) < 2:
            continue
        df = rank_sum(dl.week[key][week]['x'], dl.week[key][week]['y_eventually'], method = 'kruskal')
        df.to_csv('paper_figs/univariate_analysis/' + key + '_' + str(week) + '.csv')
        plt.figure()
        plt.hist(df['P_Val'], bins = np.arange(0,1.05,.05))
        plt.xlabel('p-values')
        plt.ylabel('Frequency')
        plt.title(key + ', week ' + str(week))
        plt.savefig('paper_figs/univariate_analysis/p_vals_kruskal_' + key + str(week) + '.pdf')
        plt.close()
    # Entropy
    entropy_otus = [[normalized_entropy(dl.week_raw['16s'][i]['x'].loc[k,:]) for k in dl.week_raw['16s'][i]['x'].index.values] for i in [0,1,2,3,4]]
    entropy_df = {i: pd.DataFrame({'Week ' + str(i): [normalized_entropy(dl.week_raw['16s'][i]['x'].loc[k,:]) for k in dl.week_raw['16s'][i]['x'].index.values], 'Outcome': dl.week_raw['16s'][i]['y_eventually']}) for i in [0,1,2,3,4]}

    fig, ax = plt.subplots(1,5,figsize = (20,10), sharey = True)
    for key, val in entropy_df.items():
        val.boxplot(column = 'Week ' + str(key), by = 'Outcome', ax = ax[key])
    plt.savefig('paper_figs/entropy.pdf')

    tmpts = [0,1,2,3]
    outcomes = ['Non-recurrer', 'Recurrer']
    all_targs = np.concatenate([list(zip(outcomes, x)) for x in itertools.permutations(tmpts, len(outcomes))])
    entropy_to_tst = {tuple(targ): entropy_df[int(targ[1])]['Week ' + targ[1]].loc[entropy_df[int(targ[1])]['Outcome'] == targ[0]] for targ in all_targs}
    pval = {}
    for df1, df2 in itertools.combinations(entropy_to_tst.keys(), 2):
        try:
            _, pval[str((df1, df2))] = st.mannwhitneyu(x = entropy_to_tst[df1], y = entropy_to_tst[df2])
        except:
            continue
    p_df = pd.DataFrame(pval, index = [0]).T
    p_df.to_csv('paper_figs/entropy_ttest.csv')

    # Heatmaps
    plot_heatmap(dl.keys['metabs']['filtered_data'], name = 'paper_figs/metab_heatmap')
    plot_heatmap(dl.keys['bile_acids']['filtered_data'], name='paper_figs/ba_heatmap')
    plot_heatmap(dl.keys['16s']['filtered_data'], name='paper_figs/otu_heatmap')

    # PCA
    pca_dF, variances, d = pcoa_custom(metabs, metric='spearman')
    plot_PCA(pca_dF, variances, [met_y, met_y_tmpts], path='paper_figs/pca/metabs_sp', target_labels=None, fig_title='Metabolites, Spearman PCoA', colors={
        'Non-recurrer': 'g', 'Recurrer': 'r'}, markers=None)

    metabs_std = standardize(metabs, override = False)
    pca_dF, variances, d = pcoa_custom(metabs_std, metric='euclidean')
    plot_PCA(pca_dF, variances, [met_y, met_y_tmpts], path='paper_figs/pca/metabs_euc', target_labels=None, fig_title='Metabolites, PCA', colors={
        'Non-recurrer': 'g', 'Recurrer': 'r'}, markers=None)


    pca_dF, variances, d = pcoa_custom(bile_acids, metric='spearman')
    plot_PCA(pca_dF, variances, [ba_y, ba_y_tmpts], path='paper_figs/pca/bile_acids_sp', target_labels=None, fig_title='Bile Acids, Spearman PCoA', colors={
        'Non-recurrer': 'g', 'Recurrer': 'r'}, markers=None)

    bile_acids_std = standardize(bile_acids, override=True)
    pca_dF, variances, d = pcoa_custom(bile_acids_std, metric='euclidean')
    plot_PCA(pca_dF, variances, [ba_y, ba_y_tmpts], path='paper_figs/pca/bile_acids_euc', target_labels=None, fig_title='Bile Acids, PCA',
             colors={'Non-recurrer': 'g', 'Recurrer': 'r'}, markers=None)

    pca_dF, variances, d = pcoa_custom(proportions, metric='braycurtis')
    plot_PCA(pca_dF, variances, [prop_y, prop_y_tmpts], path='paper_figs/pca/otus', target_labels=None,
             fig_title='OTU Relative Abundance, Bray-Curtis PCoA', colors={
            'Non-recurrer': 'g', 'Recurrer': 'r'}, markers=None)