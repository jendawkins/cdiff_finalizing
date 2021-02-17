import scipy.stats as st
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch.nn as nn
from torch.nn import functional as F
import torch
from sklearn.feature_selection import SelectFromModel
import copy
import pickle
from helper import *
import sys
from io import StringIO
from sklearn.model_selection import GridSearchCV

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.tensor as T
from sklearn.metrics import roc_curve, auc, roc_auc_score

from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score

class Net(nn.Module):
    def __init__(self, num_mols, hidden_size):
        super(Net,self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_mols, hidden_size),
            nn.BatchNorm1d(hidden_size), #applying batch norm
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(hidden_size,2),
        )

    def forward(self,x):
        x = self.classifier(x).squeeze()
        return x


class LogRegNet(nn.Module):
    def __init__(self, num_mols):
        super(LogRegNet, self).__init__()
        self.linear = nn.Linear(num_mols, 2)

    def forward(self, x):
        x = self.linear(x).squeeze()
        return x


class mlMethods():
    def __init__(self, pt_info_dict_orig, cd, lag = 1, option = 2, use16s = False, filter_percentage = 5):
        new_info_dict = copy.deepcopy(pt_info_dict_orig)
        new_info_dict2 = copy.deepcopy(pt_info_dict_orig)
        self.cd = cd

        # Change labels to be recur only at week before recurrence and cleared all other times and 
        # remove patients who don't have timepoint 1.0
        iterable_dict = copy.deepcopy(new_info_dict)
        for patient in iterable_dict:
            if 1.0 not in new_info_dict[patient].keys():
                new_info_dict.pop(patient)
                new_info_dict2.pop(patient)
                continue
            if new_info_dict[patient][1.0]['PATIENT STATUS (BWH)']=='Recur':
                tmpts = list(new_info_dict[patient].keys())
                num_labels = len(tmpts)-lag
                labs = ['Cleared']*(num_labels-lag)
                labs.extend(['Recur'])
                for i,lab in enumerate(labs):
                    new_info_dict[patient][tmpts[i]]['PATIENT STATUS (BWH)'] = lab
                if lag > 1:
                    for i in range(lag):
                        new_info_dict[patient].pop(tmpts[-i])
                        new_info_dict2[patient].pop(tmpts[-i])
                else:
                    new_info_dict[patient].pop(tmpts[-lag])
                    new_info_dict2[patient].pop(tmpts[-lag])
            else:
                tmpts = list(new_info_dict[patient].keys())
                num_labels = len(tmpts)-lag
                labs = ['Cleared']*(num_labels)
                for i,lab in enumerate(labs):
                    new_info_dict[patient][tmpts[i]]['PATIENT STATUS (BWH)'] = lab
                if lag > 1:
                    for i in range(lag):
                        new_info_dict[patient].pop(tmpts[-i])
                        new_info_dict2[patient].pop(tmpts[-i])
                else:
                    new_info_dict[patient].pop(tmpts[-lag])
                    new_info_dict2[patient].pop(tmpts[-lag])
            pt_info_dict = new_info_dict
            pt_info_dict2 = new_info_dict2
        
        tmpts = [list(pt_info_dict[i].keys())
                for i in pt_info_dict.keys()]
        all_pts = np.unique([inner for outer in tmpts for inner in outer])
        self.tmpts = all_pts
        self.week = dict()
        self.week16s = dict()
        self.targets_dict = dict()
        self.targets_dict2 = dict()
        
        if use16s:
            for ii in pt_info_dict.keys():
                for pts in pt_info_dict[ii].keys():
                    if '16s' in pt_info_dict[ii][pts].keys():
                        if str(pts) + '_16s_only' not in self.week.keys():
                            self.week[str(pts) + '_16s_only'] = []
                            self.targets_dict[str(pts) + '_16s_only'] = []
                            self.targets_dict2[str(pts) + '_16s_only'] = []

                        self.week[str(pts) + '_16s_only'].append(pt_info_dict[ii][pts]['16s'])
                        self.targets_dict[str(pts) + '_16s_only'].append(pt_info_dict[ii][pts]['PATIENT STATUS (BWH)'])
                        self.targets_dict2[str(pts) + '_16s_only'].append(pt_info_dict2[ii][pts]['PATIENT STATUS (BWH)'])
                    if '16s' in pt_info_dict[ii][pts].keys() and 'DATA' in pt_info_dict[ii][pts].keys():
                        if str(pts) + '_16s' not in self.week.keys():
                            self.week[str(pts) + '_16s'] = []
                            self.targets_dict[str(pts) + '_16s'] = []
                            self.targets_dict2[str(pts) + '_16s'] = []

                        self.week[str(pts) + '_16s'].append(pt_info_dict[ii][pts]['16s'])
                        self.targets_dict[str(pts) + '_16s'].append(pt_info_dict[ii][pts]['PATIENT STATUS (BWH)'])
                        self.targets_dict2[str(pts) + '_16s'].append(pt_info_dict2[ii][pts]['PATIENT STATUS (BWH)'])

        

        for pts in all_pts:
            self.week[pts] = pd.concat([pt_info_dict[i][pts]['DATA']
                                for i in pt_info_dict.keys()
                                if pts in pt_info_dict[i].keys()], 1).T

            indexes = [i + '-1' for i in pt_info_dict.keys() if pts in pt_info_dict[i].keys()]

            self.week[pts].index = indexes

            self.targets_dict[pts] = [pt_info_dict[i][pts]['PATIENT STATUS (BWH)'] 
                                for i in pt_info_dict.keys()
                                if pts in pt_info_dict[i].keys()]

            self.targets_dict2[pts] = [pt_info_dict2[i][pts]['PATIENT STATUS (BWH)']
                                           for i in pt_info_dict2.keys()
                                           if pts in pt_info_dict2[i].keys()]

        
        for k in self.week.keys():
            try:
                self.week[k] = pd.concat(self.week[k], 1).T
            except:
                continue

        days = [sorted(list(pt_info_dict[i].keys()))
                for i in pt_info_dict.keys()]

        N = 0.0
        days = [[day for day in sub if day != N] for sub in days]

        all_data = []
        labels = []
        labels_even = []
        patients = []

        all_data16s = []
        labels16s = []
        labels_even16s = []
        patients16s = []

        all_data16s_only = []
        labels16s_only = []
        labels_even16s_only = []
        patients16s_only = []

        iterable_dict = copy.deepcopy(pt_info_dict)

        for it, i in enumerate(iterable_dict.keys()):

            if pt_info_dict[i] and len(days[it]) > 0:

                labels.extend([pt_info_dict[i][days[it][k]]['PATIENT STATUS (BWH)'] for
                            k in range(len(days[it]))])

                labels_even.extend([pt_info_dict2[i][days[it][k]]['PATIENT STATUS (BWH)'] for k in range(len(days[it]))])

                all_data.append(pd.concat(
                    [pt_info_dict[i][days[it][k]]['DATA'] for k in range(len(days[it]))], 1))

                patients.extend([i]*(len(days[it])))
                if use16s:
                    for k in days[it]:

                        if '16s' in pt_info_dict[i][k].keys() and 'DATA' in pt_info_dict[i][k].keys():
                            to_add = pt_info_dict[i][k]['PATIENT STATUS (BWH)']
                            labels16s.append(to_add)

                            labels_even16s.append(pt_info_dict2[i][k]['PATIENT STATUS (BWH)'])

                            all_data16s.append(
                                pt_info_dict[i][k]['16s'])

                            patients16s.append(i)
                        if '16s' in pt_info_dict[i][k].keys() and 'DATA' in pt_info_dict[i][k].keys():
                            to_add = pt_info_dict[i][k]['PATIENT STATUS (BWH)']
                            labels16s_only.append(to_add)

                            labels_even16s_only.append(pt_info_dict2[i][k]['PATIENT STATUS (BWH)'])

                            all_data16s_only.append(
                                pt_info_dict[i][k]['16s'])

                            patients16s_only.append(i)
            else:
                pt_info_dict.pop(i)

            
        all_data = pd.concat(all_data, 1).T

        vals = [str(patients[j]) + '-' + str(np.concatenate(days)[j]).replace('.0','')
                for j,i in enumerate(all_data.index.values)]
        
        all_data.index = vals

        if use16s:
            all_data16s = pd.concat(all_data16s, 1).T
            all_data16s_only = pd.concat(all_data16s_only, 1).T


        self.targets_dict['all_data'] = labels
        self.targets_dict['all_data_even'] = labels_even
        self.targets_dict['week_one'] = self.targets_dict2[1.0]

        self.week['all_data'] = all_data
        self.week['week_one'] = self.week[1.0]

        if use16s:
            self.targets_dict['all_data_16s'] = labels16s
            self.targets_dict['all_data_even_16s'] = labels_even16s
            self.targets_dict['week_one_16s'] = self.targets_dict2['1.0_16s']

            self.week['all_data_16s'] = all_data16s
            self.week['week_one_16s'] = self.week['1.0_16s']

            self.targets_dict['all_data_16s_only'] = labels16s_only
            self.targets_dict['all_data_even_16s_only'] = labels_even16s_only
            self.targets_dict['week_one_16s_only'] = self.targets_dict2['1.0_16s_only']

            self.week['all_data_16s_only'] = all_data16s_only
            self.week['week_one_16s_only'] = self.week['1.0_16s_only']



        self.patient_numbers = patients
        # self.week['all'] = all_data
        # import pdb; pdb.set_trace()
        self.targets_orig = [pt_info_dict[i][1.0]
                        ['PATIENT STATUS (BWH)'] for i in pt_info_dict.keys()]
        self.targets_all_orig = labels

        temps = [self.cd.cdiff_data.T, self.cd.data16s.T]
        lb = ['','_16s']
        temp_cdiff = temps[0]
        temp_16s = temps[1]
        for i,temp in enumerate(temps):
            tmpts0 = np.array([yy.split('-')[1] for yy in temp.index.values])
            w0 = np.where(tmpts0 != '0')[0]
            w1 = np.where(tmpts0 == '1')[0]

            # for lb in ['','_16s']:
            self.week['all_data_inst' + lb[i]] = temp.iloc[w0,:]
            # ixs_for_inst = self.week['all_data_inst' + lb[i]].index.values
            ixs_for_inst = list(set(self.week['all_data_inst'].index.values).intersection(set(self.week['all_data_inst' + lb[i]].index.values)))
            # ixs_for_inst = list(set(ixs_for_inst).intersection(set(self.week['all_data' + lb[i]].index.values)))
            # import pdb; pdb.set_trace()
            self.week['all_data_inst' + lb[i]] = self.week['all_data_inst' + lb[i]].loc[ixs_for_inst]
            self.targets_dict['all_data_inst' + lb[i]] = [self.cd.targets_dict[xy] for xy in ixs_for_inst]

            ix_dict = {}
            for tm in np.unique(tmpts0):
                w1 = np.where(tmpts0 == tm)[0]
                self.week['w' + tm  + '_inst' + lb[i]] = temp.iloc[w1,:]
                ix_dict[tm] = list(set(self.week['w' + tm  + '_inst'].index.values).intersection(set(self.week['w' + tm  + '_inst' + lb[i]].index.values)))
                # ix_dict[tm] = self.week['w' + tm  + '_inst' + lb[i]].index.values
                # if tm == '1':
                #     ix_dict[tm] = list(set(ix_dict[tm]).intersection(list(set(self.week['week_one' + lb[i]].index.values))))

                self.week['w' + tm  + '_inst' + lb[i]] = self.week['w' + tm  + '_inst' + lb[i]].loc[ix_dict[tm]]
                self.targets_dict['w' + tm  + '_inst' + lb[i]] = [self.cd.targets_dict[xy] for xy in ix_dict[tm]]
                # self.targets_dict['w' + tm  + '_inst' + lb[i]] = self.cd.targets_dict
        # self.week['week_one_inst'] = self.week['w1_inst']
        # self.week['week_one_inst_16s'] = self.week['w1_inst_16s']
        
        self.data_dict_raw = {}
        self.data_dict = copy.deepcopy(self.week)
        self.data_dict_raw_filt = {}
        self.targets_int = {}
        self.data_dict_log = {}
        self.toxin_targets = {}
        self.cdiff_targets = {}
        self.toxin_amt_ng = {}
        self.toxin_amt_ml = {}
        # self.data_dict_bin = {}

        lst = list(self.data_dict.keys())
        for ls in lst:
            if isinstance(ls, str) and '16s' in ls.split('_'):
                logdat = self.filter_vars(self.data_dict[ls], self.targets_dict[ls], var_type = 'total', perc = 5)
                logdat = np.divide(logdat.T, np.sum(logdat,1)).T
                logdat = np.arcsin(logdat)
            else:
                logdat = self.log_transform(self.data_dict[ls])
            
                logdat = self.filter_vars(logdat, self.targets_dict[ls], var_type = 'total', perc = filter_percentage)

            if np.isnan(np.array(logdat)).any():
                import pdb; pdb.set_trace()

            filtdata = self.standardize(logdat, override = True)
            if '16s' in str(ls):
                bin_dat = (filtdata > 0).astype(float)
                self.data_dict[ls + '_bin'] = bin_dat


            self.data_dict_raw[ls] = self.data_dict[ls]
            self.data_dict_log[ls] = logdat
            
            rawfilt = self.filter_vars(
                self.data_dict[ls], self.targets_dict[ls], var_type='total', perc =5)
            if '16s' in str(ls):
                rawfilt = np.divide(rawfilt.T, np.sum(rawfilt,1)).T
            self.data_dict_raw_filt[ls] = rawfilt
            self.targets_int[ls] = (
                np.array(self.targets_dict[ls]) == 'Recur').astype('float')
            self.data_dict[ls] = filtdata

        self.data_dict_cgp1 = {}
        self.data_dict_cgp2 = {}
        for ls in self.data_dict.keys():
            if '16s' not in str(ls):
                xx = self.data_dict[ls].copy(deep = True)
                xx = self.standardize(xx)
                xx = self.add_cgrps(xx,proportions = False)
                self.data_dict_cgp1[ls] = xx

                xx = self.data_dict_raw_filt[ls].copy(deep = True)
                xx = self.make_cgp_proportions(xx)
                xx = self.add_cgrps(xx, proportions = True)
                self.data_dict_cgp2[ls] = xx


        if use16s:
            for dlab in ['', '_bin']:
                to_stack_all = (self.data_dict['all_data'].T[self.data_dict['all_data_16s' + dlab].index.values]).T
                to_stack_1 = (self.data_dict['week_one'].T[self.data_dict['week_one_16s' + dlab].index.values]).T

                to_stack_all = self.standardize(to_stack_all)
                to_stack_1 = self.standardize(to_stack_1)

                self.data_dict['all_data_ALL'+dlab] = pd.DataFrame(
                    np.hstack((to_stack_all, self.standardize(self.data_dict['all_data_16s'+dlab], override = True))), 
                    index=to_stack_all.index.values, 
                    columns=np.concatenate((to_stack_all.columns.values, self.data_dict['all_data_16s'+dlab].columns.values)))
                self.data_dict['week_one_ALL'+dlab] = pd.DataFrame(np.hstack(
                    (to_stack_1, self.standardize(self.data_dict['week_one_16s'+dlab], override = True))),
                    index=to_stack_1.index.values,
                    columns=np.concatenate((to_stack_1.columns.values, self.data_dict['week_one_16s'+dlab].columns.values)))
                self.targets_dict['all_data_ALL'+dlab] = self.targets_dict['all_data_16s']
                self.targets_dict['week_one_ALL'+dlab] = self.targets_dict['week_one_16s']

                self.targets_dict['all_data_16s'+dlab] = self.targets_dict['all_data_16s']
                self.targets_dict['week_one_16s'+dlab] = self.targets_dict['week_one_16s']




        # import pdb; pdb.set_trace()
        self.new_info_dict = new_info_dict
        self.new_info_dict2 = new_info_dict
        self.path = 'figs/'

        lst = list(self.data_dict.keys())
        for ls in lst:
            if 'only' not in str(ls) and not isinstance(ls,float) and '.' not in str(ls):
                # print(ls)
                if 'inst' in str(ls):
                    sa = 0
                else:
                    sa = 1
                toxin_targs = make_toxin_cdiff_targets(self.cd.toxin_data,ls,'Toxin detected', step_ahead=sa)
                cdiff_targs = make_toxin_cdiff_targets(self.cd.toxin_data,ls,'C diff isolate', step_ahead=sa)
                toxin_amt_ng = make_toxin_cdiff_targets(self.cd.toxin_data,ls,'ng/g Total toxinB in stool', step_ahead=sa)
                toxin_amt_ml = make_toxin_cdiff_targets(self.cd.toxin_data,ls,'Avg Interpolated values (ng/ml)', step_ahead=sa)
                
                self.toxin_targets[ls] = np.array(toxin_targs[self.data_dict[ls].index.values]).squeeze()
                self.toxin_amt_ng[ls] = np.array(toxin_amt_ng[self.data_dict[ls].index.values]).squeeze()
                self.toxin_amt_ml[ls] = np.array(toxin_amt_ml[self.data_dict[ls].index.values]).squeeze()
                self.cdiff_targets[ls] = np.array(cdiff_targs[self.data_dict[ls].index.values]).squeeze()


    def log_transform(self, data):
        temp = data.copy()
        temp = temp.replace(0,np.inf)
        return np.log(data + 1)
    

    def standardize(self,x,override = False):
        if not override:
            assert(x.shape[0]<x.shape[1])
        
        dem = np.std(x,0)
        if (dem == 0).any():
            dem = np.where(np.std(x, 0) == 0, 1, np.std(x, 0))
        return (x - np.mean(x, 0))/dem
    # def normalize(self, x):

    def add_cgrps(self,x, proportions = False):
        new_dat = np.zeros((x.shape[0], self.cd.carbon_gps.shape[0]))
        for i,gp in enumerate(self.cd.carbon_gps.index.values):
            metabs = self.cd.carbon_gps.loc[gp].dropna()[1:].tolist()
            metabs = set(metabs).intersection(set(x.columns.values))
            cols = x[metabs]
            c1 = np.sum(cols,1)
            if not proportions: 
                c1 = c1 / len(metabs)
            new_dat[:,i] = list(c1)
        return pd.DataFrame(new_dat, index = x.index.values, columns = self.cd.carbon_gps.index.values)

    def make_cgp_proportions(self,x):
        cgps = np.concatenate([self.cd.carbon_gps.iloc[i,:].dropna().tolist() for i in range(self.cd.carbon_gps.shape[0])])
        cgp_cols = set(x.columns.values).intersection(set(cgps))
        tot_cgp = np.sum(x[cgp_cols], 1)
        proportions = np.divide(x,np.expand_dims(tot_cgp,1))
        return proportions

    def make_metabolome_info_dict(self, metabolites, metabolome_info_dict):
        metabolome_info_dict_2 = {m: metabolome_info_dict[m] for m in metabolites}
        return metabolome_info_dict_2

    def vars(self,data, labels, normalize_data = False):
        if normalize_data:
            data = self.normalize(data)
        # labels = self.targets_dict[dat_type]
        cleared = data[np.array(labels) == 'Cleared']
        recur = data[np.array(labels) == 'Recur']
        within_class_vars = [np.var(cleared, 0), np.var(recur, 0)]
        class_means = [np.mean(cleared, 0), np.mean(cleared, 0)]

        total_mean = np.mean(data, 0)
        between_class_vars = 0
        for i in range(2):
            between_class_vars += (class_means[i] - total_mean)**2

        total_vars = np.std(data, 0)/np.mean(data,0)
        vardict = {'within':within_class_vars,'between':between_class_vars,'total':total_vars}
        return vardict

    def filter_vars(self, data, labels, perc=30, var_type = 'total', normalize_data = False):
        vardict = self.vars(data, labels, normalize_data)
        variances = vardict[var_type]
        variances = variances.replace(np.nan,0)
        
        rm2 = set(np.where(variances > np.percentile(variances, perc))[0])

        temp = data.iloc[:,list(rm2)]
        # import pdb; pdb.set_trace()
        if len(np.where(np.sum(temp,0)==0)[0]) > 0:
            import pdb; pdb.set_trace()
        return data.iloc[:,list(rm2)]

    def summarize(self,metabolome_info_dict,pt_info_dict, title, print_summary = True, metabolites = None):
        cdict = Counter([metabolome_info_dict[i]['SUPER_PATHWAY']
                         for i in metabolome_info_dict.keys()])
                
        if metabolites is not None:
            cdict = Counter([metabolome_info_dict[i]['SUPER_PATHWAY']
                             for i in metabolome_info_dict.keys() if i in set(metabolites)])

        print(cdict)
        D = cdict
        labels = list(D.keys())
        labels.sort()
        values = [D[label] for label in labels]
        plt.rcParams['figure.figsize'] = 7,7
        plt.bar(range(len(D)), values, align='center')
        plt.xticks(range(len(D)), list(labels))
        plt.xticks(rotation=45, ha='right', fontsize=18)
        plt.ylabel('Number of Molecules', fontsize=18)
        plt.yticks(fontsize=18)
        plt.title(title,fontsize = 20)
        plt.tight_layout()
        plt.savefig(self.path + title.replace(' ','_') + 'Counts.png')
        plt.show()
        

        if print_summary:
            pt_status = [[pt_info_dict[pts][i]['PATIENT STATUS (BWH)'] for i in \
                pt_info_dict[pts].keys(
            )][0] for pts in pt_info_dict.keys()]

            well = sum([p == 'Cleared' for p in pt_status])
            sick = sum([p == 'Recur' for p in pt_status])
            print('Pts with CDIFF: ' + str(sick))
            print('Pts asymptomatic: ' + str(well))

    # @profile
    def split_test_train(self, data, labels, perc=.8):

        if isinstance(data.index.values[0], str):
            patients = [int(i.split('-')[0]) for i in data.index.values]
            pdict = {}
            for i,pt in enumerate(patients):
                pdict[pt] = labels[i]
            recur_pts = [pt for pt in pdict.keys() if pdict[pt] == 1]
            cleared_pts = [pt for pt in pdict.keys() if pdict[pt]==0]



            ixtrain0 = np.concatenate(
                (np.random.choice(recur_pts, np.int(len(recur_pts)*perc), replace=False),
                 np.random.choice(cleared_pts, np.int(len(cleared_pts)*perc), replace=False)))
            ixtest0 = np.array(list(set(pdict.keys())- set(ixtrain)))

            ixtrain = np.concatenate([np.where(patients == i)[0] for i in ixtrain0])
            ixtest = np.concatenate(
                [np.where(patients == i)[0] for i in ixtest0])

            set1 = set([patients[ix] for ix in ixtest])
            set2 = set([patients[ix] for ix in ixtrain])
            set1.intersection(set2)
            assert(not set1.intersection(set2))

        else:

            classes = np.unique(labels)
            c1 = np.where(labels == classes[0])[0]
            c2 = np.where(labels == classes[1])[0]
            ixtrain = np.concatenate((np.random.choice(c1, np.int(len(c1)*perc), replace=False),
                                    np.random.choice(c2, np.int(len(c2)*perc), replace=False)))
            ixtest = np.array(list(set(range(len(labels))) - set(ixtrain)))
 

        return ixtrain, ixtest

    def pca_func(self,x,targets,n=2):
        x = (x - np.min(x, 0))/(np.max(x, 0) - np.min(x, 0))
        pca = PCA(n_components=n)
        targets = (np.array(targets) == 'Recur').astype('float')
        if x.shape[0] <= 55:
            title_name = 'Week 1'
        else:
            title_name = 'All'
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data=principalComponents, columns=[
                                'principal component 1', 'principal component 2'])

        variance = pca.explained_variance_ratio_  # calculate variance ratios

        finalDf = pd.concat([principalDf, pd.DataFrame(
            data=np.array(targets), columns=['target'])], axis=1)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1, variance explaned= ' +
                    str(np.round(variance[0], 3)), fontsize=15)
        ax.set_ylabel('Principal Component 2, variance explaned= ' +
                    str(np.round(variance[1], 3)), fontsize=15)
        ax.set_title('2 component PCA, ' + title_name + ' Data', fontsize=20)
        targets = np.unique(targets)
        colors = ['r', 'g', 'b']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                    finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
        ax.legend(targets)
        ax.grid()
        # return fig, ax
        fig.savefig(self.path + title_name.replace(' ','') + 'pca.pdf')


    def rank_sum(self,x,targets,cutoff = .05):
        if isinstance(targets[0],str):
            targets = (np.array(targets) == 'Recur').astype('float')
        else:
            targets = np.array(targets)
        pval = []
        teststat = []
        for i in range(x.shape[1]):
            xin = np.array(x)[:, i]
            X = xin[targets==1]
            Y = xin[targets==0]
            # xin1 = (xin - np.min(xin,0))/(np.max(xin,0)-np.min(xin,0))
            s, p = st.ranksums(X, Y)
            pval.append(p)
            teststat.append(s)
        df = pd.DataFrame(np.vstack((pval, teststat)).T, columns=[
                        'P_Val', 'Test_Stat'], index=x.columns.values)
                
        mols = df.index.values[np.where(np.array(df['P_Val']) < cutoff)[0]]
        # # bonferonni
        bf_cutoff = cutoff / df.shape[0]
        bf_mols = df.index.values[np.where(np.array(df['P_Val']) < bf_cutoff)[0]]

        # # ben-hoch 
        bh_df = df.copy()
        bh_df = bh_df.sort_values('P_Val', ascending=True)
        alphas = (np.arange(1, bh_df.shape[0]+1) / bh_df.shape[0])*cutoff
        out = np.where(bh_df['P_Val'] <= alphas)[0]
        if len(out > 0):
            bh_idx = out[0]
            bh_mols = bh_df['P_Val'].index.values[:bh_idx]
        else:
            bh_mols = []

        # for pv in bh_df['P_Val']:


        return df.sort_values(ascending= True, by = 'P_Val'), mols, bf_mols, bh_mols        
                
        

    def ANOVA_F(self,X=None,targets=None,n=10,features=None):
        if X is None:
            X = self.week_one_norm
        if targets is None:
            targets = self.targets
        if features is not None:
            X = X[features]

        y = self.targets
        bestfeatures = SelectKBest(k=n)
        fit = bestfeatures.fit(X, y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        #concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
        # print(featureScores.nlargest(n, 'Score'))  # print 10 best features
        sklearn_mols = set(featureScores.nlargest(n, 'Score')['Specs'])
        return sklearn_mols

    def decision_trees(self, x, targets, weight = None, n=10, standardize = True, csv_name = None, path = ''):
        if isinstance(targets[0], str):
            targets = (np.array(targets) == 'Recur').astype('float')
        else:
            targets = np.array(targets)
        dem = np.std(x, 0)
        dz = np.where(dem == 0)[0]
        dem[dz] = 1

        ixs = self.leave_one_out_cv(x, targets)
        
        if standardize:
            X = np.array((x - np.mean(x, 0))/dem)
        else:
            X = np.array(x)
        y = targets

        
        if x.shape[0] > 70:
            dattype = 'all_data'
            tmpts = np.array([ix.split('-')[1] for ix in x.index.values])
        else:
            dattype = 'week_one'
            tmpts = np.ones(x.shape[0])
        if weight == 'balanced':
            w = True
        else:
            w = False

        name = 'lr_lambdict_' + str(weight) + '.pkl'

        coefs_all = np.zeros(x.shape[1])
        ts_pred = []
        ts_true = []
        ts_probs = []
        for ix in ixs:
            train_index, test_index = ix
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            tmpts_train = tmpts[train_index]

            
            coefs_all_tr = []
            
            model = {}
            params = dict(n_estimators = np.arange(50,300,50))
            rf = RandomForestClassifier(class_weight = weight)
            # clf = RandomForestClassifier(class_weight = weight).fit(X_train, y_train)
            clf = sklearn.model_selection.GridSearchCV(rf, params).fit(X_train, y_train)
            final_model = clf.best_estimator_
            # for jc, tmpt in enumerate(np.unique(tmpts)):
            #     if dattype == 'all_data':
            #         ix_tmpt = np.where(np.array(tmpts_train)==tmpt)[0]
            #         Xt = X_train[ix_tmpt,:]
            #         yt = y_train[ix_tmpt]
            #         ws = len(yt)/(2* np.bincount([int(x) for x in yt]))
            #         wdict = {}

            #         wdict[0] = ws[0]
            #         try:
            #             wdict[1] = ws[1]
            #         except:
            #             wdict[1] = 0
            #         if weight == 'balanced':
            #             weight_n = wdict
            #         else:
            #             weight_n = weight
            #     else:
            #         Xt = X_train
            #         yt = y_train
            #         weight_n = weight

            #     clf = RandomForestClassifier(class_weight = weight_n).fit(Xt, yt)

            #     model[tmpt] = clf

            # final_model = clf
            # estimators = np.sum([val.estimators_ for val in model.values()])
            # n_estimators = len(sum_model)

            # final_model.estimators_ = estimators
            # final_model.n_estimators = n_estimators
            coefs_all += final_model.feature_importances_.squeeze()

            y_guess = final_model.predict(X_test)

            y_probs = final_model.predict_proba(X_test)
                
            if len(y_test)>1:
                ts_true.extend(y_test.squeeze())
                ts_pred.extend(y_guess.squeeze())
                ts_probs.extend(y_probs[:,1].squeeze())
            else:
                ts_true.append(int(y_test.squeeze()))
                ts_pred.append(int(y_guess.squeeze()))
                ts_probs.append(y_probs[:,1].squeeze())

        coefs_all = coefs_all / len(ixs)
        ts_pred = np.array(ts_pred)
        ts_true = np.array(ts_true)

        ts_probs = np.array(ts_probs)
        tprr = len(set(np.where(ts_pred == 1)[0]).intersection(
            set(np.where(ts_true == 1)[0])))/len(np.where(ts_true == 1)[0])

        fprr = len(set(np.where(ts_pred == 0)[0]).intersection(
            set(np.where(ts_true == 0)[0])))/len(np.where(ts_true == 0)[0])
        bac = (tprr + fprr)/2


        pos = len(np.where(ts_true == 1)[0])
        neg = len(np.where(ts_true == 0)[0])
        tp = len(set(np.where(ts_pred == 1)[0]).intersection(
            set(np.where(ts_true == 1)[0])))
        tn = len(set(np.where(ts_pred == 0)[0]).intersection(
            set(np.where(ts_true == 0)[0])))
        fn = pos - tp
        fp = neg - tn
        arr = [[tp, fn], [fp, tn]]

        f1 = sklearn.metrics.f1_score(ts_true, ts_pred)

        auc_out = sklearn.metrics.roc_curve(ts_true, ts_probs)
        auc_score = sklearn.metrics.roc_auc_score(ts_true, ts_probs)
        fpr = auc_out[0]
        tpr = auc_out[1]
        figg, axx = plt.subplots()
        axx.plot(fpr,tpr)
        axx.set_xlabel('False Positive Rate', fontsize = 20)
        axx.set_ylabel('True Positive Rate', fontsize = 20)
        axx.set_title(auc_score, fontsize = 20)
        plt.savefig('auc_' + csv_name + '.pdf')
        plt.show()

        ixl0 = np.where(coefs_all < 0)[0]
        ixg0 = np.where(coefs_all > 0)[0]

        g0coefs = copy.deepcopy(coefs_all)
        g0coefs[ixl0] = np.zeros(len(ixl0))

        l0coefs = copy.deepcopy(coefs_all)
        l0coefs[ixg0] = np.zeros(len(ixg0))
        ranks_g = np.argsort(-g0coefs)
        mols_g = x.columns.values[ranks_g]
        odds_ratio_g = np.exp(coefs_all[ranks_g])

        ranks_l = np.argsort(l0coefs)
        mols_l = x.columns.values[ranks_l]
        odds_ratio_l = np.exp(coefs_all[ranks_l])
        # import pdb; pdb.set_trace()
        df = np.vstack((mols_l, odds_ratio_l, mols_g, odds_ratio_g)).T
        df = pd.DataFrame(
            df, columns=['Metabolites', 'Odds ratio < 1', 'Metabolites', 'Odds ratio > 1'])
        name2 = 'rf_ ' + csv_name + str(np.round(auc_score,4)).replace('.','_') +'.csv'
        df.to_csv(path + name2)


        return tprr, fprr, bac, pos, neg, tp, tn, fp, fn, f1, auc_score

    def corr_fig(self,X,feats,names):
        # sns.set(font_scale=3)
        label_encoder = LabelEncoder()
        if len(names)>1:
            for i, poss_good in enumerate(feats):
                poss_good = list(poss_good)
                data = pd.DataFrame(
                    np.hstack((np.expand_dims(self.targets, 1), X[poss_good])))
                poss_good.extend(['T'])
                data.columns = poss_good
                data.iloc[:, 0] = label_encoder.fit_transform(
                    data.iloc[:, 0]).astype('float64')
                corrmat = data.corr()
                # top_corr_features = corrmat.index
                plt.figure(figsize=(30, 30))
                #plot heat map
                g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn", center=0)
                fig = g.get_figure()
                plt.title(names[i])

                fig.savefig(self.path + names[i].replace(' ', '_') + "_corr.png")
                plt.show()
        else:
                poss_good = list(feats)
                data = pd.DataFrame(
                    np.hstack((np.expand_dims(self.targets, 1), X[poss_good])))
                poss_good.extend(['T'])
                data.columns = poss_good
                data.iloc[:, 0] = label_encoder.fit_transform(
                    data.iloc[:, 0]).astype('float64')
                corrmat = data.corr()
                # top_corr_features = corrmat.index
                plt.figure(figsize=(30, 30))
                #plot heat map
                g = sns.heatmap(corrmat, annot=True, cmap="RdYlGn", center=0)
                fig = g.get_figure()
                plt.title(names[i])

                fig.savefig(self.path + names[i].replace(' ', '_') + "_corr.png")
                plt.show()

    # @profile
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def make_one_hot(self, a, num_classes):
        a = a.numpy()
        b = torch.zeros((a.shape[0], num_classes))
        b[np.arange(a.shape[0]), a] = 1
        return torch.Tensor(b)

    # @profile
    def leave_one_out_cv(self, data, labels, num_folds = None):

        if data.shape[0]>70:
            # import pdb; pdb.set_trace()
            patients = np.array([int(i.split('-')[0]) for i in data.index.values])
            pdict = {}
            for i, pt in enumerate(patients):
                pdict[pt] = labels[i]


            ix_all = []
            for ii in pdict.keys():
                pt_test = ii
                pt_train = list(set(pdict.keys()) - set([ii]))
                ixtrain = (np.concatenate(
                    [np.where(patients == j)[0] for j in pt_train]))
                ixtest = np.where(patients == pt_test)[0]
                set1 = set([patients[ix] for ix in ixtest])
                set2 = set([patients[ix] for ix in ixtrain])
                set1.intersection(set2)

                ix_all.append((ixtrain, ixtest))
                assert(not set1.intersection(set2))

        else:
            ix_all = []
            # CHANGE LINE!
            for ixs in range(len(labels)):
                ixtest = [ixs]
                ixtrain = list(set(range(len(labels))) - set(ixtest))
                ix_all.append((ixtrain, ixtest))
        return ix_all
    
    # @profile
    def split_to_folds(self, in_data, in_labels, folds):
        # Like split test-train except makes all folds at once

        # If all data, have to take into acct patients w/ multiple timepoints
        in_labels = np.array(in_labels)
        if isinstance(in_data.index.values[0], str):
            data_perc_take = 1/folds
            patients = np.array([int(ti.split('-')[1])
                                for ti in in_data.index.values])
            unique_patients = np.unique(patients)
            pts_to_take = np.copy(unique_patients)
            ix_all = []
            for f in range(folds):
                # patients to test (1/folds * len(data) patients)
                cleared_gp = patients[np.where(in_labels == 0)[0]]
                recur_gp = patients[np.where(in_labels == 1)[0]]

                cleared_gp = set(cleared_gp) - set(recur_gp)
                recur_gp = set(recur_gp)

                pts_take_cl = np.random.choice(
                    list(cleared_gp), int(data_perc_take*len(cleared_gp)))
                pts_take_re = np.random.choice(
                    list(recur_gp), int(data_perc_take*len(recur_gp)))

                pts_take = list(set(pts_take_cl) | set(pts_take_re))

                # train with rest
                pts_train = list(set(unique_patients) - set(pts_take))
                ix_ts = np.concatenate(
                    [np.where(patients == it)[0] for it in pts_take])

                ix_tr = np.concatenate(
                    [np.where(patients == it)[0] for it in pts_train])
                ix_all.append((ix_tr, ix_ts))
            zip_ixs = ix_all
        # If not, can use skf split
        else:
            skf = StratifiedKFold(folds)
            zip_ixs = skf.split(in_data, in_labels)

    # @profile
    def train_loop(self, train_data, train_labels, net, optimizer, criterion, lamb_to_test, regularizer):
        net.train()
        optimizer.zero_grad()
        out = net(train_data).double()

        reg_lambda = lamb_to_test
        l2_reg = None
        for W in net.parameters():
            if l2_reg is None:
                l2_reg = W.norm(regularizer)
            else:
                l2_reg = l2_reg + W.norm(regularizer)
        if len(out.shape) == 1:
            out = out.unsqueeze(0)
        
        loss = criterion(out, self.make_one_hot(train_labels,2)) + reg_lambda * l2_reg
        
        loss.backward()
        optimizer.step()
        return out, loss
    
    def test_loop(self,net, X_test, y_test, criterion):
        net.eval()
        test_out = net(X_test).double()
        if len(test_out.shape) == 1:
            test_out = test_out.unsqueeze(0)

        # find loss
        test_loss = criterion(
            test_out, self.make_one_hot(y_test, 2))

        m = nn.Softmax(dim=1)
        test_out_sig = m(test_out)

        y_guess = test_out_sig.detach().numpy()

        # find f1 score
        test_loss = test_loss.detach().numpy().item()
        return test_out, test_loss, y_guess

    # @profile
    def train_net(self, epochs, labels, data, loo_inner = True, loo_outer = True, folds = 3, regularizer = None, weighting = True, lambda_grid=None, train_inner = True, optimization = 'auc', perc = None, ixs = None, lrate = .001):
        # Inputs:
            # NNet - net to use (uninitialized)
            # epochs - number of epochs for training inner and outer loop
            # data & labels
            # loo - whether or not to use leave one out cross val
            # folds - number of folds for inner cv
            # shuffle - whether to shuffle 
            # regularizer - either 1 for l1, 2 for l2, or None
            # weighting - either true or false
            # lambda_grid - vector of lambdas to train inner fold over or, if train_inner = False, lambda value for outer CV
            # train_inner - whether or not to train inner fold
            # optimization - what metric ('auc','loss','f1') to use for both early stopping and deciding on lambda value. If loo = True, optimization should be 'loss'
            # perc - Remove metabolites with variance in the bottom 'perc' percent if perc is not None
        
        if loo_outer:
            assert(ixs is not None)

        if isinstance(lambda_grid, int):
            train_inner = False

        # Filter variances if perc is not none
        if perc is not None:
            data = self.filter_vars(data, labels, perc=perc)

        # Split data in outer split
        if not loo_outer:
            ixtrain, ixtest = self.split_test_train(data, labels)                                
        else:
            ixtrain, ixtest = ixs

        # Normalize data and fix instances where stdev(data) = 0
        dem = np.std(data, 0)
        dz = np.where(dem == 0)[0]
        dem[dz] = 1
        data = (data - np.mean(data, 0))/dem
        
        TRAIN, TRAIN_L, TEST, TEST_L = data.iloc[ixtrain,
                                                 :], labels[ixtrain], data.iloc[ixtest, :], labels[ixtest]

        if isinstance(ixtest, int):
            TEST, TEST_L = torch.FloatTensor([np.array(TEST)]), torch.DoubleTensor([[TEST_L]])
        else:
            TEST, TEST_L = torch.FloatTensor(
                np.array(TEST)), torch.DoubleTensor(TEST_L)

        # initialize net with TRAIN shape
        net = LogRegNet(TRAIN.shape[1])
        optimizer = torch.optim.RMSprop(net.parameters(), lr=lrate)

        # if we are doing an inner cv:
        if regularizer is not None and train_inner:
            inner_dic = dict()
            
            # split data eiter in 3 folds or leave one out and iterate over the 3 train and test datasets
            if loo_inner:
                # self.leave_one_out_cv always selects a positive example (i.e. recur) as the 1 test subject
                zip_ixs = self.leave_one_out_cv(TRAIN, TRAIN_L)
            else:
                zip_ixs = self.split_to_folds(TRAIN, TRAIN_L, folds)
  
            # iterate over lambda values
            for lamb in lambda_grid:
                # train over epochs for each lambda value

                y_test_vec = []
                y_guess_vec = []
                test_running_loss = 0
                for train_index, test_index in zip_ixs:
                    # initialize net for each new dataset
                    net.apply(self.init_weights)

                    X_train, X_test = TRAIN.iloc[train_index,:], TRAIN.iloc[test_index,:]
                    y_train, y_test = TRAIN_L[train_index], TRAIN_L[test_index]
                    if isinstance(test_index, int):
                        X_train, y_train, X_test, y_test = torch.FloatTensor(np.array(X_train)), torch.DoubleTensor(
                            y_train), torch.FloatTensor([np.array(X_test)]), torch.DoubleTensor([[y_test]])
                    else:
                        X_train, y_train, X_test, y_test = torch.FloatTensor(np.array(X_train)), torch.DoubleTensor(
                            y_train), torch.FloatTensor(np.array(X_test)), torch.DoubleTensor(y_test)
                    
                    if weighting:
                        weights = len(y_train) / (2 * np.bincount(y_train))
                        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights))

                    else:
                        criterion = nn.BCEWithLogitsLoss()

                    y_test_per_epoch = []
                    y_guess_per_epoch = []

                    running_vec = []
                    for epoch in range(epochs):

                        out = self.train_loop(X_train, y_train, net,
                                        optimizer, criterion, lamb, regularizer)
                
                        
                        # evaluate on test set
                        if epoch % 10 ==0:
                            test_out, test_loss, y_guess = self.test_loop(
                                net, X_test, y_test, criterion)
                            
                            y_test_per_epoch.append(y_test)
                            y_guess_per_epoch.append(y_guess)

                            running_vec.append(test_loss)
                            if len(running_vec) > 12:
                                bool_test = np.array([r1 >= r2 for r1, r2 in zip(
                                        running_vec[-10:], running_vec[-11:-1])]).all()
                            # perform early stopping if greater than 50 epochs and if either loss is increasing over the past 10 iterations or auc / f1 is decreasing over the past 10 iterations
                            if (len(running_vec) > 12 and bool_test):
                                y_test_vec.append(y_test_per_epoch[-11])
                                y_guess_vec.append(y_guess_per_epoch[-11])
                                test_running_loss += test_loss
                                # add record of lambda and lowest loss or highest auc/f1 associated with that lambda at this epoch
                                break

                if len(y_test_vec) ==1:
                    y_test_vec.append(y_test_per_epoch[-11])
                    y_guess_vec.append(y_guess_per_epoch[-11])
                    test_running_loss += test_loss
                y_guess_mat = np.concatenate(y_guess_vec)
                y_pred_mat = np.argmax(y_guess_mat, 1)
                if len(y_test_vec) < y_guess_mat.shape[0]:
                    y_test_vec = np.concatenate(y_test_vec)
                f1 = sklearn.metrics.f1_score(y_test_vec,y_pred_mat)
                try:
                    fpr, tpr, _ = roc_curve(y_test_vec, y_guess_mat[:, 1].squeeze())
                except:
                    import pdb; pdb.set_trace()

                roc_auc = auc(fpr, tpr)

                inner_dic[lamb] = {}
                inner_dic[lamb]['auc'] = roc_auc
                inner_dic[lamb]['f1'] = f1
                inner_dic[lamb]['loss'] = test_running_loss / (len(TRAIN_L)+1)       
                
            # find the best lambda over all splits
            if optimization == 'loss':
                max_val = np.min([inner_dic[it][optimization]
                                  for it in inner_dic.keys()])
            else:
                max_val = np.max([inner_dic[it][optimization] for it in inner_dic.keys()])
            best_lambda = [inner_dic[k][optimization] for k in inner_dic.keys() if inner_dic[k][optimization] == max_val]
            # print('Best Lambda = ' + str(best_lambda) + ', For weight ' + str(weighting) + 'and l'+ str(regularizer))
            best_lambda = np.median(best_lambda)
        else:
            if regularizer is None:
                best_lambda = 0
            elif regularizer is not None and not train_inner:
                assert(isinstance(lambda_grid, float))
                best_lambda = lambda_grid
            inner_dic = None
        
        # Now, train outer loop
        TRAIN = torch.FloatTensor(np.array(TRAIN))
        TRAIN_L = torch.DoubleTensor(np.array(TRAIN_L))
        if weighting:
            weights = len(TRAIN_L) / (2 * np.bincount(TRAIN_L))
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(weights))

        else:
            criterion = nn.BCEWithLogitsLoss()
                
        
        y_guess = []
        test_running_loss = []
        net_vec = []
        net.apply(self.init_weights)
        for epoch in range(epochs):
            out = self.train_loop(TRAIN, TRAIN_L, net,
                                  optimizer, criterion, best_lambda, regularizer)
            
            # And test outer loop
            net.eval()
            test_out = net(TEST).double()

            # calculate loss
            try:
                test_loss = criterion(test_out, self.make_one_hot(TEST_L,2))
            except:
                test_out = test_out.unsqueeze(0)
                test_loss = criterion(test_out, self.make_one_hot(TEST_L, 2))
            mm = nn.Softmax(dim=1)
            test_out_sig = mm(test_out)
            y_guess.append(test_out_sig.detach().numpy())

            test_running_loss.append(test_loss.item())
            net_vec.append(net)

            y_pred = np.argmax(y_guess[-1],1)
            
            running_vec = test_running_loss
            bool_test = np.array([r1 >= r2 for r1, r2 in zip(
                    running_vec[-10:], running_vec[-11:-1])]).all()
            
            if epoch > 50 and bool_test:
                break

        # record best net & y_guess
        net_out = net_vec[-11]
        y_guess_fin = y_guess[-11]
        y_true = TEST_L

        return inner_dic, y_guess_fin, y_true, net_out, best_lambda, running_vec
    
    def log_reg(self, x, targets, lambda_grid = np.logspace(-3, 3, 50), features=None, weight= None, 
                regularizer = 'l1', solve = 'liblinear', maxiter = 100, plot_lambdas = True, 
                plot_cf = True, use_old = False, load_old = False, optim_param = 'loss', standardize = True, csv_name = 'default',
                path = '', plot_convergence = False, tol = 0.0001, use_sgd = False, nzero_thresh = 5, stop_lambdas = True, print_coefs = True,
                seed = 0, final_res_dict = {}):

        # random.seed(seed)
        np.random.seed(seed)
        if csv_name not in final_res_dict.keys():
            final_res_dict[csv_name] = {}
        if seed not in final_res_dict[csv_name].keys():
            final_res_dict[csv_name][seed] = {}
        if isinstance(targets[0], str):
            targets = (np.array(targets) == 'Recur').astype('float')
        else:
            targets = np.array(targets)
        dem = np.std(x, 0)
        dz = np.where(dem == 0)[0]
        dem[dz] = 1

        ixs = self.leave_one_out_cv(x, targets)
        
        if standardize:
            X = np.array((x - np.mean(x, 0))/dem)
        else:
            X = np.array(x)
        y = targets

        
        if x.shape[0] > 55:
            dattype = 'all_data'
            tmpts = np.array([ix.split('-')[1] for ix in x.index.values])
        else:
            dattype = 'week_one'
            tmpts = np.ones(x.shape[0])
        if weight == 'balanced':
            w = True
        else:
            w = False
        name_old = 'outputs_june20/outputs_june18/' + dattype + '_' + \
            str(w) + '_' + str(regularizer)[1] + 'inner_dic.pkl'
        # import pdb; pdb.set_trace()
        name = 'lr_lambdict_' + str(w) + str(regularizer) + '_' + str(weight) + '.pkl'
        # print(regularizer)

        # if use_grps:
        #     temp = np.divide(X,np.sum(X,1),axis = 0)
        coef_dict = []
        if regularizer != 'none' and not isinstance(lambda_grid,float):
            if load_old:
                with open(name, 'rb') as f:
                    lambdict = pickle.load(f)
            if use_old:
                with open(name_old, 'rb') as f:
                    lambdict = pickle.load(f)
                # print('using old')
                # print(name_old)
            else:
                lambdict = {}
                for lamb in lambda_grid:
                    lambdict[lamb] = {}

                    ts_true = []
                    ts_pred = []
                    loss_vec = []
                    ts_probs = []

                    num_coefs = []
                    for ic,ix in enumerate(ixs):
                        train_index, test_index = ix
                        X_train, X_test = X[train_index, :], X[test_index, :]
                        y_train, y_test = y[train_index], y[test_index]

                        tmpts_train = tmpts[train_index]

                        coefs_all = []
                        samp_weights = np.ones(len(y_train))
                        if dattype == 'all_data':
                            weight_n = None
                            for tmpt in np.unique(tmpts):
                                ix_tmpt = np.where(np.array(tmpts_train)==tmpt)[0]
                                # Xt = X_train[ix_tmpt,:]
                                yt = y_train[ix_tmpt]

                                ones_ix = np.where(yt == 1)[0]
                                zeros_ix = np.where(yt == 0)[0]

                                ws = len(yt)/(2* np.bincount([int(x) for x in yt]))

                                if len(ws) == 1:
                                    samp_weights[ix_tmpt] = 0
                                else:
                                    samp_weights[ix_tmpt[ones_ix]] = ws[1]
                                    samp_weights[ix_tmpt[zeros_ix]] = ws[0]

                        else:
                            weight_n = weight
                            
                        
                        if regularizer == 'l1':
                            if use_sgd:
                                clf = SGDClassifier(loss = 'log', class_weight=weight_n, penalty=regularizer,
                                                            max_iter=maxiter, alpha = lamb, tol = tol).fit(X_train, y_train, sample_weight = samp_weights)
                            else:
                                clf = LogisticRegression(solver = 'liblinear', class_weight=weight_n, penalty=regularizer,
                                            max_iter=maxiter, C = 1/lamb, verbose = 0, tol = tol, warm_start=False, random_state = seed).fit(X_train, y_train, sample_weight = samp_weights)   
        
                        else:
                            if use_sgd:
                                clf = SGDClassifier(loss = 'log', class_weight=weight_n, 
                                                        max_iter=maxiter, alpha = lamb, tol = tol, warm_start=False).fit(X_train, y_train, sample_weight = samp_weights)
                            else:
                                clf = LogisticRegression(solver = 'lbfgs', class_weight=weight_n, penalty=regularizer,
                                            max_iter=maxiter, C = 1/lamb, verbose = 0, tol = tol, random_state = seed).fit(X_train, y_train, sample_weight = samp_weights)   
                        
                        coef_ind = clf.coef_ > 0
                        num_coefs.append(np.sum(coef_ind))


                        try:
                            y_guess = clf.predict(X_test)
                            y_probs = clf.predict_proba(X_test)
                        except:
                            X_test = X_test.T
                            y_guess = clf.predict(X_test)

                        y_probs = clf.predict_proba(X_test)
                        if len(y_test)>1:
                            ts_true.extend(y_test.squeeze())
                            ts_pred.extend(y_guess.squeeze())
                            ts_probs.extend(y_probs[:,1].squeeze())
                        else:
                            ts_true.append(int(y_test.squeeze()))
                            ts_pred.append(int(y_guess.squeeze()))
                            ts_probs.append(y_probs[:,1].squeeze())
  
                        loss = (-y_test * np.log(y_probs[:, 1]) - (1-y_test)*np.log(1-y_probs[:, 1]))
                        if loss.shape[0]>1:
                            loss = (np.sum(loss)/loss.shape[0]).item()
                        else:
                            loss = loss.item()
                        if np.isnan(loss):
                            try:
                                loss = 1 if y_test.item() == y_probs[:,1].item() else 0
                            except:
                                loss = np.sum(1-(y_test == y_probs[:, 1]).astype(int))/len(y_test)
                        if np.isinf(loss):
                            loss = 99
                        # import pdb; pdb.set_trace()
                        loss_vec.append(loss)
                        # assert(y_guess.item() == np.round(y_probs[:,1].item()))

                        
                    if np.mean(num_coefs)< nzero_thresh and stop_lambdas:
                        del lambdict[lamb]
                        continue
                    ts_pred = np.array(ts_pred)
                    ts_true = np.array(ts_true)
                    tprr = len(set(np.where(ts_pred == 1)[0]).intersection(
                        set(np.where(ts_true == 1)[0])))/len(np.where(ts_true == 1)[0])
                    fprr = len(set(np.where(ts_pred == 0)[0]).intersection(
                        set(np.where(ts_true == 0)[0])))/len(np.where(ts_true == 0)[0])
                    bac = (tprr + fprr)/2



                    pos = len(np.where(ts_true == 1)[0])
                    neg = len(np.where(ts_true == 0)[0])
                    tp = len(set(np.where(ts_pred == 1)[0]).intersection(
                        set(np.where(ts_true == 1)[0])))
                    tn = len(set(np.where(ts_pred == 0)[0]).intersection(
                        set(np.where(ts_true == 0)[0])))
                    fn = pos - tp
                    fp = neg - tn
                    arr = [[tp, fn], [fp, tn]]

                    auc_score = sklearn.metrics.roc_auc_score(ts_true, ts_probs)

                    # mols = self.data_dict[1.0].columns.values[ranks].squeeze()
                    lambdict[lamb]['bac'] = bac
                    lambdict[lamb]['tpr'] = tprr
                    lambdict[lamb]['fpr'] = fprr
                    lambdict[lamb]['cv'] = arr
                    lambdict[lamb]['auc'] = auc_score
                    lambdict[lamb]['loss'] = np.sum(loss_vec)/len(loss_vec)
                    # import pdb; pdb.set_trace()
            lambdas = list(lambdict.keys())
            if use_old:
                key = 'loss'
            else:
                key = optim_param
            vec = np.array([lambdict[it][key] for it in lambdas])
            nn= 5
            ma = moving_average(vec, n=nn)
            offset = int(np.floor(nn/2))
            if key == 'loss':
                max_ix  = np.argmin(ma) + offset
            else:
                max_ix = np.argmax(ma) + offset
            best_lambda = np.array(lambdas)[max_ix]

            # best_lambda = [k for k in lambdas if lambdict[k][key] == max_val]
            # print('Best Lambda = ' + str(best_lambda) + ', For weight ' + str(weighting) + 'and l'+ str(regularizer))
            best_lambda = np.min(best_lambda)
            print(' Best Lambda = ' + str(best_lambda) + ', For weight ' + str(weight) + 'and l'+ str(regularizer))
        else:
            if regularizer !='none':
                best_lambda = lambda_grid
            else:
                best_lambda = 1


        if plot_lambdas:
            fig2, ax2 = plt.subplots()
            fig2.suptitle('Weight ' + str(weight) +
                        ', regularization l' + str(regularizer))

            ax2.plot(np.array(list(lambdict.keys()))[offset:-offset], ma)
            ax2.axvline(x=best_lambda)
            for ij, k in enumerate(lambdict.keys()):
                ax2.scatter([k],
                            [lambdict[k][optim_param]])
                ax2.set_xlabel('lambda values')
                
                
                ax2.set_ylabel(optim_param)
                ax2.set_xscale('log')
                # ax2.set_title('Outer Fold ' + str(ol), fontsize=30)
                if optim_param == 'bac':
                    ax2.set_ylim(0, 1)
        ts_true = []
        ts_pred = []
        loss_vec = []
        ts_probs= []
        coefs_all = np.zeros(x.shape[1])
        ite = 0
        if plot_convergence:
            fig_c, ax_c = plt.subplots(3, len(np.unique(tmpts)))
        for ic, ix in enumerate(ixs):
            train_index, test_index = ix
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            tmpts_train = tmpts[train_index]

            datpts_train,datpts_test = x.index.values[train_index],x.index.values[test_index]
            
            np.random.seed(seed)
            tmpts_train = tmpts[train_index]

            samp_weights = np.ones(len(y_train))
            if dattype == 'all_data':
                weight_n = None
                for tmpt in np.unique(tmpts):
                    ix_tmpt = np.where(np.array(tmpts_train)==tmpt)[0]
                    # Xt = X_train[ix_tmpt,:]
                    yt = y_train[ix_tmpt]

                    ones_ix = np.where(yt == 1)[0]
                    zeros_ix = np.where(yt == 0)[0]

                    ws = len(yt)/(2* np.bincount([int(x) for x in yt]))
                    if len(ws) == 1:
                        samp_weights[ix_tmpt] = 0
                    else:
                        samp_weights[ix_tmpt[ones_ix]] = ws[1]
                        samp_weights[ix_tmpt[zeros_ix]] = ws[0]

            else:
                weight_n = weight

            try:
                if regularizer == 'l1':
                    if use_sgd:
                        clf = SGDClassifier(loss = 'log', class_weight=weight_n, penalty=regularizer,
                                            max_iter=maxiter, alpha = best_lambda, verbose = 0, tol = tol).fit(X_train, y_train)
                    else:
                        clf = LogisticRegression(solver = 'liblinear', class_weight=weight_n, penalty=regularizer,
                                            max_iter=maxiter, C = 1/best_lambda, verbose = 0, tol = tol, random_state = seed).fit(X_train, y_train)          
                else:
                    if use_sgd:
                        clf = SGDClassifier(loss = 'log', class_weight=weight_n, 
                                                max_iter=maxiter, alpha = best_lambda, verbose = 0, tol = tol).fit(X_train, y_train)
                    else:
                        clf = LogisticRegression(solver = 'lbfgs', class_weight=weight_n, penalty=regularizer,
                                            max_iter=maxiter, C = 1/best_lambda, verbose = 0, tol = tol, random_state = seed).fit(X_train, y_train)   
            except:

                continue

            coefs_all += clf.coef_.squeeze()
            coefs_dict[ic] = clf.coef_
            
            if plot_convergence and ic < 3:
                sys.stdout = old_stdout
                loss_history = mystdout.getvalue()
                loss_list = []
                for line in loss_history.split('\n'):
                    if(len(line.split("loss: ")) == 1):
                        continue
                    loss_list.append(float(line.split("loss: ")[-1]))
                
                ax_c[ic, jc].plot(np.arange(len(loss_list)), loss_list)
                ax_c[ic, jc].set_xlabel("Time in epochs")
                ax_c[ic, jc].set_ylabel("Loss")


            if x.shape[0] < 60:
                X_test = X_test.reshape(1, -1)
                y_test = y_test.reshape(1, -1)

            if len(X_test.shape) == 1:
                X_test = np.expand_dims(X_test,1)
            
            y_guess = clf.predict(X_test)

            y_probs = clf.predict_proba(X_test)
                
            loss = (-y_test*np.log(y_probs[:, 1]) - \
                (1-y_test)*np.log(1-y_probs[:, 1]))
            if loss.shape[0]>1:
                loss = (np.sum(loss)/loss.shape[0]).item()
            else:
                try:
                    loss = loss.item()
                except:
                    import pdb; pdb.set_trace()
            if np.isnan(loss):
                try:
                    loss = 0 if y_test.item() == y_probs[:, 1].item() else 1
                except:
                    if (y_test == y_probs[:,1]).all():
                        loss = 0
                    else:
                        loss = 1

            loss_vec.append(loss)
            if len(y_test)>1:
                ts_true.extend(y_test.squeeze())
                ts_pred.extend(y_guess.squeeze())
                ts_probs.extend(y_probs[:,1].squeeze())
            else:
                ts_true.append(int(y_test.squeeze()))
                ts_pred.append(int(y_guess.squeeze()))
                ts_probs.append(y_probs[:,1].squeeze())
            
            ite += 1

        ts_pred = np.array(ts_pred)
        ts_true = np.array(ts_true)

        ts_probs = np.array(ts_probs)
        tprr = len(set(np.where(ts_pred == 1)[0]).intersection(
            set(np.where(ts_true == 1)[0])))/len(np.where(ts_true == 1)[0])

        fprr = len(set(np.where(ts_pred == 0)[0]).intersection(
            set(np.where(ts_true == 0)[0])))/len(np.where(ts_true == 0)[0])
        bac = (tprr + fprr)/2


        pos = len(np.where(ts_true == 1)[0])
        neg = len(np.where(ts_true == 0)[0])
        tp = len(set(np.where(ts_pred == 1)[0]).intersection(
            set(np.where(ts_true == 1)[0])))
        tn = len(set(np.where(ts_pred == 0)[0]).intersection(
            set(np.where(ts_true == 0)[0])))
        fn = pos - tp
        fp = neg - tn
        arr = [[tp, fn], [fp, tn]]

        f1 = sklearn.metrics.f1_score(ts_true, ts_pred)

        plt.show()

        if plot_cf:
            fig3, ax3 = plt.subplots()
            df_cm = pd.DataFrame(arr, index=['Actual Recur', 'Actual Asymptomatic'], columns=[
                                 'Predicted Recur', 'Predicted Asymptomatic'])
            # sns.set(font_scale=1.4)  # for label size
            chart = sns.heatmap(df_cm, annot=True, annot_kws={
                               "size": 24})  # font size
            ax3.set_yticklabels(
                ['Actual Recur', 'Actual Asymptomatic'], rotation=45)
            ax3.xaxis.tick_top()
            ax3.xaxis.set_label_position('top')
            ax3.tick_params(length=0)
            plt.title('Weight ' + str(weight) +
                      ', regularization ' + str(regularizer))
            plt.show()

        auc_out = sklearn.metrics.roc_curve(ts_true, ts_probs)
        auc_score = sklearn.metrics.roc_auc_score(ts_true, ts_probs)
        fpr = auc_out[0]
        tpr = auc_out[1]
        figg, axx = plt.subplots()
        axx.plot(fpr,tpr)
        axx.set_xlabel('False Positive Rate', fontsize = 20)
        axx.set_ylabel('True Positive Rate', fontsize = 20)
        axx.set_title(auc_score, fontsize = 20)
        plt.savefig('auc_outputs/auc_' + csv_name + '.pdf')
        plt.show()

        if print_coefs:
            coefs_all = coefs_all / len(ixs)
            coefs_all = coefs_all.squeeze()

            ixl0 = np.where(coefs_all < 0)[0]
            ixg0 = np.where(coefs_all > 0)[0]

            g0coefs = copy.deepcopy(coefs_all)
            g0coefs[ixl0] = np.zeros(len(ixl0))

            l0coefs = copy.deepcopy(coefs_all)
            l0coefs[ixg0] = np.zeros(len(ixg0))
            ranks_g = np.argsort(-g0coefs)
            mols_g = x.columns.values[ranks_g]
            odds_ratio_g = np.exp(coefs_all[ranks_g])

            ranks_l = np.argsort(l0coefs)
            mols_l = x.columns.values[ranks_l]
            odds_ratio_l = np.exp(coefs_all[ranks_l])
            # import pdb; pdb.set_trace()
            df = np.vstack((mols_l, odds_ratio_l, mols_g, odds_ratio_g)).T
            df = pd.DataFrame(
                df, columns=['Metabolites', 'Odds ratio < 1', 'Metabolites', 'Odds ratio > 1'])
            name2 = 'lr_ ' + csv_name + str(np.round(auc_score,4)).replace('.','_') +'.csv'
            df.to_csv(path + name2)

        key = optim_param
        final_res_dict[csv_name][seed]['optimizer'] = key
        final_res_dict[csv_name][seed]['best_lambda'] = best_lambda
        final_res_dict[csv_name][seed]['auc'] = auc_score
        final_res_dict[csv_name][seed]['tpr'] = tpr
        final_res_dict[csv_name][seed]['fpr'] = fpr
        final_res_dict[csv_name][seed]['df_name'] = path + name2
        final_res_dict[csv_name][seed]['df_name'] = coefs_dict
        # ranks = np.argsort(-coefs_all)
        # mols = x.columns.values[ranks]
        # odds_ratio = np.exp(coefs_all[ranks])
        # df = np.vstack((mols, odds_ratio)).T
        # df = pd.DataFrame(df, columns = ['Metabolites', 'Odds ratio'])
        # name2 = 'lr_metabs_' + str(regularizer) + '_' + str(weight) + '.csv'
        # df.to_csv(name2)
        return tprr, fprr, bac, pos, neg, tp, tn, fp, fn, f1, auc_score, final_res_dict

    def log_reg_n4(self, x, targets, lambda_grid = np.logspace(-3, 3, 50), features=None, weight= None, 
                regularizer = 'l1', solve = 'liblinear', maxiter = 100, plot_lambdas = True, 
                plot_cf = True, use_old = False, load_old = False, optim_param = 'loss', standardize = True, csv_name = 'default',
                path = '', plot_convergence = False, tol = 0.0001, use_sgd = False, nzero_thresh = 5, stop_lambdas = True, print_coefs = True,
                seed = None, final_res_dict = {}):

        if csv_name not in final_res_dict.keys():
            final_res_dict[csv_name] = {}

        if seed not in final_res_dict[csv_name].keys():
            final_res_dict[csv_name][seed] = {}

        if isinstance(targets[0], str):
            targets = (np.array(targets) == 'Recur').astype('float')
        else:
            targets = np.array(targets)
        dem = np.std(x, 0)
        dz = np.where(dem == 0)[0]
        dem[dz] = 1

        ixs= self.leave_one_out_cv(x, targets) 
        
        if standardize:
            X = np.array((x - np.mean(x, 0))/dem)
        else:
            X = np.array(x)
        y = targets

        
        if x.shape[0] > 55:
            dattype = 'all_data'
            tmpts = np.array([ix.split('-')[1] for ix in x.index.values])
        else:
            dattype = 'week_one'
            tmpts = np.ones(x.shape[0])
        if weight == 'balanced':
            w = True
        else:
            w = False
        name_old = 'outputs_june20/outputs_june18/' + dattype + '_' + \
            str(w) + '_' + str(regularizer)[1] + 'inner_dic.pkl'
        # import pdb; pdb.set_trace()
        name = 'lr_lambdict_' + str(w) + str(regularizer) + '_' + str(weight) + '.pkl'
        # print(regularizer)

        # if use_grps:
        #     temp = np.divide(X,np.sum(X,1),axis = 0)
        ts_true = []
        ts_pred = []
        loss_vec = []
        ts_probs= []
        coefs_all = np.zeros(x.shape[1])
        coef_dict = {}
        ite = 0
        if plot_convergence:
            fig_c, ax_c = plt.subplots(3, len(np.unique(tmpts)))
        for ic, ix in enumerate(ixs):
            train_index, test_index = ix
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            Xt_pd = x.iloc[train_index,:]

            if regularizer != 'none' and not isinstance(lambda_grid,float):
                ixs_inner = self.leave_one_out_cv(Xt_pd, y_train) 
                if load_old:
                    with open(name, 'rb') as f:
                        lambdict = pickle.load(f)
                if use_old:
                    with open(name_old, 'rb') as f:
                        lambdict = pickle.load(f)
                    # print('using old')
                    # print(name_old)
                else:
                    lambdict = {}
                    for lamb in lambda_grid:
                        lambdict[lamb] = {}

                        ts_true_in = []
                        ts_pred_in = []
                        loss_vec_in = []
                        ts_probs_in = []

                        num_coefs = []
                        for ix_in in ixs_inner:
                            train_index_in, test_index_in = ix_in
                            X_train_in, X_test_in = X_train[train_index_in, :], X_train[test_index_in, :]
                            y_train_in, y_test_in = y_train[train_index_in], y_train[test_index_in]

                            tmpts_train_in = tmpts[train_index_in]

                            coefs_all_in = []
                            samp_weights = np.ones(len(y_train_in))
                            if dattype == 'all_data':
                                weight_n = None
                                for tmpt in np.unique(tmpts):
                                    ix_tmpt = np.where(np.array(tmpts_train_in)==tmpt)[0]
                                    # Xt = X_train[ix_tmpt,:]
                                    yt = y_train_in[ix_tmpt]

                                    ones_ix = np.where(yt == 1)[0]
                                    zeros_ix = np.where(yt == 0)[0]

                                    ws = len(yt)/(2* np.bincount([int(x) for x in yt]))

                                    if len(ws) == 1:
                                        samp_weights[ix_tmpt] = 0
                                    elif len(ws) == 2:
                                        samp_weights[ix_tmpt[ones_ix]] = ws[1]
                                        samp_weights[ix_tmpt[zeros_ix]] = ws[0]
                                    else:
                                        continue

                            else:
                                weight_n = weight
                                
                            
                            if regularizer == 'l1':
                                if use_sgd:
                                    clf = SGDClassifier(loss = 'log', class_weight=weight_n, penalty=regularizer,
                                                                max_iter=maxiter, alpha = lamb, tol = tol).fit(X_train_in, y_train_in, sample_weight = samp_weights)
                                else:
                                    clf = LogisticRegression(solver = 'liblinear', class_weight=weight_n, penalty=regularizer,
                                                max_iter=maxiter, C = 1/lamb, verbose = 0, tol = tol, warm_start=True, random_state = seed).fit(X_train_in, y_train_in, sample_weight = samp_weights)   
            
                            else:
                                if use_sgd:
                                    clf = SGDClassifier(loss = 'log', class_weight=weight_n, 
                                                            max_iter=maxiter, alpha = lamb, tol = tol, warm_start=True).fit(X_train_in, y_train_in, sample_weight = samp_weights)
                                else:
                                    clf = LogisticRegression(solver = 'lbfgs', class_weight=weight_n, penalty=regularizer,
                                                max_iter=maxiter, C = 1/lamb, verbose = 0, tol = tol, random_state = seed).fit(X_train_in, y_train_in, sample_weight = samp_weights)   
                            
                            coef_ind = clf.coef_ > 0
                            num_coefs.append(np.sum(coef_ind))


                            try:
                                y_guess = clf.predict(X_test_in)
                                y_probs = clf.predict_proba(X_test_in)
                            except:
                                X_test_in = X_test_in.T
                                y_guess = clf.predict(X_test_in)

                            y_probs = clf.predict_proba(X_test_in)
                            if len(y_test_in)>1:
                                ts_true_in.extend(y_test_in.squeeze())
                                ts_pred_in.extend(y_guess.squeeze())
                                ts_probs_in.extend(y_probs[:,1].squeeze())
       
                            else:
                                ts_true_in.append(int(y_test_in.squeeze()))
                                ts_pred_in.append(int(y_guess.squeeze()))
                                ts_probs_in.append(y_probs[:,1].squeeze())
    
                            loss = (-y_test_in * np.log(y_probs[:, 1]) - (1-y_test_in)*np.log(1-y_probs[:, 1]))
                            if loss.shape[0]>1:
                                loss = (np.sum(loss)/loss.shape[0]).item()
                            else:
                                loss = loss.item()
                            if np.isnan(loss):
                                try:
                                    loss = 1 if y_test_in.item() == y_probs[:,1].item() else 0
                                except:
                                    loss = np.sum(1-(y_test_in == y_probs[:, 1]).astype(int))/len(y_test_in)
                            if np.isinf(loss):
                                loss = 99
                            # import pdb; pdb.set_trace()
                            loss_vec_in.append(loss)
                            # assert(y_guess.item() == np.round(y_probs[:,1].item()))

                            
                        if np.mean(num_coefs)< nzero_thresh and stop_lambdas:
                            del lambdict[lamb]
                            continue
                        ts_pred_in = np.array(ts_pred_in)
                        ts_true_in = np.array(ts_true_in)
                        tprr = len(set(np.where(ts_pred_in == 1)[0]).intersection(
                            set(np.where(ts_true_in == 1)[0])))/len(np.where(ts_true_in == 1)[0])
                        fprr = len(set(np.where(ts_pred_in== 0)[0]).intersection(
                            set(np.where(ts_true_in == 0)[0])))/len(np.where(ts_true_in == 0)[0])
                        bac = (tprr + fprr)/2



                        pos = len(np.where(ts_true_in == 1)[0])
                        neg = len(np.where(ts_true_in == 0)[0])
                        tp = len(set(np.where(ts_pred_in == 1)[0]).intersection(
                            set(np.where(ts_true_in == 1)[0])))
                        tn = len(set(np.where(ts_pred_in == 0)[0]).intersection(
                            set(np.where(ts_true_in == 0)[0])))
                        fn = pos - tp
                        fp = neg - tn
                        arr = [[tp, fn], [fp, tn]]

                        auc_score = sklearn.metrics.roc_auc_score(ts_true_in, ts_probs_in)

                        # mols = self.data_dict[1.0].columns.values[ranks].squeeze()
                        lambdict[lamb]['bac'] = bac
                        lambdict[lamb]['tpr'] = tprr
                        lambdict[lamb]['fpr'] = fprr
                        lambdict[lamb]['cv'] = arr
                        lambdict[lamb]['auc'] = auc_score
                        lambdict[lamb]['loss'] = np.sum(loss_vec_in)/len(loss_vec_in)
                        # import pdb; pdb.set_trace()
                lambdas = list(lambdict.keys())
                if use_old:
                    key = 'loss'
                else:
                    key = optim_param
                vec = np.array([lambdict[it][key] for it in lambdas])
                nn= 5
                ma = moving_average(vec, n=nn)
                offset = int(np.floor(nn/2))
                if key == 'loss':
                    max_ix  = np.argmin(ma) + offset
                else:
                    max_ix = np.argmax(ma) + offset
                best_lambda = np.array(lambdas)[max_ix]

                # best_lambda = [k for k in lambdas if lambdict[k][key] == max_val]
                # print('Best Lambda = ' + str(best_lambda) + ', For weight ' + str(weighting) + 'and l'+ str(regularizer))
                best_lambda = np.min(best_lambda)
                print(' Best Lambda = ' + str(best_lambda) + ', For weight ' + str(weight) + 'and l'+ str(regularizer))
            else:
                if regularizer !='none':
                    best_lambda = lambda_grid
                else:
                    best_lambda = 1


            if plot_lambdas:
                fig2, ax2 = plt.subplots()
                fig2.suptitle('Weight ' + str(weight) +
                            ', regularization l' + str(regularizer))

                ax2.plot(np.array(list(lambdict.keys()))[offset:-offset], ma)
                ax2.axvline(x=best_lambda)
                for ij, k in enumerate(lambdict.keys()):
                    ax2.scatter([k],
                                [lambdict[k][optim_param]])
                    ax2.set_xlabel('lambda values')
                    
                    
                    ax2.set_ylabel(optim_param)
                    ax2.set_xscale('log')
                    # ax2.set_title('Outer Fold ' + str(ol), fontsize=30)
                    if optim_param == 'bac':
                        ax2.set_ylim(0, 1)


            tmpts_train = tmpts[train_index]

            datpts_train,datpts_test = x.index.values[train_index],x.index.values[test_index]
            
            np.random.seed(0)
            tmpts_train = tmpts[train_index]

            samp_weights = np.ones(len(y_train))
            if dattype == 'all_data':
                weight_n = None
                for tmpt in np.unique(tmpts):
                    ix_tmpt = np.where(np.array(tmpts_train)==tmpt)[0]
                    # Xt = X_train[ix_tmpt,:]
                    yt = y_train[ix_tmpt]

                    ones_ix = np.where(yt == 1)[0]
                    zeros_ix = np.where(yt == 0)[0]

                    ws = len(yt)/(2* np.bincount([int(x) for x in yt]))
                    if len(ws) == 1:
                        samp_weights[ix_tmpt] = 0
                    else:
                        samp_weights[ix_tmpt[ones_ix]] = ws[1]
                        samp_weights[ix_tmpt[zeros_ix]] = ws[0]

            else:
                weight_n = weight

            try:
                if regularizer == 'l1':
                    if use_sgd:
                        clf = SGDClassifier(loss = 'log', class_weight=weight_n, penalty=regularizer,
                                            max_iter=maxiter, alpha = best_lambda, verbose = 0, tol = tol).fit(X_train, y_train)
                    else:
                        clf = LogisticRegression(solver = 'liblinear', class_weight=weight_n, penalty=regularizer,
                                            max_iter=maxiter, C = 1/best_lambda, verbose = 0, tol = tol, random_state = seed).fit(X_train, y_train)          
                else:
                    if use_sgd:
                        clf = SGDClassifier(loss = 'log', class_weight=weight_n, 
                                                max_iter=maxiter, alpha = best_lambda, verbose = 0, tol = tol).fit(X_train, y_train)
                    else:
                        clf = LogisticRegression(solver = 'lbfgs', class_weight=weight_n, penalty=regularizer,
                                            max_iter=maxiter, C = 1/best_lambda, verbose = 0, tol = tol, random_state = seed).fit(X_train, y_train)   
            except:

                continue

            coefs_all += clf.coef_.squeeze()
            coef_dict[ic] = clf.coef_
            if plot_convergence and ic < 3:
                sys.stdout = old_stdout
                loss_history = mystdout.getvalue()
                loss_list = []
                for line in loss_history.split('\n'):
                    if(len(line.split("loss: ")) == 1):
                        continue
                    loss_list.append(float(line.split("loss: ")[-1]))
                
                ax_c[ic, jc].plot(np.arange(len(loss_list)), loss_list)
                ax_c[ic, jc].set_xlabel("Time in epochs")
                ax_c[ic, jc].set_ylabel("Loss")


            if x.shape[0] < 60:
                X_test = X_test.reshape(1, -1)
                y_test = y_test.reshape(1, -1)

            if len(X_test.shape) == 1:
                X_test = np.expand_dims(X_test,1)
            
            y_guess = clf.predict(X_test)

            y_probs = clf.predict_proba(X_test)
                
            loss = (-y_test*np.log(y_probs[:, 1]) - \
                (1-y_test)*np.log(1-y_probs[:, 1]))
            if loss.shape[0]>1:
                loss = (np.sum(loss)/loss.shape[0]).item()
            else:
                try:
                    loss = loss.item()
                except:
                    import pdb; pdb.set_trace()
            if np.isnan(loss):
                try:
                    loss = 0 if y_test.item() == y_probs[:, 1].item() else 1
                except:
                    if (y_test == y_probs[:,1]).all():
                        loss = 0
                    else:
                        loss = 1

            loss_vec.append(loss)
            if len(y_test)>1:
                ts_true.extend(y_test.squeeze())
                ts_pred.extend(y_guess.squeeze())
                ts_probs.extend(y_probs[:,1].squeeze())
            else:
                ts_true.append(int(y_test.squeeze()))
                ts_pred.append(int(y_guess.squeeze()))
                ts_probs.append(y_probs[:,1].squeeze())
            
            ite += 1

        ts_pred = np.array(ts_pred)
        ts_true = np.array(ts_true)

        ts_probs = np.array(ts_probs)
        tprr = len(set(np.where(ts_pred == 1)[0]).intersection(
            set(np.where(ts_true == 1)[0])))/len(np.where(ts_true == 1)[0])

        fprr = len(set(np.where(ts_pred == 0)[0]).intersection(
            set(np.where(ts_true == 0)[0])))/len(np.where(ts_true == 0)[0])
        bac = (tprr + fprr)/2


        pos = len(np.where(ts_true == 1)[0])
        neg = len(np.where(ts_true == 0)[0])
        tp = len(set(np.where(ts_pred == 1)[0]).intersection(
            set(np.where(ts_true == 1)[0])))
        tn = len(set(np.where(ts_pred == 0)[0]).intersection(
            set(np.where(ts_true == 0)[0])))
        fn = pos - tp
        fp = neg - tn
        arr = [[tp, fn], [fp, tn]]

        f1 = sklearn.metrics.f1_score(ts_true, ts_pred)

        plt.show()

        if plot_cf:
            fig3, ax3 = plt.subplots()
            df_cm = pd.DataFrame(arr, index=['Actual Recur', 'Actual Asymptomatic'], columns=[
                                 'Predicted Recur', 'Predicted Asymptomatic'])
            # sns.set(font_scale=1.4)  # for label size
            chart = sns.heatmap(df_cm, annot=True, annot_kws={
                               "size": 24})  # font size
            ax3.set_yticklabels(
                ['Actual Recur', 'Actual Asymptomatic'], rotation=45)
            ax3.xaxis.tick_top()
            ax3.xaxis.set_label_position('top')
            ax3.tick_params(length=0)
            plt.title('Weight ' + str(weight) +
                      ', regularization ' + str(regularizer))
            plt.show()

        auc_out = sklearn.metrics.roc_curve(ts_true, ts_probs)
        auc_score = sklearn.metrics.roc_auc_score(ts_true, ts_probs)
        fpr = auc_out[0]
        tpr = auc_out[1]
        figg, axx = plt.subplots()
        axx.plot(fpr,tpr)
        axx.set_xlabel('False Positive Rate', fontsize = 20)
        axx.set_ylabel('True Positive Rate', fontsize = 20)
        axx.set_title(auc_score, fontsize = 20)
        plt.savefig('auc_outputs/auc_' + csv_name + '.pdf')
        plt.show()

        if print_coefs:
            coefs_all = coefs_all / len(ixs)
            coefs_all = coefs_all.squeeze()

            ixl0 = np.where(coefs_all < 0)[0]
            ixg0 = np.where(coefs_all > 0)[0]

            g0coefs = copy.deepcopy(coefs_all)
            g0coefs[ixl0] = np.zeros(len(ixl0))

            l0coefs = copy.deepcopy(coefs_all)
            l0coefs[ixg0] = np.zeros(len(ixg0))
            ranks_g = np.argsort(-g0coefs)
            mols_g = x.columns.values[ranks_g]
            odds_ratio_g = np.exp(coefs_all[ranks_g])

            ranks_l = np.argsort(l0coefs)
            mols_l = x.columns.values[ranks_l]
            odds_ratio_l = np.exp(coefs_all[ranks_l])
            # import pdb; pdb.set_trace()
            df = np.vstack((mols_l, odds_ratio_l, mols_g, odds_ratio_g)).T
            df = pd.DataFrame(
                df, columns=['Metabolites', 'Odds ratio < 1', 'Metabolites', 'Odds ratio > 1'])
            name2 = 'lr_ ' + csv_name + str(np.round(auc_score,4)).replace('.','_') +'_seed_' + str(seed) + '_lambda_' + str(np.round(best_lambda,4)).replace('.','_') +'.csv'
            df.to_csv(path + name2)

        # ranks = np.argsort(-coefs_all)
        # mols = x.columns.values[ranks]
        # odds_ratio = np.exp(coefs_all[ranks])
        # df = np.vstack((mols, odds_ratio)).T
        # df = pd.DataFrame(df, columns = ['Metabolites', 'Odds ratio'])
        # name2 = 'lr_metabs_' + str(regularizer) + '_' + str(weight) + '.csv'
        # df.to_csv(name2)
        final_res_dict[csv_name][seed]['optimizer'] = key
        final_res_dict[csv_name][seed]['best_lambda'] = best_lambda
        final_res_dict[csv_name][seed]['auc'] = auc_score
        final_res_dict[csv_name][seed]['tpr'] = tpr
        final_res_dict[csv_name][seed]['fpr'] = fpr
        final_res_dict[csv_name][seed]['df_name'] = path + name2
        final_res_dict[csv_name][seed]['coefs'] = coef_dict
        return tprr, fprr, bac, pos, neg, tp, tn, fp, fn, f1, auc_score, final_res_dict


    def pls_da(self, x, targets, lim_grid = np.arange(0,1,.05), cmp_grid= np.arange(2,10,1), plot_cf = True, nest = False, csv_name = 'default', standardize = True):
        if isinstance(targets[0],str):
            targets = (np.array(targets) == 'Recur').astype('float')
        else:
            targets = np.array(targets)
        dem = np.std(x, 0)
        dz = np.where(dem == 0)[0]
        dem[dz] = 1
        if standardize:
            X = np.array((x - np.mean(x, 0))/dem)
        else:
            X = np.array(x)
        y = targets

        ixs = self.leave_one_out_cv(x, targets)
        if x.shape[0] > 50:
            dattype = 'all_data'
        else:
            dattype = 'week_one'
        
        if nest:
            lambdict = {}
            for lim in lim_grid:
                for comp in cmp_grid:

                    lambdict[(lim, comp)] = {}
                    ts_true = []
                    ts_pred = []
                    loss_vec = []
                    pls_binary = PLSRegression(n_components=comp)
                    for ix in ixs:
                        train_index, test_index = ix
                        X_train, X_test = X[train_index, :], X[test_index, :]
                        y_train, y_test = y[train_index], y[test_index]
                        pls_binary.fit(X_train, y_train)

                        binary_prediction = (pls_binary.predict(X_test)[:,0] > lim).astype('uint8')               
                        coefs = pls_binary.coef_
                        
                        y_guess = binary_prediction
                        
                        if x.shape[0] < 60:
                            X_test = X_test.reshape(1, -1)
                            y_test = y_test.reshape(1, -1)

                        if len(y_test)>1:
                            ts_true.extend(y_test.squeeze())
                            ts_pred.extend(y_guess.squeeze())
                        else:
                            ts_true.append(int(y_test.squeeze()))
                            ts_pred.append(int(y_guess.squeeze()))

                    ts_pred = np.array(ts_pred)
                    ts_true = np.array(ts_true)
                    tprr = len(set(np.where(ts_pred == 1)[0]).intersection(
                        set(np.where(ts_true == 1)[0])))/len(np.where(ts_true == 1)[0])
                    fprr = len(set(np.where(ts_pred == 0)[0]).intersection(
                        set(np.where(ts_true == 0)[0])))/len(np.where(ts_true == 0)[0])
                    bac = (tprr + fprr)/2

                    pos = len(np.where(ts_true == 1)[0])
                    neg = len(np.where(ts_true == 0)[0])
                    tp = len(set(np.where(ts_pred == 1)[0]).intersection(
                        set(np.where(ts_true == 1)[0])))
                    tn = len(set(np.where(ts_pred == 0)[0]).intersection(
                        set(np.where(ts_true == 0)[0])))
                    fn = pos - tp
                    fp = neg - tn
                    arr = [[tp, fn], [fp, tn]]

                    # mols = self.data_dict[1.0].columns.values[ranks].squeeze()
                    lambdict[(lim, comp)]['bac'] = bac
                    lambdict[(lim, comp)]['tpr'] = tprr
                    lambdict[(lim, comp)]['fpr'] = fprr
                    lambdict[(lim, comp)]['cv'] = arr
                    lambdict[(lim, comp)]['loss'] = np.sum(loss_vec)/len(loss_vec)

            lambdas = list(lambdict.keys())

            key = 'bac'
            max_val = np.max([lambdict[it][key] for it in lambdas])
            best_lambda = [k for k in lambdas if lambdict[k][key] == max_val]
            # print('Best Lambda = ' + str(best_lambda) + ', For weight ' + str(weighting) + 'and l'+ str(regularizer))
            
            best_lambda = best_lambda[0]

            print('Best Combo = ' + str(best_lambda) )
        else:
            best_lambda = (lim_grid, cmp_grid)
            lambdict = None
        ts_true = []
        ts_pred = []
        loss_vec = []
        coefs_all = np.zeros(x.shape[1])
        ite = 0
        pls_binary = PLSRegression(n_components=best_lambda[1])
        thresholds = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
        y_guess_dict = {}
        y_true_dict = {}
        bin_preds = []
        for ix in ixs:
            train_index, test_index = ix
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            np.random.seed(0)
            pls_binary.fit(X_train, y_train)
            binary_prediction = (pls_binary.predict(X_test)[:,0] > best_lambda[0]).astype('uint8')
            y_guess = binary_prediction
            coefs = pls_binary.coef_.squeeze()

            if x.shape[0] < 60:
                X_test = X_test.reshape(1, -1)
                y_test = y_test.reshape(1, -1)

            if len(y_test)>1:
                ts_true.extend(y_test.squeeze())
                ts_pred.extend(y_guess.squeeze())
                bin_preds.extend(pls_binary.predict(X_test)[:,0].squeeze())
            else:
                ts_true.append(int(y_test.squeeze()))
                ts_pred.append(int(y_guess.squeeze()))
                bin_preds.append(float(pls_binary.predict(X_test)[:,0].squeeze()))
            coefs_all += coefs
            ite += 1


        min_x = np.min(bin_preds)
        max_x = np.max(bin_preds)
        range_x = max_x - min_x
        bin_preds = (bin_preds - min_x)/range_x

        for thresh in thresholds:
            binary_prediction = (np.array(bin_preds) > thresh).astype('uint8')

            y_guess_dict[thresh] = binary_prediction.squeeze()
            y_true_dict[thresh] = ts_true
        tpr_dict = {}
        fpr_dict = {}
        for thresh in thresholds:
            ix_one = np.where(np.array(y_true_dict[thresh]) == 1)[0]
            ix_zero = np.where(np.array(y_true_dict[thresh]) == 0)[0]
            tpr_dict[thresh] = np.sum(np.array(y_guess_dict[thresh])[ix_one] == 1)/len(ix_one)
            fpr_dict[thresh] = 1-(np.sum(np.array(y_guess_dict[thresh])[ix_zero] == 1)/len(ix_zero))
        
        
        plt.plot(list(fpr_dict.values()),list(tpr_dict.values()))
        plt.show()
        roc = np.trapz(y = list(tpr_dict.values()), x = list(fpr_dict.values()))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.show()

                


        # coefs_all = coefs_all / len(ixs)
        
        ts_pred = np.array(ts_pred)
        ts_true = np.array(ts_true)
        tprr = len(set(np.where(ts_pred == 1)[0]).intersection(
            set(np.where(ts_true == 1)[0])))/len(np.where(ts_true == 1)[0])
        fprr = len(set(np.where(ts_pred == 0)[0]).intersection(
            set(np.where(ts_true == 0)[0])))/len(np.where(ts_true == 0)[0])
        bac = (tprr + fprr)/2

        pos = len(np.where(ts_true == 1)[0])
        neg = len(np.where(ts_true == 0)[0])
        tp = len(set(np.where(ts_pred == 1)[0]).intersection(
            set(np.where(ts_true == 1)[0])))
        tn = len(set(np.where(ts_pred == 0)[0]).intersection(
            set(np.where(ts_true == 0)[0])))
        fn = pos - tp
        fp = neg - tn
        arr = [[tp, fn], [fp, tn]]

        tpr = tp / len(np.where(ts_true == 1)[0])
        tnr = tn / len(np.where(ts_true == 0)[0])

        f1 = sklearn.metrics.f1_score(ts_true, ts_pred)

        # accuracy = []

        # cval = KFold(n_splits=10, shuffle=True, random_state=19)
        # for train, test in cval.split(X_binary):
            
        #     y_pred = pls_da(X_binary[train,:], y_binary[train], X_binary[test,:])
            
        #     accuracy.append(accuracy_score(y_binary[test], y_pred))


        if plot_cf:
            fig3, ax3 = plt.subplots()
            df_cm = pd.DataFrame(arr, index=['Actual Recur', 'Actual Asymptomatic'], columns=[
                                 'Predicted Recur', 'Predicted Asymptomatic'])
            # sns.set(font_scale=1.4)  # for label size
            chart = sns.heatmap(df_cm, annot=True, annot_kws={
                               "size": 24})  # font size
            ax3.set_yticklabels(
                ['Actual Recur', 'Actual Asymptomatic'], rotation=45)
            ax3.xaxis.tick_top()
            ax3.xaxis.set_label_position('top')
            ax3.tick_params(length=0)

            plt.show()

        # coefs_all = coefs_all /len(ixs)
        ixl0 = np.where(coefs_all < 0)[0]
        ixg0 = np.where(coefs_all > 0)[0]

        g0coefs = copy.deepcopy(coefs_all)
        g0coefs[ixl0] = np.zeros(len(ixl0))

        l0coefs = copy.deepcopy(coefs_all)
        l0coefs[ixg0] = np.zeros(len(ixg0))
        ranks_g = np.argsort(-g0coefs)
        mols_g = x.columns.values[ranks_g]
        odds_ratio_g = np.exp(coefs_all[ranks_g])

        ranks_l = np.argsort(l0coefs)
        mols_l = x.columns.values[ranks_l]
        odds_ratio_l = np.exp(coefs_all[ranks_l])
        # import pdb; pdb.set_trace()
        df = np.vstack((mols_l, odds_ratio_l, mols_g, odds_ratio_g)).T
        df = pd.DataFrame(
            df, columns=['Metabolites', 'Odds ratio < 1', 'Metabolites', 'Odds ratio > 1'])
        name2 = 'pls_da_metabs2_' + \
            csv_name+ '.csv'
        df.to_csv(name2)

        # ranks = np.argsort(-coefs_all)
        # mols = x.columns.values[ranks]
        # odds_ratio = np.exp(coefs_all[ranks])
        # df = np.vstack((mols, odds_ratio)).T
        # df = pd.DataFrame(df, columns = ['Metabolites', 'Odds ratio'])
        # name2 = 'lr_metabs_' + str(regularizer) + '_' + str(weight) + '.csv'
        # df.to_csv(name2)
        return lambdict, tpr, tnr, bac, f1


    def lin_reg(self, x, targets, lambda_grid = np.logspace(-3, 3, 50), maxiter = 100, plot_lambdas = True, 
                plot_cf = True, use_old = False, load_old = False, optim_param = 'loss', standardize = True, csv_name = 'default'):
        key = 'loss'
        
        dem = np.std(x, 0)
        dz = np.where(dem == 0)[0]
        dem[dz] = 1
        if standardize:
            X = np.array((x - np.mean(x, 0))/dem)
        else:
            X = np.array(x)
        y = targets

        ixs = self.leave_one_out_cv(x, targets)
        if x.shape[0] > 50:
            dattype = 'all_data'
        else:
            dattype = 'week_one'

        # if use_grps:
        #     temp = np.divide(X,np.sum(X,1),axis = 0)
        if not isinstance(lambda_grid, float):
            lambdict = {}
            for lamb in lambda_grid:
                lambdict[lamb] = {}
                ts_true = []
                ts_pred = []
                loss_vec = []
                for ix in ixs:
                    train_index, test_index = ix
                    X_train, X_test = X[train_index, :], X[test_index, :]
                    y_train, y_test = y[train_index], y[test_index]
                    clf = sklearn.linear_model.Lasso(alpha=lamb).fit(X_train, y_train)
        

                    coefs = clf.coef_
                    
                    
                    if x.shape[0] < 60:
                        X_test = X_test.reshape(1, -1)
                        y_test = y_test.reshape(1, -1)
                    y_guess = clf.predict(X_test)
                    # y_score = clf.score(X_test,y_test)

                    if len(y_test)>1:
                        ts_true.extend(y_test.squeeze())
                        ts_pred.extend(y_guess.squeeze())
                    else:
                        ts_true.append(int(y_test.squeeze()))
                        ts_pred.append(int(y_guess.squeeze()))
                                
                    loss = (np.array(ts_pred) - np.array(ts_true))**2

                    if loss.shape[0]>1:
                        loss = (np.sum(loss)/loss.shape[0]).item()
                    else:
                        loss = loss.item()

                    # import pdb; pdb.set_trace()
                    loss_vec.append(loss)
                    # assert(y_guess.item() == np.round(y_probs[:,1].item()))
                ts_pred = np.array(ts_pred)
                ts_true = np.array(ts_true)
                
                # lambdict[lamb]['score'] = y_score
                lambdict[lamb]['loss'] = np.sum(loss_vec)/len(loss_vec)

            if plot_lambdas:
                fig2, ax2 = plt.subplots()

                for ij, k in enumerate(lambdict.keys()):
                    ax2.scatter([k],
                                [lambdict[k][optim_param]])
                    ax2.set_xlabel('lambda values')
                    ax2.set_ylabel(optim_param)
                    ax2.set_xscale('log')
                    # ax2.set_title('Outer Fold ' + str(ol), fontsize=30)

            lambdas = list(lambdict.keys())

            if key == 'loss':
                max_val = np.min([lambdict[it][key]
                                    for it in lambdas])
            else:
                max_val = np.max([lambdict[it][key]
                                    for it in lambdas])
            best_lambda = [k for k in lambdas if lambdict[k][key] == max_val]
            # print('Best Lambda = ' + str(best_lambda) + ', For weight ' + str(weighting) + 'and l'+ str(regularizer))
            best_lambda = np.min(best_lambda)
            print('Best Lambda = ' + str(best_lambda))
        else:
            best_lambda = lambda_grid


        ts_true = []
        ts_pred = []
        loss_vec = []
        scores = []
        coefs_all = np.zeros(x.shape[1])
        ite = 0
        for ix in ixs:
            train_index, test_index = ix
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            np.random.seed(0)

            clf = sklearn.linear_model.Lasso(alpha=best_lambda).fit(X_train, y_train)

            coefs = clf.coef_
                    
            coefs = clf.coef_.squeeze()
            if ite == 0:
                print(X_train.shape)
            #     print(weight)
                print(clf)
            #     print(coefs)
            if x.shape[0] < 60:
                X_test = X_test.reshape(1, -1)
                y_test = y_test.reshape(1, -1)
            y_guess = clf.predict(X_test)
            # y_score = clf.score(X_test, y_test)
            loss = (y_test - y_guess)**2
            if loss.shape[0]>1:
                loss = (np.sum(loss)/loss.shape[0]).item()
            else:
                loss = loss.item()

            loss_vec.append(loss)
            if len(y_test)>1:
                ts_true.extend(y_test.squeeze())
                ts_pred.extend(y_guess.squeeze())
                # scores.extend(y_score.squeeze())
            else:
                ts_true.append(int(y_test.squeeze()))
                ts_pred.append(int(y_guess.squeeze()))
                # scores.append(y_score)
            coefs_all += coefs
            ite += 1
        # coefs_all = coefs_all / len(ixs)
        ts_pred = np.array(ts_pred)
        ts_true = np.array(ts_true)

        ranks_g = np.argsort(coefs_all)
        mols_g = x.columns.values[ranks_g]
        odds_ratio_g =coefs_all[ranks_g]

        df = np.vstack((mols_g, odds_ratio_g)).T

        df = pd.DataFrame(df, columns=['Metabolites', 'Odds ratio'])
        name2 = 'linreg_' + csv_name + '.csv'
        df.to_csv(name2)

        # ranks = np.argsort(-coefs_all)
        # mols = x.columns.values[ranks]
        # odds_ratio = np.exp(coefs_all[ranks])
        # df = np.vstack((mols, odds_ratio)).T
        # df = pd.DataFrame(df, columns = ['Metabolites', 'Odds ratio'])
        # name2 = 'lr_metabs_' + str(regularizer) + '_' + str(weight) + '.csv'
        # df.to_csv(name2)
        return loss_vec, ts_true, ts_pred, best_lambda


    def log_reg_2(self, x, targets, lambda_grid = np.logspace(-3, 3, 50), features=None, weight= None, 
                regularizer = 'l1', solve = 'liblinear', maxiter = 100, plot_lambdas = True, 
                plot_cf = True, use_old = False, load_old = False, optim_param = 'loss', standardize = True, csv_name = 'default',
                path = '', plot_convergence = False, tol = 0.0001):
        if isinstance(targets[0], str):
            targets = (np.array(targets) == 'Recur').astype('float')
        else:
            targets = np.array(targets)
        dem = np.std(x, 0)
        dz = np.where(dem == 0)[0]
        dem[dz] = 1

        ixs = self.leave_one_out_cv(x, targets)
        
        if standardize:
            X = np.array((x - np.mean(x, 0))/dem)
        else:
            X = np.array(x)
        y = targets

        
        if x.shape[0] > 70:
            dattype = 'all_data'
            tmpts = np.array([ix.split('-')[1] for ix in x.index.values])
        else:
            dattype = 'week_one'
            tmpts = np.ones(x.shape[0])
        if weight == 'balanced':
            w = True
        else:
            w = False

        name = 'lr_lambdict_' + str(w) + str(regularizer) + '_' + str(weight) + '.pkl'

        coefs_all = np.zeros(x.shape[1])
        ts_pred = []
        ts_true = []
        ts_probs = []
        log_loss_build = lambda y: sklearn.metrics.make_scorer(metrics.log_loss, greater_is_better=False, needs_proba=True, labels=sorted(np.unique(y)))
        for ix in ixs:
            train_index, test_index = ix
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            tmpts_train = tmpts[train_index]

            
            coefs_all_tr = []
            
            for jc, tmpt in enumerate(np.unique(tmpts)):
                if dattype == 'all_data':
                    ix_tmpt = np.where(np.array(tmpts_train)==tmpt)[0]
                    Xt = X_train[ix_tmpt,:]
                    yt = y_train[ix_tmpt]
                    # import pdb; pdb.set_trace()

                    # tr_pts = datpts_train[ix_tmpt]
                    # print(tmpt)
                    # print(df_train_to_print)
                    # print('')
                    # import pdb; pdb.set_trace()

                    ws = len(yt)/(2* np.bincount([int(x) for x in yt]))
                    wdict = {}

                    wdict[0] = ws[0]
                    try:
                        wdict[1] = ws[1]
                    except:
                        wdict[1] = 0
                    if weight == 'balanced':
                        weight_n = wdict
                    else:
                        weight_n = weight
                else:
                    Xt = X_train
                    yt = y_train
                    weight_n = weight
                


                if plot_convergence:
                    old_stdout = sys.stdout
                    sys.stdout = mystdout = StringIO()

                penalty = ['l1','l2']
                parameters = dict(C = 1/lambda_grid, penalty = penalty)
                

                logistic = LogisticRegression(solver = 'liblinear', class_weight=weight_n, 
                                        max_iter=maxiter, verbose = 0, tol = tol)
                clf_mod = GridSearchCV(logistic, parameters, scoring = optim_param).fit(Xt, yt)
                clf = clf_mod.best_estimator_

                best_lambda = 1/clf.C
                # print('best lambda')
                # print(best_lambda)
                coefs = clf.coef_.squeeze()

                coefs_all_tr.append(coefs)

            if dattype == 'all_data':
                coefs = np.expand_dims(np.sum(np.stack(coefs_all_tr).squeeze(),0) / len(coefs_all_tr),0)
            
            if len(coefs.shape) == 1:
                coefs = np.expand_dims(coefs,1)
            elif len(coefs.shape) < 1:
                coefs = np.expand_dims(np.expand_dims(coefs,0),1)
            
            if coefs.shape[0] > coefs.shape[1]:
                coefs = coefs.T
            
            if X_test.shape[0]> X_test.shape[1]:
                X_test = X_test.T
            
            clf.coef_ = coefs


            if x.shape[0] < 60:
                X_test = X_test.reshape(1, -1)
                y_test = y_test.reshape(1, -1)

            if len(X_test.shape) == 1:
                X_test = np.expand_dims(X_test,1)
            
            y_guess = clf.predict(X_test)

            y_probs = clf.predict_proba(X_test)
                
            # loss = (-y_test*np.log(y_probs[:, 1]) - \
            #     (1-y_test)*np.log(1-y_probs[:, 1]))
            # if loss.shape[0]>1:
            #     loss = (np.sum(loss)/loss.shape[0]).item()
            # else:
            #     loss = loss.item()

            # if np.isnan(loss):
            #     try:
            #         loss = 0 if y_test.item() == y_probs[:, 1].item() else 1
            #     except:
            #         if (y_test == y_probs[:,1]).all():
            #             loss = 0
            #         else:
            #             loss = 1

            # loss_vec.append(loss)
            if len(y_test)>1:
                ts_true.extend(y_test.squeeze())
                ts_pred.extend(y_guess.squeeze())
                ts_probs.extend(y_probs[:,1].squeeze())
            else:
                ts_true.append(int(y_test.squeeze()))
                ts_pred.append(int(y_guess.squeeze()))
                ts_probs.append(y_probs[:,1].squeeze())
            
            coefs_all += coefs.squeeze()
        # coefs_all = coefs_all / len(ixs)

        
        
        ts_pred = np.array(ts_pred)
        ts_true = np.array(ts_true)

        ts_probs = np.array(ts_probs)
        tprr = len(set(np.where(ts_pred == 1)[0]).intersection(
            set(np.where(ts_true == 1)[0])))/len(np.where(ts_true == 1)[0])

        fprr = len(set(np.where(ts_pred == 0)[0]).intersection(
            set(np.where(ts_true == 0)[0])))/len(np.where(ts_true == 0)[0])
        bac = (tprr + fprr)/2


        pos = len(np.where(ts_true == 1)[0])
        neg = len(np.where(ts_true == 0)[0])
        tp = len(set(np.where(ts_pred == 1)[0]).intersection(
            set(np.where(ts_true == 1)[0])))
        tn = len(set(np.where(ts_pred == 0)[0]).intersection(
            set(np.where(ts_true == 0)[0])))
        fn = pos - tp
        fp = neg - tn
        arr = [[tp, fn], [fp, tn]]

        f1 = sklearn.metrics.f1_score(ts_true, ts_pred)


        if plot_cf:
            fig3, ax3 = plt.subplots()
            df_cm = pd.DataFrame(arr, index=['Actual Recur', 'Actual Asymptomatic'], columns=[
                                 'Predicted Recur', 'Predicted Asymptomatic'])
            # sns.set(font_scale=1.4)  # for label size
            chart = sns.heatmap(df_cm, annot=True, annot_kws={
                               "size": 24})  # font size
            ax3.set_yticklabels(
                ['Actual Recur', 'Actual Asymptomatic'], rotation=45)
            ax3.xaxis.tick_top()
            ax3.xaxis.set_label_position('top')
            ax3.tick_params(length=0)
            plt.title('Weight ' + str(weight) +
                      ', regularization ' + str(regularizer))
            plt.show()

        auc_out = sklearn.metrics.roc_curve(ts_true, ts_probs)
        auc_score = sklearn.metrics.roc_auc_score(ts_true, ts_probs)
        fpr = auc_out[0]
        tpr = auc_out[1]
        figg, axx = plt.subplots()
        axx.plot(fpr,tpr)
        axx.set_xlabel('False Positive Rate', fontsize = 20)
        axx.set_ylabel('True Positive Rate', fontsize = 20)
        axx.set_title(auc_score, fontsize = 20)
        plt.savefig('auc_' + csv_name + '.pdf')
        plt.show()

        ixl0 = np.where(coefs_all < 0)[0]
        ixg0 = np.where(coefs_all > 0)[0]

        g0coefs = copy.deepcopy(coefs_all)
        g0coefs[ixl0] = np.zeros(len(ixl0))

        l0coefs = copy.deepcopy(coefs_all)
        l0coefs[ixg0] = np.zeros(len(ixg0))
        ranks_g = np.argsort(-g0coefs)
        mols_g = x.columns.values[ranks_g]
        odds_ratio_g = np.exp(coefs_all[ranks_g])

        ranks_l = np.argsort(l0coefs)
        mols_l = x.columns.values[ranks_l]
        odds_ratio_l = np.exp(coefs_all[ranks_l])
        # import pdb; pdb.set_trace()
        df = np.vstack((mols_l, odds_ratio_l, mols_g, odds_ratio_g)).T
        df = pd.DataFrame(
            df, columns=['Metabolites', 'Odds ratio < 1', 'Metabolites', 'Odds ratio > 1'])
        name2 = 'lr_ ' + csv_name + str(np.round(auc_score,4)).replace('.','_') +'.csv'
        df.to_csv(path + name2)

        return tprr, fprr, bac, pos, neg, tp, tn, fp, fn, f1, auc_score



    def nested_cv_func(self, x, targets, lambda_grid = np.logspace(-3, 3, 50), features=None, weight= None, 
                regularizer = 'l1', solve = 'liblinear', maxiter = 100, plot_lambdas = True, 
                plot_cf = True, use_old = False, load_old = False, optim_param = 'loss', standardize = True, csv_name = 'default',
                path = '', plot_convergence = False, tol = 0.0001, use_sgd = False, nzero_thresh = 5, stop_lambdas = True, print_coefs = True,
                seed = None, final_res_dict = {}):

        if csv_name not in final_res_dict.keys():
            final_res_dict[csv_name] = {}

        if seed not in final_res_dict[csv_name].keys():
            final_res_dict[csv_name][seed] = {}

        if isinstance(targets[0], str):
            targets = (np.array(targets) == 'Recur').astype('float')
        else:
            targets = np.array(targets)
        y = targets

        if weight == 'balanced':
            w = True
        else:
            w = False

        ts_true = []
        ts_pred = []
        loss_vec = []
        ts_probs = []
        ite = 0
        coefs_all = None
        for ic, ix in enumerate(ixs):
            train_index, test_index = ix
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            Xt_pd = x.iloc[train_index,:]

            ixs_inner = self.leave_one_out_cv(Xt_pd, y_train) 
            lambdict = {}
            for lamb in lambda_grid:
                lambdict[lamb] = {}

                ts_true_in = []
                ts_pred_in = []
                loss_vec_in = []
                ts_probs_in = []
                for ix_in in ixs_inner:
                    train_index_in, test_index_in = ix_in
                    X_train_in, X_test_in = X_train[train_index_in, :], X_train[test_index_in, :]
                    y_train_in, y_test_in = y_train[train_index_in], y_train[test_index_in]

                    tmpts_train_in = tmpts[train_index_in]
                    coefs_all_in = []
                    samp_weights = np.ones(len(y_train_in))
                    if dattype == 'all_data':
                        weight_n = None
                        samp_weights = get_class_weights(y, tmpts_train_in)
                    else:
                        weight_n = weight
                    
                    clf = LogisticRegression(solver = 'liblinear', class_weight=weight_n, penalty=regularizer,
                                        max_iter=maxiter, C = 1/lamb, verbose = 0, tol = tol, warm_start=True, random_state = seed).fit(
                                            X_train_in, y_train_in, sample_weight = samp_weights)   
                    
                    coef_ind = clf.coef_ > 0
                    num_coefs.append(np.sum(coef_ind))
                    
                    y_guess = clf.predict(X_test_in)
                    y_probs = clf.predict_proba(X_test_in)

                    if len(y_test_in)>1:
                        ts_true_in.extend(y_test_in.squeeze())
                        ts_pred_in.extend(y_guess.squeeze())
                        ts_probs_in.extend(y_probs[:,1].squeeze())
                    else:
                        ts_true_in.append(int(y_test_in.squeeze()))
                        ts_pred_in.append(int(y_guess.squeeze()))
                        ts_probs_in.append(y_probs[:,1].squeeze())

                    loss = (-y_test_in * np.log(y_probs[:, 1]) - (1-y_test_in)*np.log(1-y_probs[:, 1]))
                    if loss.shape[0]>1:
                        loss = (np.sum(loss)/loss.shape[0]).item()
                    else:
                        loss = loss.item()
                    if np.isnan(loss):
                        try:
                            loss = 1 if y_test_in.item() == y_probs[:,1].item() else 0
                        except:
                            loss = np.sum(1-(y_test_in == y_probs[:, 1]).astype(int))/len(y_test_in)
                    if np.isinf(loss):
                        loss = 99
                    # import pdb; pdb.set_trace()
                    loss_vec_in.append(loss)
                    # assert(y_guess.item() == np.round(y_probs[:,1].item()))


                    if np.mean(num_coefs)< nzero_thresh and stop_lambdas:
                        del lambdict[lamb]
                        continue
                ts_pred_in = np.array(ts_pred_in)
                ts_true_in = np.array(ts_true_in)
                tprr = len(set(np.where(ts_pred_in == 1)[0]).intersection(
                    set(np.where(ts_true_in == 1)[0])))/len(np.where(ts_true_in == 1)[0])
                fprr = len(set(np.where(ts_pred_in== 0)[0]).intersection(
                    set(np.where(ts_true_in == 0)[0])))/len(np.where(ts_true_in == 0)[0])
                bac = (tprr + fprr)/2
                auc_score = sklearn.metrics.roc_auc_score(ts_true_in, ts_probs_in)

                    # mols = self.data_dict[1.0].columns.values[ranks].squeeze()
                lambdict[lamb]['bac'] = bac
                lambdict[lamb]['tpr'] = tprr
                lambdict[lamb]['fpr'] = fprr
                lambdict[lamb]['auc'] = auc_score
                lambdict[lamb]['loss'] = np.sum(loss_vec_in)/len(loss_vec_in)
            # import pdb; pdb.set_trace()
            lambdas = list(lambdict.keys())
            key = optim_param
            vec = np.array([lambdict[it][key] for it in lambdas])
            nn= 5
            ma = moving_average(vec, n=nn)
            offset = int(np.floor(nn/2))
            if key == 'loss':
                max_ix  = np.argmin(ma) + offset
            else:
                max_ix = np.argmax(ma) + offset
            best_lambda = np.array(lambdas)[max_ix]
            best_lambda = np.min(best_lambda)
            print(' Best Lambda = ' + str(best_lambda) + ', For weight ' + str(weight) + 'and l'+ str(regularizer))


            if plot_lambdas:
                fig2, ax2 = plt.subplots()
                fig2.suptitle('Weight ' + str(weight) +
                            ', regularization l' + str(regularizer))

                ax2.plot(np.array(list(lambdict.keys()))[offset:-offset], ma)
                ax2.axvline(x=best_lambda)
                for ij, k in enumerate(lambdict.keys()):
                    ax2.scatter([k],
                                [lambdict[k][optim_param]])
                    ax2.set_xlabel('lambda values')
                    
                    
                    ax2.set_ylabel(optim_param)
                    ax2.set_xscale('log')
                    # ax2.set_title('Outer Fold ' + str(ol), fontsize=30)
                    if optim_param == 'bac':
                        ax2.set_ylim(0, 1)


            tmpts_train = tmpts[train_index]

            datpts_train,datpts_test = x.index.values[train_index],x.index.values[test_index]
            
            np.random.seed(0)
            tmpts_train = tmpts[train_index]

            samp_weights = np.ones(len(y_train))
            if dattype == 'all_data':
                weight_n = None
                samp_weights = get_class_weights(y, tmpts_train)
            else:
                weight_n = weight

            if regularizer == 'l1':
                clf = LogisticRegression(solver = 'liblinear', class_weight=weight_n, penalty=regularizer,
                                        max_iter=maxiter, C = 1/best_lambda, verbose = 0, tol = tol, random_state = seed).fit(
                                            X_train, y_train, sample_weight = samp_weights)          



            if x.shape[0] < 60:
                X_test = X_test.reshape(1, -1)
                y_test = y_test.reshape(1, -1)

            y_guess = clf.predict(X_test)
            y_probs = clf.predict_proba(X_test)
            loss = (-y_test*np.log(y_probs[:, 1]) - \
                (1-y_test)*np.log(1-y_probs[:, 1]))
            if loss.shape[0]>1:
                loss = (np.sum(loss)/loss.shape[0]).item()
            else:
                loss = loss.item()
            if np.isnan(loss):
                try:
                    loss = 0 if y_test.item() == y_probs[:, 1].item() else 1
                except:
                    if (y_test == y_probs[:,1]).all():
                        loss = 0
                    else:
                        loss = 1
            loss_vec.append(loss)
            if len(y_test)>1:
                ts_true.extend(y_test.squeeze())
                ts_pred.extend(y_guess.squeeze())
                ts_probs.extend(y_probs[:,1].squeeze())
            else:
                ts_true.append(int(y_test.squeeze()))
                ts_pred.append(int(y_guess.squeeze()))
                ts_probs.append(y_probs[:,1].squeeze())

            if coefs_all:
                coefs_all += clf.coef_.squeeze()
            else:
                coefs_all = clf.coef_.squeeze()

            ite += 1

        ts_pred = np.array(ts_pred)
        ts_true = np.array(ts_true)

        ts_probs = np.array(ts_probs)
        tprr = len(set(np.where(ts_pred == 1)[0]).intersection(
            set(np.where(ts_true == 1)[0])))/len(np.where(ts_true == 1)[0])

        fprr = len(set(np.where(ts_pred == 0)[0]).intersection(
            set(np.where(ts_true == 0)[0])))/len(np.where(ts_true == 0)[0])
        bac = (tprr + fprr)/2

        f1 = sklearn.metrics.f1_score(ts_true, ts_pred)

        auc_score = sklearn.metrics.roc_auc_score(ts_true, ts_probs)
        coefs_all = coefs_all / len(ixs)
        if print_coefs:
            df = return_log_odds(coefs_all)
            name2 = 'lr_ ' + csv_name + str(np.round(auc_score,4)).replace('.','_') \
                +'_seed_' + str(seed) + '_lambda_' + str(np.round(best_lambda,4)).replace('.','_') +'.csv'
            df.to_csv(path + name2)

        final_res_dict[csv_name][seed]['optimizer'] = key
        final_res_dict[csv_name][seed]['best_lambda'] = best_lambda
        final_res_dict[csv_name][seed]['auc'] = auc_score
        final_res_dict[csv_name][seed]['tpr'] = tpr
        final_res_dict[csv_name][seed]['fpr'] = fpr
        final_res_dict[csv_name][seed]['df_name'] = path + name2
        return tprr, fprr, bac, pos, neg, tp, tn, fp, fn, f1, auc_score, final_res_dict
