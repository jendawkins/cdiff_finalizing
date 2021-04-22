import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import copy
from sklearn.model_selection import StratifiedKFold
import os

def standardize(x,override = False):
    if not override:
        assert(x.shape[0]<x.shape[1])

    dem = np.std(x,0)
    if (dem == 0).any():
        dem = np.where(np.std(x, 0) == 0, 1, np.std(x, 0))
    return (x - np.mean(x, 0))/dem

def get_percentiles(vals, interval = 90, digits = 4, coef = False):
    col = np.array(vals)
    ix_sort = np.argsort(-np.abs(vals))
    sort = col[ix_sort]
    perc = (100-interval)/(2*100)
    ix_upper = np.int(np.floor(len(sort)*perc))
    ix_lower = np.int(np.ceil(len(sort)*(1-perc)))
    if ix_upper >= len(vals):
        ix_upper = ix_upper - 1
    if ix_lower <= len(vals):
        ix_lower = ix_lower - 1
    if coef:
        return (np.round(np.exp(sort[ix_lower]),digits), np.round(np.exp(sort[ix_upper]),digits))
    else:
        return (np.round((sort[ix_lower]),digits), np.round((sort[ix_upper]),digits))


def get_mad(vals, axis = 0):
    return np.median(np.abs(vals - np.mean(vals, axis)), axis)

def get_mad_interval(median, mad, param, digits = 4):
    if param == 'coef':
        upper = np.round(np.exp(median + mad),4)
        lower = np.round(np.exp(median - mad),4)
    else:
        upper = np.round(median + mad,4)
        lower  = np.round(median - mad,4)
    return np.round(upper, digits), np.round(lower, digits)

def get_ci(val,z=1.96):
    mean = np.mean(val)
    std = np.std(val)
    l = len(val)
    CI_top = np.round(mean + z*(std)/np.sqrt(l),4)
    CI_bot = np.round(mean - z*(std)/np.sqrt(l),4)
    return (CI_bot, CI_top)

def custom_dist(pt1, pt2, metric):
    if metric == 'e':
        dist = np.linalg.norm(pt1-pt2)
    if metric == 's':
        d1, cc = st.pearsonr(pt1, pt2)
        dist = 1-d1
    return dist

def get_metrics(pred, true, probs):
    ts_pred_in = np.array(pred)
    ts_true_in = np.array(true)
    ts_probs_in = np.array(probs)
    tprr = len(set(np.where(ts_pred_in == 1)[0]).intersection(
        set(np.where(ts_true_in == 1)[0])))/len(np.where(ts_true_in == 1)[0])
    tnrr = len(set(np.where(ts_pred_in== 0)[0]).intersection(
        set(np.where(ts_true_in == 0)[0])))/len(np.where(ts_true_in == 0)[0])
    bac = (tprr + tnrr)/2
    auc_score = sklearn.metrics.roc_auc_score(ts_true_in, ts_probs_in)
    f1 = sklearn.metrics.f1_score(ts_true_in, ts_pred_in)
    ret_dict = {'pred':pred,'true':true,'tpr':tprr,'tnr':tnrr,'bac':bac,'auc':auc_score,'f1':f1,'probs':probs,}
    return ret_dict

def get_class_weights(y, tmpts):
    samp_weights = np.ones(len(y))
    for tmpt in np.unique(tmpts):
        ix_tmpt = np.where(np.array(tmpts)==tmpt)[0]
        # Xt = X_train[ix_tmpt,:]
        yt = y[ix_tmpt]
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
    return samp_weights

def get_log_odds(coefs, names):
    coefs_all = coefs.squeeze()
    ixl0 = np.where(coefs_all < 0)[0]
    ixg0 = np.where(coefs_all > 0)[0]

    g0coefs = copy.deepcopy(coefs_all)
    g0coefs[ixl0] = np.zeros(len(ixl0))

    l0coefs = copy.deepcopy(coefs_all)
    l0coefs[ixg0] = np.zeros(len(ixg0))
    ranks_g = np.argsort(-g0coefs)
    mols_g = names[ranks_g]
    odds_ratio_g = np.exp(coefs_all[ranks_g])

    ranks_l = np.argsort(l0coefs)
    mols_l = names[ranks_l]
    odds_ratio_l = np.exp(coefs_all[ranks_l])
    # import pdb; pdb.set_trace()
    df = np.vstack((mols_l, odds_ratio_l, mols_g, odds_ratio_g)).T
    df = pd.DataFrame(
        df, columns=['Metabolites', 'Odds ratio < 1', 'Metabolites', 'Odds ratio > 1'])
    return df

def plot_lambdas_func(lambdict, optim_param, offset, ma, best_lambda):
    fig2, ax2 = plt.subplots()
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
    return fig2, ax2


def make_toxin_cdiff_targets(labels,dtype,targ_type,step_ahead = 1):
    pt_labels = labels.index.values
    pts = [x.split('-')[0] for x in pt_labels]

    if step_ahead == 0:
        end_dict = {pt_lab: labels[targ_type][pt_lab] for pt_lab in pt_labels}
    else:
        end_dict = {}
        for pt in np.unique(pts):
            ixs = np.where(np.array(pts)==pt)[0][step_ahead:]

            if 'week_one' in dtype or dtype == 'w1_inst':
                labs = [labels.iloc[ix,:][targ_type] for ix in ixs]

                if len(labs)>4:
                    lab = np.round(np.mean(labs[-len(labs):]))
                else:
                    lab = np.round(np.mean(labs[-np.int(np.round(len(labs)/2)):]))
                end_dict[pt+'-1'] = lab

            if 'all_data' in dtype or dtype == 'all_data_inst':
                labs = [labels.iloc[ix,:][targ_type] for ix in ixs]
                labs = pd.DataFrame(labs, index= [labels.index.values[ix-step_ahead] for ix in ixs])
                for i,ix in enumerate(ixs):
                    end_dict[pt_labels[ix-step_ahead]] = np.squeeze(labs.iloc[i])

            # if '16s' in dtype and '.' not in dtype:
            #     import pdb; pdb.set_trace()
            
            else:
                plabs = pt_labels[ixs]
                tmpts = [float(x.split('-')[1]) for x in plabs]
                ix = np.where(np.array(tmpts)==dtype)[0]
                if len(ix)>0:
                    lab = labels.iloc[ix+1,:][targ_type]
                    end_dict[pt_labels[ix][0]]= lab
    
    return pd.DataFrame(end_dict, index = [0])

            


def dmatrix(data, metric='e'):
    # does linkage among columns
    if metric == 's':
        data_s = data.rank()
        data_s = np.array(data_s)
    else:
        data_s = np.array(data)
    tuples = list(itertools.combinations(range(data_s.shape[1]), 2))
    vec = np.zeros(len(tuples))
    for k, (i, j) in enumerate(tuples):
        if i < j:
            vec[k] = custom_dist(data_s[:, i], data_s[:, j], metric)
    return vec

def get_percentiles(vals, interval = 90, digits = 4, coef = False):
    col = np.array(vals)
    ix_sort = np.argsort(-np.abs(vals))
    sort = col[ix_sort]
    perc = (100-interval)/(2*100)
    ix_upper = np.int(np.floor(len(sort)*perc))
    ix_lower = np.int(np.ceil(len(sort)*(1-perc)))
    if ix_upper >= len(vals):
        ix_upper = ix_upper - 1
    if ix_lower <= len(vals):
        ix_lower = ix_lower - 1
    if coef:
        CI = (np.round(np.exp(sort[ix_lower]),digits), np.round(np.exp(sort[ix_upper]),digits))
    else:
        CI = (np.round((sort[ix_lower]),digits), np.round((sort[ix_upper]),digits))
    return CI

def vars(data, labels=None, normalize_data = False):
    if normalize_data:
        data = self.normalize(data)
    # labels = self.targets_dict[dat_type]
    if labels:
        cleared = data[np.array(labels) == 'Cleared']
        recur = data[np.array(labels) == 'Recur']
        within_class_vars = [np.var(cleared, 0), np.var(recur, 0)]
        class_means = [np.mean(cleared, 0), np.mean(cleared, 0)]

        total_mean = np.mean(data, 0)
        between_class_vars = 0
        for i in range(2):
            between_class_vars += (class_means[i] - total_mean)**2
    
    else:
        within_class_vars = None
        between_class_vars = None

    total_vars = np.std(data, 0)/np.mean(data,0)
    vardict = {'within':within_class_vars,'between':between_class_vars,'total':total_vars}
    return vardict

def filter_vars(data, labels=None, perc=30, var_type = 'total', normalize_data = False):
    if labels == None:
        assert(var_type == 'total')
    vardict = vars(data, labels, normalize_data)
    variances = vardict[var_type]
    variances = variances.replace(np.nan,0)
    
    rm2 = set(np.where(variances > np.percentile(variances, perc))[0])

    temp = data.iloc[:,list(rm2)]
    # import pdb; pdb.set_trace()
    if len(np.where(np.sum(temp,0)==0)[0]) > 0:
        import pdb; pdb.set_trace()
    return data.iloc[:,list(rm2)]


# def leave_one_out_cv(data, labels, num_folds=None):

#     if isinstance(data.index.values[0], str):
#         patients = np.array([int(i.split('-')[1])
#                                 for i in data.index.values])
#         pdict = {}
#         for i, pt in enumerate(patients):
#             pdict[pt] = labels[i]

#         ix_all = []
#         for ii in pdict.keys():
#             pt_test = ii
#             pt_train = list(set(pdict.keys()) - set([ii]))
#             ixtrain = (np.concatenate(
#                 [np.where(patients == j)[0] for j in pt_train]))
#             ixtest = np.where(patients == pt_test)[0]
#             set1 = set([patients[ix] for ix in ixtest])
#             set2 = set([patients[ix] for ix in ixtrain])
#             set1.intersection(set2)

#             ix_all.append((ixtrain, ixtest))
#             assert(not set1.intersection(set2))

#     else:
#         ix_all = []
#         # CHANGE LINE!
#         for ixs in range(len(labels)):
#             ixtest = [ixs]
#             ixtrain = list(set(range(len(labels))) - set(ixtest))
#             ix_all.append((ixtrain, ixtest))
#     return ix_all


def isclose(a, b, tol=1e-03):
    return (abs(a-b) <= tol).all()

def asv_to_name(asv, tax_dat = ['inputs/dada2-taxonomy-rdp.csv','inputs/dada2-taxonomy-silva.csv']):
    tdat = [pd.read_csv(t) for t in tax_dat]
    met_class = []

    if len(asv)>100:
        classification = []
        for i,td in enumerate(tdat):
            td_out = np.array([str(x) for x in td[asv]])[-2:]
            td_out = [t for t in td_out if t != 'nan']
            classification.append(' '.join(td_out))

        
        cl = np.unique(classification)
        if len(cl) > 1:
            cl = cl[0] + ' ; ' + cl[1]
            met_class = cl
        else:
            met_class = cl[0]
    return met_class

def return_taxa_names(sequences, tax_dat = ['inputs/dada2-taxonomy-rdp.csv','inputs/dada2-taxonomy-silva.csv']):
    tdat = [pd.read_csv(t) for t in tax_dat]
    met_class = []
    for metab in sequences:
        if len(metab)>100:
            classification = []
            for i,td in enumerate(tdat):
                td_out = np.array([str(x) for x in td[metab]])[-2:]
                td_out = [t for t in td_out if t != 'nan']
                classification.append(' '.join(td_out))

            
            cl = np.unique(classification)
            if len(cl) > 1:
                cl = cl[0] + ' ; ' + cl[1]
                met_class.append(cl)
            else:
                met_class.append(cl[0])
        else:
            met_class.append(metab)
    return met_class 


def return_sig_biomarkers(dset_name,data = None, tax_dat = ['inputs/dada2-taxonomy-rdp.csv','inputs/dada2-taxonomy-silva.csv'], thresh = 10):
    df_new = pd.read_csv(dset_name) 
  
    # saving xlsx file 
    GFG = pd.ExcelWriter(dset_name.split('.')[0] + '.xlsx') 
    df_new.to_excel(GFG, index = False) 

    GFG.save() 
    xl = pd.ExcelFile(dset_name.split('.')[0] + '.xlsx')
    dset = xl.parse(header = 0, index_col = 0)
    
    lr_l0 = dset.iloc[:,:2]
    lr_l0 = lr_l0[lr_l0['Odds ratio < 1'] < 1]
    
    lr_l0 = lr_l0.iloc[:thresh,:]

    lr_g0 = dset.iloc[:,2:]
    lr_g0 = lr_g0[lr_g0['Odds ratio > 1'] > 1]
    lr_g0 = lr_g0.iloc[:thresh,:]

    metabs = lr_l0
    metabs = np.concatenate((np.array(lr_l0), np.array(lr_g0)))

    if tax_dat is not None:
        met_class = return_taxa_names(metabs[:,0])
                
        if len(metabs[1,0])> 100:
            new_df = pd.DataFrame(np.array([metabs[:,0], metabs[:,1], met_class]).T, columns = ['Biomarker', 'Log Odds', 'Genus Species'] )
        else:
            new_df = pd.DataFrame(metabs, columns = ['Biomarker', 'Log Odds'] )
    else:
        new_df = pd.DataFrame(metabs, columns = ['Biomarker', 'Log Odds'] )
    
    
    if data is not None:
        new_data = data[metabs[:,0]]

        if '16s' in dset_name or 'cdiff' in dset_name:
            new_data = np.divide(new_data.T, np.sum(new_data,1)).T
        return new_df, new_data
    else:
        return new_df


def make_fig(cd, dset_name,label, legend = False, tax_dat = None, thresh = 10):
    
    df_new = pd.read_csv(dset_name) 
  
    # saving xlsx file 
    GFG = pd.ExcelWriter(dset_name.split('.')[0] + '.xlsx') 
    df_new.to_excel(GFG, index = False) 

    GFG.save() 
    xl = pd.ExcelFile(dset_name.split('.')[0] + '.xlsx')
    dset = xl.parse(header = 0, index_col = 0)
    
    lr_l0 = dset.iloc[:,:2]
    lr_l0 = lr_l0[lr_l0['Odds ratio < 1'] < 1]
    
    lr_l0 = lr_l0.iloc[:thresh,:]

    lr_g0 = dset.iloc[:,2:]
    lr_g0 = lr_g0[lr_g0['Odds ratio > 1'] > 1]
    lr_g0 = lr_g0.iloc[:thresh,:]

    metabs = lr_l0
    metabs = np.concatenate((np.array(lr_l0), np.array(lr_g0)))

    
    if tax_dat is not None:
        met_class = return_taxa_names(metabs[:,0], tax_dat = tax_dat)

        metabs[:,0] = np.array(met_class)

        cats = np.unique(np.array(['No Carbon Group']*len(met_class)))
        gp_all = np.array(['No Carbon Group']*len(met_class))
    else:
        gp = []
        for metab in lr_l0['Metabolites']:
            ix = np.where(cd.carbon_gps.isin([metab]))[0]
            if len(ix) > 0:
                gp.append(cd.carbon_gps.index[ix][0])
            else:
                gp.append('No Carbon Group')

        gpg = []
        for metab in lr_g0['Metabolites.1']:
            ix = np.where(cd.carbon_gps.isin([metab]))[0]
            if len(ix) > 0:
                gpg.append(cd.carbon_gps.index[ix][0])
            else:
                gpg.append('No Carbon Group')

        if len(gp) == 0 and len(gpg) ==0:
            return 
        print(label)
        gp_all = gp
        gp_all.extend(gpg)
        cats = np.unique(np.array(gp_all)).tolist()
        try:
            cats.remove('No Carbon Group')
        except:
            return
        cats.append('No Carbon Group')
        cats = np.array(cats)

    fig, ax = plt.subplots(figsize = (30,40))

    matplotlib.rcParams.update({'font.size': 42})

    start = 0
    ticks = []

    metabolites = []
    
    mx = max([np.log(m) for m in metabs[:,1]])
    mn = min([np.log(m) for m in metabs[:,1]])
    
    
    for i, pl in enumerate(cats):
        ix = np.where(np.array(gp_all) == pl)[0]
        try:
            to_plot = np.log(metabs[ix.squeeze(),1].tolist())
        except:
            to_plot = np.log(metabs[ix.squeeze(),1])

        if isinstance(to_plot, float):
            cvec = ['g' if i < 0 else 'b' for i in [to_plot]]
        else:
            cvec = ['g' if i < 0 else 'b' for i in to_plot]

        ax.barh(np.arange(start, start+len(ix)),to_plot, color = cvec)
        ax.barh(np.arange(start, start+len(ix)),to_plot, color = cvec)

        last_start = start
        start = start + len(ix)
        ticks.append((start -2+ last_start )/2 + 0.5)
        if i < len(cats)-1:
            ax.hlines(start -1 + .5,mn,mx, linewidth = 2)
#         else:
#             ax.hlines(start-0.5,-2,np.log(0.6e7), linewidth = 2)

        if len(ix)>1:
            metabolites.extend(metabs[ix.squeeze(),0])
        else:
            metabolites.extend([metabs[ix.squeeze(),0]])


    # import pdb; pdb.set_trace()
    ax.vlines(0,-.5,19.5, linewidth = 0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax2 = ax.twinx()
    l = ax.get_ylim()
    l2 = ax2.get_ylim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks2 = f(ax.get_yticks())
    ticks2_means = [np.mean([ticks2[i], ticks2[i+1]]) for i in range(0,len(ticks2)-1)]
    ticks2_fin = zip(ticks2,ticks2_means)
    ticks2_fin2 = np.hstack(np.array([list(x) for x in ticks2_fin]))

    ticks2_fin2 = (ticks2_fin2 - min(ticks2_fin2)+.15)/max(ticks2_fin2 - min(ticks2_fin2)+.15)
    ticks2_fin2 = ticks2_fin2 + -.059

    ax2.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks2_fin2))
    ax2.set_yticklabels(metabolites, fontsize = 60)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)

#     ax.get_xaxis().set_ticks([])

    if len(cats) > 1:
        ax.tick_params(axis='x', which='major', labelsize=85)
        ax.set_yticks(ticks)
        ax.set_yticklabels(cats, fontsize = 70)

    ax.set_xlabel('Log Odds', fontsize = 75)

    from matplotlib.lines import Line2D

    custom_lines = [Line2D([0], [0], color='g', lw=10),
                    Line2D([0], [0], color='b', lw=10)]
    
    if legend:
        plt.legend(custom_lines, ['Decreases \n Recurrence Odds', 'Increases \n Recurrence Odds'], 
                loc = 'upper left', fontsize = 70, bbox_to_anchor = (-.2,.83))
        plt.rcParams["legend.loc"] = 'upper right'
        plt.rcParams["legend.bbox_to_anchor"] =  (0.5, 0.5)
        plt.tight_layout()
    plt.savefig(dset_name.split('.')[0] + '.pdf',bbox_inches='tight')
    plt.show()

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def merge_16s(filename1, filename2, output_fname):
    xl = pd.ExcelFile(filename1)
    df1 = xl.parse(index_col = 0)
    # new_cols = ['-'.join(x.split('-')[1:]) for x in df1.columns.values]
    # df1.columns = new_cols
    seq1 = df1.index.values

    xl = pd.ExcelFile(filename2)
    df2 = xl.parse(index_col = 0)
    new_cols = ['-'.join(['cdiff', x.split('-')[0], ''.join(x.split('-')[1:])]) for x in df2.columns.values]
    df2.columns = new_cols
    seq2 = df2.index.values

    total_seqs = np.array(np.unique(np.concatenate((seq1,seq2))))

    total_arrs = []
    for seq in total_seqs:
        ix1 = np.where(seq == seq1)[0]
        if len(ix1) > 0:
            arr1 = df1.iloc[ix1,:]
        else:
            arr1 = pd.DataFrame(np.zeros((1, len(df1.iloc[0,:]))), index = [seq], columns = df1.columns.values)
            
        ix2 = np.where(seq == seq2)[0]
        if len(ix2) > 0:
            arr2 = df2.iloc[ix2,:]
        else:
            arr2 = pd.DataFrame(np.zeros((1, len(df2.iloc[0,:]))), index= [seq], columns = df2.columns.values)
        
        total_arrs.append(pd.concat((arr1, arr2), axis = 1))

        total_arrs_f = pd.concat(total_arrs)
        total_arrs_f.to_excel(output_fname, index = True)

def leave_one_out_cv(data, labels, folds = None, ddtype = 'week_one'):
    if ddtype == 'all_data':
        assert(data.shape[0]> 70)
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
    
    if folds is not None:
        if isinstance(labels[0], str):
            cl = np.where(labels == 'Cleared')[0]
            re = np.where(labels == 'Recur')[0]
        else:
            cl = np.where(labels == 0)[0]
            re = np.where(labels == 1)[0]
        random_select_cl = np.random.choice(cl, int(folds/2))
        random_select_re = np.random.choice(re, int(folds/2))
        random_select = np.append(random_select_cl, random_select_re)
        ix_all = (np.array(ix_all)[random_select]).tolist()
    return ix_all


def split_to_folds(in_data, in_labels, folds=5, ddtype = 'week_one'):
    # Like split test-train except makes all folds at once

    # If all data, have to take into acct patients w/ multiple timepoints
    in_labels = np.array(in_labels)
    # if ddtype == 'all_data':
    data_perc_take = 1/folds
    patients = np.array([int(ti.split('-')[0])
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
    # else:
    #     # patients = in_data.index.values
    #     # for f in range(folds):

    #     skf = StratifiedKFold(folds)
    #     zip_ixs = skf.split(in_data, in_labels)
    return zip_ixs

def get_resdict_from_file(path):
    # path = 'out_test/'
    res_dict = {}
    for file in os.listdir(path):
        if file == '.DS_Store':
            continue
        key_name = file.split('.')[0].split('_')[0]
        seed = int(file.split('.')[0].split('_')[1])
        ix = int(file.split('.')[0].split('_')[2])
        with open(inner_path + file, 'rb') as f:
            # temp is a dictionary with parameter keys 
            temp = pkl.load(f)
            if key_name not in res_dict.keys():
                res_dict[key_name] = {}
            if ix not in res_dict[key_name].keys():
                res_dict[key_name][ix] = {}         
            for test_param in temp[ix].keys():
                if test_param not in res_dict[key_name][ix].keys():
                    res_dict[key_name][ix][test_param] = {}
                # res dict keys go model, held out test index, parameter to test, seed
                res_dict[key_name][ix][test_param][seed] = temp[ix][test_param]
    return res_dict

def get_best_param(res_dict, key_name):
    best_param = {}
    for ix in res_dict[key_name].keys():
        all_aucs = {}
        auc = {}
        params = list(res_dict[key_name][ix].keys())
        for param in params:
            if param not in auc.keys():
                auc[param] = []
            for seed in res_dict[key_name][ix][param].keys():
                auc[param].append(res_dict[key_name][ix][param][seed]['metrics']['auc'])
        
            all_aucs[param] = np.median(auc[param])
                
        vec = np.array([all_aucs[p] for p in params])
        ma = moving_average(vec, n=5)
        offset = int(np.floor(5./2))
        max_ix = np.argmax(ma) + offset
        
        best_param[ix] = params[max_ix]
    return best_param

# # def make_summary_fig():
#     import matplotlib as mpl

#     plt.rcParams.update({'font.size': 25})

#     img = np.zeros((len(cd.pt_info_dict.keys()), len(np.arange(0, 8.5, .5))))
#     times = np.arange(0, 8.5, .5)
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             if times[j] in ml.new_info_dict[i].keys():
#                 if ml.new_info_dict[i][times[j]]['PATIENT STATUS (BWH)'] == 'Recur':
#                     img[i, j] = 2
#                 else:
#                     img[i, j] = 1


#     # Create Cmap
#     cmap = plt.cm.jet  # define the colormap
#     # extract all colors from the .jet map
#     cmaplist = [cmap(i) for i in range(cmap.N)]
#     # force the first color entry to be grey
#     cmaplist[0] = (.5, .5, .5, 1.0)
#     cmaplist[-1] = (1, 0, 0, 1.0)
#     # create the new map
#     cmap = mpl.colors.LinearSegmentedColormap.from_list(
#         'Custom cmap', cmaplist, cmap.N)

#     # define the bins and normalize
#     bounds = np.linspace(0, 3, 4)
#     norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#     fig, ax = plt.subplots(figsize=(img.shape[0], np.round(img.shape[1]*1.5)))
#     im = ax.imshow(img, cmap=cmap)

#     # create a second axes for the colorbar
#     ax2 = fig.add_axes([0.58, 0.6, 0.01, 0.07])

#     cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
#                                 spacing='proportional', boundaries=bounds)

#     ax2.text(1.2, .05, 'NO DATA')
#     ax2.text(1.2, .39, 'NO C. DIFF DETECTED')
#     ax2.text(1.2, .73, 'C. DIFF DETECTED')

#     ax2.tick_params(which='major', length=0)
#     ax2.set_yticklabels(['', '', ''], minor=False)

#     # Major ticks
#     ax.grid(b=None, which='major')
#     ax.set_xticks(np.arange(0, len(np.arange(0, 8.5, .5)), 1), minor=False)
#     ax.set_xticklabels(np.arange(0, 8.5, .5), fontsize=18, minor=False)

#     ax.set_yticks(np.arange(.5, len(cd.pt_info_dict.keys())+.5, 1), minor=True)
#     ax.set_xticks(np.arange(0+.5, len(np.arange(0, 8.5, .5))+.5, 1), minor=True)


#     ax.grid(color='w', linestyle='-', linewidth=2, which='minor')
#     ax.set_yticks(np.arange(img.shape[0]))
#     ax.set_yticklabels(np.arange(img.shape[0]), fontsize=20)
#     ax.set_xlabel('WEEKS', fontsize=25)
#     ax.set_ylabel('PATIENTS', fontsize=25)

#     plt.tight_layout()
#     plt.savefig(path + 'pt_summary.png')
#     # plt.colorbar()
