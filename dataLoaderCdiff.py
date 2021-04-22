import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import copy
from itertools import groupby

class cdiffDataLoader():
    def __init__(self, file_name="CDiffMetabolomics.xlsx", file_name16s='seqtab-nochim-total.xlsx', 
            lod = 10, perc_met = 0.25, perc_bug = 0.15, pt_thresh = 1, input_path = 'inputs/'):
        self.filename = file_name
        self.path = input_path
    
        self.xl = pd.ExcelFile(self.path + self.filename)

        self.cdiff_raw = self.xl.parse('OrigScale', header=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        self.cdiff_raw = self.cdiff_raw.replace('Cleared','Asymptomatic')

        self.toxin_fname = pd.ExcelFile(self.path + 'Toxin B and C. difficile Isolation Results.xlsx')
        self.toxin_data = self.toxin_fname.parse('ToxB',header = 4, index_col = 0)
        self.toxin_data = self.toxin_data.drop('Crimson ID', axis=1)
        self.toxin_data = ((self.toxin_data.replace(
            '+', 1)).replace('-', 0)).replace('<0.5', 0)
        self.toxin_data = self.toxin_data.replace(
            'Not done - no sample available', 0).fillna(0)

        xl_carrier = pd.ExcelFile(self.path + self.filename.split('.')[0] + '_Carrier' + '.xlsx')
        cdiff_carrier = xl_carrier.parse('OrigScale', header=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index_col=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        header_lst_carrier = [i[4:] for i in cdiff_carrier.columns.values if 'Unnamed' not in i[0]]
        header_labels_carrier = header_lst_carrier[0]
        self.carrier_dict = {xh[0]:xh[4] for xh in header_lst_carrier}

        xl = pd.ExcelFile(self.path + "20200120_HumanCarbonSourceMap.xlsx")
        self.carbon_gps = xl.parse(header = 0, index_col = 0)

        # self.cdiff_raw.columns = self.cdiff_raw.columns.str.replace("Carrier", "Cleared")
        header_lst = [i[4:]
                      for i in self.cdiff_raw.columns.values if 'Unnamed' not in i[0]]
        self.header_labels = header_lst[0]
        self.targets_dict = {xh[0]:xh[4] for xh in header_lst}

        temp_xl = pd.ExcelFile(self.path + "cdiff_microbe_labels.xlsx")
        self.microbiome_labels = temp_xl.parse()
        # get rid of repeat values (1a, 1b, 1c, etc.)
        tmpts = [c[4].split('-')[-1] for c in self.cdiff_raw.columns.values]
        ix = []
        for i, tmpt in enumerate(tmpts):
            try:
                float(tmpt)
            except:
                ix.append(i)
        self.cdiff_raw = self.cdiff_raw.iloc[:, [
            j for j, c in enumerate(self.cdiff_raw.columns) if j not in ix]]
        
        header_lst = [i[4:] for i in self.cdiff_raw.columns.values if 'Unnamed' not in i[0]]
        self.header_names = header_lst
        # pt_info_dict = {header_names[j][0]:{header_labels[i]:header_names[j][i] for i in range(1,len(header_labels))} for j in range(len(header_names))}

        self.row_names = self.cdiff_raw.index.values[1:]
        self.row_labels = self.cdiff_raw.index.values[0]
        self.metabolome_info_dict = {self.row_names[j][1]: {self.row_labels[i]: self.row_names[j][i] for i in range(len(self.row_labels))} for j in range(len(self.row_names))}

        self.cdiff_raw = self.cdiff_raw.iloc[1:, :]

        self.targets_cd = {x[0]:x[4] for x in self.header_names}

        colnames = [x[4] for x in self.cdiff_raw.columns.values]
        self.cdiff_data = pd.DataFrame(np.array(
            self.cdiff_raw), columns=colnames, index=[x[1] for x in self.row_names])

        self.cdiff_data = self.cdiff_data.fillna(0)
        self.raw16s = pd.ExcelFile(self.path + file_name16s)
        self.raw16s = self.raw16s.parse()

        self.labels16s_dict = {101:0,102:0,103:0,105:1,106:0,108:1,109:0,114:0,115:0,116:0,119:1,120:1,121:1,123:0,124:1,\
            126:0,127:0,129:0,130:1,131:0,132:0,133:0,134:1,136:1,138:0,139:0,140:0,141:1,142:0,143:1,144:0,145:0,146:1,147:0,148:1,\
                149:1,150:0,151:0,152:0,153:0,155:0,156:1,158:1,160:0,161:1,162:1,163:1,165:1,167:1,168:'g',169:'g',170:'g',171:'g',\
                    173:1,174:0,175:0,107:1,117:1,118:1,164:0}


        self.cl_dict = {0:'Cleared',1:'Recur','g':'Unknown','b':'Unknown'}
        # dcols = ['-'.join(x.split('-')[1:])
        #          for x in self.raw16s.columns.values[1:]]
        dcols = self.raw16s.columns.values[1:]
        dcol = []
        for x in dcols:
            if len(x.split('-')) == 3:
                dcol.append('.'.join([x[:5], x[-1]]))
            elif len(x.split('-')[1]) == 2:
                dcol.append('.'.join([x[:5], x[-1]]))
            else:
                dcol.append(x)
        dcols = dcol
        self.data16s = pd.DataFrame(
            np.array(self.raw16s.iloc[:, 1:]), columns=dcols, index=self.raw16s.iloc[:, 0])

        self.raw16s = copy.deepcopy(self.data16s)
        # If sample is less than limit of detection, remove
        # temp = self.data16s.iloc[:, 1:].copy().copy()
        # temp = temp.replace(0, np.inf)
        # lod = 10
        # ixs = [np.where(temp.iloc[i, :] < lod)[0] for i in range(temp.shape[0])]
        # for i, ix in enumerate(ixs):
        #     if len(ix) > 0:
        #         self.data16s.iloc[i, ix] = 0

        

        filt_out = self.filter_by_pt(self.cdiff_data, perc_met)
        self.cdiff_data = self.cdiff_data.iloc[filt_out,:]

        # ix_zeros = np.where(self.data16s < lod)[0]
        ix_keep = self.filter_by_pt(self.data16s, perc_bug, pt_thresh = pt_thresh, meas_thresh = lod)

        # print(self.data16s.shape)
        self.data16s = self.data16s.iloc[ix_keep,:]
        # print(self.data16s.shape)

        self.counts16s = self.data16s.copy(deep=True)

        # self.data16s = self.make_proportions(self.data16s)

        self.make_pt_dict(self.cdiff_data)
        self.all_16s_info_dict = self.add_16s_to_info_dict(self.data16s, self.pt_info_dict)
        # self.counts_info_dict = self.add_16s_to_info_dict(self.counts16s, self.pt_info_dict)

        self.metabolome_pts = np.concatenate(
            [[self.pt_info_dict[i][k]['CLIENT SAMPLE ID'] for k in self.pt_info_dict[i].keys()] for i in self.pt_info_dict.keys()])
        self.microbiome_pts = dcols

        # self.info_dict_16s = self.make_16s_dict(self.data16s)
        # make simple labels dict

    def make_simple_labels(self):
        pts = self.pt_info_dict.keys()
        self.simple_labels_by_tmpt={}
        self.simple_labels_by_pt={}
        for pt in pts:
            tmpts = self.pt_info_dict[pt].keys()
            for tmpt in tmpts:

                label = self.pt_info_dict[pt][tmpt]['PATIENT STATUS (BWH)']
                if np.round(tmpt) == tmpt:
                    tf = np.int(tmpt)
                else:
                    tf = tmpt
                self.simple_labels_by_tmpt[pt + '-' + str(tf)] = label=='RECUR'
            self.simple_labels_by_pt[pt] = label=='RECUR'
        for i,ix in enumerate(self.data16s.columns.values):
            if ix not in self.simple_labels_by_tmpt.keys():
                if i != len(self.data16s.columns.values):
                    if np.float(ix.split('-')[1])>self.data16s.columns.values[i+1]:
                        self.simple_labels_by_tmpt[ix] = self.labels16s_dict[int(ix.split('-')[0])]
                    else:
                        self.simple_labels_by_tmpt[ix] = 0
                else:
                    self.simple_labels_by_tmpt[ix] = self.labels16s_dict[int(ix.split('-')[0])]
            if ix.split('-')[0] not in self.simple_labels_by_pt.keys():
                self.simple_labels_by_pt[ix.split('-')[0]] = self.labels16s_dict[int(ix.split('-')[0])]


    def make_pt_dict(self, data, idxs=None):

        pt_names = np.array([h[0].split('-')[0] for h in self.header_names])
        pts = []
        for i,n in enumerate(np.unique(pt_names)):
            pts.extend([str(i+1) + '.' + str(j) for j in range(len(np.where(pt_names == n)[0]))])

        ts = np.array([h[0].split('-')[1] for h in self.header_names])
        self.times = []
        for el in ts:
            self.times.append(float(el))
            # except:
            #     tm = str(ord(list(el)[1])-97)
            #     self.times.append(float(list(el)[0] + '.' + tm))
        idx_num = np.where(np.array(self.row_labels) == 'BIOCHEMICAL')[0][0]
        idx_val = [r[idx_num] for r in self.row_names]
        if idxs is None:
            cdiff_raw_sm = data
            cdiff_raw_sm = cdiff_raw_sm.fillna(0)
            # import pdb
            # pdb.set_trace()
            self.data = pd.DataFrame(
                np.array(data), columns=data.columns.values, index=data.index.values)
        else:
            cdiff_raw_sm = pd.DataFrame(np.array(data)[idxs, :], index = np.array(idx_val)[idxs])
            cdiff_raw_sm = cdiff_raw_sm.fillna(0)
            self.data = pd.DataFrame(
                np.array(data)[idxs, :], columns=self.header_names, index=np.array(self.row_names)[idxs])

        self.data_sm = cdiff_raw_sm
        self.pt_info_dict = {}
        pt_key = pt_names[0]
        self.pt_info_dict[pt_key] = {}
        

        for j in range(len(self.header_names)):
            if j != 0 and pt_names[j] != pt_names[j-1]:
                pt_key = pt_names[j]
                self.pt_info_dict[pt_key] = {}
                self.pt_info_dict[pt_key][self.times[j]] = {
                    self.header_labels[i]: self.header_names[j][i] for i in range(len(self.header_labels))}
                self.pt_info_dict[pt_key][self.times[j]].update(
                    {'DATA': cdiff_raw_sm.iloc[:, j]})


            else:
                self.pt_info_dict[pt_key][self.times[j]] = {
                    self.header_labels[i]: self.header_names[j][i] for i in range(len(self.header_labels))}
                self.pt_info_dict[pt_key][self.times[j]].update(
                    {'DATA': cdiff_raw_sm.iloc[:, j]})

            # self.pt_info_dict[pt_key][self.times[j]]['Toxin Status'] = \
            #     self.toxin_data[pt_key + self.times[j]]cd.toxin_data.loc['Toxin detected']
            # self.pt_info_dict[pt_key][self.times[j]]['Cdiff Status'] = \
            #     self.toxin_data[pt_key + self.times[j]]cd.toxin_data.loc['C diff isolate']

    def add_16s_to_info_dict(self, data16s, pt_info_dict):
        pt_info_dict_all16s = copy.deepcopy(pt_info_dict)
        all_pts = np.unique(list(self.cdiff_data)+ list(self.data16s))
        for key in all_pts:
            key_pt = key.split('-')[0]
            key_tmpt = float(key.split('-')[1])
            if key_pt in pt_info_dict.keys() and key_tmpt in pt_info_dict[key_pt].keys():
                pt_label = key
                try:
                    col_16s = data16s[pt_label]
                    pt_info_dict[key_pt][key_tmpt]['16s'] = col_16s
                    pt_info_dict_all16s[key_pt][key_tmpt]['16s'] = col_16s
                except:
                    continue

            else:

                pt_label = key
                col_16s = data16s[pt_label]
                try:
                    pt_info_dict[key_pt][key_tmpt]['16s'] = col_16s
                    pt_info_dict_all16s[key_pt][key_tmpt]['16s'] = col_16s
                except:
                    if key_pt not in pt_info_dict_all16s.keys():
                        pt_info_dict_all16s[key_pt] = {}
                    if key_tmpt not in pt_info_dict_all16s[key_pt].keys():
                        pt_info_dict_all16s[key_pt][key_tmpt] = {}
                    pt_info_dict_all16s[key_pt][key_tmpt]['16s'] = col_16s
                
                if key_tmpt == list(pt_info_dict_all16s[key_pt].keys())[-1]:
                    lab = self.labels16s_dict[int(pt_label.split('-')[0])]
                    label = self.cl_dict[lab]
                    pt_info_dict_all16s[key_pt][key_tmpt]['PATIENT STATUS (BWH)'] = label
                else:
                    pt_info_dict_all16s[key_pt][key_tmpt]['PATIENT STATUS (BWH)'] = 'Cleared'

        return pt_info_dict_all16s
    
    def make_proportions(self, data):
        total_counts = np.sum(data, 0)
        data = data / total_counts
        return data

    def index_by(self, colname, rowname):
        cols = self.data.columns.values
        ix1 = np.where(np.array(self.header_labels) == colname)[0][0]

        c_map = {cols[i]: cols[i][ix1] for i in range(len(cols))}
        # colnames = [cols[i][ix1] for i in range(len(cols))]

        rows = self.data.index.values
        ix2 = np.where(np.array(self.row_labels) == rowname)[0][0]

        # rownames = [rows[i][ix2] for i in range(len(rows))]

        r_map = {rows[i]: rows[i][ix2] for i in range(len(rows))}
        data = self.data.rename(columns=c_map, index=r_map)
        return data

    def filter_by_pt(self, dataset, perc = .25, pt_thresh = 2, meas_thresh = 0):
        pts = [x.split('-')[0] for x in dataset.columns.values]
        tmpts = [x.split('-')[1] for x in dataset.columns.values]

        # mets is dataset with ones where data is present, zeros where it is not
        mets = np.zeros(dataset.shape)
        mets[dataset > meas_thresh] = 1

        # if measurement of a microbe/metabolite only exists in less than pt_thresh timepoints, set that measurement to zero
        for pt in pts:
            ixs = np.where(np.array(pts)==pt)[0]
            tmpts_pt = np.array(tmpts)[ixs]
            mets_counts = np.sum(mets,1).astype('int')
            met_rm_ixs = np.where(mets_counts < pt_thresh)[0]
            for ix in ixs:
                mets[met_rm_ixs, ix] = 0
        labels = np.array([x[4] for x in self.header_names])
        lab_cats = np.unique(labels)
        mets_all_keep = []
        # For each class, count how many measurements exist within that class and keep only measurements in X perc in each class
        for lab_cat in lab_cats:
            mets_1 = mets[:,np.where(labels == lab_cat)[0]]
            met_counts = np.sum(mets_1,1)
            met_keep_ixs = np.where(met_counts >= np.round(perc*mets_1.shape[1]))[0]
            mets_all_keep.extend(met_keep_ixs)
        return np.unique(mets_all_keep)


    # def filter_by_pt(self,dataset, perc=None, val = None, tol = 1e-4):
    #     ix_keep_all = []
    #     pts = [x.split('-')[0] for x in dataset.columns.values]

    #     if val is None and perc is not None:
    #         val_1 = np.int(dataset.shape[1]*(1-perc))
    #         val_2 = np.int(len(np.unique(pts))*(1-perc))
    #         val = [val_1, val_2]
    #     else:
    #         val = [val, np.int(dataset.shape[1]*(1-perc))]

    #     cnts_vec = []
    #     cnts_vec_pts = []
    #     for jj, use_pts in enumerate([True, False]):
    #         ix_keep=[]
    #         for i in range(dataset.shape[0]):
    #             met_row = dataset.iloc[i, :]

    #             ixs = [[ij]*len(list(j)) for ij, (k, j) in enumerate(groupby(pts))]
    #             # import pdb; pdb.set_trace()
    #             met_counts = [np.sum([met_row.iloc[ii] for ii in ix]) for ix in ixs]

    #             if use_pts:
    #                 ix_zero = np.where(np.array(met_counts) < tol)[0]
    #                 counts = len(np.unique(pts))-len(ix_zero)
    #             else:
    #                 ix_zero = np.where(np.array(met_row) < tol)[0]
    #                 tmpts = [x.split('-')[1]
    #                          for x in dataset.columns.values[ix_zero]]
    #                 cnts_minus = len(np.where(np.array(tmpts) == '0')[0])
    #                 counts = dataset.shape[1] - len(ix_zero)
    #                 counts = counts - cnts_minus
    #             if counts >= val[jj]:
    #                 ix_keep.append(i)
    #             if use_pts:
    #                 cnts_vec_pts.append(counts)
    #             else:
    #                 cnts_vec.append(counts)
    #         ix_keep_all.append(ix_keep)
            
    #     return ix_keep_all, cnts_vec, cnts_vec_pts


    def amt_over_time(self, mols, dict_name=None, pts=None, save = None, filename = None):
        mapping = {'Cleared': 'green', 'Recur': 'red'}
        if pts is None:
            pts = self.pt_info_dict.keys()
        for molecule in mols:
            # plt.figure()
            fig, ax = plt.subplots()
            for patient in pts:
                if dict_name is None:
                    tmpts = list(self.pt_info_dict[patient].keys())
                    data = [self.pt_info_dict[patient][t]['DATA'][molecule] for t in tmpts]
                    label = [self.pt_info_dict[patient][t]
                            ['PATIENT STATUS (BWH)'] for t in tmpts]
                else:
                    tmpts = list(dict_name[patient].keys())
                    data = [dict_name[patient][t]['DATA'][molecule]
                            for t in tmpts]
                    label = [dict_name[patient][t]
                            ['PATIENT STATUS (BWH)'] for t in tmpts]

                from matplotlib.lines import Line2D
                custom_lines = [Line2D([0], [0], color='g', lw=4),
                                Line2D([0], [0], color='r', lw=4)]

                
                ax.legend(custom_lines, ['Asymptomatic', 'Recur'],fontsize=12)

                labels = [colors.to_rgb(mapping[l]) for l in label]
                ax.scatter(tmpts, data, color=labels, linewidth=5)
                ax.plot(tmpts, data, '--k', linewidth=.5)
                ax.set_yscale('log')
                plt.xlabel('Week',fontsize=15)
                plt.ylabel('Amt',fontsize = 15)
                plt.title(molecule,fontsize=18)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
            filename = 'lineplot_' + molecule + '.pdf'
            if save:
                plt.savefig(filename)
            plt.show()

