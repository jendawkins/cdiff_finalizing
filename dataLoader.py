from helper import *
import pandas as pd
import numpy as np

class dataLoader():
    def __init__(self, path = "inputs", filename_cdiff = "CDiffMetabolomics.xlsx",
        filename_16s = 'seqtab-nochim-total.xlsx', filename_ba = 'MS FE BANEG PeakPantheR',
            filename_toxin = 'Toxin B and C. difficile Isolation Results.xlsx',
                 filename_CSgps = '20200120_HumanCarbonSourceMap.xlsx', pt_perc = 0.25, meas_thresh = 10, var_perc = 5, pt_tmpts = 1):
        
        self.path = path
        self.filename_cdiff = filename_cdiff
        self.filename_16s = filename_16s
        self.filename_ba = filename_ba
        self.filename_toxin = filename_toxin
        self.filename_CSgps = filename_CSgps

        self.pt_perc = pt_perc
        self.meas_thresh = meas_thresh
        self.var_perc = var_perc
        self.pt_tmpts = pt_tmpts

        self.load_cdiff_data()
        self.load_ba_data()
        self.load_16s_data()
        self.keys = {'metabs':self.cdiff_data_dict,'16s':self.data16s_dict,'bile_acids':self.ba_data}
        # self.week_one = {}
        # for key, value in keys.items():
        #     temp = self.get_week_x(value['data'],value['targets_by_pt'], week = 1)
        #     self.week_one[key] = self.filter_transform(temp['x'], value['targets_by_pt'], key), temp['y']

        self.week = {}
        self.week_one = {}
        self.week_raw = {}
        self.week_filt = {}
        for key, value in self.keys.items():
            self.week[key] = {}
            self.week_raw[key] = {}
            self.week_filt[key] = {}
            value['data'] = value['data'].fillna(0)
            value['targets'] = value['targets'].replace('Recur', 'Recurrer').replace('Cleared','Non-recurrer')
            value['targets_by_pt'] = value['targets_by_pt'].replace('Recur', 'Recurrer').replace('Cleared', 'Non-recurrer')
            value['filtered_data'] = self.filter_transform(value['data'], targets_by_pt = None, key = key)
            temp = self.get_week_x(value['filtered_data'], value['targets_by_pt'], week=1)

            self.week_one[key] = temp['x'], temp['y']
            temp_filt = filter_by_pt(value['data'], targets=None, perc=self.pt_perc, pt_thresh=self.pt_tmpts,
                                     meas_thresh=self.meas_thresh)
            for week in [0,1,1.5,2,2.5,3,3.5,4]:
                self.week[key][week] = self.get_week_x_step_ahead(value['filtered_data'], value['targets_by_pt'], week = week)
                self.week_filt[key][week] = self.get_week_x_step_ahead(temp_filt, value['targets_by_pt'],
                                                                  week=week)
                self.week_raw[key][week] = self.get_week_x_step_ahead(value['data'], value['targets_by_pt'],
                                                                      week=week)

        ix_both = list(set(self.week_one['metabs'][0].index.values).intersection(set(self.week_one['16s'][0].index.values)))
        ix_pt = [ix.split('-')[0] for ix in ix_both]
        joint = np.hstack((self.week_one['metabs'][0].loc[ix_both,:],
                                            self.week_one['16s'][0].loc[ix_both,:]))
        cols = list(self.week_one['metabs'][0].columns.values)
        cols.extend(self.week_one['16s'][0].columns.values)
        self.week_one['joint'] = pd.DataFrame(joint, index = ix_both, columns = cols), self.cdiff_data_dict['targets_by_pt'][ix_pt]


    def load_cdiff_data(self):
        xl = pd.ExcelFile(self.path + '/' + self.filename_cdiff)
        self.cdiff_raw = xl.parse('OrigScale', header = None, index_col = None)
        act_data = self.cdiff_raw.iloc[11:, 13:]
        feature_header = self.cdiff_raw.iloc[11:, :13]
        pt_header = self.cdiff_raw.iloc[:11,13:]
        pt_names = list(self.cdiff_raw.iloc[:11,12])
        pt_names[-1] = 'GROUP'
        feat_names = list(self.cdiff_raw.iloc[10,:13])
        feat_names[-1] = 'HMDB'

        self.col_mat_mets = feature_header
        self.col_mat_mets.columns = feat_names
        self.col_mat_mets.index = np.arange(self.col_mat_mets.shape[0])

        self.col_mat_pts = pt_header.T
        self.col_mat_pts.columns = pt_names
        self.col_mat_pts.index = np.arange(self.col_mat_pts.shape[0])

        self.targets_dict = pd.Series(self.col_mat_pts['PATIENT STATUS (BWH)'].values, index = self.col_mat_pts['CLIENT SAMPLE ID'].values).to_dict()

        self.cdiff_dat = pd.DataFrame(np.array(act_data), columns = self.col_mat_pts['CLIENT SAMPLE ID'].values,
                          index = self.col_mat_mets['BIOCHEMICAL'].values).fillna(0).T

        self.targets_by_pt = {key.split('-')[0]:value for key, value in self.targets_dict.items() if key.split('-')[1].isnumeric()}

        # # Filter by low prevelance across timepoints and class
        # self.cdiff_data_filt1 = filter_by_pt(self.cdiff_dat, pd.Series(self.targets_by_pt), perc = 0.15, pt_thresh = 1, meas_thresh = 10)
        # # Log transform & standardize
        # epsilon = get_epsilon(self.cdiff_data_filt1)
        # self.cdiff_data_log = np.log(self.cdiff_data_filt1 + epsilon)
        # self.cdiff_data_stand = standardize(self.cdiff_data_log)
        # # Filter by low variance
        # self.cdiff_data_filt2 = filter_vars(self.cdiff_data_stand, perc = 5)
        self.cdiff_data_dict = {'sampleMetadata':self.col_mat_pts, 'featureMetadata':self.col_mat_mets,
                                'data':self.cdiff_dat, 'targets':pd.Series(self.targets_dict),
                                'targets_by_pt': pd.Series(self.targets_by_pt)}

    def load_16s_data(self):
        self.file16s = pd.ExcelFile(self.path + '/' + self.filename_16s)
        self.raw16s = self.file16s.parse(index_col = 0)
        dcol = []
        for x in self.raw16s.columns.values:
            if len(x.split('-')) == 3:
                dcol.append('.'.join([x[:5], x[-1]]))
            elif len(x.split('-')[1]) == 2:
                dcol.append('.'.join([x[:5], x[-1]]))
            else:
                dcol.append(x)
        self.data16s = pd.DataFrame(
            np.array(self.raw16s), columns=dcol, index=self.raw16s.index.values)
        self.targets_16s = {key: val for key, val in self.targets_dict.items() if key in self.data16s.columns.values}
        pt_both = [x for x in dcol if x in self.cdiff_dat.index.values]
        self.data16s_both = (self.data16s[pt_both]).T
        pts = np.unique([x.split('-')[0] for x in self.data16s_both.index.values])
        targets_by_pt = pd.Series(self.cdiff_data_dict['targets_by_pt'][pts], index = pts)

        # filter by low prevelance over time and classes
        # self.data16s_filt1 = filter_by_pt(self.data16s_both, targets_by_pt, perc = 0.15, pt_thresh = 1, meas_thresh = 10)
        # # transform to proportions and then centered log transform
        # self.data16s_prop = np.divide(self.data16s_filt1.T, np.sum(self.data16s_filt1, 1)).T
        # epsilon = get_epsilon(self.data16s_prop)
        # geom_means = np.exp(np.mean(np.log(self.data16s_filt1 + epsilon),1))
        # temp = np.divide(self.data16s_filt1.T, geom_means).T
        # epsilon = get_epsilon(temp)
        # self.data16s_clr = np.log(temp + epsilon)
        # # Filter by low variance
        # self.data16s_filt2 = filter_vars(self.data16s_clr, perc = 5)

        # taxa_dict = {taxa: asv_to_name(taxa) for taxa in self.data16s_filt2.columns.values}
        self.data16s_dict = {'data': self.data16s_both, 'targets': pd.Series(self.targets_16s),
                             'targets_by_pt': pd.Series(self.targets_by_pt)}

    def load_ba_data(self):
        self.ba_data = {}
        path = self.path + '/' + self.filename_ba + '/' 
        for file in os.listdir(path):
            if 'csv' not in file:
                continue
            key_name = file.split('_')[1].split('.')[0]
            if key_name == 'intensityData':
                self.ba_data[key_name] = pd.read_csv(path + file, header = None)
            else:
                self.ba_data[key_name] = pd.read_csv(path + file, index_col = 0)

        self.ba_data['sampleMetadata'] = self.ba_data['sampleMetadata'].replace(to_replace = ['Recurrer','Nonrecurrer'], value = ['Recur','Cleared'])
        self.targets_dict_ba = pd.Series(self.ba_data['sampleMetadata']['Futher Subject info?'].values, 
                                    index = self.ba_data['sampleMetadata']['Sample ID'].values).to_dict()

        mapper = {'Further_further_sample_info': 'PATIENT STATUS (BWH)','Sample ID':'CLIENT SAMPLE ID'}
        self.ba_data['sampleMetadata'] = self.ba_data['sampleMetadata'].rename(columns = mapper)
        self.ba_data['featureMetadata'] = self.ba_data['featureMetadata'].rename(columns = {'Feature Name': 'BIOCHEMICAL'})

        self.targets_by_pt_ba = {key.split('-')[0]: value for key, value in self.targets_dict_ba.items() if
                         key.split('-')[0].isnumeric()}
        intensity_data = self.ba_data.pop('intensityData')
        temp = pd.DataFrame(np.array(intensity_data), index =np.array(list(self.ba_data['sampleMetadata']['CLIENT SAMPLE ID'])), columns = list(self.ba_data['featureMetadata']['BIOCHEMICAL']))
        self.ba_data['data'] = temp.drop('Study Pool Sample')
        # filt1 = filter_by_pt(self.ba_data['data_raw'], pd.Series(self.targets_by_pt_ba), perc = 0.15, pt_thresh = 1, meas_thresh = 10)
        # # Log transform & standardize
        # epsilon = get_epsilon(filt1)
        # data_log = np.log(filt1 + epsilon)
        # data_stand = standardize(data_log, override = True)
        # Filter by low variance
        # filt2 = filter_vars(data_stand, perc=5)
        # self.ba_data['data'] = filt2
        self.ba_data['targets_by_pt'] = pd.Series(self.targets_by_pt_ba)
        self.ba_data['targets'] = pd.Series(self.targets_dict_ba).drop(labels = 'Study Pool Sample')

    def filter_transform(self, data, targets_by_pt, key = 'metabs'):
        filt1 = filter_by_pt(data, targets_by_pt, perc=self.pt_perc, pt_thresh=self.pt_tmpts,
                             meas_thresh=self.meas_thresh)
        epsilon = get_epsilon(filt1)

        if '16s' not in key:
            transformed = np.log(filt1 + epsilon)
        else:
            data_prop = np.divide(filt1.T, np.sum(filt1, 1)).T
            epsilon = get_epsilon(data_prop)
            geom_means = np.exp(np.mean(np.log(data_prop + epsilon), 1))
            temp = np.divide(data_prop.T, geom_means).T
            epsilon = get_epsilon(temp)
            transformed = np.log(temp + epsilon)
        stand = standardize(transformed, override=True)
        filt2 = filter_vars(stand, perc=self.var_perc)
        return filt2


    def get_week_x(self, data, targets, week = 1):
        ixs = data.index.values
        pts = [x.split('-')[0] for x in ixs]
        tmpts = [x.split('-')[1] for x in ixs]
        rm_ix = []
        for pt in np.unique(pts):
            ix_pt = np.where(pt == np.array(pts))[0]
            tm_floats = [float(tmpts[ix]) for ix in ix_pt if tmpts[ix].replace('.','').isnumeric()]
            if max(tm_floats) == week:
                rm_ix.append(pt)
        week_one = np.where(np.array(tmpts)==str(week))[0]
        pt_keys = np.array(pts)[week_one]
        pt_keys = np.array(list(set(pt_keys) - set(rm_ix)))
        pt_keys_1 = np.array([pt + '-' + str(week) for pt in pt_keys])
        data_w1 = data.loc[pt_keys_1]
        targs = targets[pt_keys]
        return {'x':data_w1,'y':targs}

    def get_week_x_step_ahead(self, data, targets, week = 1):

        ixs = data.index.values
        pts = [x.split('-')[0] for x in ixs]
        tmpts = [x.split('-')[1] for x in ixs]
        week_one = np.where(np.array(tmpts) == str(week))[0]
        pt_keys = np.array(pts)[week_one]

        rm_ix = []
        targets_out = {}
        event_time = {}
        for pt in np.unique(pts):
            targets_out[pt] = 'Non-recurrer'
            ix_pt = np.where(pt == np.array(pts))[0]
            tm_floats = [float(tmpts[ix]) for ix in ix_pt if tmpts[ix].replace('.', '').isnumeric()]
            event_time[pt] = tm_floats[-1]
            if targets[pt] == 'Recurrer':
                ix_pt = np.where(pt == np.array(pts))[0]
                tm_floats = [float(tmpts[ix]) for ix in ix_pt if tmpts[ix].replace('.','').isnumeric()]
                if week not in tm_floats:
                    continue
                if max(tm_floats) == week:
                    rm_ix.append(pt)
                    continue
                tm_floats.sort()
                tmpt_step_before = tm_floats[-2]
                if tmpt_step_before == week:
                    targets_out[pt] = 'Recurrer'

        pt_keys = np.array(list(set(pt_keys) - set(rm_ix)))
        pt_keys_1 = np.array([pt + '-' + str(week) for pt in pt_keys])
        data_w1 = data.loc[pt_keys_1]
        targs = pd.Series(targets_out)[pt_keys]
        return {'x':data_w1,'y_step_ahead':targs, 'y':targets[pt_keys], 'event_times': pd.Series(event_time)[pt_keys]}

    def get_step_ahead(self, data, targets):
        ixs = targets.index.values
        data_all = data.iloc[ixs,:]
        targets_out = targets[data_all.index.values]
        return {'x':data_all,'y':targets_out}

if __name__ == "__main__":
    dl = dataLoader()