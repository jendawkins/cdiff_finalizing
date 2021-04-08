from helper import *
import pandas as pd
import numpy as np

class dataLoader():
    def __init__(self, path = "inputs", filename_cdiff = "CDiffMetabolomics.xlsx", \
        filename_16s = 'seqtab-nochim-total.xlsx', filename_ba = 'MS FE BANEG PeakPantheR', \
            filename_toxin = 'Toxin B and C. difficile Isolation Results.xlsx'):
        
        self.path = path
        self.filename_cdiff = filename_cdiff
        self.filename_16s = filename_16s
        self.filename_ba = filename_ba
        self.filename_toxin = filename_toxin
        self.load_cdiff_data()
        self.load_ba_data()
        
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

        self.cdiff_dat = pd.DataFrame(np.array(act_data), columns = self.col_mat_pts['CLIENT SAMPLE ID'], 
                          index = self.col_mat_mets['BIOCHEMICAL']).fillna(0).T

        self.cdiff_data = {'sampleMetadata':self.col_mat_pts, 'featureMetadata':self.col_mat_mets, 'data':self.cdiff_dat, 'targets':self.targets_dict}

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
                self.ba_data[key_name] = pd.read_csv(path + file)

        self.ba_data['sampleMetadata'] = self.ba_data['sampleMetadata'].replace(to_replace = ['Recurrer','Nonrecurrer'], value = ['Recur','Cleared'])
        self.targets_dict_ba = pd.Series(self.ba_data['sampleMetadata']['Futher Subject info?'].values, 
                                    index = self.ba_data['sampleMetadata']['Sample ID'].values).to_dict()

        mapper = {'Further_further_sample_info': 'PATIENT STATUS (BWH)','Sample ID':'CLIENT SAMPLE ID'}
        self.ba_data['sampleMetadata'] = self.ba_data['sampleMetadata'].rename(columns = mapper)
        self.ba_data['featureMetadata'] = self.ba_data['featureMetadata'].rename(columns = {'Feature Name': 'BIOCHEMICAL'})

        self.ba_data['data'] = self.ba_data.pop('intensityData')