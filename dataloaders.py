import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

def standardise(x:pd.DataFrame, dtypes:dict):
    '''zero mean and unit variance'''

    # split into numerical and categorical features
    numerical = x.iloc[:, dtypes['numeric']].reset_index(drop=True) #(drop=True, inplace=True)
    categorical = x.iloc[:, dtypes['categorical']].reset_index(drop=True) #(drop=True, inplace=True)
    binary = x.iloc[:, dtypes['binary']].reset_index(drop=True) #(drop=True, inplace=True)

    # standardizing training data
    # TODO: should not do it for test data!
    cols = x.columns[dtypes['numeric']]
    sc = StandardScaler(copy=False)
    x_sc = sc.fit_transform(numerical)
    numerical = pd.DataFrame(x_sc, columns=cols)

    # might want to pass this to the dataloader as well to use the inverse_transform method
    on = OneHotEncoder()
    categoricalf = on.fit_transform(categorical).toarray()
    columns_f = on.get_feature_names_out()
    categorical = pd.DataFrame(categoricalf, columns=columns_f)
    categorical = pd.concat([categorical, binary], axis=1)
    return numerical, categorical 

class Boston(Dataset):
    """Boston dataset"""

    def __init__(self, mode='train', transforms=None):
        boston_dtype = {'numeric': [0,1,2,4,5,6,7,8,9,10,11,12,13],
                        #'categorical': [3],
                        'categorical': [],
                        'binary': [3]}
        data = pd.read_csv('datasets/boston.csv')
        N = len(data)

        # TODO: make more random and seed dependent?
        if mode == 'train':
            self.data_num, self.data_cat = standardise(data.iloc[:int(0.6*N)], boston_dtype)
        elif mode == 'val':
            self.data_num, self.data_cat = standardise(data.iloc[int(0.6*N):int(0.8*N)], boston_dtype)
        else:
            self.data_num, self.data_cat = standardise(data.iloc[int(0.8*N):], boston_dtype)
        # self.transforms = transforms

    def __len__(self):
        return len(self.data_num)

    def __getitem__(self, idx):
        sample_num = torch.tensor(self.data_num.iloc[idx].values)
        sample_cat = torch.tensor(self.data_cat.iloc[idx].values).float()
        if sample_cat[0] == torch.tensor(np.nan):
            stop = 0
        # if self.transforms:
        #     sample = self.transforms(sample)
        return sample_num, sample_cat

class Avocado(Boston, Dataset):
    """Avocado dataset
    inherits __len__ and __getitem__ from Boston dataset 
    """

    def __init__(self, mode='train', transforms=None):
        avocado_dtype = {'numeric': [1, 2, 3, 4, 5, 6, 7, 8, 9], 
                        'categorical': [10, 11], 
                        'binary': [10]}
        data = pd.read_csv('datasets/avocado.csv',index_col=0)
        data['Date'] = pd.to_datetime(data['Date'])
        N = len(data)
        if mode == 'train':
            self.data_num, self.data_cat = standardise(data.iloc[:int(0.6*N)], avocado_dtype)
        elif mode == 'val':
            self.data_num, self.data_cat = standardise(data.iloc[int(0.6*N):int(0.8*N)], avocado_dtype)
        else:
            self.data_num, self.data_cat = standardise(data.iloc[int(0.8*N):], avocado_dtype)


class Energy(Boston, Dataset):
    """Energy dataset
    inherits __len__ and __getitem__ from Boston dataset 
    """

    def __init__(self, mode='train', transforms=None):
        dtypes = {'numeric': [0, 1, 2, 3, 4, 8, 9], 
                'categorical': [5, 6, 7], 
                'binary': []}
        data = pd.read_csv('datasets/energy.csv')
        data.columns = ['relative_compactness', 'surface_area', 'wall_area', 'roof_area', 'overall_height','orientation', 'glazing_area', 'glazing_area_distribution', 'heating_load', 'cooling_load']
        N = len(data)
        if mode == 'train':
            self.data_num, self.data_cat = standardise(data.iloc[:int(0.6*N)], dtypes)
        elif mode == 'val':
            self.data_num, self.data_cat = standardise(data.iloc[int(0.6*N):int(0.8*N)], dtypes)
        else:
            self.data_num, self.data_cat = standardise(data.iloc[int(0.8*N):], dtypes)

class Bank(Boston, Dataset):
    """Bank dataset
        inherits __len__ and __getitem__ from Boston dataset 
    """

    def __init__(self, mode='train', transforms=None):
        dtypes = {'numeric': [0, 5, 9, 11, 12, 13, 14], 
                    'categorical': [1, 2, 3,4,6,7, 8, 10, 15,16], 
                    'binary': [4, 6, 7, 16]}
        data = pd.read_csv('datasets/bank-full.csv', sep=';')
        N = len(data)
        if mode == 'train':
            self.data_num, self.data_cat = standardise(data.iloc[:int(0.6*N)], dtypes)
        elif mode == 'val':
            self.data_num, self.data_cat = standardise(data.iloc[int(0.6*N):int(0.8*N)], dtypes)
        else:
            self.data_num, self.data_cat = standardise(data.iloc[int(0.8*N):], dtypes)


if __name__ == '__main__':
    train_data = Boston(mode='train')
    val_data = Boston(mode='val')
    test_data = Boston(mode='test')

    training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    
    # create dummy dataloader to get a batch of 1
    tester_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    print(next(iter(tester_loader)))