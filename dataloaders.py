import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def standardise(x:pd.DataFrame, dtypes:dict):
    '''zero mean and unit variance'''
    numerical = x.iloc[:, dtypes['numeric']]
    categorical = x.iloc[:, dtypes['categorical']]
    # split into numecical and categorical features
    cols = x.columns[dtypes['numeric']]
    sc = StandardScaler(copy=False)
    x = sc.fit_transform(numerical)
    numerical = pd.DataFrame(x, columns=cols)

    # might want to pass this to the dataloader as well to use the inverse_transform method
    on = OneHotEncoder()
    categoricalf = on.fit_transform(categorical).toarray()
    columns_f = on.get_feature_names_out()
    categorical = pd.DataFrame(categoricalf, columns=columns_f)

    return numerical, categorical 

class Boston(Dataset):
    """Boston dataset"""

    def __init__(self, mode='train', transforms=None):
        boston_dtype = {'numeric': [0, 1, 2, 4,5,6,7,8,9,10,11,12,13], 
                      'categorical': [3],
                       'binary': [3]}
        data = pd.read_csv('datasets/boston.csv')
        if mode == 'train':
            self.data_num, self.data_cat = standardise(data.iloc[:300], boston_dtype)
        elif mode == 'val':
            self.data_num, self.data_cat = standardise(data.iloc[300:350], boston_dtype)
        else:
            self.data_num, self.data_cat = standardise(data.iloc[350:], boston_dtype)
        # self.transforms = transforms

    def __len__(self):
        return len(self.data_num)

    def __getitem__(self, idx):
        sample_num = list(self.data_num.iloc[idx].values) # is list necessary?
        sample_cat = list(self.data_cat.iloc[idx].values) # is list necessary?
        # if self.transforms:
        #     sample = self.transforms(sample)
        return sample_num, sample_cat

class Avocado(Dataset):
    """Avocado dataset"""

    def __init__(self, mode='train', transforms=None):
        boston_dtype = {'numeric': [0, 1, 2, 4,5,6,7,8,9,10,11,12,13], 
                      'categorical': [3],
                       'binary': [3]}
        data = pd.read_csv('datasets/boston.csv')
        if mode == 'train':
            self.data_num, self.data_cat = standardise(data.iloc[:300], boston_dtype)
        elif mode == 'val':
            self.data_num, self.data_cat = standardise(data.iloc[300:350], boston_dtype)
        else:
            self.data_num, self.data_cat = standardise(data.iloc[350:], boston_dtype)
        # self.transforms = transforms

    def __len__(self):
        return len(self.data_num)

    def __getitem__(self, idx):
        sample_num = list(self.data_num.iloc[idx].values) # is list necessary?
        sample_cat = list(self.data_cat.iloc[idx].values) # is list necessary?
        # if self.transforms:
        #     sample = self.transforms(sample)
        return sample_num, sample_cat

class Energy(Dataset):
    """Energy dataset"""

    def __init__(self, mode='train', transforms=None):
        boston_dtype = {'numeric': [0, 1, 2, 4,5,6,7,8,9,10,11,12,13], 
                      'categorical': [3],
                       'binary': [3]}
        data = pd.read_csv('datasets/boston.csv')
        if mode == 'train':
            self.data_num, self.data_cat = standardise(data.iloc[:300], boston_dtype)
        elif mode == 'val':
            self.data_num, self.data_cat = standardise(data.iloc[300:350], boston_dtype)
        else:
            self.data_num, self.data_cat = standardise(data.iloc[350:], boston_dtype)
        # self.transforms = transforms

    def __len__(self):
        return len(self.data_num)

    def __getitem__(self, idx):
        sample_num = list(self.data_num.iloc[idx].values) # is list necessary?
        sample_cat = list(self.data_cat.iloc[idx].values) # is list necessary?
        # if self.transforms:
        #     sample = self.transforms(sample)
        return sample_num, sample_cat

class Bank(Dataset):
    """Bank dataset"""

    def __init__(self, mode='train', transforms=None):
        boston_dtype = {'numeric': [0, 1, 2, 4,5,6,7,8,9,10,11,12,13], 
                      'categorical': [3],
                       'binary': [3]}
        data = pd.read_csv('datasets/boston.csv')
        if mode == 'train':
            self.data_num, self.data_cat = standardise(data.iloc[:300], boston_dtype)
        elif mode == 'val':
            self.data_num, self.data_cat = standardise(data.iloc[300:350], boston_dtype)
        else:
            self.data_num, self.data_cat = standardise(data.iloc[350:], boston_dtype)
        # self.transforms = transforms

    def __len__(self):
        return len(self.data_num)

    def __getitem__(self, idx):
        sample_num = list(self.data_num.iloc[idx].values) # is list necessary?
        sample_cat = list(self.data_cat.iloc[idx].values) # is list necessary?
        # if self.transforms:
        #     sample = self.transforms(sample)
        return sample_num, sample_cat


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