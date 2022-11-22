import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def load_dataset(dataset_name, batch_size, shuffle, seed):


    # TODO: sep=';'??
    data = pd.read_csv(f'datasets/{dataset_name}.csv')

    # TODO: pre-process this instead?
    if dataset_name == 'avocado':
        data = data.drop(columns=['Unnamed: 0'])
        data = data.drop(columns=['Date'])

    columns = list(data.columns)
    # getting information about each variable and restructuring categorical to numbers
    data, var_info, var_dtype = dataset_info_restructure(dataset_name, data)
    # splitting data
    # TODO: hparam test size
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed, shuffle=shuffle)# , stratify=None)
    # again, split to get val data (on train data)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=seed, shuffle=shuffle)# , stratify=None)

    # normalize training data (similar to Ma et al.), however only the numerical columns
    #    - TODO: then use sigmoid activation function?
    #    - TODO: not normalizing categorical values
    numeric_columns = [columns[idx] for idx in var_dtype['numeric']]
    train_min = train_data[numeric_columns].min()
    train_max = train_data[numeric_columns].max()
    train_data[numeric_columns] = (train_data[numeric_columns] - train_min) / (train_max - train_min)
    val_data[numeric_columns] = (val_data[numeric_columns] - train_min) / (train_max - train_min)
    test_data[numeric_columns] = (test_data[numeric_columns] - train_min) / (train_max - train_min)


    # create a data class with __getitem__, i.e. iterable
    train_data = iterate_data(train_data)
    val_data = iterate_data(val_data)
    test_data = iterate_data(test_data)


    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

    return ((var_info, var_dtype), (train_loader, val_loader, test_loader))


def dataset_info_restructure(dataset_name, data):

    boston_dtype = {'numeric': [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                    'categorical': [3]}

    # TODO: index start with 1?
    #avocado_dtype = {'numeric': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    #                 'categorical': [10, 11]}

    avocado_dtype = {'numeric': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                     'categorical': [9, 10, 11]}

    energy_dtypes = {'numeric': [0, 1, 2, 3, 4, 8, 9],
                     'categorical': [5, 6, 7]}

    bank_dtypes = {'numeric': [0, 5, 9, 11, 12, 13, 14],
                   'categorical': [1, 2, 3, 4, 6, 7, 8, 10, 15, 16]}

    # what dataset and then the variables
    dataset_info_container = {
        'boston': boston_dtype,
        'avocado': avocado_dtype,
        'energy': energy_dtypes,
        'bank': bank_dtypes}

    # getting dtype variable information
    var_dtype = dataset_info_container[dataset_name]
    #inv_var_dtype = {v: k for k, v in var_dtype.items()}
    inv_var_dtype = {}
    for k, v in var_dtype.items():
        for x in v:
            inv_var_dtype[x] = k
    # finding number of values per variable
    var_info = {}

    for idx, variable_name in enumerate(list(data.columns)):
        if inv_var_dtype[idx] == 'categorical':
           new_columns = pd.get_dummies(data[variable_name])
           new_columns_names = list(new_columns.columns)
           data[new_columns_names] = new_columns
           num_unique = len(new_columns_names) # num unique values
           # dropping original dataframe
           data.drop(columns=variable_name, inplace=True)
           var_info[idx] = {'name': variable_name, 'dtype': 'categorical', 'num_vals': num_unique}

        else:
            # normal distribution:
            var_info[idx] = {'name': variable_name, 'dtype': 'numerical', 'num_vals': 2}


    return data, var_info, var_dtype


class iterate_data(Dataset):

    def __init__(self, data):
        # converting to tensor
        self.N = len(data)
        self.torch_data = torch.tensor(data.values)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        batch = self.torch_data[idx, :]

        return batch







### PREVIOUS ###

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
    numerical_OH = pd.DataFrame(x_sc, columns=cols)

    # might want to pass this to the dataloader as well to use the inverse_transform method
    on = OneHotEncoder()
    categoricalf = on.fit_transform(categorical).toarray()
    columns_f = on.get_feature_names_out()
    categorical_OH = pd.DataFrame(categoricalf, columns=columns_f)
    categorical_OH = pd.concat([categorical_OH, binary], axis=1)

    # categorical class-to-idx dict
    class_idxs = {}
    for col in columns_f:
        # get class and corresponding attribute
        attr, cl = col.split('_')
        # if new attribute
        if attr not in class_idxs:
            class_idxs[attr] = {}
        # insert class index
        class_idxs[attr][cl] = len(class_idxs[attr])

    return numerical_OH, categorical_OH, {'numerical': numerical, 'categorical': categorical, 'class_idxs': class_idxs}

class Boston(Dataset):
    """Boston dataset"""

    def __init__(self, mode='train', transforms=None):
        boston_dtype = {'numeric': [0,1,2,4,5,6,7,8,9,10,11,12,13],
                        'categorical': [3],
                        'binary': [3]}
        data = pd.read_csv('datasets/boston.csv')
        N = len(data)

        # TODO: make more random and seed dependent?
        if mode == 'train':
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[:int(0.6*N)], boston_dtype)
        elif mode == 'val':
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[int(0.6*N):int(0.8*N)], boston_dtype)
        else:
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[int(0.8*N):], boston_dtype)
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
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[:int(0.6*N)], avocado_dtype)
        elif mode == 'val':
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[int(0.6*N):int(0.8*N)], avocado_dtype)
        else:
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[int(0.8*N):], avocado_dtype)

#inverse_transform -> onehot_ncoder

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
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[:int(0.6*N)], dtypes)
        elif mode == 'val':
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[int(0.6*N):int(0.8*N)], dtypes)
        else:
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[int(0.8*N):], dtypes)

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
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[:int(0.6*N)], dtypes)
        elif mode == 'val':
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[int(0.6*N):int(0.8*N)], dtypes)
        else:
            self.data_num, self.data_cat, self.data_dict = standardise(data.iloc[int(0.8*N):], dtypes)



'''
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

'''


