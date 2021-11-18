import pandas as pd
import pickle

class DBpediaReader:
    def __init__(self, path='data/text'):
        self.data_path = f"{path}/DBP_wiki_data.csv"
        self.test_path = f"{path}/DBPEDIA_test.csv"
        self.train_path = f"{path}/DBPEDIA_train.csv"
        self.val_path = f"{path}/DBPEDIA_val.csv"
    
    def read_data(self):
        data_df = pd.read_csv(self.data_path)
        test_df = pd.read_csv(self.test_path)
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        return data_df, test_df, train_df, val_df

class CIFAR100Reader:
    def __init__(self, path='data/image'):
        self.meta_path = f"{path}/meta"
        self.test_path = f"{path}/test"
        self.train_path = f"{path}/train"

    def read_data(self):
        with open(self.meta_path, 'rb') as fo:
            meta_dict = pickle.load(fo, encoding='bytes')
        with open(self.train_path, 'rb') as fo:
            train_dict = pickle.load(fo, encoding='bytes')
        with open(self.test_path, 'rb') as fo:
            test_dict = pickle.load(fo, encoding='bytes')
        return meta_dict, train_dict, test_dict