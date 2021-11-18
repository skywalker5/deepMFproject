import pandas as pd

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
        pass