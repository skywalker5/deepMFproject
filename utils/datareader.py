import pandas as pd
import numpy as np
import pickle
import gensim
from pathlib import Path
import utils.doctovec as doctovec
import scipy.io as sio
from scipy.sparse import csc_matrix
from gensim.matutils import corpus2dense, corpus2csc

class DBpediaReader:
    def __init__(self, path='data/text'):
        self.data_path = Path(path) / "DBP_wiki_data.csv"
        self.dict_path_sm = Path(path) / "DBPEDIA_dictionary_sm.dict"
        self.dict_path_lg = Path(path) / "DBPEDIA_dictionary_lg.dict"
        self.dictionary = {}
        self.data_df = None

        self.data_path_sm = Path(path) /"DBP_wiki_data_sm.csv"
        self.data_path_lg = Path(path) /"DBP_wiki_data_lg.csv"

        self.tfidf_path_sm = Path(path) /"DBP_wiki_data_tfidf_sm"
        self.tfidf_path_lg = Path(path) /"DBP_wiki_data_tfidf_lg"
    
    def read_data(self):
        self.data_df = pd.read_csv(self.data_path)

    def sample_data(self, option='sm'):
        if option == 'sm':
            n = 1000
            target_path = self.data_path_sm
        elif option == 'lg':
            n = 3000
            target_path = self.data_path_lg
        else:
            return
        n1 = n//3
        n2 = n - n1
        df_c1 = self.data_df[self.data_df['l1'] == 'Place']
        df_c2 = self.data_df[self.data_df['l1'] == 'Agent']
        df_c1_sample = df_c1.sample(n=n1, random_state=42)
        df_c2_sample = df_c2.sample(n=n2, random_state=42)
        sample = pd.concat([df_c1_sample, df_c2_sample])
        sample.to_csv(target_path)
        

    def get_data_matrix(self):
        if type(self.data_df) == type(None):
            self.read_data()
        if not self.data_path_sm.exists():
            self.sample_data('sm')
        if not self.data_path_lg.exists():
            self.sample_data('lg')
        sample_sm = pd.read_csv(self.data_path_sm)
        self.sample_sm_arr = (doctovec.vectorize(doc) for doc in sample_sm.text)
        sample_lg = pd.read_csv(self.data_path_lg)
        self.sample_lg_arr = (doctovec.vectorize(doc) for doc in sample_lg.text)
        if not self.dict_path_sm.exists():
            self.dictionary_sm = gensim.corpora.Dictionary(self.sample_sm_arr)
            self.dictionary_sm.filter_extremes(2, 1, len(self.dictionary_sm))
            self.dictionary_sm.save(str(self.dict_path_sm))
        else:
            self.dictionary_sm = gensim.corpora.Dictionary.load(str(self.dict_path_sm))
        if not self.dict_path_lg.exists():
            self.dictionary_lg = gensim.corpora.Dictionary(self.sample_lg_arr)
            self.dictionary_lg.filter_extremes(2, 1, len(self.dictionary_lg))
            self.dictionary_lg.save(str(self.dict_path_lg))
        else:
            self.dictionary_lg = gensim.corpora.Dictionary.load(str(self.dict_path_lg))

        if not self.tfidf_path_sm.exists():
            bow_corpus = [self.dictionary_sm.doc2bow(doc) for doc in self.sample_sm_arr]
            tfidf = gensim.models.TfidfModel(bow_corpus)
            self.tfidf_corpus_sm = tfidf[bow_corpus]
            gensim.corpora.MmCorpus.serialize(str(self.tfidf_path_sm), self.tfidf_corpus_sm)
        else:
            self.tfidf_corpus_sm = gensim.corpora.MmCorpus(str(self.tfidf_path_sm))
        if not self.tfidf_path_lg.exists():
            bow_corpus = [self.dictionary_lg.doc2bow(doc) for doc in self.sample_lg_arr]
            tfidf = gensim.models.TfidfModel(bow_corpus)
            self.tfidf_corpus_lg = tfidf[bow_corpus]
            gensim.corpora.MmCorpus.serialize(str(self.tfidf_path_lg), self.tfidf_corpus_lg)
        else:
            self.tfidf_corpus_lg = gensim.corpora.MmCorpus(str(self.tfidf_path_lg))

        # use corpus2csc for sparse matrix
        num_terms, num_docs = len(self.dictionary_sm.keys()), self.dictionary_sm.num_docs
        self.X_sm = corpus2dense(self.tfidf_corpus_sm, num_terms, num_docs)
        num_terms, num_docs = len(self.dictionary_lg.keys()), self.dictionary_lg.num_docs
        self.X_lg = corpus2dense(self.tfidf_corpus_lg, num_terms, num_docs)
        

class CIFAR100Reader:
    def __init__(self, path='data/image'):
        self.train_path = f"{path}/train"
        self.data_dict = None

        self.label_path_sm = Path(path) /"cifar100_sm_label.csv"
        self.label_path_lg = Path(path) /"cifar100_lg_label.csv"

        self.matrix_path_sm = Path(path) /"cifar100_sm_matrix.mat"
        self.matrix_path_lg = Path(path) /"cifar100_lg_matrix.mat"

    def read_data(self):
        with open(self.train_path, 'rb') as fo:
            self.train_dict = pickle.load(fo, encoding='bytes')

    def sample_data(self, option='sm'):
        if option == 'sm':
            n = 1000
            target_label_path = self.label_path_sm
            target_matrix_path = self.matrix_path_sm
        elif option == 'lg':
            n = 3000
            target_label_path = self.label_path_lg
            target_matrix_path = self.matrix_path_lg
        else:
            return

        fine_labels = np.array(self.train_dict[b'fine_labels'])
        coarse_labels = np.array(self.train_dict[b'coarse_labels'])
        n_each = n//100
        fine_label_sample = []
        coarse_label_sample = []
        data_sample = []
        np.random.seed(42)
        for i in range(100):
            ind = np.where(fine_labels == i)[0]
            ind_sample = np.random.choice(ind, n_each,replace=False)    
            fine_label_sample.append(list(fine_labels[ind_sample]))
            coarse_label_sample.append(list(coarse_labels[ind_sample]))
            data_sample.append(self.train_dict[b'data'][ind_sample])
        label_df = pd.DataFrame({'fine_labels':fine_label_sample, 'coarse_label':coarse_label_sample})
        label_df.to_csv(target_label_path)

        data_mat = np.vstack(data_sample)
        sio.savemat(target_matrix_path, {'X':data_mat})

    def get_data_matrix(self):
        if type(self.data_dict) == type(None):
            self.read_data()
        if not self.label_path_sm.exists():
            self.sample_data('sm')
        if not self.label_path_lg.exists():
            self.sample_data('lg')
        
        self.label_sm = pd.read_csv(self.label_path_sm)
        self.label_lg = pd.read_csv(self.label_path_lg)
        self.X_sm = sio.loadmat(self.matrix_path_sm)['X'].T
        self.X_lg = sio.loadmat(self.matrix_path_lg)['X'].T
        M = 255
        self.X_sm = self.X_sm / M
        self.X_lg = self.X_lg / M