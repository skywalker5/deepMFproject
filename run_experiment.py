from utils.datareader import DBpediaReader
from utils.datareader import CIFAR100Reader
from deepmf.deepmf import deepmf
from hiernmf2.hier8_neat import hier8_neat

if __name__ == "__main__":
    text_reader = DBpediaReader()
    image_reader = CIFAR100Reader()

    # text_reader.get_data_matrix()
    image_reader.get_data_matrix()

    # text_X_sm = text_reader.X_sm
    # text_X_lg = text_reader.X_lg
    image_X_sm = image_reader.X_sm
    # image_X_lg = image_reader.X_lg

    # k=5
    # options = {
    #     'inneriters':50,
    #     'tol':0.001,
    #     'verbose':True
    # }
    # Ws, H, errs = deepmf(text_X_sm, 2, [10,5], options)
    # print("DeepMF done!")
    k=100
    tree, splits, is_leaf, clusters, timings, Ws, priorities = hier8_neat(image_X_sm, k)
    print("HierNMF done!")