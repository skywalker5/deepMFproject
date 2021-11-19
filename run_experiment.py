from utils.datareader import DBpediaReader
from utils.datareader import CIFAR100Reader
import utils.doctovec as doctovec


if __name__ == "__main__":
    text_reader = DBpediaReader()
    image_reader = CIFAR100Reader()

    text_reader.get_data_matrix()

    # data_df, test_df, train_df, val_df = text_reader.read_data()
    meta_dict, train_dict, test_dict = image_reader.read_data()

    # 1. Convert text data to X, 
    a = doctovec.vectorize(text_reader.data_df.text.iloc[0])
    print(1)