from utils.datareader import DBpediaReader
from utils.datareader import CIFAR100Reader
import utils.doctovec as doctovec


if __name__ == "__main__":
    text_reader = DBpediaReader()
    image_reader = CIFAR100Reader()

    # text_reader.get_data_matrix()

    # data_df, test_df, train_df, val_df = text_reader.read_data()
    image_reader.get_data_matrix()

    # 1. Convert text data to X, 
    a = doctovec.vectorize(text_reader.data_df.text.iloc[0])
    print(1)