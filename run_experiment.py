from utils.datareader import DBpediaReader
from utils.datareader import CIFAR100Reader



if __name__ == "__main__":
    text_reader = DBpediaReader()
    image_reader = CIFAR100Reader()

    data_df, test_df, train_df, val_df = text_reader.read_data()
    meta_dict, train_dict, test_dict = image_reader.read_data()
    print(1)