from utils.datareader import DBpediaReader
from utils.datareader import CIFAR100Reader
import utils.doctovec as doctovec


if __name__ == "__main__":
    text_reader = DBpediaReader()
    image_reader = CIFAR100Reader()

    text_reader.get_data_matrix()
    image_reader.get_data_matrix()

    print(1)