import numpy as np
import pickle

RANDOM_SEED = 42
VAL_SIZE_RATIO = 0.15
TEST_SIZE_RATIO = 0.15


class DatasetRetriever:
    """
    Manages the training, validation and test-dataset
    """

    __instance = None

    @classmethod
    def instance(cls):
        if cls.__instance is None:
            cls()
        return cls.__instance

    def __init__(self):
        if self.__instance is not None:
            raise Exception("Singleton instantiated multiple times!")

        DatasetRetriever.__instance = self

        print("Loading datasets...")
        with open('data/graphs.dat', 'rb') as file:
            self.all_graphs = np.asarray(pickle.load(file))

        print("Splitting training, evaluation and test-sets...")
        # train, validation and test split
        np.random.seed(42)
        self._idxs = np.arange(len(self.all_graphs))
        self._idxs = np.random.permutation(self._idxs)

        self._val_size = int(len(self._idxs) * VAL_SIZE_RATIO)
        self._test_size = int(len(self._idxs) * TEST_SIZE_RATIO)
        self._train_size = len(self._idxs) - self._val_size - self._test_size

        print("Finished loading data!")


    def get_training_graphs(self) -> np.array:
        """
        Loads all training subset of graphs
        """
        return self.all_graphs[self._idxs[:self._train_size]]

    def get_evaluation_graphs(self) -> np.array:
        """
        Loads all evaluation subset of graphs
        """
        return self.all_graphs[self._idxs[:self._val_size]]

    def get_test_graphs(self) -> np.array:
        """
        Loads all testing subset of graphs
        """
        return self.all_graphs[self._idxs[:self._test_size]]