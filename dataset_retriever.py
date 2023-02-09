import numpy as np
import pickle
from graph import Graph

RANDOM_SEED = 315
# Val and Test sizes decreased on purpose in comparison to the ususal 15 - 20 %
# Reason is the non-performant permutations calculation
VAL_SIZE_RATIO = 0.05
TEST_SIZE_RATIO = 0.05

ONLY_SPECIFIC_NODE_AMOUNT = None

class DatasetRetriever:
    """
    Manages the training, validation and test-dataset
    """

    __instance = None

    @classmethod
    def instance(cls, override_path = None):
        if cls.__instance is None:
            cls(override_path=override_path)
        return cls.__instance

    def __init__(self, override_path = None):
        if self.__instance is not None:
            raise Exception("Singleton instantiated multiple times!")

        DatasetRetriever.__instance = self

        print("Loading datasets...")
        path = 'data/graphs.dat'
        if override_path is not None:
            path = override_path

        with open('data/graphs.dat', 'rb') as file:
            all_graphs_list = pickle.load(file)

        if ONLY_SPECIFIC_NODE_AMOUNT is not None:
            pass
            graph: Graph
            all_graphs_list = [graph for graph in all_graphs_list if len(graph.get_nodes()) == ONLY_SPECIFIC_NODE_AMOUNT]

        self.all_graphs = np.asarray(all_graphs_list)

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
        return self.all_graphs[self._idxs[self._test_size + self._val_size:]]

    def get_validation_graphs(self) -> np.array:
        """
        Loads all evaluation subset of graphs
        """
        return self.all_graphs[self._idxs[self._test_size:self._test_size + self._val_size]]

    def get_test_graphs(self) -> np.array:
        """
        Loads all testing subset of graphs
        """
        return self.all_graphs[self._idxs[:self._test_size]]

    def get_random_graph(self) -> Graph:
        return np.random.choice(self.all_graphs)
