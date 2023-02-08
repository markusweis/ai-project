from dataset_retriever import DatasetRetriever
from prediction_models.gnn.dataset import CustomGraphDataset
import lovely_tensors as lt


def test_dataset():
    dataset_retriever = DatasetRetriever.instance()
    ds = CustomGraphDataset(dataset_retriever.all_graphs[:200])
    for (pl, g) in ds:
       print(pl, g)



if __name__ == "__main__":
    lt.monkey_patch()
    test_dataset()
        