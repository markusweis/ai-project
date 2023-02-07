from dataset_retriever import DatasetRetriever
from prediction_models.gnn.dataset import CustomGraphDataset
import torch 
from torch.utils.data import DataLoader

def test_dataset():
    dataset_retriever = DatasetRetriever.instance()
    ds = CustomGraphDataset(dataset_retriever.all_graphs)
    val_dataloader = DataLoader(ds, batch_size=1, shuffle=False)
    max_parts_id, max_family_id = 0, 0 
    for (pl, g) in ds: 
        x = pl[0]
*        if torch.max(pl[0]) > max_parts_id:
            max_parts_id = torch.max(pl[0])
        if torch.max(pl[1]) > max_family_id:
            max_family_id = torch.max(pl[1])

    print(max_parts_id, max_family_id)

# max_part_id = 959, max_familiy_id = 88

if __name__ == "__main__":
    test_dataset()
        