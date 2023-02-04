# from prediction_models.gnn.gnn_stack import GNNStack
# from prediction_models.gnn.dataset import CustomGraphDataset
# from torch.utils.data import DataLoader
# import numpy as np
# import pickle
# import lovely_tensors as lt

# lt.monkey_patch()





# # Load train data
# with open('data/graphs.dat', 'rb') as file:
#     graphs = np.asarray(pickle.load(file))

#     dataset = CustomGraphDataset(graphs)
#     train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

#     for i in train_dataloader:
#         print("---")
#         print(i)    