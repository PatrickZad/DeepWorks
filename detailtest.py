import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

'''
array = np.array([7, 3, 8, 4, 6, 3, 6, 8])
print(np.sort(array))
print(array)
'''


class RandMatrces(Dataset):
    def __init__(self):
        self.data = np.random.random((512, 32, 32))

    def __len__(self):
        return 512

    def __getitem__(self, index):
        tensor = torch.from_numpy(self.data[index, :, :])
        label = torch.ones(6)
        return tensor, label


if __name__ == '__main__':
    dataset = RandMatrces()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(10):
        for i, data in enumerate(dataloader):
            print(i)
            print(data[0].shape)
            print(data[1].shape)
