import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

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
    '''
    dataset = RandMatrces()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for epoch in range(10):
        for i, data in enumerate(dataloader):
            print(i)
            print(data[0].shape)
            print(data[1].shape)
    '''
    in_array = np.ones((3, 256, 256))
    target = np.ones((3, 256, 256)) * 2
    in_tensor = torch.tensor(in_array, requires_grad=True)
    target_tensor = torch.from_numpy(target)
    loss_func = nn.L1Loss()
    loss = loss_func(in_tensor,target_tensor)
    print(loss)
