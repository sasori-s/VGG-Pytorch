import torch
import os

tensor_path = ''
torch.save(
    torch.tensor([0.4531, 0.4512, 0.3915]),
    os.path.join(tensor_path, 'dataset_mean.pt')
)
            
torch.save(
    torch.tensor([0.2573, 0.2421, 0.2585]),
    os.path.join(tensor_path, 'dataset_std.pt')
)