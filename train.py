import cv2
import torch

from dataset import Dataset, Preprocessing
from torchvision import transforms, utils

def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    params = {'batch_size': 50,
              'shuffle': True,
              'num_workers': 4}
    max_epochs = 100
    
    # Load dataset
    data_path = '../../../generated_data/'
    my_dataset = Dataset(data_path, 
                    transform=transforms.Compose([
                        Preprocessing()]))

    training_generator = torch.utils.data.DataLoader(my_dataset, **params)
    #data_iter = iter(training_generator)
    #in_, out_ = data_iter.next()
    #print(in_.shape)
    #print(out_.shape)
    for epoch in range(max_epochs):
        for local_batch, local_labels in training_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        

if __name__ == '__main__':
    main()
