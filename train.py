import cv2
import torch

from dataset import Dataset, Preprocessing
from torchvision import transforms, utils
import unet.resunet_model as unet

def dice_loss(pred,target):
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    return 1 - (numerator + 1) / (denominator + 1)

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
    data_path = '../generated_data/'
    my_dataset = Dataset(data_path, 
                    transform=transforms.Compose([
                        Preprocessing()]))

    training_generator = torch.utils.data.DataLoader(my_dataset, **params)

    model = unet.ResUNet(2, 1, n_size=16)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(max_epochs):
        for local_batch, local_labels in training_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            print(local_batch.type())
            y_pred = model(local_batch)
            loss = dice_loss(y_pred, local_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        

if __name__ == '__main__':
    main()
