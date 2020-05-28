import cv2
import torch
from torch.utils.data.dataset import random_split
from torchvision import transforms, utils

from dataset import Dataset, Preprocessing
import unet.resunet_model as unet
import loggs


def dice_loss(pred,target):
    numerator = 2 * torch.sum(pred * target)
    denominator = torch.sum(pred + target)
    return 1 - (numerator + 1) / (denominator + 1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def main():
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    train_params = {'batch_size': 50,
              'shuffle': True,
              'num_workers': 4}
    valid_params = {'batch_size': 100,
              'shuffle': True,
              'num_workers': 4}
    
    # Load dataset
    data_path = '../generated_data/'
    my_dataset = Dataset(data_path, 
                    transform=transforms.Compose([
                        Preprocessing()]))

    lengths = [int(len(my_dataset)*0.8), int(len(my_dataset)*0.2)]
    train_dataset, val_dataset = random_split(my_dataset, lengths)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_params)
    val_loader = torch.utils.data.DataLoader(val_dataset, **valid_params)

    # Training params
    learning_rate = 1e-3
    max_epochs = 4

    # Model
    model = unet.ResUNet(2, 1, n_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = AverageMeter('Training loss', ':.6f')
    val_loss = AverageMeter('Validation loss', ':.6f')
    best_loss = float('inf')

    nb_of_batches = lengths[0] //train_params['batch_size']
    # Training loop
    for epoch in range(max_epochs):
        if not epoch:
            logg_file = loggs.Loggs(['epoch', 'train_loss', 'val_loss'])
        for i, (x_batch, y_labels) in enumerate(train_loader):
            x_batch, y_labels = x_batch.to(device), y_labels.to(device)
            y_pred = model(x_batch)
            #y_pred = torch.round(y_pred[0])
            loss = dice_loss(y_pred, y_labels)
            train_loss.update(loss.item(), x_batch.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loggs.training_bar(i, nb_of_batches, prefix='Epoch: %d/%d'%(epoch,max_epochs), suffix='Loss: %.6f'%loss.item())
        print(train_loss)
        
        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(val_loader):
                x_val, y_val = x_val.to(device), y_val.to(device)
                model.eval()
                yhat = model(x_val)
                loss = dice_loss(yhat, y_val)
                val_loss.update(loss.item(), x_val.size(0))
                if i == 10: break
            print(val_loss)
            logg_file.save([epoch, train_loss.avg, val_loss.avg])

            # Save the best model with minimum validation loss
            if best_loss > val_loss.avg:
                print('Updated model with validation loss %.6f ---> %.6f' %(best_loss, val_loss.avg))
                best_loss = val_loss.avg
                torch.save(model, 'best_model.pt')
                
            

        

if __name__ == '__main__':
    main()
