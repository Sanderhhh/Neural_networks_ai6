import torch
from torch.utils.data import random_split
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import os

if __name__ == "__main__":
    print("HEllo world!")
    train_folder_path = "subset-data"
    test_folder_path = "subset-data"
    train_folder = os.listdir(train_folder_path)
    test_folder = os.listdir(test_folder_path)

    # make it so that half is for training, half is for validation, can adjust this ratio later
    apple_folder_path = "subset-data/Apple"
    pear_folder_path = "subset-data/Pear"
    apple_folder = os.listdir(apple_folder_path)
    pear_folder = os.listdir(pear_folder_path)
    image_total = len(apple_folder) + len(pear_folder)
    training_size = int(image_total / 2)
    validation_size = image_total - training_size

    dataset = ImageFolder(train_folder_path, transform = ToTensor())
    print(dataset)
    training_dataset, validation_dataset = torch.utils.data.random_split(dataset, [training_size, validation_size])
    
    print(len(training_dataset), len(validation_dataset))
    
    batch_size = 128
    
    training_dl = DataLoader(training_dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    validation_dl = DataLoader(validation_dataset, batch_size*2, num_workers = 4, pin_memory = True)
    
    #base class of the NN
    
    def accuracy(outputs, labels):
        _, predictions = torch.max(outputs, dim = 1)
        return torch.tensor(torch.sum(preds == labels).item() / len(predictions))
    
    class ImageClassBase(nn.Module):
        
        def train_step(self, batch):
            images, labels = batch
            # generate the predictions
            out = self(images)
            # calculate the loss with the cross entropy
            loss = F.cross_entropy(out, labels)
            return loss
        
        def validation_step(self, batch):
            images, labels = batch
            # generate the predictions
            out = self(images)
            # calculate the loss with the cross entropy
            loss = F.cross_entropy(out, labels)
            # calculate the accuracy
            acc = accuracy(out, labels)
            return {'val_loss': loss.detach(), 'val_acc': acc}
        
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            # combine losses
            epoch_loss = torch.stack(batch_losses).mean()
            batch_accs = [x['val_acc'] for x in outputs]
            # combine accuracies
            epoch_acc = torch.stack(batch_accs).mean()
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            print("Epoch [{}], train_loss: {.4f}, val_loss: {.4f}, val_acc: {.4f}".format(epoch, result['train_loss'], result['val_loss'], result['val_acc']))
            
    print("Finished the class of nn")
