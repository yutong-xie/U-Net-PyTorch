import glob
import os
import numpy as np
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, average_precision_score

from PIL import Image
import torch
from torch import nn
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable

import copy
import torchvision.models as models

from model import MyModel
from evaluate import segmentation_eval, compute_confusion_matrix, compute_ap
from plot import plot_results 
from dataset import SegmentationDataset 

IS_GPU = True 
EPOCHS = 40

# load dataset 
train_dataset = SegmentationDataset(split='train')
train_dataloader = data.DataLoader(train_dataset, batch_size=1, 
                                    shuffle=True, num_workers=4, 
                                    drop_last=True)

dataset = SegmentationDataset(split='val', data_dir=DATASET_PATH)
dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False, 
                            num_workers=0, drop_last=False)

test_dataset = SegmentationDataset(split='test', data_dir=DATASET_PATH)
test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, 
                            num_workers=0, drop_last=False)


def eval(model, data_loader, is_gpu):
    gts, preds = [], []
    with torch.no_grad():
      for i, batch in enumerate(tqdm(dataloader)):
        img, gt = batch
        if is_gpu:
            img = img.cuda()
            
        outputs = model(img).data.cpu().numpy()
        gt = gt.numpy()
        gts.append(gt[0,:,:,:])
        preds.append(outputs[0,:,:,:])

    gts = np.array(gts)
    preds = np.array(preds)
    return gts, preds, list(test_dataset.classes)
                            
def training(model, criterion, optimizer):
    mAP_over_epochs = []
    mIoU_over_epochs = []
    best_loss = float('inf')

    for epoch in tqdm(range(EPOCHS), total=EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
          inputs, labels = data
          
          if IS_GPU:
            inputs = inputs.cuda()
            labels = labels.cuda()

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = model(inputs)
          labels = labels.squeeze(1)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()

        # Normalizing the loss by the total number of train batches
        running_loss/=len(train_dataloader)
        print('Training Epoch [%d] loss: %.3f' %(epoch + 1, running_loss))

        gts, preds, _ = eval(model, dataloader, IS_GPU)

        ious, counts = compute_confusion_matrix(gts, preds)
        aps = compute_ap(gts, preds)
        print('Test result on Validation images:')
        print('{:>0s}: AP: {:0.2f}, IoU: {:0.2f}'.format('mean', np.mean(aps), np.mean(ious)))

        if running_loss < best_loss:
          print("saving best model \n")
          best_loss = running_loss
          best_model_wts = copy.deepcopy(model.state_dict())


        mAP_over_epochs.append(np.mean(aps))
        mIoU_over_epochs.append(np.mean(ious))

    # Plot train loss over epochs and val set accuracy over epochs
    plt.ioff()
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.ylabel('mAP')
    plt.plot(np.arange(EPOCHS), mAP_over_epochs, 'k-')
    plt.title('mAP and mIoU')
    plt.xticks(np.arange(EPOCHS, dtype=int))
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(EPOCHS), mIoU_over_epochs, 'b-')
    plt.ylabel('mIoU')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(EPOCHS, dtype=int))
    plt.grid(True)
    plt.savefig("plotq2.png")
    plt.close(fig)
    print('Finished Training')

    model.load_state_dict(best_model_wts)

    return model

if __name__ == "__main__":
    device = torch.device("cuda:0")

    # set training                                
    model = MyModel().to(device) 
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # train and evaluate 
    best_model = training(model, criterion, optimizer)
    gts, preds, classes  = eval(best_model, test_loader, IS_GPU)
    aps, ious = segmentation_eval(gts, preds, classes, 'evaluation.pdf')