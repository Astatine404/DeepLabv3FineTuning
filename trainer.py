import copy
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm

batch_size = 2

def train_model(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            stack = False
            prev_inputs = None
            prev_masks = None
            for sample in tqdm(iter(dataloaders[phase])):
                if isinstance(sample, torch.Tensor) and sample.nelement() == 0:
                    continue
                    
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
    
                if stack:
                    inputs = torch.cat([prev_inputs, inputs], 0)
                    masks = torch.cat([prev_masks, masks], 0)
                    stack = False
                if inputs.shape[0] % 2 == 1:
                    prev_inputs = inputs
                    prev_masks = masks
                    stack = True
                    continue
                
                loss = None
                for i in range(0, inputs.shape[0], batch_size):
                    ip = inputs[i:i+batch_size]
                    msk = masks[i:i+batch_size]
                    if ip.nelement() == 0:
                        continue
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'Train'):
                        outputs = model(ip)
                        loss = criterion(outputs['out'], msk)
                        y_pred = outputs['out'].data.cpu().numpy().ravel()
                        y_true = msk.data.cpu().numpy().ravel()
                        #print(np.unique(y_true))
                        for name, metric in metrics.items():
                            if name == 'f1_score':
                                # Use a classification threshold of 0.1
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(y_true > 0, y_pred > 0.1))
                            else:
                                batchsummary[f'{phase}_{name}'].append(
                                    metric(y_true.astype('uint8'), y_pred))

                        # backward + optimize only if in training phase
                        if phase == 'Train':
                            loss.backward()
                            optimizer.step()
                            #print(loss.item())
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
