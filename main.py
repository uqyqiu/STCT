import torch
import torch.nn as nn
from scr.Dataload import DataLoad
from scr.util import train, evaluation,gpu_available,get_results
from scr.plot import rocplot
from scr.stct_networks import STCT


device = gpu_available()

# add data path
train_path = '/Users/andyq/Downloads/AHA/AHA_train.npy'
test_path = '/Users/andyq/Downloads/AHA/AHA_test.npy'
validate_path = '/Users/andyq/Downloads/AHA/AHA_validate.npy'

# load dataloader
data = DataLoad(train_path, test_path, validate_path)

# load the model
if device == True:
    model = STCT().cuda()
else:
    model = STCT()

# initialize learning rate
lr = 0.001

# creat the optimizer & loss
optimizer = torch.optim.Adam(model.parameters(),lr = lr)
criterion = nn.CrossEntropyLoss()

# train model
model = train(Epoch=1,model=model, 
     criterion=criterion, 
     optimizer=optimizer, 
     train_loader=data[0], 
     validate_loader=data[2], 
     device=device)

# evaluation
target_pred, target_true, output = evaluation(model,
                                                    test_loader=data[1])
get_results(target_true, target_pred)

# plot roc
n_classes = 5
rocplot(n_classes,target_true,output)