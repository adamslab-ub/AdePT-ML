import numpy as np
import torch.utils.data
# import matplotlib.pyplot as plt
import time
from src.OPTMA import *
import config1 as c
c.filename = 'No_Args'
trainset = np.load(c.trainset)
testset = np.load(c.testset)
output=[]
error = []
xtest = torch.Tensor(testset[:,:c.D_in])
ytest = torch.Tensor(testset[:,c.D_in:])
xtrain = torch.Tensor(trainset[:,:c.D_in])
ytrain = torch.Tensor(trainset[:,c.D_in:])
x_train = torch.Tensor(xtrain).to(c.device)
y_train = torch.Tensor(ytrain).to(c.device)
x_test= torch.Tensor(xtest).to(c.device)
y_test = torch.Tensor(ytest).to(c.device)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=17, shuffle=True, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=c.batch_size, shuffle=True, drop_last=True)
#%% Initalize model and optimizer
# If MLP, dict should have the following dict config=dict(D_in= , D_out= , Hidden_layer_size= , Num_layers= )
# If Physics, dict should have the following fields "Forward", "Jacobian"
architecture = [dict(Type="MLP",config=dict(D_in=c.D_in , D_out= c.D_PP, Hidden_layer_size= 100, Num_layers= 3),args =None),
                dict(Type="Physics", Forward = c.finite_diff_simple, Jacobian = c.partial_jacobian,args = None),
                dict(Type="Physics", Forward = c.post_process, Jacobian = c.partial_jacobian,args = None),]


model = OPTMA(architecture)
model.to(c.device)
optimizer = torch.optim.Adam(model.parameters(),c.lr,weight_decay=c.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30 ,gamma=0.95)
loss_fn = torch.nn.MSELoss()
model, train_loss, test_loss = train_model(train_loader, test_loader, model, optimizer, scheduler, loss_fn, c.filename, c.epochs)
