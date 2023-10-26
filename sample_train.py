import numpy as np
import torch
import torch.utils.data
# import matplotlib.pyplot as plt
from Part_Physics import batch_physics, batch_jacobian
import plotly.graph_objects as go
from hypyml.model import HyPyML

#%% Setting up model config
filename = 'Trial_Run'
trainset = np.load("train_data.npy")
testset = np.load("test_data.npy")
D_in = 1
D_PP = 1
D_out = 1
batch_size = 10
lr = 1e-4
epochs = 300


#%% Loading Data
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
xtest = torch.Tensor(testset[:,:D_in])
ytest = torch.Tensor(testset[:,D_in:])
xtrain = torch.Tensor(trainset[:,:D_in])
ytrain = torch.Tensor(trainset[:,D_in:])
x_train = xtrain.to(device)
y_train = ytrain.to(device)
x_test= xtest.to(device)
y_test = ytest.to(device)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=100, shuffle=True, drop_last=True)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True, drop_last=True)
#%% Initalize model and optimizer

# If MLP, dict should have the following dict config=dict(D_in= , D_out= , Hidden_layer_size= , Num_layers= )
# If Physics, dict should have the following fields "Forward", "Jacobian"

architecture = [dict(Type="Physics", Forward = batch_physics, Jacobian = batch_jacobian,args = None), 
                dict(Type="MLP",config=dict(D_in=D_in , D_out= D_PP, Hidden_layer_size= 50, Num_layers= 2),args =None)]

model = HyPyML(architecture)
model.optma.to(device)
optimizer = torch.optim.Adam(model.optma.parameters(),lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=30 ,gamma=0.95)
loss_fn = torch.nn.MSELoss()

#%% Train
train_loss, test_loss = model.train_model(train_loader, test_loader, optimizer, loss_fn,scheduler, filename, epochs)

#%% Testing Model and Generating Plots
predictions = model.forward(x_test)
fig = go.Figure()
fig.add_trace(go.Scatter(x=xtest.detach().numpy().ravel(), y = predictions.detach().cpu().numpy().ravel(),mode ="markers", name="PIML Predictions"))
fig.add_trace(go.Scatter(x=xtest.detach().numpy().ravel(), y = ytest.detach().numpy().ravel(),mode ="markers", name="Truth"))
fig.show()
