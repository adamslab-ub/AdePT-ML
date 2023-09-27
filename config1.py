import torch
from Part_Physics import *
device = torch.device('cuda')
D_in = 2
D_out = 2
D_PP = 5
D_invout = 23
lr = 1e-4
dropout = 0.0
batch_size = 10
epochs = 1000
Hidden_layer_size = 100
Num_layers = 2
weight_decay = 0#1e-5
filename = 'Models/extrapolation.pt'
trainset = "Data/trainset_50.npy"
testset = "Data/testset_50.npy"

########### Partial Physics Details #########################

partial_physics_v1 = lambda x: fun_vmap(x,Heat_Conduction.pp_model)

partial_jacobian_v1 =lambda x: grad_fun_vmap(x,Heat_Conduction.jacobian)

partial_physics_v2 = lambda x: fun_vmap(x,Heat_Conduction.pp_model_V2)

partial_jacobian_v2 = lambda x: grad_fun_vmap(x,Heat_Conduction.jacobian_v2)

partial_physics_v3 = lambda x: fun_vmap(x,Heat_Conduction.pp_model_V3)

partial_jacobian_v3 = lambda x: grad_fun_vmap(x,Heat_Conduction.jacobian_v3)

partial_physics_v5 = lambda x: fun_vmap(x,Heat_Conduction.pp_model_V5)

partial_jacobian_v5 = lambda x: grad_fun_vmap(x,Heat_Conduction.jacobian_v5)