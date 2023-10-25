
import torch
import torch.nn
import numpy as np
from joblib import dump
import os
    
class MLP(torch.nn.Module):
    def __init__(self,config):
        super(MLP, self).__init__()
        self.layers = torch.nn.ModuleList()
        H = config['Hidden_layer_size']
        self.norm = torch.nn.BatchNorm1d(config['D_in'])
        self.linear_in = torch.nn.Linear(config['D_in'], H)
        for i in range(config["Num_layers"]):
            self.layers.append(torch.nn.Linear(H,H))
        self.linear_out = torch.nn.Linear(H, config['D_out'])
        self.nl1 = torch.nn.LeakyReLU()
        self.nl2 = torch.nn.Tanh()
        self.nl3 = torch.nn.Sigmoid()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        out = self.linear_in(x)
        for i in range(len(self.layers)-1):
            net = self.layers[i]
            out = self.nl1(net(out))
        out = self.linear_out(out)
        return out
    

class CNN(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(config["C_in"], config["C_out"], kernel_size=config["kern_size"], stride=1, padding=(int((config["kern_size"][0]-1)/2),int((config["kern_size"][1]-1)/2)))
        self.conv2 = torch.nn.Conv2d(config["C_out"], config["C_out"], kernel_size=config["kern_size"], stride=1, padding=(int((config["kern_size"][0]-1)/2),int((config["kern_size"][1]-1)/2)))
        self.conv3 = torch.nn.Conv2d(config["C_out"], config["C_out"], kernel_size=config["kern_size"])
        # self.act1 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=config["kern_size"],stride=1)
        self.pool = config["Pool"]
        self.flat = torch.nn.Flatten()
        self.nl1 = torch.nn.LeakyReLU()
        self.nl2 = torch.nn.Tanh()
        self.H = config['Hidden_layer_size']
        if self.H!=0:
            self.layers = torch.nn.ModuleList()
            self.linear_in = torch.nn.Linear(config['D_Size'], self.H)
            for i in range(config["Num_layers"]):
                self.layers.append(torch.nn.Linear(self.H,self.H))
            self.linear_out = torch.nn.Linear(self.H, config['D_out'])

    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.nl1(self.conv1(x))
        x = self.nl1(self.conv2(x))
        x = self.nl1(self.conv3(x))
        if self.pool:
            x = self.pool2(x)
        out = self.flat(x)
        if self.H !=0:
            out = self.linear_in(out)
            for i in range(len(self.layers)-1):
                net = self.layers[i]
                out = self.nl1(net(out))
            out = self.nl1(self.linear_out(out))
        return out
           
class Physics(torch.autograd.Function):
    @staticmethod
    def forward(ctx,input,forward_fun, jacobian_fun, args=None):
        if torch.cuda.is_available(): 
            device = "cuda"
        else: 
            device = "cpu"
        # ctx.needs_input_grad = (True,False)
        ctx.save_for_backward(input, args)
        ctx.jacobian_fun = jacobian_fun
        input = input.detach().cpu().numpy() 
        if args!=None:
            args = args.detach().cpu().numpy()
            out = forward_fun(input, args)
            out = torch.Tensor(out).to(device)
            return out
        else: 
            out = forward_fun(input)
            out = torch.Tensor(out).to(device)
            return out#.reshape(input.shape[0],-1)
    
    @staticmethod
    def backward(ctx,grad_output):
        if torch.cuda.is_available(): 
            device = "cuda"
        else: 
            device = "cpu"
        # We return as many input gradients as there were arguments.
        input,args, = ctx.saved_tensors
        jacobian_fun = ctx.jacobian_fun
        jac_final = None
        if ctx.needs_input_grad[0]:
            input = input.detach().cpu().numpy()
            if args!=None:
                args = args.detach().cpu().numpy()
                jac_final = jacobian_fun(input, args)
                jac_final = torch.Tensor(jac_final).to(device)
                grad_final = torch.zeros(input.shape[0],input.shape[1]).to(device)
                grad_output = grad_output.reshape(input.shape[0],-1)
                for i in range(grad_final.shape[0]): 
                    grad_final[i,:] = torch.matmul(grad_output[i,:].reshape(1,-1), jac_final[i].reshape(-1,input.shape[1]) )
                return grad_final,None,None,None
            else: 
                jac_final = jacobian_fun(input)
                jac_final = torch.Tensor(jac_final).to(device)
                grad_final = torch.zeros(input.shape[0],input.shape[1]).to(device)
                grad_output = grad_output.reshape(input.shape[0],-1)
                for i in range(grad_final.shape[0]): 
                    grad_final[i,:] = torch.matmul(grad_output[i,:].reshape(1,-1), jac_final[i].reshape(-1,input.shape[1]) )
                return grad_final,None, None