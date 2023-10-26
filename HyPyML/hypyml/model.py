import torch
import torch.nn
import numpy as np
from joblib import dump
import os
from hypyml.sub_models import MLP, Physics, CNN
    
class Torch_Wrapper(torch.nn.Module): 
    """
    Architecture is a list of dicts containing the following fields: Model type: "Physics" or "MLP"
    """
    def __init__(self,architecture):
        super(Torch_Wrapper, self).__init__()
        self.models_mlp=torch.nn.ModuleList()
        self.models_cnn=torch.nn.ModuleList()
        self.models_physics = []
        self.args = []
        self.arch = architecture
        self.model_ref = []
        # self.params = []
        for i in architecture: 
            if i["Type"] == 'Physics': 
                self.models_physics.append(Physics.apply)
                self.model_ref.append(['Physics', len(self.models_physics)-1 ])
            elif i["Type"] == "MLP": 
                self.models_mlp.append(MLP(i["config"]))
                self.model_ref.append(['MLP', len(self.models_mlp)-1])
            elif i["Type"] == "CNN": 
                self.models_cnn.append(CNN(i["config"]))
                self.model_ref.append(['CNN', len(self.models_cnn)-1])
            self.args.append(i["args"])
    
    def forward(self,x):
        out = x
        phy_count = 0
        mlp_count = 0
        cnn_count = 0
        for i in range(len(self.arch)): 
            ## Choosing the correct model from list
            if self.arch[i]["Type"] == "Physics":
                model = self.models_physics[phy_count]
                phy_count+=1
                if self.args[i] != None:
                    out = model(out,self.arch[i]['Forward'],self.arch[i]['Jacobian'], x[:,self.args[i]])
                else: 
                    out = model(out,self.arch[i]['Forward'],self.arch[i]['Jacobian'])
            elif self.arch[i]["Type"] == "CNN" : 
                model = self.models_cnn[cnn_count]
                cnn_count +=1
                if self.args[i] != None:
                    # out = model(out,x[:,self.args[i]])
                    out = model(torch.hstack((x[:,self.args[i]],out)))
                else:
                    out = model(out)

            else: 
                model = self.models_mlp[mlp_count]
                mlp_count +=1
                if self.args[i] != None:
                    # out = model(out,x[:,self.args[i]])
                    out = model(torch.hstack((x[:,self.args[i]],out)))
                else:
                    out = model(out)
        return out
    
class HyPyML():
    def __init__(self,architecture):
        self.optma = Torch_Wrapper(architecture)
    
    def forward(self,x):
        return self.optma.forward(x)

    def fwd_till(self,x,end_block):
        return self.fwd_selective(x,start_block=0, end_block=end_block)
    
    def fwd_from(self,x,start_block):
        return self.fwd_selective(x,start_block=start_block,end_block=len(self.optma.arch)-1)
    
    def fwd_selective(self,x,start_block,end_block):
        """
        Runs inference from any specified start module in architecture to any end point
        """
        out = x
        phy_count = 0
        mlp_count = 0
        cnn_count = 0
        
        assert end_block < len(self.optma.arch), "Max possible value of end_block is %d, but recieved %d" %(len(self.optma.arch)-1, end_block)

        for i in range(start_block, end_block): 
            ## Choosing the correct model from list
            count = self.optma.model_ref[i][0]
            if self.optma.arch[i]["Type"] == "Physics":
                tmp_model = self.optma.models_physics[count]
                if self.args[i] != None:
                    out = tmp_model(out,self.optma.arch[i]['Forward'],self.optma.arch[i]['Jacobian'], x[:,self.args[i]])
                else: 
                    out = tmp_model(out,self.optma.arch[i]['Forward'],self.optma.arch[i]['Jacobian'])
            elif self.optma.arch[i]["Type"] == "CNN" : 
                tmp_model = self.optma.models_cnn[count]
                if self.args[i] != None:
                    out = tmp_model(torch.hstack((x[:,self.args[i]],out)))
                else:
                    out = tmp_model(out)
            else: 
                tmp_model = self.optma.models_mlp[count]
                if self.args[i] != None:
                    out = tmp_model(torch.hstack((x[:,self.args[i]],out)))
                else:
                    out = tmp_model(out)
        return out

    # @staticmethod
    def train_step(self,optimizer,loss_fn,scheduler=None ):
        # Builds function that performs a step in the train loop
        def train_step(x, y,test=False):
            if not test:
                yhat = self.forward(x)
                loss = loss_fn(yhat,y)#torch.mean(torch.abs(yhat-y))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            else:
                a = self.optma.eval()
                with torch.no_grad():
                    yhat = a(x)
                    loss = loss_fn(yhat,y) 
            return loss.item()
        # Returns the function that will be called inside the train loop
        return train_step

    def train_model(self,train_loader, test_loader, optimizer, loss_fn, scheduler, filename, epochs):

        data_dir = os.path.join(os.getcwd(),"Models/Training_History_%s" % filename)
        if not os.path.exists(data_dir): 
            os.system("mkdir %s" % data_dir)
        train_step_obj = self.train_step(optimizer, loss_fn,scheduler)
        training_loss = []
        test_loss =[]
        for epoch in range(epochs):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                loss = train_step_obj(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss.append(np.mean(batch_losses))
            batch_losses =[]
            for x_batch,y_batch in test_loader:
                loss = train_step_obj(x_batch, y_batch,test=True)
                batch_losses.append(loss)
            test_loss.append(np.mean(batch_losses))
            print('epoch',epoch,'training loss',training_loss[-1], 'test loss',test_loss[-1])
            if epoch%50 ==0:
                torch.save(self.optma.state_dict(), "%s/Model_%d.pt" %(data_dir,epoch))
        torch.save(self.optma.state_dict(), data_dir+"/Model_final.pt")
        tmp = self.optma.arch
        reduced_arch =[]
        for i in tmp: 
            if i["Type"] == "Physics": 
                reduced_arch.append(dict(Type="Physics"))
            else: 
                reduced_arch.append(i)
        dump(dict(train_loss=training_loss, test_loss=test_loss, model_architecture = reduced_arch),"%s/Training_History.joblib" % data_dir)

        return training_loss, test_loss