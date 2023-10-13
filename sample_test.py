import numpy as np
from src.OPTMA import *
import config1 as c
import plotly.graph_objects as go

layout = go.Layout(
    paper_bgcolor='rgb(255,255,255)',
    plot_bgcolor='rgb(255,255,255)'

)


def test_model(arc, filename, testdata= None):
    model = OPTMA(arc)
    model.load_state_dict(torch.load("Models/Training_History_"+filename))
    model.to(c.device)
    if testdata:
        preds = model.forward(testdata)
    else:
        preds = model.forward(x_test)
    preds = preds.detach().cpu().numpy()
    return preds, np.abs(preds-ytest)


def make_surface_plot(models):
    names = models.keys()
    fig = go.Figure(layout=layout)
    fig1 = go.Figure(layout=layout)
    w = 0
    ticktext = []
    x_grid, y_grid = np.meshgrid(np.linspace(0,1,20),np.linspace(0,1,20))
    input = np.hstack((x_grid.reshape(-1,1), y_grid.reshape(-1,1)))
    input_1 = torch.Tensor(input).to(c.device)
    bounds = np.load("Data/Input_bounds.npy")

    inputx = bounds[0,0] + (np.linspace(0,1,20) * (bounds[0,1]-bounds[0,0]))
    inputy = bounds[1,0] + (np.linspace(0,1,20) * (bounds[1,1]-bounds[1,0]))
    colorscale = ["sunset","teal", "fall"]
    out_bounds = np.load("Data/Output_bounds.npy")
    speeds = np.array([5.00E-02,	1.00E-01,	1.50E-01,	2.00E-01,	2.50E-01	,3.00E-01,	3.50E-01,	4.00E-01,	4.50E-01,	5.00E-01,	5.50E-01,	6.00E-01,	6.50E-01])
    power = np.array([40,50,55,60,70,75,85,90,95,100])
    depth = np.loadtxt("Data/HF_Depth.txt"); width = np.loadtxt("Data/HF_Width.txt")
    depth_lf = np.loadtxt("Data/py_lf_width.txt"); width_lf = np.loadtxt("Data/py_lf_width.txt")
    fig.add_trace(go.Surface(y = speeds, x = power, z = depth.T,name="Depth_HF",colorscale="Blues",showlegend=True,showscale=False))
    fig.add_trace(go.Surface(y = speeds, x = power, z = depth_lf.T,name="Depth_LF",colorscale="Greens",showlegend=True,showscale=False))

    fig1.add_trace(go.Surface(y = speeds, x = power, z = width.T,name="Width_HF",colorscale="Blues",showlegend=True,showscale=False))
    fig1.add_trace(go.Surface(y = speeds, x = power, z = width_lf.T,name="Width_LF",colorscale="Greens",showlegend=True,showscale=False))

    for name in names:
        model = OPTMA(models[name][0])
        model.load_state_dict(torch.load("Models/Training_History_"+models[name][1]))
        model.to(c.device)
        preds = model.forward(input_1)
        preds = preds.detach().cpu().numpy()
        preds =  (preds* (out_bounds[:,1]-out_bounds[:,0])) + out_bounds[:,0]
        fig.add_trace(go.Surface(x = inputx, y = inputy, z = preds[:,0].reshape(x_grid.shape) ,name=name,colorscale=colorscale[int(w/2)],showlegend=True,showscale=False))
        fig1.add_trace(go.Surface(x = inputx, y = inputy, z = preds[:,1].reshape(x_grid.shape) ,name=name,colorscale=colorscale[int(w/2)],showlegend=True,showscale=False))
        # fig.update_coloraxes(showscale=False)
        w+=2
    fig.update_scenes(yaxis_title_text='Speed',  
                    xaxis_title_text='Laser Power Percentage',  
                    zaxis_title_text='Depth of Cut')
    fig.update_layout(legend=dict(font=dict(size= 20)))
    fig.show()

    fig1.update_scenes(yaxis_title_text='Speed',  
                    xaxis_title_text='Laser Power Percentage',  
                    zaxis_title_text='Width of Cut')
    fig1.update_layout(legend=dict(font=dict(size= 20)))
    fig1.show()

def gen_results(models):
    names = models.keys()
    fig = go.Figure(layout=layout)
    w = 0
    ticktext = []
    for name in names:
        preds, err = test_model(models[name][0], models[name][1])
        print("Model: %s, Mean Error: %f, Standard Deviation: %f" % (name, np.mean(err), np.std(err)))

        fig.add_trace(go.Box(x = w*np.ones(err.shape[0],) ,y = err[:,0],name=name+"_Depth",showlegend=False))
        fig.add_trace(go.Box(x = 1 + w*np.ones(err.shape[0],) , y = err[:,1],name=name+"_Width",showlegend=False))
        fig.add_trace(go.Scatter(x = [w,w+1] ,y =np.mean(err,axis=0),showlegend=False,mode="markers", marker=dict(color="red")))
        ticktext = ticktext + [name+"_Depth",name+"_Width"]
        w+=2
    print("Model: LF, Mean Error: %f, Standard Deviation: %f" % (np.mean(err), np.std(err)))
    err = np.abs(lf_test - ytest)
    fig.add_trace(go.Box(x = w*np.ones(err.shape[0],) ,y = err[:,0],name="LF_Depth",showlegend=False))
    fig.add_trace(go.Box(x = 1 + w*np.ones(err.shape[0],) , y = err[:,1],name="LF_Width",showlegend=False))
    fig.add_trace(go.Scatter(x = [w,w+1] ,y =np.mean(err,axis=0),showlegend=False,mode="markers", marker=dict(color="red")))
    ticktext = ticktext + ["LF_Depth","LF_Width"]

    fig.update_xaxes(showgrid=False, showline=True, linecolor="black", mirror=True, tickmode="array", tickvals = np.arange(2*(len(names)+1)), ticktext = ticktext)
    fig.update_yaxes(showgrid=False, showline=True, linecolor="black", mirror=True, title="Absolute Error (Normalized)")
    fig.show()




testset = np.load(c.testset)
xtest = torch.Tensor(testset[:,:c.D_in])
ytest = testset[:,c.D_in:]
x_test= torch.Tensor(xtest).to(c.device)

lf_test = np.load("Data/testset_lf.npy")

architecture_V1 = [dict(Type="MLP",config=dict(D_in=c.D_in , D_out= c.D_PP, Hidden_layer_size= 700, Num_layers= 5),args =None),
                dict(Type="Physics", Forward = c.partial_physics_v1, Jacobian = c.partial_jacobian_v1,args = None)]

architecture_V2 = [dict(Type="MLP",config=dict(D_in=c.D_in , D_out= 3, Hidden_layer_size= 600, Num_layers= 5),args =None),
                dict(Type="Physics", Forward = c.partial_physics_v2, Jacobian = c.partial_jacobian_v2,args =None),
                dict(Type="MLP",config=dict(D_in=3 , D_out= c.D_out, Hidden_layer_size= 600, Num_layers= 5),args =[1])]


architecture_V3 = [dict(Type="MLP",config=dict(D_in=c.D_in , D_out= 3, Hidden_layer_size= 200, Num_layers= 5),args =None),
                dict(Type="Physics", Forward = c.partial_physics_v3, Jacobian = c.partial_jacobian_v3,args =None),
                dict(Type="CNN",config=dict(C_in=1,C_out = 1,Pool=False, kern_size = (3,3), D_Size = 231, D_out= 2, Hidden_layer_size= 400, Num_layers= 5),args =None), 
                dict(Type="MLP",config=dict(D_in=4 , D_out= c.D_out, Hidden_layer_size= 500, Num_layers= 3),args =[0,1])]

architecture_V4 =[dict(Type="MLP",config=dict(D_in=c.D_in , D_out= 3, Hidden_layer_size= 600, Num_layers= 5),args =None),
                dict(Type="Physics", Forward = c.partial_physics_v3, Jacobian = c.partial_jacobian_v3,args =None),
                dict(Type="CNN",config=dict(C_in=1,C_out = 2,Pool=True, kern_size = (1,3) , D_Size = 238, D_out= 2, Hidden_layer_size= 0, Num_layers= 0),args =None),
                dict(Type="MLP",config=dict(D_in=29 , D_out= c.D_out, Hidden_layer_size= 600, Num_layers= 5),args =[1])]


architecture_V5 = [dict(Type="Physics", Forward = c.partial_physics_v5, Jacobian = c.partial_jacobian_v5,args =None),
                dict(Type="MLP",config=dict(D_in=3 , D_out= c.D_out, Hidden_layer_size= 600, Num_layers= 5),args =[1])]

architecture_ANN = [dict(Type="MLP",config=dict(D_in=c.D_in , D_out= c.D_out, Hidden_layer_size= 700, Num_layers= 5),args =None)]


models = dict(
    V1 = [architecture_V1,"V1/Model_final.pt"],
    # V2 = [architecture_V2,"V2/Model_final.pt"], 
    V4 = [architecture_V4,"V4/Model_final.pt"],
    # V5 = [architecture_V5,"V5/Model_final.pt"],
    ANN = [architecture_ANN, "Pure_ANN/Model_final.pt"], 
)

gen_results(models)
# make_surface_plot(models)


# # V1_preds,V1_err = test_model(architecture_V1, "V1/Model_100.pt")
# V2_preds,V2_err = test_model(architecture_V2, "V2/Model_final.pt")
# # V3_preds,V3_err = test_model(architecture_V3, "V3/Model_1000.pt")
# # V4_preds,V4_err = test_model(architecture_V4, "V4/Model_final.pt")
# V5_preds,V5_err = test_model(architecture_V4, "V4/Model_final.pt")
# ANN_preds,ANN_err = test_model(architecture_ANN,"Pure_ANN/Model_final.pt")


## GPR
# from sklearn.ensemble import RandomForestRegressor
# import numpy as np
# traindata = np.load("Data/trainset_normal.npy")
# testdata = np.load("Data/testset_normal.npy")

# model = RandomForestRegressor(n_estimators=300)

# model.fit(traindata[:,:2], traindata[:,2:])
# preds = model.predict(testdata[:,:2])
# rf_err = np.abs(preds-testdata[:,2:])



# fig = go.Figure(layout=layout)
# # fig.add_trace(go.Box(y = V1_err[:,0],name="V1_Depth",showlegend=False))
# # fig.add_trace(go.Box(y = V1_err[:,1],name="V1_Width",showlegend=False))
# fig.add_trace(go.Box(y = V2_err[:,0],name="V2_Depth",showlegend=False))
# fig.add_trace(go.Box(y = V2_err[:,1],name="V2_Width",showlegend=False))
# # fig.add_trace(go.Box(y = V4_err[:,0],name="V4_Depth",showlegend=False))
# # fig.add_trace(go.Box(y = V4_err[:,1],name="V4_Width",showlegend=False))
# fig.add_trace(go.Box(y = ANN_err[:,0],name="ANN_Depth",showlegend=False))
# fig.add_trace(go.Box(y = ANN_err[:,1],name="ANN_Width",showlegend=False))
# # fig.add_trace(go.Box(y = rf_err[:,0],name="RF_Depth",showlegend=False))
# # fig.add_trace(go.Box(y = rf_err[:,1],name="RF_Width",showlegend=False))

# fig.update_xaxes(showgrid=False, showline=True, linecolor="black", mirror=True)
# fig.update_yaxes(showgrid=False, showline=True, linecolor="black", mirror=True, title="Absolute Error (Normalized)")
# fig.show()


# from joblib import load

# dat = load("Training_History_V2/Training_History.joblib")

# fig = go.Figure()
# fig.add_trace(go.Scatter(x = np.arange(len(dat["train_loss"])), y = dat["train_loss"], name = "Train Loss"))
# fig.add_trace(go.Scatter(x = np.arange(len(dat["test_loss"])), y = dat["test_loss"], name = "Test Loss"))
# fig.update_yaxes(type="log")
# fig.show()