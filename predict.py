import importlib
import torch
from torch import nn
from torch.utils.data import DataLoader
import ct_utils
device='cpu';

import numpy as np
from rdkit import Chem
import os


#***********************************************************************************
def get_data( global_features, labels ):

    bPKs= [];
    global_feat_list = [];

    for i, tmp_label in enumerate(labels):
        
        bPKs.append(int(tmp_label));
        global_feat_list.append(torch.from_numpy(global_features[i,:])); 
        
    return  global_feat_list, bPKs;
    
#***********************************************************************************   
def collate(sample):
    global_feats, labels = map(list,zip(*sample));
    global_feats = torch.stack([torch.tensor(tmp) for tmp in global_feats]);

    return  global_feats, torch.tensor(labels);

#**********************************************************************************
def scale_features(df, model_path):
    # scale the data
    from sklearn.preprocessing import StandardScaler
    import pickle
    scalerfile = model_path + "/scaler.sav";
    final_train_scaler = pickle.load(open(scalerfile, 'rb'));

    df[features_to_be_scaled] = final_train_scaler.transform(df[features_to_be_scaled]);

    return df;

#***********************************************************************************
def predict(df, model_path, scale=False, num_cpu=1):
    
    df = df.copy();
    
    #print("Scaling the features ...");
    if scale:
        df = scale_features(df, model_path);

    import sys
    sys.path.insert(1, model_path);
    mod = importlib.import_module('DeepCt');
    importlib.reload(mod);
    DeepCt = getattr(mod, 'DeepCt');

    num_models = 10;
    num_tasks = len(targets_combined);
    model_list = [];

    for i in range(num_models):
        tmp_model = DeepCt(global_feats=len(global_features), num_layers=5, n_tasks=num_tasks, predictor_hidden_feats=256);

        tmp_model.load_state_dict(torch.load(model_path + "/weights_" + str(i) + ".pth", map_location=torch.device('cpu')));
        tmp_model.eval();
        model_list.append(tmp_model);

    global_feat_list, bPK = get_data(df[global_features].to_numpy(dtype=np.float32), [1]*df.shape[0]);
    
    test_data = list(zip( global_feat_list, bPK)).copy();
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, collate_fn=collate, drop_last=False);

    model_list_device = [tmp_model.to(device) for tmp_model in model_list];
    
    predictions = np.array([], dtype=np.float).reshape(0, num_tasks);
    for i, ( global_feats, labels) in enumerate(test_loader):            
        labels = labels.to(device);
        global_feats = global_feats.to(device);

        tmp_predictions = np.array([], dtype=np.float).reshape(0, 0);
        for tmp_model in model_list_device:
            if tmp_predictions.shape[0] == 0:
                tmp_predictions = tmp_model( global_feats).detach().numpy();         
            else:
                tmp_predictions = tmp_predictions + tmp_model(global_feats).detach().numpy();

        tmp_predictions = tmp_predictions/float(len(model_list_device)); 
        predictions = np.concatenate((predictions, tmp_predictions), axis=0)
    
    return predictions;

#***********************************************************************************
def predict_single_model(df, model_path, model_id, scale=False, num_cpu=1):
    
    df = df.copy();
    
    #print("Scaling the features ...");
    if scale:
        df = scale_features(df, model_path);

    import sys
    sys.path.insert(1, model_path);
    mod = importlib.import_module('DeepCt');
    importlib.reload(mod);
    DeepCt = getattr(mod, 'DeepCt');

    num_models = 1;
    num_tasks = len(targets_combined);
    model_list = [];

    tmp_model = DeepCt(global_feats=len(global_features), num_layers=5, n_tasks=num_tasks, predictor_hidden_feats=256);
    tmp_model.load_state_dict(torch.load(model_path + "/weights_" + str(model_id) + ".pth", map_location=torch.device('cpu')));
    tmp_model.eval();
    model_list.append(tmp_model);

    global_feat_list, bPK = get_data(df[global_features].to_numpy(dtype=np.float32), [1]*df.shape[0]);
    
    test_data = list(zip( global_feat_list, bPK)).copy();
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, collate_fn=collate, drop_last=False);

    model_list_device = [tmp_model.to(device) for tmp_model in model_list];
    
    predictions = np.array([], dtype=np.float).reshape(0, num_tasks);
    for i, ( global_feats, labels) in enumerate(test_loader):            
        labels = labels.to(device);
        global_feats = global_feats.to(device);

        tmp_predictions = np.array([], dtype=np.float).reshape(0, 0);
        for tmp_model in model_list_device:
            if tmp_predictions.shape[0] == 0:
                tmp_predictions = tmp_model( global_feats).detach().numpy();         
            else:
                tmp_predictions = tmp_predictions + tmp_model(global_feats).detach().numpy();

        tmp_predictions = tmp_predictions/float(len(model_list_device)); 
        predictions = np.concatenate((predictions, tmp_predictions), axis=0)
    
    return predictions;


#**************************************************************
#************** predict C-t curves from ensemble **************
#**************************************************************
def predict_ct_from_ensemble(df, doses, model_path, num_models=10, meas_times=None, num_cmpts=3, scale=False):
    
    if meas_times is None:
        meas_times = torch.linspace(0.0,200,1000).repeat(df.shape[0], 1);
        
    for tmp_model in range(num_models):

        pred_consts = predict_single_model(df, model_path, tmp_model, scale=scale, num_cpu=1);
        tmp_times, po_curve_pred = ct_utils.ODE_solutions_torch(torch.tensor(pred_consts[:, :7]), route="p.o.", times=meas_times, dose=torch.tensor(doses), num_cmpts=num_cmpts);
        _, iv_curve_pred = ct_utils.ODE_solutions_torch(torch.tensor(pred_consts[:, 7:]), route="i.v.", times=meas_times, dose=torch.tensor(doses),
                                                       num_cmpts=num_cmpts);        
        
        if tmp_model == 0:
            po = po_curve_pred;
            iv = iv_curve_pred;
        else:
            po = po + po_curve_pred;
            iv = iv + iv_curve_pred;

    po = po*1.0/num_models;
    iv = iv*1.0/num_models;

    return tmp_times, po, iv;

#**************************************************************
#********** predict derived readouts from ensemble ************
#**************************************************************
def predict_derived_from_ensemble(df, doses, model_path, species, num_models=10, num_cmpts=2, scale=False):
    
    for tmp_model in range(num_models):

        pred_consts = predict_single_model(df, model_path, tmp_model, scale=scale, num_cpu=1);
        tmp_drv = ct_utils.get_derived_parameters_from_ct_curve_torch(torch.tensor(pred_consts[:,:7]), torch.tensor(pred_consts[:,7:]), species, dose=torch.tensor(doses), num_cmpts=num_cmpts);
        tmp_drv = tmp_drv.numpy();
        
        if tmp_model == 0:
            drv = tmp_drv;
        else:
            drv = drv + tmp_drv;
            
    drv = drv*1.0/num_models;
        
    return drv;
    
    
#****************************************************************************
#****************** some pre-defined lists of features **********************
#****************************************************************************
global_features = ["melloddy_emb_" + str(i) for i in range(2000)]

features_to_be_scaled = ["Dose_trf"];


# targets for c-t modelling
targets_po = ["ka_po", "Cl_po", "Vc_po", "Q1_po", "Vp1_po", "Q2_po", "Vp2_po"];
targets_iv = ["CL_iv", "Vc_iv", "Q1_iv", "Vp1_iv", "Q2_iv", "Vp2_iv"];
targets_combined =  targets_po + targets_iv;