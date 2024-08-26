import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from scipy.integrate import ode
from sklearn.metrics import r2_score, auc
import torch
from torch.utils.data import DataLoader
from torcheval.metrics.aggregation.auc import AUC
import os, sys
import contextlib
torch.pi = torch.tensor(3.141592653589793)

if torch.cuda.is_available():
    print('use GPU')
    device='cuda'
else:
    print('use CPU')
    device='cpu'

#****************************************************************************
#********************* get descriptastorus descriptors **********************
#****************************************************************************
def get_descriptastorus_properties(df, name_smiles_col="Structure"):
    
    from descriptastorus.descriptors import rdNormalizedDescriptors
    #----------------------------------------------------
    #------------- ----- descriptastorus ----------------
    #----------------------------------------------------
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    cdf_norm_cols = [i[0]+"_cdf_norm" for i in generator.columns];
    # example for converting a smiles string into the values
    def rdkit_2d_normalized_features(smiles: str):
        # n.b. the first element is true/false if the descriptors were properly computed
        results = generator.process(smiles)
        processed, features = results[0], results[1:]
        if processed is None:
            logging.warning("Unable to process smiles %s", smiles)
        # if processed is None, the features are are default values for the type
        return features

    smiles = df[name_smiles_col].to_numpy();
    num_compounds = smiles.size;
    inds = np.arange(num_compounds);
    unique_smis, rec_inds = np.unique(smiles, return_inverse=True);

    tmp_data_arr = np.zeros((num_compounds, len(cdf_norm_cols)));

    for count, tmp_smi in enumerate(unique_smis):

            if count % 10000 == 0:
                print("Analyzing compound " + repr(count));

            tmp_inds = inds[rec_inds==count];
            try:
                tmp_props = rdkit_2d_normalized_features(tmp_smi);
                tmp_data_arr[tmp_inds, :] = np.array(tmp_props);
            except:
                tmp_data_arr[tmp_inds, :] = np.nan

    #append the columns to the original dataframe   
    for tmp_descriptor_ind in range(len(cdf_norm_cols)):
        df[cdf_norm_cols[tmp_descriptor_ind]] = tmp_data_arr[:, tmp_descriptor_ind];
       
    return df, cdf_norm_cols;
      
#****************************************************************************
#************************** get RDKit descriptors ***************************
#****************************************************************************
def get_rdkit_properties(df, name_smiles_col="Structure"):
    
    #calculate the features
    smiles = df[name_smiles_col].to_list();
    num_compounds = len(smiles);

    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit.Chem import Descriptors
    des_list = [x[0] for x in Descriptors._descList]
    des_list = des_list + ["SA"];
    destype_list = ["RDKit descriptors"] * len(des_list);
    from rdkit.Chem import RDConfig
    import os
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer

    smiles = df[name_smiles_col].to_numpy();
    num_compounds = smiles.size;
    inds = np.arange(num_compounds);
    unique_smis, rec_inds = np.unique(smiles, return_inverse=True);

    tmp_data_arr = np.zeros((num_compounds, len(des_list)))*np.nan;

    for count, tmp_smi in enumerate(unique_smis):

            if count % 10000 == 0:
                print("Analyzing compound " + repr(count));

            tmp_inds = inds[rec_inds==count];
            try:
                tmp_mol = Chem.MolFromSmiles(tmp_smi);
            except:
                continue;
            
            for tmp_descriptor_ind in range(len(des_list)):

                tmp_descriptor = des_list[tmp_descriptor_ind];

                if tmp_descriptor == "QED":
                    tmp_descriptor_val = Chem.QED.qed(tmp_mol);
                elif tmp_descriptor == "SA":
                    tmp_descriptor_val = sascorer.calculateScore(tmp_mol);
                else:
                    calc = MoleculeDescriptors.MolecularDescriptorCalculator([tmp_descriptor])
                    descriptors = calc.CalcDescriptors(tmp_mol);
                    tmp_descriptor_val = descriptors[0];          

                tmp_data_arr[tmp_inds, tmp_descriptor_ind] = tmp_descriptor_val;

    #append the columns to the original dataframe   
    for tmp_descriptor_ind in range(len(des_list)):
        df[des_list[tmp_descriptor_ind]] = tmp_data_arr[:, tmp_descriptor_ind];

    #now norm all descriptors with respect to the number of atoms
    for tmp_descriptor in des_list:

        tmp_data = df[tmp_descriptor].to_numpy();

        tmp_data = 1000*tmp_data/df["HeavyAtomCount"].to_numpy();
        df[tmp_descriptor + "/1000 HeavyAtoms"] = tmp_data;

        des_list = des_list + [tmp_descriptor + "/1000 HeavyAtoms"];
        destype_list = destype_list + ["RDKit descriptors/1000 atoms"];
    
    return df, des_list; 
        
#****************************************************************************
#***************************** get fingerprints *****************************
#****************************************************************************
def get_fingerprints(df, name_smiles_col="Structure"):
    #get fingerprints
    num_bits=1024

    #get list of fingerprints from smiles
    fp_list = [];
    for count, tmp_smi in enumerate(df[name_smiles_col].to_list()):

        if count % 10000 == 0:
            print("Analyzing compound " + repr(count));

        tmp_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(tmp_smi), 2, nBits=num_bits);
        fp_list.append(tmp_fp);

    #make array of fingerprints
    num_fp = len(fp_list)
    fp_array = np.zeros((num_fp, num_bits), dtype=np.int8)

    for tmp_fp in range(num_fp):
        tmp_array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp_list[tmp_fp], tmp_array)
        fp_array[tmp_fp, :] = tmp_array

    #append them to dataframe
    fingerprint_cols = ["fingerprint_" + str(i) for i in range(fp_array.shape[1])];
    for i, tmp_col in enumerate(fingerprint_cols):
        df[tmp_col] = fp_array[:,i];
        
    return df, fingerprint_cols;
    
#*********************************************************
#******* extract derived PK endpoints from curves ********
#*********************************************************
def get_derived_parameters_from_ct_curve_torch(params_po, params_iv, species, dose=1.0, num_cmpts=3):
        
    BW = torch.ones(species.shape[0]);
    BW[species=="Mouse"] = torch.tensor(0.025) #kg
    BW[species=="Rat"] = torch.tensor(0.25) #kg
    BW[species=="Dog"] = torch.tensor(10) #kg
    BW[species=="Monkey"] = torch.tensor(4) #kg
        
    t_po, Ct_po = ODE_solutions_torch(params_po, route="p.o.", dose=dose, num_cmpts=num_cmpts);
    t_iv, Ct_iv = ODE_solutions_torch(params_iv, route="i.v.", dose=dose, num_cmpts=num_cmpts);
        
    metric = AUC(n_tasks=params_po.shape[0])
    metric.update(t_po, Ct_po)
    AUC_po = metric.compute()
    metric.reset()
    
    metric.update(t_iv, Ct_iv)
    AUC_iv = metric.compute()
    metric.reset()
    
    metric.update(t_iv, Ct_iv*t_iv)
    AUMC_iv = metric.compute()
    metric.reset()
    
    Cmax_iv = torch.max(Ct_iv, dim=1)[0];
    Cmax_po = torch.max(Ct_po, dim=1)[0];
 
    #get AUC last IV
    tmp_Ct_iv = Ct_iv.clone();
    tmp_Ct_iv = tmp_Ct_iv/Cmax_iv[:,None];
    tmp_Ct_iv[tmp_Ct_iv<=0.1] = torch.tensor(0.0);
    tmp_Ct_iv = tmp_Ct_iv*Cmax_iv[:,None];
    Clast_iv = Cmax_iv*torch.tensor(0.1);
    metric.update(t_iv, tmp_Ct_iv)
    AUClast_iv = metric.compute()
    metric.reset()    
    
    #get AUC last PO
    tmp_Ct_po = Ct_po.clone();
    tmp_Ct_po = tmp_Ct_po/Cmax_po[:,None];
    tmp_Ct_po[tmp_Ct_po<=0.1] = torch.tensor(0.0);
    tmp_Ct_po = tmp_Ct_po*Cmax_po[:,None];
    Clast_po = Cmax_po*torch.tensor(0.1);
    metric.update(t_po, tmp_Ct_po)
    AUClast_po = metric.compute()
    metric.reset()  
    
    #get clearance
    CL = dose/(BW*AUC_iv);    
    CLout = (CL/torch.tensor(60.0))*torch.tensor(1000.0)

    #calc half-life from AUC_tail
    AUC_diff_iv = AUC_iv-AUClast_iv
    AUC_diff_iv = torch.clamp(AUC_diff_iv, min=10e-5 ,max= 10e5 );
    t_half_iv = torch.log(torch.tensor(2))/  (Clast_iv/AUC_diff_iv)
    t_half_iv[(AUC_diff_iv==10e-5)  | (AUC_diff_iv==10e5)] = torch.nan;
    
    AUC_diff_po = AUC_po-AUClast_po
    AUC_diff_po = torch.clamp(AUC_diff_po, min=10e-5 ,max= 10e5 );
    t_half_po = torch.log(torch.tensor(2))/  (Clast_po/AUC_diff_po)
    t_half_po[(AUC_diff_po==10e-5)  | (AUC_diff_po==10e5)] = torch.nan;

    #get vss, mrt and F from the other predicted props
    MRT = AUMC_iv/AUC_iv;
    Vout = MRT*CL;

    F = AUC_po/AUC_iv;
    
    #set-up of output list
    dervived_endpoints = torch.cat((torch.unsqueeze(AUC_po, 0), torch.unsqueeze(Cmax_po, 0), torch.unsqueeze(t_half_po, 0), torch.unsqueeze(AUC_iv, 0), torch.unsqueeze(Cmax_iv, 0), torch.unsqueeze(t_half_iv, 0), torch.unsqueeze(CLout, 0), torch.unsqueeze(Vout, 0), torch.unsqueeze(F, 0), torch.unsqueeze(MRT, 0)),0);
    
    return torch.transpose(dervived_endpoints, 0, 1);

#*******************************************
#**************** calc Dose ****************
#*******************************************
def recalc_dose(dose, mw, species):
    
    trf_doses = [];
    for i, tmp_dose in enumerate(dose):
        tmp_mw = mw[i]
        tmp_species = species[i]
    
        if tmp_species == "Mouse":
            BW = 0.025 #kg
        elif tmp_species == "Rat":
            BW = 0.25 #kg
        elif tmp_species == "Dog":
            BW = 10 #kg
        elif tmp_species == "Monkey":
            BW = 4 #kg
        else:
            BW = np.nan;

        trf_doses.append(tmp_dose*10**6/tmp_mw*BW);
    
    return trf_doses;
    
#***************************************
#********** Analytical sol *************
#***************************************
def ODE_solutions_torch(params, route="p.o.", dose=None, times=None, num_cmpts=3):
    
    if len(params.shape)==2:
        num_samples = params.shape[0];
        if dose is None:
            dose = torch.ones(num_samples).to(device);
    else:
        num_samples = 1;
        if dose is None:
            dose = torch.tensor(1.0).to(device);
    
    if times is not None:
        times = times.clone()
        times[torch.isnan(times)] = 100000;
    
    #-----------------------------------------
    #------------- i.v. solution -------------
    #-----------------------------------------
    if route == "i.v.":
        
        if len(params.shape)==2:
            CL, V1, Q2, V2, Q3, V3 = params[:,0], params[:,1], params[:,2], params[:,3], params[:,4], params[:,5];
        else:    
            CL, V1, Q2, V2, Q3, V3 = params;
          
        if num_cmpts==2:
            Q3 = torch.tensor(0.0)*Q3;
            V3 = torch.tensor(0.000001)*V3 + torch.tensor(1.0); 
        elif num_cmpts==1:
            Q3 = torch.tensor(0.0)*Q3;
            V3 = torch.tensor(0.000001)*V3 + torch.tensor(1.0); 
            Q2 = torch.tensor(0.0)*Q2;
            V2 = torch.tensor(0.000001)*V2 + torch.tensor(1.0); 
        
        a0 = (CL*Q2*Q3)/(V1*V2*V3);
        a1 = ((CL*Q3)/(V1*V3)) + ((Q2*Q3)/(V2*V3)) + ((Q2*Q3)/(V2*V1)) + ((CL*Q2)/(V1*V2)) + ((Q3*Q2)/(V3*V1));
        a2 = (CL/V1) + (Q2/V1) + (Q3/V1) + (Q2/V2) + (Q3/V3);
        
        p = a1 - (a2**torch.tensor(2))/torch.tensor(3);
        q = (torch.tensor(2.0)*(a2**torch.tensor(3))/torch.tensor(27)) - (a1*a2/torch.tensor(3)) + a0;
        
        r1 = torch.sqrt(-(p**torch.tensor(3))/torch.tensor(27));
        r2 = torch.tensor(2)*(r1**(torch.tensor(1)/torch.tensor(3)));

        l = -q/(torch.tensor(2)*r1)
        if len(params.shape)==2:
            l[l<=-1.0] = torch.tensor(-0.999999);
            l[l>=1.0] = torch.tensor(0.999999);
        else:
            if l<=-1.0: 
                l = torch.tensor(-0.999999);
            elif l>=1.0:
                l = torch.tensor(0.999999);
        
        phi = torch.arccos(l)/torch.tensor(3);
        alpha = -(torch.cos(phi)*r2 - (a2/torch.tensor(3)));
        beta = -(torch.cos(phi+(torch.tensor(2)*torch.pi/torch.tensor(3)))*r2 - a2/torch.tensor(3))
        gamma = -(torch.cos(phi+(torch.tensor(4)*torch.pi/torch.tensor(3))) *r2 - a2/torch.tensor(3))

        A = (((Q2/V2) - alpha)*((Q3/V3) - alpha))/(V1*(alpha-beta)*(alpha-gamma));
        B = (((Q2/V2) - beta)*((Q3/V3) - beta))/(V1*(beta-alpha)*(beta-gamma));
        C = (((Q2/V2) - gamma)*((Q3/V3) - gamma))/(V1*(gamma-beta)*(gamma-alpha));

        if len(params.shape)==2:
            if times is None:
                times = torch.linspace(0.0,200,1000).repeat(num_samples, 1).to(device);
            D = torch.clamp(-alpha[:,None]*times,  max=50);
            E = torch.clamp(-beta[:,None]*times,  max=50);
            F = torch.clamp(-gamma[:,None]*times, max=50);    
            conc =  dose[:,None]*(A[:,None]*torch.exp(D) + B[:,None] * torch.exp(E) + C[:,None]*torch.exp(F))
        else:
            if times is None:
                times = torch.linspace(0.0,200,1000).to(device);
            D = torch.clamp(-alpha*times,  max=50);
            E = torch.clamp(-beta*times,  max=50);
            F = torch.clamp(-gamma*times, max=50);    
            conc = dose *  (A*torch.exp(D) + B * torch.exp(E) + C*torch.exp(F));
            
    #-----------------------------------------
    #------------- p.o. solution -------------
    #-----------------------------------------
    elif route == "p.o.":
        if len(params.shape)==2:   
            ka, CL, V1, Q2, V2, Q3, V3 = params[:,0], params[:,1], params[:,2], params[:,3], params[:,4], params[:,5], params[:,6];
        else:    
            ka, CL, V1, Q2, V2, Q3, V3 = params;
            
        if num_cmpts==2:
            Q3 = torch.tensor(0.0).to(device)*Q3;
            V3 = torch.tensor(1.0).to(device); 
        elif num_cmpts==1:
            Q3 = torch.tensor(0.0).to(device)*Q3;
            V3 = torch.tensor(1.0).to(device);
            Q2 = torch.tensor(0.0).to(device)*Q2;
            V2 = torch.tensor(1.0).to(device); 

        a0 = (CL*Q2*Q3)/(V1*V2*V3);
        a1 = ((CL*Q3)/(V1*V3)) + ((Q2*Q3)/(V2*V3)) + ((Q2*Q3)/(V2*V1)) + ((CL*Q2)/(V1*V2)) + ((Q3*Q2)/(V3*V1));
        a2 = (CL/V1) + (Q2/V1) + (Q3/V1) + (Q2/V2) + (Q3/V3);    
        
        p = a1 - (a2**torch.tensor(2).to(device))/torch.tensor(3).to(device);
        q = (torch.tensor(2)*(a2**torch.tensor(3).to(device))/torch.tensor(27).to(device))   -   (a1*a2/torch.tensor(3).to(device)) + a0
        r1 = torch.sqrt(-(p**torch.tensor(3).to(device))/torch.tensor(27).to(device));
        r2 = torch.tensor(2).to(device)*(r1**(torch.tensor(1).to(device)/torch.tensor(3).to(device)));
        
        l = -q/(torch.tensor(2).to(device)*r1)
        
        if len(params.shape)==2:
            l[l<=-1.0] = torch.tensor(-0.999999 );
            l[l>=1.0] = torch.tensor(0.999999 );
        else:
            if l<=-1.0: 
                l = torch.tensor(-0.999999);
            elif l>=1.0:
                l = torch.tensor(0.999999);

        phi = torch.arccos(l)/torch.tensor(3).to(device);
        alpha = -(torch.cos(phi)*r2 - (a2/torch.tensor(3).to(device)));
        beta = -(torch.cos(phi+(torch.tensor(2).to(device)*torch.pi/torch.tensor(3).to(device)))*r2 - a2/torch.tensor(3).to(device))
        gamma = -(torch.cos(phi+(torch.tensor(4).to(device)*torch.pi/torch.tensor(3).to(device))) *r2 - a2/torch.tensor(3).to(device) )
        
        A = (ka*((Q2/V2) - alpha)*((Q3/V3) - alpha))/((ka-alpha)*(alpha-beta)*(alpha-gamma));
        B = (ka*((Q2/V2) - beta)*((Q3/V3) - beta))/((ka-beta)*(beta-alpha)*(beta-gamma));
        C = (ka*((Q2/V2) - gamma)*((Q3/V3) - gamma))/((ka-gamma)*(gamma-beta)*(gamma-alpha));

        if len(params.shape)==2:
            if times is None:
                times = torch.linspace(0.0,200,1000).repeat(num_samples, 1).to(device);
            D = torch.clamp(-alpha[:,None]*times,  max=50);
            E = torch.clamp(-beta[:,None]*times,  max=50);
            F = torch.clamp(-gamma[:,None]*times, max=50);    
            conc =  dose[:,None]*(A[:,None]*torch.exp(D) + B[:,None] * torch.exp(E) + C[:,None]*torch.exp(F) - ((A[:,None]+B[:,None]+C[:,None])*torch.exp(-ka[:,None]*times)));
        else:
            if times is None:
                times = torch.linspace(0.0,200,1000).to(device);
            D = torch.clamp(-alpha*times,  max=50);
            E = torch.clamp(-beta*times,  max=50);
            F = torch.clamp(-gamma*times, max=50);    
            conc = dose *  (A*torch.exp(D) + B * torch.exp(E) + C*torch.exp(F) - ((A+B+C)*torch.exp(-ka*times)));
    
    conc[torch.isinf(conc)] = torch.tensor(5*10**10);
    conc[conc>5*10**10] = torch.tensor(5*10**10);
        
    if times is not None:
        times = times.clone();
        times[times==100000] = np.nan;
        
    return times, conc;

#*************************************************************************
#**************** L2**2 for model using derived endpoints ****************
#*************************************************************************
def L2_derived_loss(preds, observed):
    
    preds = preds[~torch.isnan(observed)];
    observed = observed[~torch.isnan(observed)];

    L2_loss = torch.absolute((torch.log(preds)-observed));
    L2_loss = torch.mean(L2_loss);
    
    return L2_loss;

#***************************************
#********** exp. curve loss ************
#***************************************
def L2_expcurve_and_readout_loss(preds, observed, num_cmpts=3, curve_weight=1.0, readout_weight=0.0, compartment_penalty_wight=0.1, weighting="abs_log_error"):
    
    #Structure of preds
    # these are the predicted compartmental constants, i.e. 13 columns: "ka_po", "Cl_po", "Vc_po", "Q1_po", "Vp1_po", "Q2_po", "Vp2_po", "CL_iv", "Vc_iv", "Q1_iv", "Vp1_iv", "Q2_iv", "Vp2_iv"
    
    #Structure of observed:
    # columns 0-99: time of each measure concentration 
    # columns 100-199: measured p.o. concentrations 
    # columns 200-299: measured i.v. concentrations 
    # columns 300: dose 
    # columns 301-311: derived PK readouts 'AUCinf_p.o.','Cmax_p.o.', 'T(12)_p.o.', 'AUCinf_i.v.','Cmax_i.v.', 'T(12)_i.v.', 'In vivo CL', 'Vss', "F", "MRT", these can be ignored when  readout_weight=0

    meas_times = observed[:, :100].clone()
    meas_conc_po = observed[:, 100:200].clone()
    meas_conc_iv = observed[:, 200:300].clone()
    doses = observed[:, 300].clone();

    tmp_params_pred_po = preds[:, :7].clone();
    tmp_params_pred_iv = preds[:, 7:].clone();

    times, conc_pred_po = ODE_solutions_torch(tmp_params_pred_po.to(device), route="p.o.", dose=doses, times=meas_times, num_cmpts=num_cmpts)
    _, conc_pred_iv = ODE_solutions_torch(tmp_params_pred_iv.to(device), route="i.v.", dose=doses, times=meas_times, num_cmpts=num_cmpts);
    ct_exp_inds = torch.arange(conc_pred_po.shape[0])[:,None].repeat(1,conc_pred_po.shape[1]);
    t_last = torch.max(times, dim=1)[0];
    #rel_times = times/t_last[:,None];
    
    #*************************************
    #******* first the curve loss ********
    #*************************************
    tmp_preds = torch.cat((torch.flatten(conc_pred_po), torch.flatten(conc_pred_iv)), 0);
    tmp_observed = torch.cat((torch.flatten(meas_conc_po), torch.flatten(meas_conc_iv)), 0);
    tmp_ct_inds = torch.cat((torch.flatten(ct_exp_inds), torch.flatten(ct_exp_inds)), 0);
    #times_weights = torch.cat((torch.flatten(rel_times), torch.flatten(rel_times)), 0);

    #remove unmeasured data
    tmp_preds_filt = tmp_preds[(~torch.isnan(tmp_observed)) & (~torch.isinf(tmp_observed)) & (~torch.isnan(tmp_preds)) & (~torch.isinf(tmp_preds)) ];
    tmp_observed_filt = tmp_observed[(~torch.isnan(tmp_observed)) & (~torch.isinf(tmp_observed)) & (~torch.isnan(tmp_preds)) & (~torch.isinf(tmp_preds))];
    tmp_ct_inds_filt = tmp_ct_inds[(~torch.isnan(tmp_observed)) & (~torch.isinf(tmp_observed)) & (~torch.isnan(tmp_preds)) & (~torch.isinf(tmp_preds))];
    #times_weights_filt = times_weights[(~torch.isnan(tmp_observed)) & (~torch.isinf(tmp_observed)) & (~torch.isnan(tmp_preds)) & (~torch.isinf(tmp_preds))];
    
    tmp_preds_filt2 = tmp_preds_filt[(tmp_preds_filt>=1e-5) & (tmp_observed_filt>=1e-5)];
    tmp_observed_filt2 = tmp_observed_filt[(tmp_preds_filt>=1e-5) & (tmp_observed_filt>=1e-5)];
    tmp_ct_inds_filt2 = tmp_ct_inds_filt[(tmp_preds_filt>=1e-5) & (tmp_observed_filt>=1e-5)];
    #times_weights_filt2 = times_weights_filt[(tmp_preds_filt>=1e-5) & (tmp_observed_filt>=1e-5)];
    
    if weighting == "abs_log_error":
        errs_curve = torch.absolute(torch.log(tmp_preds_filt2) - torch.log(tmp_observed_filt2));
    elif weighting == "Y_hat**2":
        errs_curve = torch.absolute(torch.sqrt(tmp_preds_filt2) - torch.sqrt(tmp_observed_filt2));
    elif weighting == "abs_error":
        errs_curve = torch.absolute(tmp_preds_filt2 - tmp_observed_filt2);

    #get the mean of each curve
    means = torch.zeros(conc_pred_po.shape[0]);
    for i in torch.arange(conc_pred_po.shape[0]):
        means[i] = torch.mean(errs_curve[tmp_ct_inds_filt2==i]);
        
    #get the mean over the means of the curves
    L2_loss_curve = torch.mean(means[~torch.isnan(means) & ~torch.isinf(means)]);
    
    #*************************************
    #******* now the readout loss ********
    #*************************************
    if readout_weight != 0:
        drv = get_derived_parameters_from_ct_curve_torch(tmp_params_pred_po, tmp_params_pred_iv, species="rat", dose=doses, num_cmpts=num_cmpts);
        drv_exp_inds = torch.arange(drv.shape[0])[:,None].repeat(1,drv.shape[1]);

        readouts = observed[:, 301:311].clone();

        pred_readout_filt = drv[~torch.isnan(readouts) & ~torch.isnan(drv)  &  ~torch.isinf(readouts) & ~torch.isinf(drv)];
        observed_readout_filt = readouts[~torch.isnan(readouts) & ~torch.isnan(drv)  &  ~torch.isinf(readouts) & ~torch.isinf(drv)];
        drv_exp_inds_filt = drv_exp_inds[~torch.isnan(readouts) & ~torch.isnan(drv)  &  ~torch.isinf(readouts) & ~torch.isinf(drv)];

        #log transform the derived parameters
        pred_readout_filt = torch.log(pred_readout_filt);
        errors_readout = torch.absolute(pred_readout_filt - observed_readout_filt);

        means = torch.zeros(drv.shape[0]);
        for i in torch.arange(drv.shape[0]):
            means[i] = torch.mean(errors_readout[drv_exp_inds_filt==i]);
        L2_loss_readout = torch.mean(means[~torch.isnan(means)  &  ~torch.isinf(means)]);

        if torch.isnan(L2_loss_readout) | torch.isinf(L2_loss_readout):
            L2_loss_readout = torch.tensor(0.0);
    else:
        L2_loss_readout = torch.tensor(0.0);
       
    #*****************************************************************************************
    #now the loss on the compartmental constats, penalize multiple compartments
    #*****************************************************************************************
    penalty_cmpt1 = torch.mean(tmp_params_pred_po[:,:3]) + torch.mean(tmp_params_pred_iv[:,:2]);
    penalty_cmpt2 = torch.mean(tmp_params_pred_po[:,3:5]) + torch.mean(tmp_params_pred_iv[:,2:4]);
    penalty_cmpt3 = torch.mean(tmp_params_pred_po[:,5:]) + torch.mean(tmp_params_pred_iv[:,4:]);
    weigth_penalty = torch.tensor(compartment_penalty_wight);
    
    if num_cmpts == 3:
        ca_const_penalty =  weigth_penalty*penalty_cmpt2 + weigth_penalty*penalty_cmpt3;
    elif num_cmpts == 2:
        ca_const_penalty =  weigth_penalty*penalty_cmpt2;
    else:
        ca_const_penalty = weigth_penalty*penalty_cmpt1;
        
    L2_loss = torch.tensor(readout_weight)*L2_loss_readout + torch.tensor(curve_weight)*L2_loss_curve + ca_const_penalty;
    
    return L2_loss;
        
#**************************************
#*** some pre-defined feature lists ***
#**************************************
descriptastorus_features = ['BalabanJ_cdf_norm',
 'BertzCT_cdf_norm',
 'Chi0_cdf_norm',
 'Chi0n_cdf_norm',
 'Chi0v_cdf_norm',
 'Chi1_cdf_norm',
 'Chi1n_cdf_norm',
 'Chi1v_cdf_norm',
 'Chi2n_cdf_norm',
 'Chi2v_cdf_norm',
 'Chi3n_cdf_norm',
 'Chi3v_cdf_norm',
 'Chi4n_cdf_norm',
 'Chi4v_cdf_norm',
 'EState_VSA1_cdf_norm',
 'EState_VSA10_cdf_norm',
 'EState_VSA11_cdf_norm',
 'EState_VSA2_cdf_norm',
 'EState_VSA3_cdf_norm',
 'EState_VSA4_cdf_norm',
 'EState_VSA5_cdf_norm',
 'EState_VSA6_cdf_norm',
 'EState_VSA7_cdf_norm',
 'EState_VSA8_cdf_norm',
 'EState_VSA9_cdf_norm',
 'ExactMolWt_cdf_norm',
 'FpDensityMorgan1_cdf_norm',
 'FpDensityMorgan2_cdf_norm',
 'FpDensityMorgan3_cdf_norm',
 'FractionCSP3_cdf_norm',
 'HallKierAlpha_cdf_norm',
 'HeavyAtomCount_cdf_norm',
 'HeavyAtomMolWt_cdf_norm',
 'Ipc_cdf_norm',
 'Kappa1_cdf_norm',
 'Kappa2_cdf_norm',
 'Kappa3_cdf_norm',
 'LabuteASA_cdf_norm',
 'MaxAbsEStateIndex_cdf_norm',
 'MaxAbsPartialCharge_cdf_norm',
 'MaxEStateIndex_cdf_norm',
 'MaxPartialCharge_cdf_norm',
 'MinAbsEStateIndex_cdf_norm',
 'MinAbsPartialCharge_cdf_norm',
 'MinEStateIndex_cdf_norm',
 'MinPartialCharge_cdf_norm',
 'MolLogP_cdf_norm',
 'MolMR_cdf_norm',
 'MolWt_cdf_norm',
 'NHOHCount_cdf_norm',
 'NOCount_cdf_norm',
 'NumAliphaticCarbocycles_cdf_norm',
 'NumAliphaticHeterocycles_cdf_norm',
 'NumAliphaticRings_cdf_norm',
 'NumAromaticCarbocycles_cdf_norm',
 'NumAromaticHeterocycles_cdf_norm',
 'NumAromaticRings_cdf_norm',
 'NumHAcceptors_cdf_norm',
 'NumHDonors_cdf_norm',
 'NumHeteroatoms_cdf_norm',
 'NumRadicalElectrons_cdf_norm',
 'NumRotatableBonds_cdf_norm',
 'NumSaturatedCarbocycles_cdf_norm',
 'NumSaturatedHeterocycles_cdf_norm',
 'NumSaturatedRings_cdf_norm',
 'NumValenceElectrons_cdf_norm',
 'PEOE_VSA1_cdf_norm',
 'PEOE_VSA10_cdf_norm',
 'PEOE_VSA11_cdf_norm',
 'PEOE_VSA12_cdf_norm',
 'PEOE_VSA13_cdf_norm',
 'PEOE_VSA14_cdf_norm',
 'PEOE_VSA2_cdf_norm',
 'PEOE_VSA3_cdf_norm',
 'PEOE_VSA4_cdf_norm',
 'PEOE_VSA5_cdf_norm',
 'PEOE_VSA6_cdf_norm',
 'PEOE_VSA7_cdf_norm',
 'PEOE_VSA8_cdf_norm',
 'PEOE_VSA9_cdf_norm',
 'RingCount_cdf_norm',
 'SMR_VSA1_cdf_norm',
 'SMR_VSA10_cdf_norm',
 'SMR_VSA2_cdf_norm',
 'SMR_VSA3_cdf_norm',
 'SMR_VSA4_cdf_norm',
 'SMR_VSA5_cdf_norm',
 'SMR_VSA6_cdf_norm',
 'SMR_VSA7_cdf_norm',
 'SMR_VSA8_cdf_norm',
 'SMR_VSA9_cdf_norm',
 'SlogP_VSA1_cdf_norm',
 'SlogP_VSA10_cdf_norm',
 'SlogP_VSA11_cdf_norm',
 'SlogP_VSA12_cdf_norm',
 'SlogP_VSA2_cdf_norm',
 'SlogP_VSA3_cdf_norm',
 'SlogP_VSA4_cdf_norm',
 'SlogP_VSA5_cdf_norm',
 'SlogP_VSA6_cdf_norm',
 'SlogP_VSA7_cdf_norm',
 'SlogP_VSA8_cdf_norm',
 'SlogP_VSA9_cdf_norm',
 'TPSA_cdf_norm',
 'VSA_EState1_cdf_norm',
 'VSA_EState10_cdf_norm',
 'VSA_EState2_cdf_norm',
 'VSA_EState3_cdf_norm',
 'VSA_EState4_cdf_norm',
 'VSA_EState5_cdf_norm',
 'VSA_EState6_cdf_norm',
 'VSA_EState7_cdf_norm',
 'VSA_EState8_cdf_norm',
 'VSA_EState9_cdf_norm',
 'fr_Al_COO_cdf_norm',
 'fr_Al_OH_cdf_norm',
 'fr_Al_OH_noTert_cdf_norm',
 'fr_ArN_cdf_norm',
 'fr_Ar_COO_cdf_norm',
 'fr_Ar_N_cdf_norm',
 'fr_Ar_NH_cdf_norm',
 'fr_Ar_OH_cdf_norm',
 'fr_COO_cdf_norm',
 'fr_COO2_cdf_norm',
 'fr_C_O_cdf_norm',
 'fr_C_O_noCOO_cdf_norm',
 'fr_C_S_cdf_norm',
 'fr_HOCCN_cdf_norm',
 'fr_Imine_cdf_norm',
 'fr_NH0_cdf_norm',
 'fr_NH1_cdf_norm',
 'fr_NH2_cdf_norm',
 'fr_N_O_cdf_norm',
 'fr_Ndealkylation1_cdf_norm',
 'fr_Ndealkylation2_cdf_norm',
 'fr_Nhpyrrole_cdf_norm',
 'fr_SH_cdf_norm',
 'fr_aldehyde_cdf_norm',
 'fr_alkyl_carbamate_cdf_norm',
 'fr_alkyl_halide_cdf_norm',
 'fr_allylic_oxid_cdf_norm',
 'fr_amide_cdf_norm',
 'fr_amidine_cdf_norm',
 'fr_aniline_cdf_norm',
 'fr_aryl_methyl_cdf_norm',
 'fr_azide_cdf_norm',
 'fr_azo_cdf_norm',
 'fr_barbitur_cdf_norm',
 'fr_benzene_cdf_norm',
 'fr_benzodiazepine_cdf_norm',
 'fr_bicyclic_cdf_norm',
 'fr_diazo_cdf_norm',
 'fr_dihydropyridine_cdf_norm',
 'fr_epoxide_cdf_norm',
 'fr_ester_cdf_norm',
 'fr_ether_cdf_norm',
 'fr_furan_cdf_norm',
 'fr_guanido_cdf_norm',
 'fr_halogen_cdf_norm',
 'fr_hdrzine_cdf_norm',
 'fr_hdrzone_cdf_norm',
 'fr_imidazole_cdf_norm',
 'fr_imide_cdf_norm',
 'fr_isocyan_cdf_norm',
 'fr_isothiocyan_cdf_norm',
 'fr_ketone_cdf_norm',
 'fr_ketone_Topliss_cdf_norm',
 'fr_lactam_cdf_norm',
 'fr_lactone_cdf_norm',
 'fr_methoxy_cdf_norm',
 'fr_morpholine_cdf_norm',
 'fr_nitrile_cdf_norm',
 'fr_nitro_cdf_norm',
 'fr_nitro_arom_cdf_norm',
 'fr_nitro_arom_nonortho_cdf_norm',
 'fr_nitroso_cdf_norm',
 'fr_oxazole_cdf_norm',
 'fr_oxime_cdf_norm',
 'fr_para_hydroxylation_cdf_norm',
 'fr_phenol_cdf_norm',
 'fr_phenol_noOrthoHbond_cdf_norm',
 'fr_phos_acid_cdf_norm',
 'fr_phos_ester_cdf_norm',
 'fr_piperdine_cdf_norm',
 'fr_piperzine_cdf_norm',
 'fr_priamide_cdf_norm',
 'fr_prisulfonamd_cdf_norm',
 'fr_pyridine_cdf_norm',
 'fr_quatN_cdf_norm',
 'fr_sulfide_cdf_norm',
 'fr_sulfonamd_cdf_norm',
 'fr_sulfone_cdf_norm',
 'fr_term_acetylene_cdf_norm',
 'fr_tetrazole_cdf_norm',
 'fr_thiazole_cdf_norm',
 'fr_thiocyan_cdf_norm',
 'fr_thiophene_cdf_norm',
 'fr_unbrch_alkane_cdf_norm',
 'fr_urea_cdf_norm',
 'qed_cdf_norm'];

# targets for c-t modelling
targets_po = ["ka_po", "Cl_po", "Vc_po", "Q1_po", "Vp1_po", "Q2_po", "Vp2_po"];
targets_iv = ["CL_iv", "Vc_iv", "Q1_iv", "Vp1_iv", "Q2_iv", "Vp2_iv"];
targets_combined =  targets_po + targets_iv;