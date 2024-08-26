# DeepCt: 
## Predicting pharmacokinetic compartmental models and concentration-time curves from chemical structure using deep learning

This is the underlying code to train the DeepCt models as well as functionality to predict the concentration-time curves from chemical structures using SMILES representations.

**1. Create environment and kernel**

Dependencies are provided in environment_DeepCt.yml. 

A conda environment can be created by running the following line from the command line:

```
conda env create -f environment_DeepCt.yml
```

Activate the environment and create a jupyter kernel:

```
conda activate DeepCt
python -m ipykernel install --user --name=DeepCt
```

**2. Model training**

Now you are ready to train a model using the jupyter notebook [`train.ipynb`](train.ipynb).

By default, the model generates some dummy data to test if everything is running. Please, follow the descriptions in the notebook to setup your data accordingly when you apply the model to your own datasets. Create a ```tmp_folder``` before executing the notebook.

**3. Model testing**

The model can be tested using the jupyter notebook [`test.ipynb`](test.ipynb). In the notebook you can also see how the model can be called and predictions are be made using SMILES representations of chemical structures.

