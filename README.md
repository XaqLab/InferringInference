# Inferring-TAPBrain
Code for training and inferring the TAP model brain.  

Dependencies: Python 3.5.6 or newer, PyTorch 1.5.1 or newer. 

## Description of contents
### code
This folder contains the main python code: 
- `rnnmodel.py` : RNN model for the TAP brain
- `tapdynamics.py` : functions for generating the TAP dynamics
- `particlefilter.py` : particle filter and Q function required for particle-EM
- `notebookutils.py` : functions for loading a TAP brain, generating model parameters, etc
- `utils.py` : all other required utils

### data/brains
This folder contains the parameters for a few example TAP brains.

### notebooks
- `TrainTAPbrain.ipynb` : example notebook that illustrates the training of a TAP brain model
- `InferTAPbrain-brain-2022-Nr-500.ipynb` : example notebook that illustrates inferring inference in a TAP brain model

### scripts
- `InferTAPbrain.py` : script for inferring inference in a TAP brain model
- `runscripts.sh`: shell script for launching multiple inference jobs in parallel
