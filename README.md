# ntire2020_entry
This repository contains our entry for the NTIRE 2020 challenge on spectral reconstruction from RGB.
The processed test images are located for both tracks in /out_clean and /out_real.

## Recreating the results
The pretrained networks including the code to run them are provided.
Note that binaries for custom external dependencies were compiled for Windows 10.
Thus, you cannot run this code on MacOS or Linux.
However, you can easily load our pretrained networks (the .tar files) into your own framework, as they are stored using standard Pytorch serialization.

### Basic Requirements:
* Windows 10
* Cuda capable graphics card with at least 4Gb Ram (tested for cuda version>=10)
* A basic understanding of Conda and Python

### Step-by-Step Guide
1. Clone the repo into a location, in the following called root_dir and modify the script 'spectral_recovery.py' according to your needs
2. Open the command console
3. Navigate to root_dir: 
```
cd root_dir
```
4. Create the necessary conda environment: 
```
conda env create -f env.yml
```
5. Activate the environment: 
```
activate ntire2020
```
6. Run the script: 
```
python spectral_recovery.py
```
