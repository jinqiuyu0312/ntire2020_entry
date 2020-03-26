# ntire2020_entry
This repository contains our entry for the NTIRE 2020 challenge on spectral reconstruction from RGB.
The processed test images by our algoritm are located for both tracks in /model_clean and /model_real.

## Recreating the results
The pretrained networks including the code to run them are provided.
Note that binaries for custom external dependencies were compiled for Windows 10.
Thus, you cannot run the code on MacOS or Linux.

Follow these steps, in order to run the script 'spectral_recovery.py' in your own.

Basic Requirements:
* Windows 10
* Cuda version >= 10 and cuda capable graphics card with at least 4Gb Ram
* A basic understanding of Conda and Python

1. Clone the repo into a location, in the following called root_dir
2. Open the command console
3. Navigate to root_dir: cd root_dir
4. Create the necessary conda environment: conda env create -f env.yml
5. Activate the environment: activate ntire2020
6. Modify the script 'spectral_recovery.py' according to your needs
6. Run the script: python spectral_recovery.py
