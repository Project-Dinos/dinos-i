# Dinos I 

This repository is used to host data products and notebooks for the analysis in [Tan et. al 2024](https://ui.adsabs.harvard.edu/abs/2023arXiv231109307T/abstract), which is the first paper of the Project Dinos series whose aims to study elliptical galaxy evolution using strong-lensing galaxies.

The 1_HST_lens_processing folder contains all the notebooks used to download and preprocess the HST images used in the analysis. All the HST data used in the analysis can be found in 2_dolphin_modelling/data.

The 2_dolphin_modelling folder contains the directory used to run the lens modelling software [dolphin](https://github.com/ajshajib/dolphin).  

The 3_model_parameters folder contains the lens model parameters tables (in the Data subfolder) and additional analysis notebooks used to obtain the lens parameters from the dolphin output files.

The 4_hierarc_analysis folders contains the directory used to run the hierarchical analysis software [hierArc](https://github.com/sibirrer/hierArc/).

All the lens models from this project, including MCMC chains, are released at https://www.projectdinos.com/dinos-i.


