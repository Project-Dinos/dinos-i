# dinos-i 

This repository is used to host data products and notebooks for the analysis in [Tan et. al 2024](https://arxiv.org/abs/2311.09307), which is the first paper of the Project Dinos series whose aims to study elliptical galaxy evolution using strong-lensing galaxies.

The HST_lens_processing folder contains all the notebooks used to download and preprocess the HST images used in the analysis. All the HST data used in the analysis can be found in dolphin_modelling/data.

The dolphin_modelling folder contains the directory used to run the lens modelling software [dolphin](https://github.com/ajshajib/dolphin).  The lens model parameters can be also found in posterior_tables.

The hierarc_analysis folders contains the directory used to run the hierarchical analysis software [hierArc](https://github.com/sibirrer/hierArc/). The posterior of the population-level parameters hierarc can be found in hierarc_analysis/Analysis/all_lenses_chain_const.h5 and hierarc_analysis/Analysis/all_lenses_chain_OM.h5. The additional analysis folder include other miscellaneous notebooks that aid in the analysis.


