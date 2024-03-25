This folder contains lens model parameters (Found in the Data Folder) the notebooks to obtain lens model parameters from the output files. A more accessible table of the lens model parameters can be found in https://www.projectdinos.com/dinos-i . The notebooks in the directory contains includes:

* extract_mass_params_from_chains.ipynb : Extract lens mass parameters from the dolphin output file (see ../2_dolphin_modelling/)
* extract_light_params_from_chains.ipynb : Extract lens light parameters from lens light fits (see the sersic_light_fits/ )
* make_corner_plot.ipynb : Produce corner plot with lens model parameters
* calculate_gamma_error_systematic.ipynb: Calculate the  modelling systematic uncertainty and plot Figure 7 of the dinos-i paper
* calculate_mass_sheet_trans.ipynb: Produce Figure 11 of the dinos-i paper, which shows the effect of the MST on the mass density profile and mass density slope of lens system


