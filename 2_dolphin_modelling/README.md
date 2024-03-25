This folder contains the directory used to run the lens modelling software dolphin.

* data :  Contains all the HST images and PSF needed for lens modelling
* log : Contains the modelling logs (empty in this repo)
* output : Contains an example lens modelling output. The complete lens modelling output files can be found at (https://www.projectdinos.com/dinos-i)
* settings : contains all the  modelling settings for all the DINOS-I lenses

The notebook folder contains various notebooks used to run the dolphin modelling pipeline the notebooks are describe below.

* dolphin_example_notebook.ipynb :  provides a simple example to produce and read the output files to make model overview plots
* deepCR_demo.ipynb :  provides a demostration of Cosmic Ray masking with deepCR
* Simple_Lens_Previewer.ipynb :  provides a notebook to quickly look at a HST image with the mask from the settings.