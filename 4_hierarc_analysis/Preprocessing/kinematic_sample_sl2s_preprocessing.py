from hierarc.LensPosterior.kin_constraints import KinConstraints
from hierarc.Util import ifu_util
from lenstronomy.Util import constants as const
from astropy.io import fits
import numpy as np
import pickle
import os
import csv
import matplotlib.pyplot as plt
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from lenstronomy.Util import param_util
import lenstronomy.Util.multi_gauss_expansion as mge

dir_path = os.getcwd()


# BOSS spectra observational conditions, https://arxiv.org/pdf/0805.1931.pdf Section 6.3


# numerical settings (not needed if power-law profiles with Hernquist light distribution is computed)
kwargs_numerics_galkin = {'interpol_grid_num': 1000,  # numerical interpolation, should converge -> infinity
                          'log_integration': True,  # log or linear interpolation of surface brightness and mass models
                          'max_integrate': 100, 'min_integrate': 0.001}  # lower/upper bound of numerical integrals

num_sample_model_slit = 100  # number of draws from the model to generate a kinematic prediction for the slit data

# anisotropy model
anisotropy_model = 'OM' # 'OM', 'GOM' 

# population distribution in the power-law slope
# this distribution is assumed on lenses within the sample that do not have a reliably measured slope from imaging data

#gamma_mean_pop, gamma_error_pop = 2.10, 0.16


# file containing all the information about the sample including lens model parameters from imaging data
file_name_param_catalogue = 'SL2S_all_params.csv'
file_name_param = os.path.join(dir_path, file_name_param_catalogue)
print(file_name_param)


def process_sample_slit(file_name):
    """
    SDSS single slit (or fiber) measurement
    """
    posterior_list = []

    with open(file_name, newline='') as myFile:
        reader = csv.DictReader(myFile)
        for row in reader:
            if np.float(row['z_lens'])>0 and np.float(row['z_source'])>0 and np.float(row['sigma_v'])>0:
                name = row['name']
                print(name)
                r_eff = float(row['r_eff'])
                r_eff_error = float(row['r_eff_error'])
                gamma, gamma_error = float(row['gamma']), float(row['gamma_error'])
                kwargs = {'z_lens': float(row['z_lens']), 'z_source': float(row['z_source']),
                         'r_eff': r_eff, 'r_eff_error': r_eff_error,
                         'theta_E': float(row['theta_E']), 'theta_E_error': float(row['theta_E_error']),
                         'gamma': gamma, 'gamma_error': gamma_error}
                
                # MGE light models if available
                light_model = str(row['light_model'])
                if light_model == 'double_sersic':
                    print("Using Double sersic")
                    r_sersic_1, n_sersic_1, amp_sersic_1, r_sersic_2, n_sersic_2, amp_sersic_2 = float(row['r_sersic_1']), float(row['n_sersic_1']), float(row['amp_sersic_1']), float(row['r_sersic_2']), float(row['n_sersic_2']), float(row['amp_sersic_2'])
                    kwargs_lens_light = [{'R_sersic': r_sersic_1, 'amp': amp_sersic_1, 'n_sersic': n_sersic_1, 'center_x': 0, 'center_y': 0}, 
                                         {'R_sersic': r_sersic_2, 'amp': amp_sersic_2, 'n_sersic': n_sersic_2,'center_x': 0, 'center_y': 0}]
                    lens_light_model_list=['SERSIC', 'SERSIC']
                    MGE_light = True
                    kwargs_mge_light = {'grid_spacing': 0.01, 'grid_num': 100, 'n_comp': 20, 'center_x': None, 'center_y': None}
                    print('MGE is active for %s' % name)
                elif light_model == 'single_sersic':
                    print("Using Single sersic")
                    r_sersic_1, n_sersic_1, amp_sersic_1 = float(row['r_eff']), float(4.0), float(row['amp_sersic_1'])
                    kwargs_lens_light = [{'R_sersic': r_sersic_1, 'amp': amp_sersic_1, 'n_sersic': n_sersic_1}]
                    lens_light_model_list=['SERSIC']
                    MGE_light = True
                    kwargs_mge_light = {'grid_spacing': 0.01, 'grid_num': 100, 'n_comp': 20, 'center_x': None, 'center_y': None}
                    print('MGE is active for %s' % name)

                sigma_v = float(row['sigma_v'])
                sigma_v_error_independent = float(row['sigma_v_error'])
                sigma_v = [sigma_v]
                sigma_v_error_independent = [sigma_v_error_independent]
                
                print('Using slit rectangular info, slit: ',(row['slit']))
                
                
                kwargs_aperture_slit = {'aperture_type': 'slit',
                                        'length': np.float(row['slit']), 'width': np.float(row['width']),
                                        'center_ra':0, 'center_dec':0, 'angle':0}
                kwargs_seeing_slit   = {'psf_type': 'GAUSSIAN', 'fwhm': np.float(row['seeing'])}

                kin_constraints = KinConstraints(sigma_v_measured=sigma_v, 
                                 sigma_v_error_independent=sigma_v_error_independent, 
                                 sigma_v_error_covariant=0, 
                                 kwargs_aperture=kwargs_aperture_slit, kwargs_seeing=kwargs_seeing_slit, 
                                 kwargs_numerics_galkin=kwargs_numerics_galkin, 
                                 anisotropy_model=anisotropy_model, kwargs_lens_light=kwargs_lens_light, 
                                 lens_light_model_list=lens_light_model_list, MGE_light=MGE_light,
                                 kwargs_mge_light=kwargs_mge_light, num_kin_sampling=2000, num_psf_sampling=100, **kwargs)

                kwargs_posterior = kin_constraints.hierarchy_configuration(num_sample_model=num_sample_model_slit)
                kwargs_posterior['name'] = name
                posterior_list.append(kwargs_posterior)
                print(kwargs_posterior)
    return posterior_list

if anisotropy_model == 'OM':
    file_name_slit = 'sl2s_slit_om_processed.pkl'
    file_name_ifu = 'sl2s_ifu_om_processed.pkl'
if anisotropy_model == 'GOM':
    file_name_slit = 'sl2s_slit_gom_processed.pkl'
    file_name_ifu = 'sl2s_ifu_gom_processed.pkl'



posterior_list_slit = process_sample_slit(file_name_param)

file = open(file_name_slit, 'wb')
pickle.dump(posterior_list_slit, file)
file.close()

  