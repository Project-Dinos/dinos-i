import numpy as np
import os
import h5py
import yaml

import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.util as util
from astroObjectAnalyser.DataAnalysis.analysis import Analysis
from astroObjectAnalyser.astro_object_superclass import StrongLensSystem
from lenstronomy.Workflow.fitting_sequence import FittingSequence
#from lenstronomy.Plots.output_plots import ModelPlot
#import lenstronomy.Plots.output_plots as out_plot
from lenstronomy.Plots.model_plot import ModelPlot

from lenstronomy.Util import param_util


from photutils import detect_threshold
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from photutils import detect_sources
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import deblend_sources
import numpy as np
import astropy.units as u
from photutils import source_properties, EllipticalAperture
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize


import matplotlib
matplotlib.use('Agg')

import warnings
warnings.filterwarnings('ignore')
import cv2
import copy
import h5py

import paperfig as pf
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import coloripy as cp
from dolphin.processor.config import ModelConfig

import os 
import sys

survey = sys.argv[2]
if survey == 'SLACS':
    os.chdir('/home/s1/chinyi/80bdata/dinos/midway_backup/Gold_run/SLACS/notebooks')
elif survey == 'SL2S':
    os.chdir('/home/s1/chinyi/80bdata/dinos/midway_backup/Gold_run/SL2S/notebooks')
elif survey == 'BELLS':
    os.chdir('/home/s1/chinyi/80bdata/dinos/midway_backup/Gold_run/BELLS/notebooks')
else:
    raise('Invalid Survey')


from dolphin.processor import Processor

dir_path = '../'
processor = Processor(dir_path)
#processor.swim('SDSSJ0330-0020', model_id='dolph_test', log=False)

from dolphin.analysis.output import Output

output = Output(dir_path)


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def band_select(lens,survey):
    a_yaml_file = open("../settings/{}_config.yml".format(lens))
    settings = yaml.load(a_yaml_file, yaml.FullLoader)
    
    reading_band = []
    if 'F606W' in settings['band']:
        print('F606W')
        return settings['band'].index('F606W')
    elif 'F600LP' in settings['band']:
        return settings['band'].index('F600LP')
        print('F600LP')
    elif 'F555W' in settings['band']:
        return settings['band'].index('F555W')
        print('F555W')
    elif  survey=='BELLS':
        return settings['band'].index('F814W')
        print('F555W')      
    else:
        raise "No Visible band available !"

def get_mask(large_image, bkg_rms, lens_name,mask_index = 0,kernel_size=11,mask_radius=None):
    #threshold = detect_threshold(data, nsigma=2.)

    threshold = (3.0 * bkg_rms)  

    sigma = 6.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    segm = detect_sources(large_image, threshold, npixels=5, filter_kernel=kernel)


    #norm = ImageNormalize(stretch=SqrtStretch())
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    #ax1.imshow(large_image, origin='lower', cmap='Greys_r', norm=norm)
    #ax1.set_title('Data')
    cmap = segm.make_cmap(random_state=12345)
    #ax2.imshow(segm, origin='lower', cmap=cmap)
    #ax2.set_title('Segmentation Image')


    segm_deblend = deblend_sources(large_image, segm, npixels=9,
                                   filter_kernel=kernel, nlevels=32,
                                   contrast=0.01)


    cat = source_properties(large_image, segm_deblend)
    r = 3.  # approximate isophotal extent
    apertures = []
    for obj in cat:
        position = np.transpose((obj.xcentroid.value, obj.ycentroid.value))
        a = obj.semimajor_axis_sigma.value * r
        b = obj.semiminor_axis_sigma.value * r
        theta = obj.orientation.to(u.rad).value
        apertures.append(EllipticalAperture(position, a, b, theta=theta))


    #norm = ImageNormalize(stretch=SqrtStretch())
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12.5))
    #ax1.imshow(large_image, origin='lower', cmap='Greys_r', norm=norm)
    #ax1.set_title('Data')
    #cmap = segm_deblend.make_cmap(random_state=12345)
    #ax2.imshow(segm_deblend, origin='lower', cmap=cmap)
    #ax2.set_title('Segmentation Image')
    #for aperture in apertures:
    #    aperture.plot(axes=ax1, color='white', lw=1.5)
    #    aperture.plot(axes=ax2, color='white', lw=1.5)

    a = np.array(segm_deblend)

    s = set(a.flatten())
    area = []

    for num in s:
        if num == 0:
            continue
        counter = np.zeros_like(a)
        counter[a == num] = 1
        area.append(np.sum(counter))

    sorted_area = sorted(area,reverse=True)   
    mask_arg = area.index(sorted_area[mask_index])
        
  #  central_id = list(s)[np.argsort(np.max(area, axis=0))[-2]]+1
    central_id = list(s)[mask_arg]+1
    print(central_id)

    a[a == central_id] = 0

    #plt.matshow(a, origin='lower')
    #plt.title('initial segmented mask')
    #plt.show()

    kernel = np.ones((kernel_size, kernel_size),np.uint8)

    dilation = cv2.dilate(a.astype(np.uint8), kernel, iterations=1)

    #plt.matshow(dilation, origin='lower')
    #plt.title('dilated mask')
    #plt.show()
    
    large_mask = np.zeros_like(dilation)
    large_mask[dilation == 0] = 1
    
    if mask_radius==None:
        mask1 = create_circular_mask(len(large_image),len(large_image),radius=2.0*np.sqrt(sorted_area[mask_index]))
    else:
        mask1 = create_circular_mask(len(large_image),len(large_image),radius=2.0*mask_radius)    

    plt.matshow(np.log10(large_image*large_mask*mask1), origin='lower')
    if survey=='SLACS':
        plt.savefig("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/SLACS/single_sersic/plots/{}_mask.png".format(lens_name))
    elif survey=='SL2S':
        plt.savefig("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/SL2S/single_sersic/plots/{}_mask.png".format(lens_name))
    elif survey=='BELLS':
        plt.savefig("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/BELLS/single_sersic/plots/{}_mask.png".format(lens_name))
    else:
        raise("Bad Survey")
    plt.show()
    
    return large_mask*mask1

from scipy.special import gamma

def get_total_flux(kwargs):
    n_sersic = kwargs['n_sersic']
    e1 = kwargs['e1']
    e2 = kwargs['e2']
    amp = kwargs['amp']
    r_sersic = kwargs['R_sersic']
    
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    
    b_n = 1.9992 * n_sersic - 0.3271
    flux = q * 2*np.pi * n_sersic * amp * r_sersic**2 * np.exp(b_n) * b_n**(-2*n_sersic) * gamma(2*n_sersic)
    
    return flux


from scipy.special import gamma, gammainc
from scipy.optimize import brentq

def _flux(r, amp, n, r_s):
    bn = 1.9992 * n - 0.3271
    x = bn * (r/r_s)**(1./n)
    
    return amp * r_s**2 * 2 * np.pi * n * np.exp(bn) / bn**(2*n) * gammainc(2*n, x) * gamma(2*n)

def _total_flux(amp, n, r_s):
    bn = 1.9992 * n - 0.3271

    return amp * r_s**2 * 2 * np.pi * n * np.exp(bn) / bn**(2*n) * gamma(2*n)

def get_half_light_radius(kwargs_light):
    tot_flux = _total_flux(kwargs_light[0]['amp'], kwargs_light[0]['n_sersic'], 
                                  kwargs_light[0]['R_sersic']) + \
                        _total_flux(kwargs_light[1]['amp'], kwargs_light[1]['n_sersic'], 
                                  kwargs_light[1]['R_sersic'])
    def func(r):
        return _flux(r, kwargs_light[0]['amp'], kwargs_light[0]['n_sersic'], 
                                  kwargs_light[0]['R_sersic']) + \
                        _flux(r, kwargs_light[1]['amp'], kwargs_light[1]['n_sersic'], 
                                  kwargs_light[1]['R_sersic']) - tot_flux/2.
    
    return brentq(func, 0.01, 10) #min(kwargs_light[0]['R_sersic'], kwargs_light[1]['R_sersic']),
                  #max(kwargs_light[0]['R_sersic'], kwargs_light[1]['R_sersic'])
                 #)


def fit_sersic(lens_name, kwargs_data, kwargs_psf, mask, ra_offset, dec_offset, bound=0.5):
    """
    """
    #lens_light_model_list = ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']
    lens_light_model_list = ['SERSIC_ELLIPSE']

    kwargs_model = {'lens_light_model_list': lens_light_model_list,
                    #'joint_len'
                   }
    kwargs_constraints = {}
   # kwargs_constraints = {'joint_lens_light_with_lens_light': [[0, 1, ['center_x', 'center_y', 'e1', 'e2']]]}
    kwargs_numerics_galfit = {'supersampling_factor': 1}
    kwargs_likelihood = {'check_bounds': True, 'image_likelihood_mask_list': [mask], 
                         'check_positive_flux': True,
                        }

    image_band = [kwargs_data, kwargs_psf, kwargs_numerics_galfit]
    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

    # lens light model choices
    fixed_lens_light = []
    kwargs_lens_light_init = []
    kwargs_lens_light_sigma = []
    kwargs_lower_lens_light = []
    kwargs_upper_lens_light = []

    # first Sersic component
    fixed_lens_light.append({'n_sersic': 4.})
    kwargs_lens_light_init.append({'R_sersic': .1, 'n_sersic': 4, 'e1': 0, 'e2': 0, 'center_x': ra_offset, 'center_y': dec_offset})
    kwargs_lens_light_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.5, 'center_y': 0.5})
    kwargs_lower_lens_light.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.5, 'center_x': ra_offset-bound, 'center_y': dec_offset-bound})
    kwargs_upper_lens_light.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 8, 'center_x': ra_offset+bound, 'center_y': dec_offset+bound})

  #  # second Sersic component
  #  fixed_lens_light.append({'n_sersic': 1.})
  #  kwargs_lens_light_init.append({'R_sersic': .5, 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0})
  #  kwargs_lens_light_sigma.append({'n_sersic': 0.5, 'R_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
  #  kwargs_lower_lens_light.append({'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.5, 'center_x': -10, 'center_y': -10})
  #  kwargs_upper_lens_light.append({'e1': 0.5, 'e2': 0.5, 'R_sersic': 20, 'n_sersic': 8, 'center_x': 10, 'center_y': 10})

    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]

    kwargs_params = {'lens_light_model': lens_light_params}

    fitting_seq = FittingSequence(kwargs_data_joint, 
                                  kwargs_model, kwargs_constraints, 
                                  kwargs_likelihood, kwargs_params)

    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 150, 'n_iterations': 200}],
                          ['MCMC', {'n_burn': 0, 'n_run': 500, 'sigma_scale': .01, 'threadCount': 1,'walkerRatio':16}]]
   # fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 10, 'n_iterations': 10}]] 
    #fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 2, 'n_iterations': 2}]]

    chain_list = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    
    lens_result = kwargs_result['kwargs_lens']
    lens_light_result = kwargs_result['kwargs_lens_light']
    source_result = kwargs_result['kwargs_source']
    ps_result = kwargs_result['kwargs_ps']
    
#     lensPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string=cmap,
#                              likelihood_mask_list=[mask], multi_band_type='multi-linear',
#                              bands_compute=None)
    
    cmap = pf.cmap
    lensPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result,
                        arrow_size=0.02, cmap_string=cmap,
                        likelihood_mask_list=[mask], #kwargs_likelihood['image_likelihood_mask_list'],
                        multi_band_type='multi-linear'
                        )
    

    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)

    lensPlot.data_plot(ax=axes[0], band_index=0, cmap=cmap)
    lensPlot.model_plot(ax=axes[1], band_index=0, cmap=cmap)
    lensPlot.normalized_residual_plot(ax=axes[2], v_min=-6, v_max=6, cmap=msh_cmap2)
    f.tight_layout()
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    logL = round(lensPlot._logL_list[0],2)
    plt.title("chi2 : {:.2f} ".format(logL))
    if survey=='SLACS':
        plt.savefig("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/SLACS/single_sersic/plots/{}_model_{:.2f}.png".format(lens_name,logL))
    elif survey=='SL2S':
        plt.savefig("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/SL2S/single_sersic/plots/{}_model_{:.2f}.png".format(lens_name,logL))
    elif survey=='BELLS':
        plt.savefig("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/BELLS/single_sersic/plots/{}_model_{:.2f}.png".format(lens_name,logL))
    else:
        raise "Invalid Survey"
    plt.show()

    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)

    lensPlot.decomposition_plot(ax=axes[0], text='Lens light', lens_light_add=True, unconvolved=True, cmap=cmap)
    lensPlot.decomposition_plot(ax=axes[1], text='Lens light convolved', lens_light_add=True, cmap=cmap)
    lensPlot.subtract_from_data_plot(ax=axes[2], text='Data - Lens Light', lens_light_add=True)
    f.tight_layout()
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    plt.show()
    
#     pert_y = system.ra + lens_light_result[0]['center_x']/3600
#     pert_x = system.dec + lens_light_result[0]['center_y']/3600/np.cos(pert_y*np.pi/3600/180)
    flux1 = get_total_flux(lens_light_result[0])
  #  flux2 = get_total_flux(lens_light_result[1])
    
    
    
    #print('lens light result: ', lens_light_result)
    #print('total flux: ', flux)
    
    #n_sersic = kwargs_result['kwargs_lens_light'][0]['n_sersic']
    #r_sersic = kwargs_result['kwargs_lens_light'][0]['R_sersic']
    
   
    
    return kwargs_result['kwargs_lens_light'], chain_list

# n_sersics = []
# R_sersics = []

bd_ratios = []
R_effs = []
results = []


R_eff_df = {}

msh_cmap2 = cp.get_msh_cmap(rgb1=np.array([0.085, 0.532, 0.201])*256, rgb2=np.array([0.436, 0.308, 0.631])*256, 
                            ref_point=(160, 160, 160), 
                            num_bins=501, rescale='power', power=1.5)

nrow = 1
ncol = 6

pf.set_fontscale(1.6)
fontsize = pf.mnras_figcaption_fontsize*1.6

# fig, axes = plt.subplots(nrow, 4, 
#                          figsize=(pf.get_fig_size(width=pf.mnras_textwidth*2, height_ratio=0.2*nrow)), 
#                          sharex=False, sharey=False, constrained_layout=False)


def Sersic_Fitter(lens_name,mask_index=0,kernel_size=10,mask_radius=None):
        
    fig = plt.figure(figsize=(pf.get_fig_size(width=pf.mnras_textwidth*2, height_ratio=1/(ncol+1)*nrow)))
    gs = GridSpec(nrow, ncol, left=0, right=.95, hspace=0.1, wspace=0.)

    axes = []

    for n in range(nrow):
        axs = []
        for i in range(ncol):
            axs.append(fig.add_subplot(gs[n:n+1, i:i+1]))
        axes.append(axs)

    counter = 0

    first = True
    name = lens_name
    
    print('#######################################')
    print(name)
    print('#######################################')
    

    
    model_id = 'Gold'
    

        
    if True:
        output = Output(dir_path)
        _ = output.load_output(name, model_id)

        walker_ratio = 16

        num_walker = int(output.num_params_mcmc * walker_ratio)
        
        median_args = np.median(output.samples_mcmc[-num_walker:, :], axis=0)
        
        kwargs_median = output.get_kwargs_from_args(name, model_id, median_args)
        
        model_plot, v_max = output.get_model_plot(lens_name, model_id=model_id, kwargs_result=kwargs_median)
        
        
        band_index = band_select(lens_name,survey)
        
        v_min = v_max - 2.5
        
        config = ModelConfig(settings=output.model_settings)
        mask = config.get_masks()[band_index]
        
        model_plot.data_plot(ax=axes[counter][0], band_index=band_index, v_max=v_max, v_min=v_min, 
                             text=r'{}${}${}'.format(lens_name[4:9], lens_name[9:10], lens_name[10:]), font_size=fontsize)
        
        if first:
            text = 'Reconstructed'
        else:
            text = ''
        model_plot.model_plot(ax=axes[counter][1], band_index=band_index, v_max=v_max, v_min=v_min,
                              font_size=fontsize, no_arrow=True, no_scale=True, text=text)
        
        if first:
            text = 'Residual'
        else:
            text = ''
        model_plot.normalized_residual_plot(ax=axes[counter][2],
                                           band_index=band_index, text=text,
                                           cmap='RdBu_r', v_max=4, no_arrow=True,
                                           v_min=-4, font_size=fontsize)
        #model_plot.convergence_plot(ax=axes[1, 1], band_index=band_index,
        #                           cmap=convergence_cmap)
        
        source, _ = model_plot.source(deltaPix=0.02, numPix=100)
        
        source_max = np.log10(np.max(source)) + 0.1
        source_min = source_max - 1.1
        
        if first:
            text = r'Data $-$ deflector'
        else:
            text = ''
        subtracted = model_plot.subtract_from_data_plot(ax=axes[counter][3], text=text, font_size=fontsize,
                                                        lens_light_add=False, source_add=True,
                                                        band_index=band_index,v_max=v_max,
                                                        v_min=v_min-0.5,get_image=True)
        
        data = model_plot.subtract_from_data_plot(ax=axes[counter][3], text=text, font_size=fontsize,
                                                  lens_light_add=False, band_index=band_index,
                                                  source_add=False, v_max=v_max, 
                                                  v_min=v_min-0.5,get_image=True)
        
        
        if first:
            text = 'Source'
        else:
            text = ''
            
        model_plot.source_plot(ax=axes[counter][4], deltaPix_source=0.02, numPix=100, 
                               font_size=fontsize, text=text,
                               band_index=band_index, v_max=source_max, v_min=source_min, scale_size=0.5)
        
        if first:
            text = 'Magnification'
        else:
            text = ''
        model_plot.magnification_plot(ax=axes[counter][5], font_size=fontsize,
                                     band_index=band_index, text=text,
                                     cmap=pf.msh_cmap, no_arrow=True)
        
        #plt.show()
        plt.close() #Dont show plots (see above)

        
        lensed_source = data - subtracted          
        plt.matshow(np.log10(lensed_source))
        #plt.show()
        plt.close() #Dont show plots (see above)

      
        
     #   try
        with open("../settings/{}_config.yml".format(name)) as f:
            settings = yaml.load(f, yaml.FullLoader)
        filt = settings['band'][band_index]  

        
        data_filename = '../large_cutout/{}/image_{}_{}.h5'.format(name, name, filt)
        local_data_filename = data_filename #os.path.join(base_path, 'data', data_filename)
        f = h5py.File(local_data_filename, "r")
     #   except:
     #       filt = 'F555W'
     #       data_filename = 'dolph_dir/large_cutouts/{}/image_{}_{}.h5'.format(name, name, filt)
     #       local_data_filename = data_filename #os.path.join(base_path, 'data', data_filename)
     #       f = h5py.File(local_data_filename, "r")
            
        dset = f['image_data'][()]
        bkg_rms = f['background_rms'][()]
        wht_map = f['exposure_time'][()]
        ra_at_xy_0 = f['ra_at_xy_0'][()]
        dec_at_xy_0 = f['dec_at_xy_0'][()]
        Matrix_inv = f['transform_pix2angle'][()]
        f.close()

        kwargs_data = {'image_data': dset, 
                       'background_rms': bkg_rms,
                       'noise_map': None,
                       'exposure_time': wht_map,
                       'ra_at_xy_0': ra_at_xy_0,
                       'dec_at_xy_0': dec_at_xy_0, 
                       'transform_pix2angle': Matrix_inv
                       }

        data_filename = '../data/{}/psf_{}_{}.h5'.format(name, name, filt)
        local_data_filename = data_filename #os.path.join(base_path, 'data', data_filename)

        f = h5py.File(local_data_filename, "r")

        kernel_point_source = f['kernel_point_source'][()]
        kernel_point_source_init = f['kernel_point_source'][()]

        kwargs_psf = {'psf_type': "PIXEL", 
                       'kernel_point_source': kernel_point_source ,
                       'kernel_point_source_init': kernel_point_source_init,
                       'psf_error_map': None
                     }
        f.close()

        #plt.matshow(np.log10(dset), origin='lower')
        #plt.show()
        
        result = cv2.matchTemplate(data.astype(np.float32), dset.astype(np.float32), cv2.TM_CCOEFF_NORMED)
        match = np.unravel_index(result.argmax(),result.shape)

        large_image = copy.deepcopy(dset)
        large_image[match[0]:match[0]+len(lensed_source), match[1]:match[1]+len(lensed_source)] -= data

        #plt.matshow(np.log10(np.abs(large_image)), origin='lower')
        #plt.title('check subimage placement')
        #plt.show()

        large_image = copy.deepcopy(dset)
        large_image[match[0]:match[0]+len(lensed_source), match[1]:match[1]+len(lensed_source)] -= lensed_source*mask

        #plt.matshow(np.log10(np.abs(large_image)), origin='lower')
        #plt.title('source subtracted large image')
        #plt.show()
        
        large_mask = get_mask(large_image, bkg_rms,lens_name, mask_index = mask_index , kernel_size=kernel_size, mask_radius=mask_radius)
        
        kwargs_data['image_data'] = large_image
        
        
        kwargs_result, chain_list = fit_sersic(lens_name,kwargs_data, kwargs_psf, large_mask, 0, 0, 1.)
        if survey =='SLACS':
            np.save("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/SLACS/single_sersic/models/{}_results".format(lens_name),kwargs_result)
            np.save("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/SLACS/single_sersic/models/{}_chains".format(lens_name),chain_list)
        elif survey =='SL2S':
            np.save("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/SL2S/single_sersic/models/{}_results".format(lens_name),kwargs_result)
            np.save("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/SL2S/single_sersic/models/{}_chains".format(lens_name),chain_list)
        elif survey =='BELLS':
            np.save("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/BELLS/single_sersic/models/{}_results".format(lens_name),kwargs_result)
            np.save("/home/s1/chinyi/80bdata/dinos/Sersic_Fitter/v2/BELLS/single_sersic/models/{}_chains".format(lens_name),chain_list)
        else :
            raise "Invalid Survey"
        
     #   bd_ratios.append(bd_ratio)
     #   R_effs.append(R_eff)
     #   results.append(kwargs_result)
     #   R_eff_df[lens_name]=R_eff
     #   print('Reff',R_eff)
        print("\n\n")
        
        
        return None
def Sersic_Wrapper(lens_name):  
    if lens_name == 'SL2SJ0226-0420':
        Sersic_Fitter('SL2SJ0226-0420',kernel_size=9)
    elif lens_name == 'SL2SJ0217-0513':
        Sersic_Fitter('SL2SJ0217-0513',kernel_size=1)  
    elif lens_name == 'SL2SJ0858-0143':
        Sersic_Fitter('SL2SJ0858-0143',kernel_size=2)  
    elif lens_name == 'SL2SJ1359+5535':
        Sersic_Fitter('SL2SJ1359+5535',mask_radius=30)  
    elif lens_name == 'SL2SJ0901-0259':
        Sersic_Fitter('SL2SJ0901-0259',mask_index =3,kernel_size=4)
    elif lens_name == 'SL2SJ0904-0059':
        Sersic_Fitter('SL2SJ0904-0059',mask_index =1,kernel_size=20)
    elif lens_name == 'SL2SJ1359+5535':
        Sersic_Fitter('SL2SJ1359+5535',mask_index =0)
    elif lens_name == 'SL2SJ1406+5226':
        Sersic_Fitter('SL2SJ1406+5226',mask_index =2,kernel_size=2,mask_radius=30)
    elif lens_name == 'SL2SJ2214-1807':
        Sersic_Fitter('SL2SJ2214-1807',mask_index=1)
    else:
        Sersic_Fitter(lens_name)
Sersic_Wrapper(sys.argv[1])
#'SDSSJ1112+0826')




