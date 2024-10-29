import DH_array as dharray
import DH_path as dhpath
import DH_MC2fit_sub as dhmc2fitsub
import DH_multicore as mulcore

import numpy as np
import os, glob, shutil
import random
import time
import pickle
import copy
from astropy.io import fits
from dataclasses import dataclass, field
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd
from scipy import stats

from astropy.modeling.functional_models import Sersic1D, Sersic2D
from astropy.convolution import Gaussian2DKernel
from astropy.utils.data import download_file, get_pkg_data_filename
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval,ImageNormalize,AsinhStretch
from scipy.special import gamma, gammainc, gammaincinv ## Sersic post processing

import subprocess
from multiprocessing import Pool
import warnings

@dataclass
class MC2fitSetting:
    """
    Setting for MC2fit

    ## Path
        proj_folder:    str = 'galfit_g_base/' ## Galfit results will be saved here
        fn_galfit:      str = dhpath.fn_galfit() ## Galfit program path
        fni_image:      str = 'image_g.fits' ## input image
        fni_psf:        str = 'psf_g.fits' ## PSF image
        fni_masking:    str = 'masking_c_d.fits' ## combined masking file
        fni_sigma:      str = 'sigma_g.fits' ## No sigma image
        is_run_galfit_dir_work: bool = False ## If true, run GALFIT in each working directory. Use True if you want to extract sigma.fits

    ## Info
        band:           str = 'g'
        est_sky:        float = 0 ## estimated sky value [ADU]
        zeromag:        float = 22.5 ## Photometric zeropoint MAG=-2.5*log(data)+zeromag
        plate_scale:    float = 0.2 ## Input float or array (dx, dy). [arcsec / pix]
        image_size:     int = 500 ## Input int or array (x, y). [pixel]

    ## Run option
        Ncore:          int = 16 ## Multi-core processing
        fast_skip:      bool = False ## Skip running GALFIT if result files already exist
        overwrite:      bool = True  ## Overwrite the result
        remove_galfit_interval: float = 30 ## Remove GALFIT log files when the number of files reaches the given value.
        debug:          bool = False ## Print internal messages, use debug mode
        use_try:        bool = True ## Skip all error message
        extract_sigma:  bool = False ## Extract internal Sigma image from GALFIT (sigma.fits)

    ## Print Output
        show_progress:  float = -1 ## Interval for printing the progress. -1 to hide
        silent:         bool = True ## Print internal messages

    """
    ## Path
    proj_folder:    str = 'galfit_g_base/' ## Galfit results will be saved here
    fn_galfit:      str = dhpath.fn_galfit() ## Galfit program path
    fni_image:      str = 'image_g.fits' ## input image
    fni_psf:        str = 'psf_g.fits' ## PSF image
    fni_masking:    str = 'masking_c_d.fits' ## combined masking file
    fni_sigma:      str = 'sigma_g.fits' ## No sigma image
    is_run_galfit_dir_work: bool = False ## If true, run GALFIT in each working directory. Use True if you want to extract sigma.fits

    ## Info
    band:           str = 'g'
    est_sky:        float = 0 ## estimated sky value [ADU]
    zeromag:        float = 22.5 ## Photometric zeropoint MAG=-2.5*log(data)+zeromag
    plate_scale:    float = 0.2 ## Input float or array (dx, dy). [arcsec / pix]
    image_size:     int = 500 ## Input int or array (x, y). [pixel]

    ## Run option
    Ncore:          int = 16 ## Multi-core processing
    fast_skip:      bool = False ## Skip running GALFIT if result files already exist
    overwrite:      bool = True  ## Overwrite the result
    remove_galfit_interval: float = 30 ## Remove GALFIT log files when the number of files reaches the given value.
    debug:          bool = False ## Print internal messages, use debug mode
    use_try:        bool = True ## Skip all error message
    extract_sigma:  bool = False ## Extract internal Sigma image from GALFIT (sigma.fits). Set is_run_galfit_dir_work as True
    output_block:   bool = True ## Extract the output file. If False, it does not run Galfit

    ## Print Output
    show_progress:  float = -1 ## Interval for printing the progress. -1 to hide
    silent:         bool = True ## Print internal messages

    ## Chi2 calculation
    tolerance:      float = 1e-3 # Chi2 tolerance for "Good results" (|Min chi2 - chi2 |/(Min chi2) < tolerance)
    comp_crit:      str   ='RChi_50' ## Comparing value between results

#     np.ndarray = field(default_factory=lambda:
#                                   np.array(['sersic2', 'sersic2'])) # list of components (Ncomp = len(components))


    def __post_init__(self):
        self.plate_scale=dharray.value_repeat_array(self.plate_scale, 2)
        self.image_size=dharray.value_repeat_array(self.image_size, 2)
        if(self.proj_folder[-1]!='/'): self.proj_folder=self.proj_folder+"/" ## For directory format
        print("Setting completed")

class MC2fitRunlist():
    """
    Descr - Runlist
    INPUT
     * complist : array_like (Default = np.array(['sersic2', 'sersic2']))
        List of components (Ncomp = len(components))
     * namelist : array_like (Default=np.array(['ss_23', 'ss_24', 'ss_25', 'ss_26']))
        Namelist (Nset = len(namelist))
     * est_params_array: Ndarray - object (Ncomp * 7)
        (Default: np.array([[np.nan, np.nan, np.array([23, 24, 25, 26]), 30, 1, 1, 0],
        [np.nan, np.nan, np.array([23, 24, 25, 26]), 10, 4, 1, 0]],
        dtype=object) # See note)
        ** The list of estimated parameters for compoments.
                        These values are initial guesses for Galfit.
                        The single value in the list will be copied.
                        For values of nan, they will be converted to default values.
                  index    [  0   |  1   | 2 |  3 | 4 |5 |6 ]
                  params   [ xpos | ypos |mag|reff| n |ar|pa]
                  default  [center|center|20 | 20 | 4 |1 |0 ]
         ** for values of np.nan --> It will be converted into default values (See class AutoGalfit)
     * lim_pos, lim_mag, lim_reff, lim_n, lim_ar, lim_pa: Ndarray (Ncomp * 2)  or list (,2)
        Constraints (min and max values)
        ** for values of np.nan --> Galfit will not use the constraint for the parameter
        ** e.g., [[0, 2], [0.1, 5]]  or [0,2]
        ** You can use lim_params_array
     * lim_params_array: Ndarray (self.Ncomp, self.Nparam, 2) (Default: None)
        Constraints table.
        ** If None, it will be automatically generated based on lim_pos, lim_mag ...
     * use_constraint: Bool (Default: True)
        Whether use the constraint or not.
        The single value in the list will be duplicated.
     * group_id: int (default: -1)
        Set the group ID
     * size_conv, size_fitting: int (default: -1, -1)
        Galfit parameters

    """
    def __init__(self,
                 MC2fitSettingClass,
                 **kwargs):

        ## default setting
        self.complist = np.array(['sersic2', 'sersic2'])  # list of components (Ncomp = len(components))
        self.namelist = np.array(['ss_23', 'ss_24', 'ss_25', 'ss_26'])   # Namelist (Nset = len(namelist))

        ## Initial guess
              # Nan -> Default
        self.est_params_array = np.array([[np.nan, np.nan, np.array([23, 24, 25, 26]), 30, 1, 1, 0],
                                          [np.nan, np.nan, np.array([23, 24, 25, 26]), 10, 4, 1, 0]],
                                          dtype=object) # See note

        self.use_constraint = True ## We will use constraints

        self.lim_pos = np.array([[80, 120]]) # position constraint
        self.lim_mag = np.nan # magnitude constraint
        self.lim_reff = np.array([[5, 200]]) # Effective radius constraint
        self.lim_n = np.array([[0.2, 2]]) # sersic index (n) constraint
        self.lim_ar = np.array([[0.1, 1]])
        self.lim_pa = np.nan
        self.lim_params_array = None ## Instead of above, you can put in the values at once.

        self.group_id = 0   ## Fitting group
        self.size_conv = 101  ## Convolution box size
        self.size_fitting = -1


        try: self.__dict__.update(MC2fitSettingClass.__dict__)
        except: pass
        self.__dict__.update(kwargs)
        self._post_init()


    def _post_init(self):
        self.plate_scale=dharray.value_repeat_array(self.plate_scale, 2)
        self.image_size=dharray.value_repeat_array(self.image_size, 2)
        self.size_fitting=dharray.value_repeat_array(self.size_fitting, 2)
        self.size_conv=dharray.value_repeat_array(self.size_conv, 2)

        if(type(self.complist)==str): print("♣ Warning! ♣ complist is not array type!")
        if(type(self.namelist)==str):
            print("♣ Warning! ♣ namelist is not array type!")
            self.namelist=np.array([self.namelist])
        self.Ncomp = len(self.complist)
        self.Nset = len(self.namelist)
        self.Nparam = len(self.est_params_array[0])
        self._setup_params_table()
        self._generate_runlist()

    def _setup_params_table(self):
        """
        Repeat est_params --> Generate table
        Skip for arrays
        est_params_table: Nset * Ncomp * params
        """
        self.est_params_table=np.full((self.Nset, self.Ncomp, self.Nparam), np.nan)
        for comp in range (self.Ncomp):
            for i in range(self.Nparam):
                self.est_params_table[:,comp,i]=dharray.repeat_except_array(self.est_params_array[comp,i], self.Nset)
        self.use_constraint_list=dharray.repeat_except_array(self.use_constraint, self.Nset)

        # All sets will share the same constraints
        # Ncomp * params
        if(hasattr(self.lim_params_array, "__len__")==False):
            self.lim_params_array=np.full((self.Ncomp, self.Nparam, 2), np.nan)
            self.lim_params_array[:,0]=self.value2array(self.lim_pos, self.Ncomp)
            self.lim_params_array[:,1]=np.copy(self.lim_params_array[:,0])
            self.lim_params_array[:,2]=self.value2array(self.lim_mag, self.Ncomp)
            self.lim_params_array[:,3]=self.value2array(self.lim_reff, self.Ncomp)
            self.lim_params_array[:,4]=self.value2array(self.lim_n, self.Ncomp)
            self.lim_params_array[:,5]=self.value2array(self.lim_ar, self.Ncomp)
            self.lim_params_array[:,6]=self.value2array(self.lim_pa, self.Ncomp)

        # Nset * Ncomp * params
        self.lim_params_table = np.repeat(self.lim_params_array[None,:,:], self.Nset, axis=0)

    def value2array(self, value, repeat):
        """
        Value -> array by repeating
        """
        if(hasattr(value, "__len__")): #e.g., [[1,2], [3,4]]
            for i in range (len(value)):
                if(hasattr(value[i], "__len__")==False): value[i]=[np.nan, np.nan]  #e.g., [[1,2], np.nan]
            if(len(value)==1): return np.repeat(value, repeat, axis=0)
            elif(len(value)>repeat): return np.array(value[:repeat])
            elif(len(value)<repeat): print("Error! Size is not enough!")
            else: return np.array(value)
        else: return np.repeat([[np.nan, np.nan]], repeat, axis=0)  #np.nan

    def _generate_runlist(self):
        """
        Make runlist
        """
        ## Generate array
        runlist_dtype=[('name', '<U20'),('complist', object), ('ncomp', '<i4'),
                       ('est_params', object), ('lim_params', object),
                       ('size_conv', object), ('size_fitting', object),
                       ('group_ID', '<i4'), ('use_lim', bool),
                      ]
        self.runlist=np.zeros(self.Nset, dtype=runlist_dtype)

        ## Input values
        for i in range(self.Nset):
            self.runlist['est_params'][i]=self.est_params_table[i]
            self.runlist['lim_params'][i]=self.lim_params_table[i]
            self.runlist['complist'][i]=self.complist
            self.runlist['ncomp'][i]=len(self.complist)
            self.runlist['size_conv'][i]=self.size_conv
            self.runlist['size_fitting'][i]=self.size_fitting
        self.runlist['name']=self.namelist
        self.runlist['use_lim']=self.use_constraint_list
        self.runlist['group_ID'] = self.group_id

    def add_runlist(self, add_runlist):
        self.runlist=np.append(self.runlist, add_runlist.runlist)
        self.namelist=self.runlist['name']

    def show_runlist(self, show_lim=True):
        """
        Display runlist
        """
        Nset_imsi=len(self.runlist['est_params'])
        dum, Nparams_imsi=np.shape(self.runlist['est_params'][0])

        ## Front | est | (lim) | End
        runlist_show_front=self.runlist[['name', 'complist', 'ncomp']]
        runlist_show_front=dharray.array_remove_void(runlist_show_front)
        runlist_show_end=self.runlist[['use_lim', 'group_ID']]
        runlist_show_end=dharray.array_remove_void(runlist_show_end)

        ## Make names
        params_list=['xpos', 'ypos', 'mag', 'reff', 'n', 'ar', 'pa']
        params_list_est=dharray.array_attach_string(params_list, 'est_', add_at_head=True)
        params_list_lim=dharray.array_attach_string(params_list, 'lim_', add_at_head=True)

        ## Make est_list, lim_list
        runlist_show_estlist=np.zeros((Nset_imsi, Nparams_imsi), dtype=object) # Ncomp : already merged into
        runlist_show_limlist=np.zeros((Nset_imsi, Nparams_imsi), dtype=object) # Ncomp : already merged into
        for i in range (Nset_imsi):
            runlist_show_estlist[i]=np.apply_along_axis(', '.join, 0, self.runlist['est_params'][i].astype(str))
            temp=np.apply_along_axis(' '.join, 2, self.runlist['lim_params'][i].astype(str)).astype(str).T
            temp=np.char.add('(', temp.astype(str))
            temp=np.char.add(temp.astype(str), ')')
            runlist_show_limlist[i]=np.apply_along_axis(', '.join, 1, temp)

        ## Add names
        runlist_show_comp=dharray.array_quickname(runlist_show_estlist, names=params_list_est, dtypes=object)
        runlist_show_lim=dharray.array_quickname(runlist_show_limlist, names=params_list_lim, dtypes=object)

        ## Merge
        res=dharray.array_add_columns(runlist_show_front, runlist_show_comp)
        if(show_lim): res=dharray.array_add_columns(res, runlist_show_lim)
        res=dharray.array_add_columns(res, runlist_show_end)

        display(pd.DataFrame(res))

class PyGalfit():
    """
    Automatically run Galfit with input presets
    See also -> Runlist_Galfit (Generate presets)
    """
    def __init__(self,
                 MC2fitSettingClass,
                 dir_work,
                 **kwargs):

        ## default setting
        self.size_fitting = -1  # Fix the fitting area       ## -1 -> Full image
        self.size_conv = -1     # Convolution size.             ## -1 --> Auto (size of the PSF image)
        self.prefix=''
        self.suffix=''
        self.outputname='galfit'
        self.is_run_galfit_dir_work=False
        self.check_files=True
        self.dir_work=dir_work
        self.extract_sigma=False
        self.psf_size=10


        try: self.__dict__.update(MC2fitSettingClass.__dict__)
        except: pass
        self.__dict__.update(kwargs)
        self._post_init()


    def _post_init(self):
        self.plate_scale=dharray.value_repeat_array(self.plate_scale, 2)
        self.image_size=dharray.value_repeat_array(self.image_size, 2)
        self.size_fitting=dharray.value_repeat_array(self.size_fitting, 2)
        self.size_conv=dharray.value_repeat_array(self.size_conv, 2)
        self.default_est_params=np.array([self.image_size[0]/2, self.image_size[1]/2, 20, 20, 4, 1, 0])

        self.n_comp=int(0)
        self.comp_dtype=[('fitting_type', '<U10'),
                         ('est_xpos', '<f8'), ('est_ypos', '<f8'),
                         ('est_mag', '<f8'), ('est_reff', '<f8'), ('est_n', '<f8'),
                         ('est_axisratio', '<f8'), ('est_pa', '<f8')]

        #Filenames
        self._set_path_name(self.check_files)
        self.generate_galconfig()
        self.generate_constraints()

    def _set_path_name(self, check_files=False):
        if(self.is_run_galfit_dir_work): adding_dir_work='' ## Run Galfit in each folder
        else: adding_dir_work=self.dir_work                 ## Run Galfit in a specific folder
        os.makedirs(self.dir_work+self.proj_folder, exist_ok=True)

        ## Configuration file
            # fn : file name
            # fnc: file name in the configuration file (galfit running folder)
        self.fn_gal_conf = self.dir_work+self.proj_folder + 'param_' + self.prefix \
                           + self.outputname+ self.suffix + ".dat"
        self.fnc_gal_conf = adding_dir_work+self.proj_folder + 'param_' + self.prefix \
                           + self.outputname+ self.suffix + ".dat"

        ## We use a single constraint file
            # fn : file name
            # fnc: file name in the configuration file
        self.fn_constraint = self.dir_work+self.proj_folder + 'const_' + self.prefix \
                           + self.outputname+ self.suffix + ".dat"
        self.fnc_constraint = adding_dir_work+self.proj_folder + 'const_' + self.prefix \
                           + self.outputname+ self.suffix + ".dat"

        ## Image
        self.fn_image=self.dir_work+self.fni_image
        self.fnc_image=adding_dir_work+self.fni_image
        if(check_files):
            try: fits.open(self.fn_image)
            except: raise Exception(">> ♣♣♣ Warning! ♣♣♣ Image file does not exist! -"+str(self.fn_image))

        ## PSF
        self.fn_psf = self.dir_work + self.fni_psf
        self.fnc_psf = adding_dir_work + self.fni_psf
        if(check_files):
            try: self.psf_size=np.shape(fits.getdata(self.fn_psf)) # Get PSF size
            except: raise Exception(">> ♣♣♣ Warning! ♣♣♣ PSF file does not exist! -"+str(self.fn_psf))

        ## Sigma
        if((self.fni_sigma==None) | (self.fni_sigma=='none')):
            self.fn_sigma = 'none'
            self.fnc_sigma = 'none'
        else:
            self.fn_sigma = self.dir_work + self.fni_sigma
            self.fnc_sigma = adding_dir_work + self.fni_sigma
        if((check_files) & (self.fn_sigma!='none')):
            try: fits.open(self.fn_sigma)
            except: raise Exception(">> ♣♣♣ Warning! ♣♣♣ Sigma file does not exist! -"+str(self.fn_sigma))

        ## Mask
        if((self.fni_masking==None) | (self.fni_masking=='none')):
            self.fn_masking = 'none'
            self.fnc_masking = 'none'
        else:
            self.fn_masking = self.dir_work + self.fni_masking
            self.fnc_masking = adding_dir_work + self.fni_masking
        if(check_files):
            try: fits.open(self.fn_masking)
            except: raise Exception(">> ♣♣♣ Warning! ♣♣♣ Masking file does not exist! -"+str(self.fn_masking))

        ## Output
        self.fn_output_imgblock=self.dir_work+self.proj_folder+'output_'+ self.prefix + self.outputname + self.suffix + ".fits"
        self.fnc_output_imgblock=adding_dir_work+self.proj_folder+'output_'+ self.prefix + self.outputname + self.suffix + ".fits"



    def add_fitting_comp(self, fitting_type='sersic',
                        est_params=np.full(7, np.nan), is_allow_vary=True, fix_psf_mag=False):
        """
        Descr - Add fitting component
        INPUT
         * fitting_type: (Default='sersic')
         * est_params: Estimated parameters.
                       For values nan, they will be default values.
                 index    [  0   |  1   | 2 |  3 | 4 |5 |6 ]
                 params   [ xpos | ypos |mag|reff| n |ar|pa]
                 default  [center|center|20 | 20 | 4 |1 |0 ]

         * is_allow_vary: If false, the value is fixed. Galfit will not change the parameter
           1) True/False -> Apply the value for all parameters (Default=True)
           2) Array -> Apply the values for each parameter
                 index    [  0   |  1   | 2(X) |  3 | 4 |5 |6 ]
                 params   [ xpos | ypos |mag(X)|reff| n |ar|pa]  ## Mag always changes (True)!
        """

        self.n_comp+=1
        if(self.silent==False): print("● Added a fitting component ", self.n_comp)
        index=int(self.n_comp-1)

        ## Allow_vary_array
        if(hasattr(is_allow_vary, "__len__")==False):
            allow_vary_array=np.full(7, is_allow_vary)
        else: allow_vary_array=is_allow_vary


        ## Generate Array
        if(self.n_comp==1): ## Make a new one
            self.comp=np.zeros(self.n_comp, dtype=self.comp_dtype)
        else: ## Attach a new row
            temp_comparray=np.zeros(self.n_comp, dtype=self.comp_dtype)
            temp_comparray[:-1]=np.copy(self.comp)
            self.comp=np.copy(temp_comparray)


        ## ========Est params Setting===========
        ## default values
        nans=np.isnan(est_params)
        est_params[nans]=np.copy(self.default_est_params[nans])

        ## setting
        for i, item in enumerate(self.comp.dtype.names):
            if(i==0): self.comp[index][item]=np.copy(fitting_type)
            else: self.comp[index][item]=np.copy(est_params[i-1])

        if(self.silent==False):
            print(">> Input Data : ")
            print(self.comp.dtype)
            print(self.comp)

        ## =============== INPUT values ==================
        fconf = open(self.fn_gal_conf, 'a')
        fconf.write('\n# Component number: %d\n'%int(self.n_comp+1))
        fconf.write(' 0) %s                     #  object type\n'%fitting_type)
        fconf.write(' 1) %f  %f   %d %d         #  position x, y\n'%(est_params[0], est_params[1], allow_vary_array[0], allow_vary_array[1]))
        if(fitting_type=='psf'):
            if(self.silent==False): print("PSF")
            if(fix_psf_mag): fconf.write(' 3) %f      0              #  Total magnitude\n'  %est_params[2])
            else: fconf.write(' 3) %f      1              #  Total magnitude\n'  %est_params[2])
        else:
            fconf.write(' 3) %f      1              #  Integrated magnitude (sersic2 - mu_Reff)\n'  %est_params[2])
            fconf.write(' 4) %f      %d             #  R_e (half-light radius)   [pix]\n'  %(est_params[3], allow_vary_array[3]))
            fconf.write(' 5) %f      %d             #  Sersic index n (exponential n=1)\n'  %(est_params[4], allow_vary_array[4]))
            fconf.write(' 6) 0.0         0          #     -----\n')
            fconf.write(' 7) 0.0         0          #     -----\n')
            fconf.write(' 8) 0.0         0          #     -----\n')
            fconf.write(' 9) %f      %d             #  axis ratio (b/a)\n' %(est_params[5], allow_vary_array[5]))
            fconf.write('10) %f      %d             #  position angle (PA) [deg: Up=0, Left=90]\n'  %(est_params[6], allow_vary_array[6]))
        fconf.write(' Z) 0                      #  output option (0 = resid., 1 = Do not subtract)\n')

        fconf.close()




    def generate_galconfig(self):
        if(self.silent==False): print("● Generated config file to ", self.fn_gal_conf)

        ## Conv size
        if(self.size_conv[0]<=0): size_conv=self.psf_size

        ## Fitting box
        size_xmin, size_xmax = self.get_size(self.image_size[0], self.size_fitting[0])
        size_ymin, size_ymax = self.get_size(self.image_size[1], self.size_fitting[1])


        # Create name and open configuration file for writing
        fconf = open(self.fn_gal_conf, 'w')
        fconf.write('# Automatically generated by Auto_galfit.ipynb _ DJ Khim\n')
        fconf.write('# IMAGE and GALFIT CONTROL PARAMETERS\n')

        fconf.write('A) %s                     # Input data image (FITS file)\n' %self.fnc_image)
        fconf.write('B) %s                     # Output data image block\n' %self.fnc_output_imgblock)
        fconf.write('C) %s                   # Sigma image name (made from data if blank or "none")\n'%self.fnc_sigma)
        fconf.write('D) %s                     # Input PSF image and (optional) diffusion kernel\n' %self.fnc_psf)
        fconf.write('E) 1                      # PSF fine sampling factor relative to data\n')
        fconf.write('F) %s                     # Bad pixel mask (FITS image or ASCII coord list)\n' %self.fnc_masking)
        fconf.write('G) %s                     # File with parameter constraints (ASCII file)\n' %self.fnc_constraint)
        fconf.write('H) %d    %d   %d    %d      # Image region to fit (xmin xmax ymin ymax)\n' %(size_xmin, size_xmax, size_ymin, size_ymax))
        fconf.write('I) %d    %d               # Size of the convolution box (x y)\n'  %(self.size_conv[0], self.size_conv[1]))
        fconf.write('J) %f                     # Magnitude photometric zeropoint\n' % self.zeromag)
        fconf.write('K) %f    %f               # Plate scale (dx dy)  (arcsec per pixel)\n'  %(self.plate_scale[0], self.plate_scale[1]))
        fconf.write('O) regular                # Display type (regular, curses, both\n')
        fconf.write('P) 0                      # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomp\n\n')
        fconf.write('# -----------------------------------------------------------------------------\n')
        fconf.write('#   par)    par value(s)    fit toggle(s)    # parameter description \n')
        fconf.write('# -----------------------------------------------------------------------------\n\n')

        #========================= SKY ===================================
        fconf.write('# Component number: 1\n')
        fconf.write(' 0) sky                    #  object type\n')
        fconf.write(' 1) %f      1              #  sky background at center of fitting region [ADUs]\n' %self.est_sky)
        fconf.write(' 2) 0.0         0          #  dsky/dx (sky gradient in x)\n')
        fconf.write(' 3) 0.0         0          #  dsky/dy (sky gradient in y)\n')
        fconf.write(' Z) 0                      #  output option (0 = resid., 1 = Do not subtract)\n')

        fconf.close()

    def get_size(self, image_size, fitting_size):
        if(fitting_size>=0):
            half_size=np.around(fitting_size/2)
            size_min=np.around(image_size/2-half_size)
            size_max=np.around(image_size/2+half_size)
            if(size_min<0): size_min=0
            if(size_max>image_size): size_max=image_size
        else:
            size_min=0
            size_max=image_size

            return size_min, size_max


    def generate_constraints(self):
        if(self.silent==False): print("● Generated constraints file to ", self.fn_constraint)

        # Create name and open configuration file for writing
        fconf = open(self.fn_constraint, 'w')
        fconf.write('# Automatically generated by Auto_galfit.ipynb _ DJ Khim\n')
        fconf.write('# GALFIT CONSTRAINT PARAMETERS\n')
        fconf.write('# Component     Parameter    Constraint range       Comment\n\n')


    def add_constraints(self, id_comp, lim_params=np.full((7,2), np.nan), re_min=None, n_ratio=None):
        """
        Descr - Add fitting constraint
        INPUT
         - id_comp: Component ID
         * lim_params: Parameter constraints
                       7 parameters * (min, max) -> (7*2) array
                       For values nan, it will not used
                 index    [  0   |  1   | 2 |  3 | 4 |5 |6 ]
                 params   [ xpos | ypos |mag|reff| n |ar|pa]
                 default  [center|center|20 | 20 | 4 |1 |0 ]

        """
        id_comp=int(id_comp)
        if(self.silent==False): print("● Added constraint for ", id_comp)
        fconf = open(self.fn_constraint, 'a')
        check_valid=np.sum(lim_params, axis=1)
        par_in_list=['x', 'y', 'mag', 'reff', 'n', 'q', 'pa']

        if(hasattr(re_min, "__len__")):
            lim_params[3]+=re_min

        for i, par_in in enumerate(par_in_list):
            if(np.isfinite(check_valid[i])):
                # if(i<2):
                #     descr='# Soft constraint: %s-position to within %d and %d of the >>INPUT<< value.'%(par_in, lim_params[i,0], lim_params[i,1])
                #     fconf.write('  %d        %s        %d  %d     %s\n\n'%(id_comp, par_in, lim_params[i,0], lim_params[i,1], descr))

                if(i<2):
                    descr='# Soft constraint: Constrains the %s-position to '
                    descr+='within values from %.2f to %.2f.'%(lim_params[i,0], lim_params[i,1])
                if(i==2):
                    descr='# Soft constraint: Constrains the magnitude to '
                    descr+='within values from %.2f to %.2f.'%(lim_params[i,0], lim_params[i,1])
                if(i==3):
                    descr='# Soft constraint: Constrains the effective radius to '
                    descr+='within values from %.2f to %.2f.'%(lim_params[i,0], lim_params[i,1])
                if(i==4):
                    descr='# Soft constraint: Constrains the Sersic index n to '
                    descr+='within values from %.2f to %.2f.'%(lim_params[i,0], lim_params[i,1])
                if(i==5):
                    descr='# Soft constraint: Constrains the axis ratio (b/a) to '
                    descr+='within values from %.2f to %.2f.'%(lim_params[i,0], lim_params[i,1])
                if(i==6):
                    descr='# Soft constraint: Constrains the position angle to '
                    descr+='within values from %.2f to %.2f.'%(lim_params[i,0], lim_params[i,1])
                fconf.write('  %d        %s        %.2f to %.2f     %s\n\n'%(id_comp, par_in, lim_params[i,0], lim_params[i,1], descr))
        if(n_ratio!=None):
                descr='# Soft constraint: Sersic indice n ratio to be within values from  %.2f to %.2f.'%(n_ratio, 1000)
                fconf.write('  3/2       n        %.2f    %.2f     %s\n\n'%(n_ratio, 1000, descr))
        fconf.close()


    def run_galfit(self, print_galfit=False, use_dir=os.getcwd()):
        '''
        Create and execute Galfit shell commands
        '> /dev/null' suppresses output
        '''
        ### Command
        if(self.silent==False): print("● Run galfit")
        cmd = self.fn_galfit

        if(self.extract_sigma): cmd += ' -outsig '
        if(self.output_block): cmd += ' -c'
        else: cmd += ' -o2'
        cmd+= ' %s ' % self.fnc_gal_conf #+ ' > /dev/null'

        if(self.silent==False): print(">> Command : ", cmd)

        ### Run
        if(print_galfit==True): result=subprocess.run(cmd, shell=True, cwd=use_dir)
        else: result=subprocess.run(cmd, shell=True, cwd=use_dir, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if(self.silent==False):
            print(">> Result : ", result)
            if(result.returncode==0): print(">> Galfit done!")
            else: raise Exception(">> ♣♣♣ Warning! ♣♣♣ Galfit has a problem!")
        return result


class MC2fitRun():
    """
    Descr - Run galfit with Runlist (See class AutoGalfit)
    INPUT
     *** Working dir ***
     * dir_work : str (default: '')
     * proj_folder: str (default: 'galfit/')
     * group_id : int (default: -1)
        Run Galfit only for the given group ID.
        If the value is -1, run galfit for all group IDs.
     * fni_image, fni_imputmask, fni_psf, fni_sigma: path for input images.
     * fn_image, fn_imputmask, fn_psf, fn_sigma: absolute path for input images. (if 'fni', it follows above)

     *** parameters ***
     * add_params_array : array_like
        It will be added/replaced to the est_params. For None, it will be ignored.
        See, add_params_mode
        If length of array is 1, the array will be applied to the all runlists.
        (Default=None)
             index    [  0   |  1   | 2 |  3 | 4 |5 |6 ]
             params   [ xpos | ypos |mag|reff| n |ar|pa]

     * add_params_mode : str or array (default: 'add')
        Add or replace the est_params.
        list or array is also possible.
        (e.g., 'add', ['add', 'add', 'replace'])
     * is_allow_vary: If false, the value is fixed. Galfit will not change the parameter
        1) True/False -> Apply the value for all parameters (Default=True)
        2) Array -> Apply the values for each parameter
             index    [  0   |  1   | 2(X) |  3 | 4 |5 |6 ]
             params   [ xpos | ypos |mag(X)|reff| n |ar|pa]  ## Mag always changes (True) except for PSF
     * fix_psf_mag: If True, mag for PSF is fixed (is_allow_vary[2]=False for PSF)
     * is_nth_mag_offset: If True, the magnitude of the nth comp. would be respective to 1st comp.
                          The input values are being offsets. (2nd mag = 1st mag + input values)
                          (Default=False)

    """
    def __init__(self,
                 MC2fitSettingClass,
                 RunlistClass,
                 dir_work,
                 **kwargs):

        ## path setting
        self.dir_work=dir_work
        self.proj_folder='galfit/'
        self.prefix=''
        self.suffix=''
        self.fni_image='image.fits'
        self.fni_masking=None
        self.fni_psf='psf_area_g.fits'
        self.fni_sigma='sigma_g.fits'
        self.is_run_galfit_dir_work=False
        self.check_files=True

        ## Run setting
        self.group_id=-1

        ## Output setting
        self.output_block=True
        self.overwrite=False
        self.extract_sigma=False

        ## Parameter array
        self.add_params_array=None
        self.Ncomp_add_params_array=0
        self.add_params_mode='add'
        self.is_nth_mag_offset=False
        self.fix_psf_mag=False
        self.add_constraints_array=None
        self.is_allow_vary=True

        ## print setting
        self.silent=False
        self.print_galfit=False
        self.debug=False

        ## default values
        self.plate_scale=0.262
        self.zeromag=22.5
        self.est_sky=10
        self.re_min=None
        self.n_ratio=None

        ## Class
        self.MC2fitSettingClass=MC2fitSettingClass
        self.RunlistClass=RunlistClass

        try: self.__dict__.update(MC2fitSettingClass.__dict__)
        except: pass
        self.__dict__.update(kwargs)
        self._post_init()


    def _post_init(self):
        self.plate_scale=dharray.value_repeat_array(self.plate_scale, 2)
        self.image_size=dharray.value_repeat_array(self.image_size, 2)
        self._select_run()


    def _select_run(self):
        ## Using group ID, Select runlist
        if(self.group_id<0): usingrunlist=copy.deepcopy(self.RunlistClass.runlist)
        else: usingrunlist=copy.deepcopy(self.RunlistClass.runlist[np.isin(self.RunlistClass.runlist['group_ID'], self.group_id)])

        ## If add_params_array is given --> Add values to runlist
        if(hasattr(self.add_params_array, "__len__")==False):
            self.Ncomp_add_params_array=0
        if(self.Ncomp_add_params_array!=0):
            self.add_params_mode=dharray.repeat_except_array(self.add_params_mode, self.Ncomp_add_params_array)
            if(self.silent==False): print(">> Additional params array will be considered! Ncomp:", self.Ncomp_add_params_array)


        ## ================Loop=================
        for i_run in range (len(usingrunlist)):
            thisrunlist=usingrunlist[i_run]
            self._each_run(thisrunlist)

    def _each_run(self, thisrunlist):

        outputname=thisrunlist['name']

        ## Check if the result is exist --> Skip
        if(self.overwrite==False):
            if(os.path.exists(self.dir_work+self.proj_folder+'output_'+self.prefix+self.outputname+self.suffix+".fits")==True): return  # Skip
            if(os.path.exists(self.dir_work+self.proj_folder+'result_'+self.prefix+self.outputname+self.suffix+".fits")==True): return  # Skip


        PyGal=PyGalfit(outputname=outputname,
                       size_conv=thisrunlist['size_conv'],
                       size_fitting=thisrunlist['size_fitting'],
                       **self.__dict__)

        for comp in range (thisrunlist['ncomp']):
            if(comp<(self.Ncomp_add_params_array)):
                ## Add / replace est params
                if(self.add_params_mode[comp]=='add'): thisrunlist['est_params'][comp]+=self.add_params_array[comp]
                else: thisrunlist['est_params'][comp]=self.add_params_array[comp]


            if(comp>0): # Except for the first one
                if(self.is_nth_mag_offset):
                    if(self.debug):
                        print(">> Offset mag")
                        print(thisrunlist['est_params'][comp,2], thisrunlist['est_params'][0,2])
                    thisrunlist['est_params'][comp,2]+=thisrunlist['est_params'][0,2]

            if(self.debug):
                print("\n>> j", comp)
                print(">>>> EST params: ",thisrunlist['est_params'][comp])
                print(">>>> LIM params: ",thisrunlist['lim_params'][comp])
                print(">>>> is_allow_vary: ",self.is_allow_vary)

            ## Add fitting components
            PyGal.add_fitting_comp(fitting_type=thisrunlist['complist'][comp],
                                          est_params=thisrunlist['est_params'][comp],
                                          is_allow_vary=self.is_allow_vary, fix_psf_mag=self.fix_psf_mag)
            ## Add constraints
            if(thisrunlist['use_lim']==True):
                PyGal.add_constraints(id_comp=comp+2, ## 1: Sky, 2: Comp0, 3: Comp1, ...
                                             lim_params=thisrunlist['lim_params'][comp],
                                             re_min=self.re_min,
                                             n_ratio=self.n_ratio
                                   )

        ## Run
        if(self.is_run_galfit_dir_work): use_dir=os.path.join(os.getcwd(), self.dir_work)
        else: use_dir=os.getcwd()

        if(self.debug):
            print(">>>> use_dir", use_dir)
        PyGal.run_galfit(print_galfit=self.print_galfit, use_dir=use_dir)

class MC2fitRun_Mulcore():
    """
    Descr - Run galfit with Runlist (See class AutoGalfit)
    INPUT
     *** Working dir ***
     * dir_work : str (default: '')
     * proj_folder: str (default: 'galfit/')
     * group_id : int (default: -1)
        Run Galfit only for the given group ID.
        If the value is -1, run galfit for all group IDs.
     * fni_image, fni_imputmask, fni_psf, fni_sigma: path for input images.
     * fn_image, fn_imputmask, fn_psf, fn_sigma: absolute path for input images. (if 'fni', it follows above)

     *** parameters ***
     * add_params_array : array_like
        It will be added/replaced to the est_params. For None, it will be ignored.
        See, add_params_mode
        If length of array is 1, the array will be applied to the all runlists.
        (Default=None)
             index    [  0   |  1   | 2 |  3 | 4 |5 |6 ]
             params   [ xpos | ypos |mag|reff| n |ar|pa]

     * add_params_mode : str or array (default: 'add')
        Add or replace the est_params.
        list or array is also possible.
        (e.g., 'add', ['add', 'add', 'replace'])
     * is_allow_vary: If false, the value is fixed. Galfit will not change the parameter
        1) True/False -> Apply the value for all parameters (Default=True)
        2) Array -> Apply the values for each parameter
             index    [  0   |  1   | 2(X) |  3 | 4 |5 |6 ]
             params   [ xpos | ypos |mag(X)|reff| n |ar|pa]  ## Mag always changes (True) except for PSF
     * fix_psf_mag: If True, mag for PSF is fixed (is_allow_vary[2]=False for PSF)
     * is_nth_mag_offset: If True, the magnitude of the nth comp. would be respective to 1st comp.
                          The input values are being offsets. (2nd mag = 1st mag + input values)
                          (Default=False)

    """
    def __init__(self,
                 MC2fitSettingClass,
                 RunlistClass,
                 DirInfoClass,
                 remove_galfit_interval=300,
                 add_params_array_set=None,
                 re_cut_list=None,
                 skiplist=None,
                 **kwargs):

        ## Multicore setting
        self.use_try=True
        self.Ncore=2
        self.show_multicore=True
        self.show_progress=-1

        ## Additional parameters
        self.add_params_array_set=add_params_array_set
        self.remove_galfit_interval=remove_galfit_interval
        self.re_cut_list=re_cut_list
        self.skiplist=skiplist

        ## Class
        self.MC2fitSettingClass=MC2fitSettingClass
        self.RunlistClass=RunlistClass
        self.DirInfoClass=DirInfoClass

        try: self.__dict__.update(MC2fitSettingClass.__dict__)
        except: pass
        self.__dict__.update(kwargs)
        self._post_init()
        if(self.Ncore!=1):
            self._show_multicore_result()


    def _show_multicore_result(self):
        Nfailed=np.sum(np.isnan(self.multi_res))
        print("======== Multicore result ========")
        print("● Succeed: %d / %d"%(np.sum(self.multi_res==0), len(self.DirInfoClass.dir_work_list)))
        print("● Failed: %d / %d"%(Nfailed, len(self.DirInfoClass.dir_work_list)))
        if(Nfailed>0):
            print(">> See where: self.multi_res")
            print(">> To debug, use Ncore=1, and use_try = False")

    def _post_init(self):
        self.plate_scale=dharray.value_repeat_array(self.plate_scale, 2)
        self.image_size=dharray.value_repeat_array(self.image_size, 2)

        if(hasattr(self.add_params_array_set, "__len__")==False): self.add_params_array_set=np.full(len(self.DirInfoClass.dir_work_list), None)
        if(hasattr(self.skiplist, "__len__")==False): self.skiplist=np.full(len(self.DirInfoClass.dir_work_list), False)
        if(hasattr(self.re_cut_list, "__len__")==False): self.re_cut_list=np.full(len(self.DirInfoClass.dir_work_list), None)
        if(len(self.add_params_array_set)!=len(self.DirInfoClass.dir_work_list)):
            print("Warning! Check add_params_array!")
            print("Stop the function")
            return

        global sub_run_process

        def sub_run_process(i):
            dir_work=self.DirInfoClass.dir_work_list[i]
            fn_done=dir_work+self.proj_folder+"done_galfit.dat"
            if(self.fast_skip==True):
                if(os.path.exists(fn_done)): return 0
            if(self.skiplist[i]==True): return 0

            if(i%self.remove_galfit_interval==0):
                existlist1=glob.glob('./galfit.*')
                for fb in existlist1:
                    try: os.remove(fb)
                    except: pass
                try: os.remove('./fit.log')
                except: pass


            MC2fitRun(dir_work=dir_work,
                      add_params_array=self.add_params_array_set[i],
                      re_cut=self.re_cut_list[i],
                      **self.__dict__
                     )


            np.savetxt(fn_done, np.array(['done']), fmt="%s")
            return 0

        self.multi_res=mulcore.multicore_run(sub_run_process,
                                             len(self.DirInfoClass.dir_work_list),
                                             Ncore=self.Ncore,
                                             use_try=self.use_try,
                                             show_progress=self.show_progress,
                                             debug=self.show_multicore)
        self.multi_res=np.array(self.multi_res)

class MC2fitResult():
    """
    Descr - Galfit result function
    INPUT
      0) For all input types
      * tolerance: Chi2 tolerance for "Good results" (|Min chi2 - chi2 |/(Min chi2) < tolerance) (Default=1e-3)
      * silent: Print text (Default=False)
      * ignore_no_files: Run the code even though there is no any files. (Default=False)
      * auto_input: Auto_input mode (sequence 1 -> 2 -> 3) (Default=True)
      * group_id: If None, load all. If not, load input group ID only. (Default=None)

      1) Load data
      * fn_load: (Default='')
      * fna_load: (Default='submodels.dat')
      ** if None / or if it cannot find the data, go next
      2) Runlist
      * runlist: (Default=None)
      * dir_work: working directory (Default=None)
      ** if runlist is None, go next
      3) Manual
      * fnlist, namelist
    """
    def __init__(self,
                 MC2fitSettingClass,
                 runlist,
                 dir_work,
                 **kwargs):

        self.runlist=runlist
        self.dir_work=dir_work
        self.paramslist=['XC', 'YC', 'MU_E', 'RE', 'N', 'AR', 'PA', 'MAG']

#         runlist=None, dir_work=None,  ## 2) For input_runlist mode
#         fnlist=None, namelist=None,  ## 3) For input_manual mode
        self.group_id=None
        self.tolerance=1e-3 # Chi2 tolerance for "Good results" (|Min chi2 - chi2 |/(Min chi2) < tolerance)
        self.ignore_no_files=False
        self.comp_crit='RChi_50'
        self.input_only=False
        self.reff_snr_crit=2
        self.reff_snr_n_cut=100
        self.n_crit=2
        self.re_cut=[0,0],
        self.extended_itemlist=['war_tot', 'is_swap', 'group_ID']
        self.chi2_itemlist= ['CHI2NU', 'AIC_F', 'RChi_50', 'AIC_50']
        self.comp_autoswap=False
        self.swapmode='N'
        self.comp_4th_nsc_autoswap=True
        self.comp_4th_ss_autoswap=True
        self.img_center=100


        try: self.__dict__.update(MC2fitSettingClass.__dict__)
        except: pass
        self.__dict__.update(kwargs)
        self._post_init()

    def _post_init(self):
        self.galfit_itemlist_2nd=dharray.array_attach_string(self.paramslist, '2_', add_at_head=True)
        self.galfit_itemlist_3rd=dharray.array_attach_string(self.paramslist, '3_', add_at_head=True)
        self.galfit_itemlist_4th=dharray.array_attach_string(self.paramslist, '4_', add_at_head=True)

        # basically, galfit_itemlist= 2nd + 3rd.
        # If there is 4th component, it will be added at 'input_runlist'
        self.galfit_itemlist=self.galfit_itemlist_2nd+self.galfit_itemlist_3rd
        self.extended_itemlist=self.chi2_itemlist+self.extended_itemlist

#         self.dir_work=str(dir_work)
        self.dir_work_galfit=os.path.join(self.dir_work, self.proj_folder)

        self.plate_scale=dharray.value_repeat_array(self.plate_scale, 2)
        self.image_size=dharray.value_repeat_array(self.image_size, 2)
        self._input()
        self._dataswap()

        ## Find best
        if(self.Data.Ndata!=0):
            self.best=self.find_best(remove_warn=True)
            self.best_warn=self.find_best(remove_warn=False)
            try: self.best_near=self.find_near_best(self.Data.val[comp_crit][self.best],
                                               tolerance=self.tolerance, comp_crit=comp_crit)
            except: self.best_near=self.best
        else: ## No data
            self.best, self.best_warn, self.best_near = np.array([np.nan]), np.array([np.nan]), np.array([np.nan])


    def _input(self):

        if(self.silent==False): print("=========== Run ResultGalfit ===========\n")
        ## ========= INPUT ===================
        ## INPUT 2) Runlist
        if(hasattr(self.runlist, "__len__")):  ## If runlist is not empty, use runlist
            self._input_runlist(runlist=self.runlist, group_id=self.group_id, dir_work=self.dir_work_galfit)
        ## INPUT 3) Manual
        elif(hasattr(self.fnlist, "__len__")):   ## If fnlist is not empty, use fnlist
            self._input_manual(fnlist=self.fnlist, namelist=self.namelist)

        if(self.input_only): return

    def _dataswap(self):
        ## Swap 2nd and 3rd
        self.Data.val['is_swap']=0
        self.Data.err['is_swap']=0
        self.Data.war['is_swap']=0
        if(self.comp_autoswap):
            self.swaplist=self.make_swaplist(self.swapmode)
            self.Data.swap_data(self.swaplist, self.galfit_itemlist_2nd, self.galfit_itemlist_3rd)

        if((self.comp_4th_nsc_autoswap) & (np.isin(['4_XC'], self.Data.val.dtype.names)[0])):
            list_4th=np.isfinite(self.Data.val['4_XC'])
            ## Double PSF - closer
            dpsf=np.isnan(self.Data.val['3_N']) & np.isnan(self.Data.val['4_N'])
            dist23=(self.Data.val['2_XC']-self.Data.val['3_XC'])**2 + (self.Data.val['2_YC']-self.Data.val['3_YC'])**2
            dist24=(self.Data.val['2_XC']-self.Data.val['4_XC'])**2 + (self.Data.val['2_YC']-self.Data.val['4_YC'])**2
            targetlist=dpsf & list_4th & (dist24<dist23)
            self.Data.swap_data(targetlist, self.galfit_itemlist_3rd, self.galfit_itemlist_4th)

        if((self.comp_4th_ss_autoswap) & (np.isin(['4_XC'], self.Data.val.dtype.names)[0])):
            ## SS + PSF - closer to the center (except for N=5 gal)
            ds=np.isfinite(self.Data.val['2_N']) & np.isfinite(self.Data.val['3_N'])
            dist2=(self.Data.val['2_XC']-img_center)**2 + (self.Data.val['2_YC']-img_center)**2
            dist3=(self.Data.val['3_XC']-img_center)**2 + (self.Data.val['3_YC']-img_center)**2
            is_not_psf= self.Data.val['3_N']!=5
            targetlist=ds & list_4th & (dist3<dist2) & is_not_psf
            self.Data.swap_data(targetlist, self.galfit_itemlist_2nd, self.galfit_itemlist_3rd)


    def _input_runlist(self, runlist, group_id=None, dir_work=None):
        """
        Descr - Input data using the given runlist
        ** It will extract fnlist, namelist, and run input_manual
        ** See input_manual as well
        INPUT
         - runlist
         * group_id : (None : see all group_id)
         * dir_work : working dir for data
        """
        if(self.silent==False):
            print("\n● Input data with Runlist")
            print(">> Group ID : ", group_id)

        ## Cut runlist
        if(group_id==None): pass
        else: runlist=runlist[np.isin(runlist['group_ID'], group_id)]

        ## Maximum Ncomponent
        self.Ncomp_max=np.nanmax(runlist['ncomp'])
        if(self.Ncomp_max>2):
            self.galfit_itemlist=[]
            for i in range (self.Ncomp_max):
                galfit_itemlist=dharray.array_attach_string(self.paramslist, str(int(i+2))+'_', add_at_head=True)
                self.galfit_itemlist+=galfit_itemlist

        namelist=runlist['name']
        fnlist=dharray.array_attach_string(namelist, 'result_', add_at_head=True)
        fnlist=dharray.array_attach_string(fnlist, dir_work, add_at_head=True)
        fnlist=dharray.array_attach_string(fnlist, '.fits')
        self._input_manual(fnlist, namelist, runlist=runlist)

    def _input_manual(self, fnlist, namelist, runlist=None):
        """
        Descr - Manually input data
        ** See input_runlist as well
        INPUT
         - fnlist
         - namelist
        """
        if(self.silent==False):
            if(hasattr(runlist, "__len__")==False):
                print("\n● Input data with the manual mode")

        self.Data=ResGalData(namelist=namelist, fnlist=fnlist,
                            data_type=self.galfit_itemlist+self.extended_itemlist,
                            chi2_itemlist=self.chi2_itemlist
                            )

        if(self.silent==False): print(">> # of data : ", len(fnlist))
        if(len(self.Data.namelist)==0):
            if(self.ignore_no_files==False): raise Exception(">> ♣♣♣ Warning! ♣♣♣ Namelist is empty!")
            else: print(">> ♣♣♣ Warning! ♣♣♣ Namelist is empty!")
        self.Data.read_data(itemlist=self.galfit_itemlist)

        if(self.Data.Ndata==0):
            if(self.ignore_no_files==False): raise Exception(">> ♣♣♣ Warning! ♣♣♣ No files!")
            else: print(">> ♣♣♣ Warning! ♣♣♣ No files! .... e.g.)", fnlist[0])
        if(self.silent==False): print(">> # of read data : ", self.Data.Ndata)

        ## If it has runlist
        if(hasattr(runlist, "__len__")):
            self.Data.val['group_ID']=runlist['group_ID']
            self.Data.err['group_ID']=runlist['group_ID']
            self.Data.war['group_ID']=runlist['group_ID']
            self.check_bound(runlist=runlist)
            self.check_strange_radius(snr_crit=self.reff_snr_crit, reff_snr_n_cut=self.reff_snr_n_cut)
            self.check_strange_n(n_crit=self.n_crit)

        ## If not --> Manual check
        else:
            try: self.check_bound()
            except: pass
        self.safe_itemlist=self.Data.clean_data(self.galfit_itemlist)


    def check_bound(self, runlist=None):

        """
        Descr - Check parameter at the limit
        INPUT
         ** runlist : Automatically get limits (default : None)
        """
        self.bound=np.copy(self.Data.val)
        self.bound.fill(0)
        if(self.silent==False):
            print("\n◯ Check bound")
            if(hasattr(runlist, "__len__")): print(">>>> Runlist exists")
            else: print(">>>> Runlist does not exist")

        using_itemlist=np.copy(self.galfit_itemlist)
        mag_pos=np.where(np.char.find(using_itemlist, 'MAG')>0)[0]
        using_itemlist=using_itemlist[np.isin(np.arange(len(using_itemlist)), mag_pos, invert=True)]

        x_pos=np.where(np.char.find(using_itemlist, 'XC')>0)[0]
        y_pos=np.where(np.char.find(using_itemlist, 'YC')>0)[0]

        value_data=dharray.array_flexible_to_simple(self.Data.val[using_itemlist])
        pos_pos=np.append(x_pos, y_pos) #Center position
        for i_run in range(len(runlist)):

            thislim=copy.deepcopy(runlist['lim_params'][i_run])
            thislim[:,3]+=self.re_cut ## Add manual cut
            thislim=np.vstack(thislim)
            #this_pos_pos=pos_pos[pos_pos<len(thislim)]  # cut higher components
            #(e.g,) Total galfit_itemlist is up to 3rd comp. but this run has only 2nd comp.
            #thislim[this_pos_pos]+=100 # Center position

            thisvals=value_data[i_run][:len(thislim)] # cut higher components
            boundlist=np.where((thisvals<=thislim[:,0]) | (thisvals>=thislim[:,1]))[0]

            if(len(boundlist)!=0):
                boundlist=np.array(using_itemlist)[boundlist]
                self.bound[boundlist][i_run]=2

        ## Whether use constraint or not
        if(hasattr(runlist, "__len__")):
            removelist=np.where(runlist['use_lim']==0)
            self.bound[removelist]=0

        self.Data.war=dharray.array_quickname(dharray.array_flexible_to_simple(self.Data.war)
                                      + dharray.array_flexible_to_simple(self.bound),
                                      names=self.Data.war.dtype.names, dtypes=self.Data.war.dtype)


    def check_strange_radius(self, snr_crit=1, reff_snr_n_cut=1e9):

        """
        Descr - Check Reff by comparing uncertainty
        INPUT
         ** runlist : Automatically get limits (default : None)
        """
        if(snr_crit==None):
            self.Data.reff_strange=np.full(len(self.Data.val), False)
            self.Data.reff_snr=np.full(len(self.Data.val), np.nan)
            return
        if(self.silent==False):
            print("\n◯ Check Reff")

        using_itemlist=np.copy(self.galfit_itemlist)
        n_pos=np.where(np.char.find(using_itemlist, 'N')>0)[0]
        cut_using_itemlist=using_itemlist[n_pos]
        n_data=dharray.array_flexible_to_simple(self.Data.val[cut_using_itemlist]) ## n val

        reff_pos=np.where(np.char.find(using_itemlist, 'RE')>0)[0]
        cut_using_itemlist=using_itemlist[reff_pos]

        value_data=dharray.array_flexible_to_simple(self.Data.val[cut_using_itemlist]) ## Reff val
        error_data=dharray.array_flexible_to_simple(self.Data.err[cut_using_itemlist]) ## Reff err


        with np.errstate(divide='ignore'):
            self.Data.reff_snr=value_data/error_data # SNR
        with warnings.catch_warnings():  ## Ignore warning. Nan warning is natural in this case.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            usingdat=copy.deepcopy(self.Data.reff_snr)
            usingdat[np.isnan(error_data) & np.isfinite(value_data)]=0   # It has the value, but error is inf.
            usingdat[n_data>reff_snr_n_cut]=np.nan ## sersic index higher than the given value -> not apply SNR cut
            for i in range (len(usingdat[0])): ## Loop for a component
                self.Data.war[cut_using_itemlist[i]][usingdat[:,i]<snr_crit]+=100
            sumarray=np.nanmin(usingdat, axis=1)


        self.Data.reff_strange=(sumarray<snr_crit) # Lower than the criteria
        self.Data.reff_nan=np.isnan(error_data) & np.isfinite(value_data) # It has the value, but error is inf.
        self.Data.reff_nan=np.sum(self.Data.reff_nan, axis=1)
        self.Data.reff_strange[self.Data.reff_nan!=0]=True
        # above: We want to avoid PSF only data (no Re) being catagorized as an error
        # cf) np.isnan(sumarray) --> PSF is now reff_strange

    def check_strange_n(self, n_crit=2):
        """
        Descr - Check specific value of n
        INPUT
         ** runlist : Automatically get limits (default : None)
        """
        if(n_crit==None):
            self.Data.n_strange=np.full(len(self.Data.val), False)
            return

        using_itemlist=np.copy(self.galfit_itemlist)
        n_pos=np.where(np.char.find(using_itemlist, 'N')>0)[0]
        cut_using_itemlist=using_itemlist[n_pos]  ##"2_N", "3_N", "4_N", ...
        n_data=dharray.array_flexible_to_simple(self.Data.val[cut_using_itemlist]) ## n val (array)
        model_ss = (np.isfinite(n_data[:,0]) & np.isfinite(n_data[:,1])) ## Only SS (& SS+PSF)model
        n_strange_raw=(n_data==n_crit)
        self.Data.n_strange=(np.sum(n_strange_raw, axis=1).astype(bool) & model_ss)

        for i in range (len(n_strange_raw[0])):
            # target=(n_strange_raw[:,i]==True)
            target=((n_strange_raw[:,i]==True) & model_ss)
            self.Data.war[cut_using_itemlist[0]][target]+=100



    def find_best(self, remove_warn=True, comp_crit=None):
        if(self.silent==False): print("\n● Find the best submodel")
        if(comp_crit==None): comp_crit=self.comp_crit
        chi2list=self.Data.val[comp_crit]
        usingindex=np.arange(len(chi2list))
        if(remove_warn==True):
            if(self.silent==False): print(">> Without warning")
            usingindex=usingindex[(self.Data.war_tot==0)
                                   & (self.Data.reff_strange==False)
                                   & (self.Data.n_strange==False)]
            ## If no results
            if(len(usingindex)==0):
                if(self.silent==False): print(">> No results. Finding the best regardless of the warning")
                return self.find_best(remove_warn=False)
            ## If All data is NaN
            checknan=np.sum(np.isnan(chi2list[usingindex]))
            if(checknan==len(usingindex)):
                if(self.silent==False): print(">> No results. Finding the best regardless of the warning")
                return self.find_best(remove_warn=False)
        else:
            ## Reff_Strange filter
            good_data=usingindex[(self.Data.reff_strange==False)
                                 & (self.Data.n_strange==False)
                                 & np.isfinite(chi2list[usingindex])]
            ## If no results
            if(len(good_data)==0):
                if(self.silent==False): print(">> All data is ruled out by the Reff SNR criterion.")
                pass
            else: usingindex=usingindex[(self.Data.reff_strange==False) & (self.Data.n_strange==False)]

        min_chi=np.nanmin(chi2list[usingindex])
        minpos=np.where(chi2list[usingindex]==min_chi)[0]
        return usingindex[minpos]

    def find_near_best(self, min_chi, tolerance=0, comp_crit='RChi_50'):
        chi2list=self.Data.val[comp_crit]
        if(hasattr(min_chi, "__len__")): min_chi=np.nanmin(min_chi)
        self.chi2_diff=np.abs(chi2list-min_chi)/min_chi
        nearpos=np.where(self.chi2_diff<tolerance)[0] ## Fraction
        return nearpos

    def save_data(self, fna_output='saved_resgal.dat', fn_output='', return_array=False):
        if(fna_output==None): fn_output=None
        elif(fn_output==''): fn_output=self.dir_work_galfit+fna_output
        if(self.silent==False): print("\n● Save the result : ", fn_output)
        return self.Data.save_data(fn_output=fn_output, add_header=self.dir_work_galfit, return_array=return_array)

    ##========================== Show data & display ============================
    def show_data(self, only_val=False, hide_nan=True, hide_PA=True, hide_fail_read=True):
        if(np.sum(self.Data.is_file_exist)==0):
            print(">> No data!")
            return
        if(hide_nan==True): newitemlist=self.safe_itemlist
        else: newitemlist=np.array(self.itemlist) ## All the items
        if(hide_PA==True):
            drop=np.where((newitemlist=='2_AR') | (newitemlist=='2_PA')
            | (newitemlist=='3_AR') | (newitemlist=='3_PA'))
            newitemlist=np.delete(newitemlist, drop)

        newitemlist=np.append(['group_ID', 'is_swap'], newitemlist)
        if(hide_fail_read==True): newfilelist=self.Data.is_file_exist
        else: newfilelist=np.full(len(self.Data.is_file_exist), True) ## All the files
        errorlist=self.Data.namelist[self.Data.war_tot.astype(bool)]
        lowsnrlist=self.Data.namelist[self.Data.reff_strange.astype(bool) | self.Data.n_strange.astype(bool)]

        newitemlist_chi2=np.append(newitemlist, self.chi2_itemlist)
        self.df_display(self.Data.val[newfilelist][newitemlist_chi2], newfilelist, errorlist, lowsnrlist, caption='Values')

        if(only_val==False):
            self.df_display(self.Data.err[newfilelist][newitemlist], newfilelist, errorlist, lowsnrlist, caption='Error')
            self.df_display(self.Data.war[newfilelist][newitemlist], newfilelist, errorlist, lowsnrlist, caption='Warning', is_limit_format=True)

    def df_display(self, data, newfilelist, errorlist, lowsnrlist, caption='datatype', is_limit_format=False):
        df=pd.DataFrame(data, index=self.Data.namelist[newfilelist])
        if(is_limit_format==True): formatting='{:.0f}'
        else: formatting='{:f}'
        df=df.style.apply(lambda x: ['background: lightyellow' if x.name in self.Data.namelist[self.best_near]
                                      else '' for i in x],
                           axis=1)\
            .apply(lambda x: ['background: bisque' if x.name in self.Data.namelist[self.best_warn]
                              else '' for i in x],
                   axis=1)\
            .apply(lambda x: ['background: lightgreen' if x.name in self.Data.namelist[self.best]
                                      else '' for i in x],
                           axis=1)\
            .apply(lambda x: ['color: red' if x.name in errorlist
                          else '' for i in x],
               axis=1)\
            .apply(lambda x: ['font-style: italic; font-weight:bold' if x.name in lowsnrlist
                          else '' for i in x],
               axis=1)\
            .format(formatting)\
            .set_caption(caption).set_table_styles([{
            'selector': 'caption',
            'props': [
                ('color', 'red'),
                ('font-size', '16px')
            ]
        }])
        display(df)

    def make_swaplist(self, mode='RE'):
        if(mode=='RE'):
            return np.where(self.Data.val['2_RE']<self.Data.val['3_RE'])[0] # Larger Re
        elif(mode=='N'):
            return np.where(self.Data.val['3_N']<self.Data.val['2_N'])[0] #Lower N
        elif(mode=='MU'):
            swaplist=np.zeros(len(self.Data.val), dtype=bool)
            for i in range (len(self.Data.val)):
                if(np.isnan(self.Data.val[i]['3_N'])): continue
                s2=Sersic1D(amplitude=mu2pix(self.Data.val[i]['2_MU_E'], plate_scale=self.plate_scale, zeromag=self.zeromag),
                                            r_eff=self.Data.val[i]['2_RE'], n=self.Data.val[i]['2_N'])
                s3=Sersic1D(amplitude=mu2pix(self.Data.val[i]['3_MU_E'], plate_scale=self.plate_scale, zeromag=self.zeromag),
                            r_eff=self.Data.val[i]['3_RE'], n=self.Data.val[i]['3_N'])
                re=np.nanmax([self.Data.val[i]['2_RE'], self.Data.val[i]['3_RE']])
                swaplist[i]=(s2(re)<s3(re))  #Outer region -> brighter
            return np.where(swaplist==True)
        elif(mode=='2MU'):
            swaplist=np.zeros(len(self.Data.val), dtype=bool)
            for i in range (len(self.Data.val)):
                if(np.isnan(self.Data.val[i]['3_N'])): continue
                s2=Sersic1D(amplitude=mu2pix(self.Data.val[i]['2_MU_E'], plate_scale=self.plate_scale, zeromag=self.zeromag),
                            r_eff=self.Data.val[i]['2_RE'], n=self.Data.val[i]['2_N'])
                s3=Sersic1D(amplitude=mu2pix(self.Data.val[i]['3_MU_E'], plate_scale=self.plate_scale, zeromag=self.zeromag),
                            r_eff=self.Data.val[i]['3_RE'], n=self.Data.val[i]['3_N'])
                re=2*np.nanmax([self.Data.val[i]['2_RE'], self.Data.val[i]['3_RE']])
                swaplist[i]=(s2(re)<s3(re))  #Outer region -> brighter
            return np.where(swaplist==True)
        elif(mode=='3MU'):
            swaplist=np.zeros(len(self.Data.val), dtype=bool)
            for i in range (len(self.Data.val)):
                if(np.isnan(self.Data.val[i]['3_N'])): continue
                s2=Sersic1D(amplitude=mu2pix(self.Data.val[i]['2_MU_E'], plate_scale=self.plate_scale, zeromag=self.zeromag),
                            r_eff=self.Data.val[i]['2_RE'], n=self.Data.val[i]['2_N'])
                s3=Sersic1D(amplitude=mu2pix(self.Data.val[i]['3_MU_E'], plate_scale=self.plate_scale, zeromag=self.zeromag),
                            r_eff=self.Data.val[i]['3_RE'], n=self.Data.val[i]['3_N'])
                re=3*np.nanmax([self.Data.val[i]['2_RE'], self.Data.val[i]['3_RE']])
                swaplist[i]=(s2(re)<s3(re))  #Outer region -> brighter
            return np.where(swaplist==True)


class Chi2Update():
    """
    Descr - Calculate chi2 and update Galfit fits
    ** Input files in the list should share the same simga & masking images
    ** See cal_chi2()
    INPUT
     - fnlist : file list (those share the same sigma & masking files)
     - fn_sigma : Sigma file (A signle file for all files in fnlist).
                  If None : The code will make a flat image
     * fn_masking : Masking file (A single file for all files in fnlist).
                    If None : Do not mask
     * radius : The region for calculating chi2 (Radius)
                     If -1, calculate for all region (default = -1)
     * image_size : Size of images (default = 200)
     * repair_fits : Repair the Galfit header if it has problem (Default : True)
     * chi_item_name : Input chi2 name as a new header. (default : 'F')
       ** 'Chi_'+'chi_item_name' will be added for the chi2 value
       ** 'RChi_'+'chi_item_name' will be added for the reduced chi2 value
       ** 'NDOF_'+'chi_item_name' will be added for the number of DOF
       ** 'LIK_'+'chi_item_name' will be added for the likehood value
     * fn_base_noext : Base Path when the input file does not have extensions
       ** In this case, image file will be loaded from 'fn_base_noext' + header['DATAIN']
     * Overwrite : (default : False)
    """
    def __init__(self,
                 MC2fitSettingClass,
                 fnlist,
                 fn_sigma,
                 fn_masking=None,
                 **kwargs):


        self.fnlist=fnlist
        self.fn_sigma=fn_sigma
        self.fn_masking=fn_masking
        self.repair_fits=True
        self.image_size=200
        self.chi2_radius_list=[-1, 50]
        self.chi_item_namelist=['F', '50']
        self.fn_base_noext=''
        self.overwrite=False,
        self.silent=True

        try: self.__dict__.update(MC2fitSettingClass.__dict__)
        except: pass
        self.__dict__.update(kwargs)
        self._post_init()


    def _post_init(self):
        self.image_size=dharray.value_repeat_array(self.image_size, 2)
        self._get_data()
        for fn in self.fnlist:
            self._main_work(fn)

    def _get_data(self):
        ## Multi-radius
        if(hasattr(self.chi2_radius_list, "__len__")):
            self.Nseq=len(self.chi2_radius_list)
        else:
            self.chi2_radius_list=np.array([self.chi2_radius_list])
            self.Nseq=1

        if(self.silent==False): print("Radius", self.chi2_radius_list)
        self.chi_item_namelist=dharray.repeat_except_array(self.chi_item_namelist, self.Nseq)

        ## Input sigma
        if(self.fn_sigma==None): # If no sigma -> Make a flat image
            self.sig=np.full((self.image_size), 1)
        else: self.sig=fits.getdata(self.fn_sigma)

        ## Input masking
        if(self.fn_masking==None): self.mask=None
        else: self.mask=fits.getdata(self.fn_masking)

    def _main_work(self, fn):
        ## File read
        try:
            GDat=ReadGalfitData(fn, fn_base_noext=self.fn_base_noext, repair_fits=self.repair_fits)
        except:
            if(self.silent==False): print(">> Cannot read the file:", fn)
            return

        ## Check previous work
        if(self.overwrite==False):
            try:  # Header exists --> Next file
                GDat.header['LIK_'+self.chi_item_namelist]
                return
            except: # Do work!
                pass


        if(GDat.Next>2): ext_save=2
        else: ext_save=0

        ## ============= Save Log =============
        fits.setval(fn, 'postpro', value="============== Post Processing ==============", after='LOGFILE', ext=ext_save)
        fits.setval(fn, 'pfn_sig', value=self.fn_sigma, after='postpro', ext=ext_save)
        fits.setval(fn, 'pfn_mask', value=self.fn_masking, after='pfn_sig', ext=ext_save)

        ## ============= Save results =======================
        for seq in range (self.Nseq):
            fits.setval(fn, 'Field'+self.chi_item_namelist[seq], value="--"*20, ext=ext_save)

            ## Main calc
            chi2, Npixel=cal_chi2(GDat.resi, self.sig, self.mask, self.chi2_radius_list[seq])

            if(Npixel<=0): chi2=1e7-1
            if(np.isinf(chi2)): chi2=1e7-1
            fits.setval(fn, 'Chi_'+self.chi_item_namelist[seq], value=chi2, ext=ext_save)

            p=GDat.header['NFREE']
            ndof=Npixel-p
            fits.setval(fn, 'NDOF_'+self.chi_item_namelist[seq], value=ndof, ext=ext_save)

            if((ndof>10) & (np.isfinite(chi2))):
                fits.setval(fn, 'RChi_'+self.chi_item_namelist[seq], value=chi2/ndof, ext=ext_save)
                aic=chi2+2*p+(2*p*(p+1))/(Npixel-p-1)
                fits.setval(fn, 'AIC_'+self.chi_item_namelist[seq], value=aic, ext=ext_save)
                # https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Statistics_with_Technology_2e_(Kozak)/11%3A_Chi-Square_and_ANOVA_Tests/11.02%3A_Chi-Square_Goodness_of_Fit
                likelihood=1-stats.chi2.cdf(chi2, ndof)
                fits.setval(fn, 'LIK_'+self.chi_item_namelist[seq], value=likelihood, ext=ext_save)

            else:
                fits.setval(fn, 'RChi_'+self.chi_item_namelist[seq], value=1e7-1, ext=ext_save)
                fits.setval(fn, 'AIC_'+self.chi_item_namelist[seq], value=1e7-1, ext=ext_save)
                fits.setval(fn, 'LIK_'+self.chi_item_namelist[seq], value=-(1e7-1), ext=ext_save)

        if(self.silent==False): print(">>", fn, "Done")

def cal_chi2(dat, sig, mask=None, radius=-1):
    """
    Descr - calculate chi2 (See also. cal_update_chi2)
    INPUT
     - dat : data array
     - sig : sigma array
     * mask : masking array. It will mask all pixels having non-zero. If None, do not mask. (Default : None)
     * radius : The region for calculating chi2 (Radius)
                     If -1, calculate for all region (default : -1)
    """
    if(hasattr(mask, "__len__")): mask=mask.astype(bool)
    mdat=np.ma.masked_array(dat, mask)
    if(radius<0): innerpart=np.where(mdat.mask==False) #return np.sum((mdat**2/sig**2)),
    else:
        center=int(len(dat)/2)
        radius=int(radius)
        mx, my=np.mgrid[0:len(dat),0:len(dat)]
        innerpart=np.where(((mx-center)**2 + (my-center)**2 <= radius**2) & (mdat.mask==False))
        if(len(innerpart[0])==0): return 0,0

    try: return np.sum((mdat[innerpart]**2/sig[innerpart]**2)), len(innerpart[0])
    except: return np.nan,0


class MC2fitPostProcessing():
    """

    """
    def __init__(self,
                 MC2fitSettingClass,
                 RunlistClass,
                 dir_work,
                 **kwargs):


        ## path setting
        self.dir_work=dir_work
        self.proj_folder='galfit/'
        self.fn_base_noext = ''

        ## print setting
        self.silent=False

        ## Parameters
        self.chi2_radius_list = [-1, 50]
        self.chi_item_namelist = ['F', '50']
        self.repair_fits = True
        self.centermag = 'sersic'
        self.sersic_res = 500

        ## Class
        self.MC2fitSettingClass=MC2fitSettingClass
        self.RunlistClass=RunlistClass

        try: self.__dict__.update(MC2fitSettingClass.__dict__)
        except: pass
        self.__dict__.update(kwargs)
        self._post_init()


    def _post_init(self):
        self.plate_scale=dharray.value_repeat_array(self.plate_scale, 2)
        self.image_size=dharray.value_repeat_array(self.image_size, 2)


        ## Sigma
        if((self.fni_sigma==None) | (self.fni_sigma=='none')):
            self.fn_sigma = None
        else: self.fn_sigma=self.dir_work+self.fni_sigma
        ## Mask
        if((self.fni_masking==None) | (self.fni_masking=='none')):
            self.fni_masking = None
        else: self.fn_masking=self.dir_work+self.fni_masking

        self.check_params()
        self.post_chi2() ## Calc chi2

        ##================ Centermag ========================
        if(self.centermag!=None):
            self.ResThis=MC2fitResult(group_id=None,
                                      runlist=self.RunlistClass.runlist,
                                      ignore_no_files=True,
                                      comp_autoswap=False, comp_4th_nsc_autoswap=False, comp_4th_ss_autoswap=False,
                                      **self.__dict__)


            if(self.centermag=='sersic'):
                self.centermag_sersic()
            elif(self.centermag=='galfit'):
                self.centermag_galfit()

    def check_params(self):
        if(self.silent==False): print(">> Check params")
        if(hasattr(self.chi2_radius_list, "__len__")):
            self.Nseq=len(self.chi2_radius_list)
        else: self.Nseq=1
        if(self.silent==False): print(">>>> Nseq : ", self.Nseq)
        if(type(self.chi_item_namelist)==str):
            Nname=1
        else: Nname=len(self.chi_item_namelist)
        if(self.silent==False): print(">>>> Nchi_item_name : ", Nname)

        if(Nname!=self.Nseq):
            print("Warning! The input length is strange!")
            print(">> chi_radius : ", self.Nseq)
            print(">> chi_item_name : ", Nname)

    def post_chi2(self):
        ##chi2
        if(self.silent==False): print(">> Post chi2")
        fnlist=dharray.array_attach_string(self.RunlistClass.runlist['name'], 'result_', add_at_head=True)
        fnlist=dharray.array_attach_string(fnlist, self.dir_work+self.proj_folder, add_at_head=True)
        fnlist=dharray.array_attach_string(fnlist, '.fits')
        self.fnlist=np.array(fnlist)
        Chi2Update(**self.__dict__)

    def centermag_sersic(self):
        if(self.silent==False): print(">> Sersic Centermag")
        tmag2, sersiclist2=add_totalmag(self.ResThis.Data.val, comp=2,
                                        sersic_res=self.sersic_res, plate_scale=self.plate_scale[0], zeromag=self.zeromag)
        tmag3, sersiclist3=add_totalmag(self.ResThis.Data.val, comp=3,
                                        sersic_res=self.sersic_res, plate_scale=self.plate_scale[0], zeromag=self.zeromag)
        ## INPUT

        for i, fn in enumerate (self.fnlist[sersiclist2]):
            hdul=fits.open(fn)
            if(len(hdul)>1): save_ext=2
            else: save_ext=0
            inputvalue="%.4f"%tmag2[i]
            inputvalue+=" +/- 0.0000"
            fits.setval(fn, '2_MAG', value=inputvalue, ext=save_ext, after='2_MU_E', comment='Integrated magnitude (post-processed)')

        for i, fn in enumerate (self.fnlist[sersiclist3]):
            if(len(hdul)>1): save_ext=2
            else: save_ext=0
            inputvalue="%.4f"%tmag3[i]
            inputvalue+=" +/- 0.0000"
            fits.setval(fn, '3_MAG', value=inputvalue, ext=save_ext, after='3_MU_E', comment='Integrated magnitude (post-processed)')

def add_totalmag(datarray, comp=2, sersic_res=500, plate_scale=0.262, zeromag=22.5):
    ## Integral mag
    if(comp==2):
        sersiclist=np.where(np.isfinite(datarray['2_MU_E']))
        usingdata=datarray[sersiclist]
        ## Old version
        # intmag=sersic_integral2D(mu_e=usingdata['2_MU_E'], reff=usingdata['2_RE'],
        #                   n=usingdata['2_N'], ar=usingdata['2_AR'], plate_scale=plate_scale, res=sersic_res)
        intmag = dhmc2fitsub.mu_to_mag(mu_e=usingdata['2_MU_E'], Re=usingdata['2_RE'],
                          n=usingdata['2_N'], ar=usingdata['2_AR'], plate_scale=plate_scale, zeromag=zeromag)

    elif(comp==3):
        sersiclist=np.where(np.isfinite(datarray['3_MU_E']))
        usingdata=datarray[sersiclist]
        # intmag=sersic_integral2D(mu_e=usingdata['3_MU_E'], reff=usingdata['3_RE'],
        #                   n=usingdata['3_N'], ar=usingdata['3_AR'], plate_scale=plate_scale, res=sersic_res)
        intmag = dhmc2fitsub.mu_to_mag(mu_e=usingdata['3_MU_E'], Re=usingdata['3_RE'],
                          n=usingdata['3_N'], ar=usingdata['3_AR'], plate_scale=plate_scale, zeromag=zeromag)
    return intmag, sersiclist[0]

class MC2fitPostProcessing_Mulcore():
    """

    """
    def __init__(self,
                 MC2fitSettingClass,
                 RunlistClass,
                 DirInfoClass,
                 **kwargs):

        ## Multicore setting
        self.use_try=True
        self.Ncore=2
        self.show_multicore=True
        self.show_progress=-1

        ## Class
        self.MC2fitSettingClass=MC2fitSettingClass
        self.RunlistClass=RunlistClass
        self.DirInfoClass=DirInfoClass

        try: self.__dict__.update(MC2fitSettingClass.__dict__)
        except: pass
        self.__dict__.update(kwargs)
        self._post_init()
        if(self.Ncore!=1):
            self._show_multicore_result()


    def _show_multicore_result(self):
        Nfailed=np.sum(np.isnan(self.multi_res))
        print("======== Multicore result ========")
        print("● Succeed: %d / %d"%(np.sum(self.multi_res==0), len(self.DirInfoClass.dir_work_list)))
        print("● Failed: %d / %d"%(Nfailed, len(self.DirInfoClass.dir_work_list)))
        if(Nfailed>0):
            print(">> See where: self.multi_res")
            print(">> To debug, use Ncore=1, and use_try = False")


    def _post_init(self):
#         self.plate_scale=dharray.value_repeat_array(self.plate_scale, 2)
#         self.image_size=dharray.value_repeat_array(self.image_size, 2)
#         if(hasattr(self.chi2_radius_list, "__len__")==False): ## Const
#             self.chi2_radius_list=np.repeat(self.chi2_radius_list, len(DirInfo.dir_work_list)) ## 1D array
#         if(hasattr(self.chi2_radius_list, "__len__")):
#             if(np.ndim(self.chi2_radius_list)==1): ## 1D array
#                 self.chi2_radius_list=np.array([self.chi2_radius_list]).T ## 2D array (N*1)

        global sub_run_process

        def sub_run_process(i):
            dir_work=self.DirInfoClass.dir_work_list[i]
            fn_done=dir_work+self.proj_folder+"done_post.dat"
            if(self.fast_skip==True):
                if(os.path.exists(fn_done)): return 0

            if(i%self.remove_galfit_interval==0):
                existlist1=glob.glob('./galfit.*')
                for fb in existlist1:
                    try: os.remove(fb)
                    except: pass
                try: os.remove('./fit.log')
                except: pass

#             dum=dhgalfit.PostProcessing(proj_folder=proj_folder, Runlist=Runlist, dir_work=dir_work,
#                                 fna_sigma=fna_sigma, fna_masking=fna_masking,
#                                 fna_image=fna_image, fna_psf=fna_psf,
#                                 chi_radius=chi_radius[i], repair_fits=repair_fits, image_size=image_size,
#                                 chi_item_name=chi_item_name,
#                                 fn_base_noext=fn_base_noext,
#                                 plate_scale=plate_scale, zeromag=zeromag,
#                                 chi2_full_image=chi2_full_image,
#                                 centermag=centermag, sersic_res=500,
#                                 overwrite=overwrite, silent=silent, print_galfit=print_galfit)

            MC2fitPostProcessing(dir_work=dir_work,
                      **self.__dict__
                     )


            np.savetxt(fn_done, np.array(['done']), fmt="%s")
            return 0

        self.multi_res=mulcore.multicore_run(sub_run_process,
                                             len(self.DirInfoClass.dir_work_list),
                                             Ncore=self.Ncore,
                                             use_try=self.use_try,
                                             show_progress=self.show_progress,
                                             debug=self.show_multicore)
        self.multi_res=np.array(self.multi_res)

##=============================================


class ReadGalfitData():
    def __init__(self, fn, ext_list=[1,2,3], fn_base_noext='', repair_fits=False,
                 replace_folder=None):
        """
        Descr - Read galfit output file
        INPUT
         - fn : file name
         - ext : Extensions. int or list of ints.
           ** (0 : No data)
           ** 1 : Original image
           ** 2 : Model image
           ** 3 : Residual image ([2] - [1])
         * fn_base_noext : Base Path when the input file does not have extensions
           ** In this case, image file [1] will be loaded from 'fn_base_noext' + header['DATAIN']
         * replace_folder: Change the path for image file.
           ** e.g.) replace_folder = ['SMDG', 'SMDG_original']
        """

        hdul=fits.open(fn)
        self.Next=len(hdul)
        if(repair_fits==True):
            hdul.verify('silentfix')
            ## Re-write the file if the code above does not work
            #hdul.writeto(fn, overwrite=True)
        hdul.close()
        if(self.Next>2): ## If the input file has extensions
            self.header=fits.getheader(fn, ext=2)
            if(np.isin(1, ext_list)):
                self.image=fits.getdata(fn, ext=1)
                self.header_img=fits.getheader(fn, ext=1)
            if(np.isin(2, ext_list)): self.model=fits.getdata(fn, ext=2)
            if(np.isin(3, ext_list)): self.resi=fits.getdata(fn, ext=3)

        else:
            self.header=fits.getheader(fn)
            path=fn_base_noext+self.header['DATAIN']
            if(hasattr(replace_folder, "__len__")==True):
                path=path.replace(replace_folder[0], replace_folder[1])
            if(np.sum(np.isin([1,3], ext_list))):
                self.image=fits.getdata(path)
                self.header_img=fits.getheader(path)
            if(np.sum(np.isin([2,3], ext_list))): self.model=fits.getdata(fn)
            if(np.isin(3, ext_list)): self.resi = self.image - self.model



#============================= Drawing ==============================
def show_FITSimage(fn, ext_data=0, ext_wcs=0, use_zscale=True, use_wcs=True):
    image_data = fits.getdata(fn, ext=ext_data)
    image_header = fits.getheader(fn, ext=ext_wcs)
    if(use_wcs==True):
        wcs = WCS(image_header)
        ax = plt.subplot(projection=wcs)
    else: ax = plt.subplot()

    if(use_zscale==True):
        norm = ImageNormalize(image_data, interval = ZScaleInterval())
        im = ax.imshow(image_data, origin='lower', norm=norm, cmap='bone')
    else:
        image_data=np.log10(image_data)
        im = ax.imshow(image_data, origin='lower', cmap='bone')


class MC2fitDrawing():
    """

    """
    def __init__(self,
                 MC2fitSettingClass,
                 **kwargs):

        ## path setting
        self.fn='./TestRun/result_test.fits'
        self.fn_masking=None,

        ## Figure setting
        self.gs=None  ## previous gridspec
        self.data_index=0 ## Starting point
        self.figsize=(13,4)

        self.plot_residual=True
        self.titlelist=['Image', 'Model', 'Residual']
        self.cmap='bone'
        self.resi_scale=0.2 ## Residual scale (color range compared to that of the original)
        self.resi_scale_center=None  ## If none, use the median value as the center

        self.image_offset=0
        self.logscale=False
        self.log_image_offset=50

        ## ReadGalfitData
        self.fn_base_noext=''
        self.replace_folder=None

        ## Masking
        self.show_masking=True
        self.show_masking_center=False
        self.masking_center_color='r'
        self.masking_center_size=10
        self.fill_masking=True    #Masking


        ## Panel Info
        self.show_htitle=True
        self.show_vtitle=True
        self.descr=''
        self.c_vtitle='k'
        self.header_vtitle=None     #descr
        self.comp_crit='CHI2NU'
        self.show_ticks=True
        self.scale_percentile=None
        self.vmin=None
        self.vmax=None
        self.show_colorbar=True
        self.xlim=None
        self.ylim=None
        self.Nminor_tick=5

        try: self.__dict__.update(MC2fitSettingClass.__dict__)
        except: pass
        self.__dict__.update(kwargs)
        self._post_init()


    def _post_init(self):
        self._setting()


    def _setting(self):
        ## Gridspec Setting
        if(self.gs==None):
            self.plot=plt.figure(figsize=self.figsize)
            if(self.plot_residual==True): self.gs=gridspec.GridSpec(1,3)
            else: self.gs=gridspec.GridSpec(1,1)

        ## File open
        self.Gdat=ReadGalfitData(self.fn, fn_base_noext=self.fn_base_noext, replace_folder=self.replace_folder)

        ## Masking file
        if(self.fn_masking!=None): self.masking=fits.getdata(self.fn_masking)
        else:
            self.show_masking=False
            self.show_masking_center=False

        ## Positions of Masking centers
        if(self.show_masking_center==True):
            maskid=np.unique(self.masking)
            maskid=maskid[maskid!=0] ## 0: Non-masked

            centerpos=np.full((len(maskid), 2), np.nan)
            for i, thismask in enumerate(maskid): # loop for [1, maxmask]
                target=np.where(self.masking==thismask)
                centerpos[i]=np.nanmean(target, axis=1)


        #=============== Galfit subfiles - 1 : Obs / 2 : Model / 3 : Residual ============
        if(self.plot_residual):
            usingmethod_list=['image', 'model', 'resi']
        else: usingmethod_list=['image']

        self.axlist=np.zeros(len(usingmethod_list), dtype=object)
        for i, method in enumerate(usingmethod_list):
            ax=plt.subplot(self.gs[self.data_index*3+i])
            self.axlist[i]=ax
            imagedata=getattr(self.Gdat, method)

            if(self.show_masking_center==True): plt.scatter(centerpos[:,1], centerpos[:,0],
                                                            c=self.masking_center_color,
                                                            s=self.masking_center_size)

            if(method=='resi'): self._draw_panel(ax, imagedata, is_resi=True)
            else: self._draw_panel(ax, imagedata, is_resi=False) ## Main drawing
            self._panel_setting(ax, i)


    def _draw_panel(self, ax, imagedata, is_resi=False):

        if(is_resi==False): ## Not residual map
            imagedata=imagedata + self.image_offset
            if(self.logscale==True):
                imagedata=np.log10(imagedata+self.log_image_offset)

        if(self.show_masking==True):
            median=np.nanmedian(imagedata)
            maskedimage=np.ma.masked_array(imagedata, self.masking, fill_value=median) #masking --> median value
            if(self.fill_masking==True): imagedata=maskedimage.filled()
            else: imagedata=maskedimage


        ## Plot & Scale Min, Max
        if(is_resi): ## For residual map
            if(self.resi_scale<0):
                self.resi_scale=-self.resi_scale
                #self.cmap='bone_r'

            if(self.logscale): newscale=(10**self.vmax-10**self.vmin)*self.resi_scale/2 ## Linear scale
            else: newscale=(self.vmax-self.vmin)*self.resi_scale/2

            if(self.resi_scale_center==None): self.resi_scale_center=np.nanmedian(imagedata)
            vmin=self.resi_scale_center-newscale
            vmax=self.resi_scale_center+newscale

        else:
            vmin=self.vmin
            vmax=self.vmax

        if(self.scale_percentile==None):
            a=ax.imshow(imagedata, cmap=self.cmap, vmin=vmin, vmax=vmax, origin='lower')
        else:
            newscale=[np.percentile(imagedata, 50-self.scale_percentile), np.percentile(imagedata, 50+self.scale_percentile)]
            a=ax.imshow(imagedata, cmap=self.cmap, origin='lower', vmin=newscale[0], vmax=newscale[1])

        self.vmin, self.vmax=a.get_clim() ## Save min, max -> Use it in next plots

        if(self.show_colorbar): plt.colorbar(a)


    def _panel_setting(self, ax, i):
        # Show titles
        if(self.show_htitle==True): plt.title(self.titlelist[i], size=15, fontname='DejaVu Serif', fontweight='semibold')

        if(i==0): ## Only the leftmost plot
            if(self.show_vtitle==True):
                try:
                    if(self.header_vtitle==None): self.header_vtitle=self.comp_crit
                    text=self.descr+"\n"+self.header_vtitle+r" = %.2f"%self.Gdat.header[self.comp_crit]
                    plt.ylabel(text, size=15, c=self.c_vtitle, fontname='DejaVu Serif', fontweight='semibold')
                except: pass


        imgsize=np.shape(self.Gdat.image)
        if(self.xlim==None): self.xlim=(0, imgsize[0])
        if(self.ylim==None): self.ylim=(0, imgsize[1])
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)

        if(self.show_ticks==True):
            ax.set_xticks([self.xlim[0], np.mean(self.xlim), self.xlim[1]])
            ax.set_yticks([self.ylim[0], np.mean(self.ylim), self.ylim[1]])
            ax.tick_params(labelsize=10, width=1, length=6, axis='both', direction='in', right=True, top=True)
            ax.xaxis.set_minor_locator(AutoMinorLocator(self.Nminor_tick))
            ax.yaxis.set_minor_locator(AutoMinorLocator(self.Nminor_tick))
        else:
            ax.set_xticks([])
            ax.set_yticks([])





def auto_drawing(fnlist, descr_list, suptitle='Name : Gal 1',                                        # Descr
                 fn_masking=None, show_masking=True, show_masking_center=True, fill_masking=True,    # Masking
                 scale_percentile=None, fix_row=False, dpi=100, fn_save_image='UDG_example.png'):

    # Check chi^2
    chi_list=np.full(3, np.nan)
    for i in range (len(fnlist)):
        try:
            hdu=fits.open(fnlist[i])
            chi_list[i]=hdu[2].header['CHI2NU']
        except: chi_list[i]=np.nan

    # Drawing
    fig=plt.figure(figsize=(13,12))
    gs=gridspec.GridSpec(3,3)
    gs.update(hspace=0.05)

    if(fix_row==True): Nrow=3
    else: Nrow=len(fnlist)

    for i in range (Nrow):
        if(i==0): show_htitle=True
        else: show_htitle=False
        if(chi_list[i]==np.nanmin(chi_list)): c_vtitle='r'
        else: c_vtitle='k'
        try: drawing_galfit(fnlist[i], gs=gs, data_index=i, descr=descr_list[i],
                       show_htitle=show_htitle, c_vtitle=c_vtitle, scale_percentile=scale_percentile,
                        fn_masking=fn_masking, show_masking=show_masking,
                            show_masking_center=show_masking_center, fill_masking=fill_masking
                           )
        except:
            ax0=plt.subplot(gs[i*3:(i+1)*3]) ##Empty box
            plt.axis('off')

    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    plt.suptitle(suptitle, fontname='DejaVu Serif', fontweight='semibold', verticalalignment='center')
    if(fn_save_image!=None): plt.savefig(fn_save_image, bbox_inches='tight', facecolor='w', dpi=dpi)





#===================== Codes for ResultGalfit ==================
def read_galfit_header(fn, itemlist):
    result=np.full(len(itemlist), np.NaN, dtype=object)
    Next=0
    try:
        hdul=fits.open(fn)
        Next=len(hdul)
        if(len(hdul)>2):
            header=hdul[2].header
        else: header=hdul[0].header
        hdul.close()

        for i, item in enumerate(itemlist):
            try: result[i]=header[item]
            except: continue
    except: None
    return result, Next

def split_galfit_header(rawdata):
    """
    Descr - Split rawdata to [value, error, warning]
    """

    ##========== Check warning ==================
    dum=rawdata.split('*') # When fitting is strange, it has *
    warning=0
    if(len(dum)==1):  # the value does not have '*'
        imsi=rawdata.split()
    else:  # the value has '*'
        imsi=rawdata.replace('*', '').split()
        warning=1

    ##========== Check fixed value ==============
    is_fixed=rawdata.split('[') # When fitting is fixed, it has [, ]
    if(len(is_fixed)==1): # Not fixed
        value=imsi[0]
        err=imsi[2]
    else: # Fixed     e.g.) '[101.1474]'
        value=imsi[0].replace('[', '').replace(']', '')
        err=-999
    return value, err, warning

def read_galfit_header_singlevalues(fnlist, itemlist=['CHI2NU'], return_success=True):
    result_array=np.full((len(fnlist),len(itemlist)), np.NaN)
    result_array=dharray.array_quickname(result_array, names=itemlist)
    read_success=np.zeros(len(fnlist)).astype(bool) ## Retuen whether file is exist
    #errorcount=0
    for i , fn in enumerate (fnlist):
        try:
            hdul=fits.open(fn)
            if(len(hdul)>2): header=hdul[2].header
            else: header=hdul[0].header
            hdul.close()

            read_success[i]=True
            for j, name in enumerate(itemlist):
                try:
                    result_array[i][name]=header[name]
                except:
                    #errorcount+=1
                    continue
        except:
            read_success[i]=False
            continue
    #if(errorcount>0): print(">> ♣♣♣ Warning! ♣♣♣ Header has problems!", fnlist)
    if(return_success==False): return result_array
    else: return result_array, read_success


#============================== Result Galfit =====================
@dataclass
class ResGalData:
    """
    Data array
    """
    namelist : np.ndarray # Data namelist
    fnlist : np.ndarray # Result file namelist
    data_type : np.ndarray # data type
    chi2_itemlist : list['CHI2NU', 'AIC_F', 'RChi_50', 'AIC_50']
    #================== Will be input or made ===========
    val : np.ndarray = field(default_factory=lambda:
                             np.zeros(1)) # Value
    err : np.ndarray = field(default_factory=lambda:
                             np.zeros(1)) # Error
    war : np.ndarray = field(default_factory=lambda:
                             np.zeros(1)) # Warning
    war_tot : np.ndarray = field(default_factory=lambda:
                                 np.zeros(1)) # Warning total
    is_file_exist : np.ndarray = field(default_factory=lambda:
                                       np.zeros(1)) # File exist or not
    Next : np.ndarray = field(default_factory=lambda:
                              np.zeros(1)) # Number of extensions
    Ndata : int = 0 # Total number of data = sum(is_file_exist)


    def __post_init__(self): # make empty tables
        self.namelist=np.array(self.namelist)
        self.fnlist=np.array(self.fnlist)
        itemlist=self.data_type
        datalen1=len(self.namelist)
        datalen2=len(self.fnlist)
        if(datalen1!=datalen2): print("Error! Check data size!")

        self.val=np.full((datalen1,len(itemlist)), np.NaN)
        self.val=dharray.array_quickname(self.val, names=itemlist)
        self.err=np.full((datalen1,len(itemlist)), np.NaN)
        self.err=dharray.array_quickname(self.err, names=itemlist)
        self.war=np.full((datalen1,len(itemlist)), np.NaN)
        self.war=dharray.array_quickname(self.war, names=itemlist)

    # def bulk_input(self, val, err, war):
    #     self.val=dharray.array_quickname(val, names=self.val.dtype.names, dtypes=self.val.dtype)
    #     self.err=dharray.array_quickname(err, names=self.val.dtype.names, dtypes=self.val.dtype)
    #     self.war=dharray.array_quickname(war, names=self.val.dtype.names, dtypes=self.val.dtype)
    #     self.war_tot=self.val['war_tot']
    #     self.is_file_exist=np.zeros(len(self.val)).astype(bool)
    #     self.is_file_exist[np.isfinite(self.val['CHI2NU'])]=True
    #     self.Ndata=np.sum(self.is_file_exist)

    def clean_data(self, galfit_itemlist=None):
        ## Remove void fields
        self.val=dharray.array_remove_void(self.val)
        self.err=dharray.array_remove_void(self.err)
        self.war=dharray.array_remove_void(self.war)

        self.is_file_exist=np.zeros(len(self.val)).astype(bool)
        self.is_file_exist[np.isfinite(self.val['CHI2NU'])]=True
        self.Ndata=np.sum(self.is_file_exist)

        ## Check save itemlist
        ## If any of data is not NaN -> save (discard the data if all var are NaN)
        if(hasattr(galfit_itemlist, "__len__")):
            safe_itemlist=[]  # Save list
            for item in galfit_itemlist:
                check=np.sum(np.isfinite(self.val[item]))
                if(check!=0):
                    safe_itemlist+=[item] # If any of data is not NaN -> save (discard the data if all var are NaN)

            safe_itemlist=np.array(safe_itemlist)
            if(len(safe_itemlist)==0): print(">> Warning! No safe_itemlist!")
            else: self._make_war_tot(safe_itemlist)
            return safe_itemlist

    def _make_war_tot(self, safe_itemlist):
        #if(is_use_bwar==True): imsidat=np.copy(self.bwar[self.safe_itemlist])
        #else: imsidat=np.copy(self.war[self.safe_itemlist])

        if(self.Ndata==0): ## No data
            self.war_tot=np.full(len(self.war), 0)
        else:
            imsidat=np.copy(self.war[safe_itemlist])
            imsidat=dharray.array_flexible_to_simple(imsidat, remove_void=True)
            with warnings.catch_warnings():  ## Ignore warning. Nan warning is natural in this case.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.war_tot=np.nanmax(imsidat, axis=1)
        self.val['war_tot']=self.war_tot
        self.err['war_tot']=self.war_tot
        self.war['war_tot']=self.war_tot

    def cut_data(self, cutlist):
        cutlist=np.array(cutlist)
        self.namelist=self.namelist[cutlist]
        self.fnlist=self.fnlist[cutlist]
        self.val=self.val[cutlist]
        self.err=self.err[cutlist]
        self.war=self.war[cutlist]
        self.war_tot=self.war_tot[cutlist]
        self.Next=self.Next[cutlist]
        self.is_file_exist=self.is_file_exist[cutlist]
        self.Ndata=np.sum(self.is_file_exist)
        self.reff_strange=self.reff_strange[cutlist]
        self.n_strange=self.n_strange[cutlist]
        self.reff_snr=self.reff_snr[cutlist]

    def read_data(self, itemlist):
        fnlist=self.fnlist
        chi2_itemlist=self.chi2_itemlist

        ## Read data
        self.Next=np.zeros(len(fnlist)).astype(int)
        for i, fn in enumerate(fnlist):
            rawdata, self.Next[i]=read_galfit_header(fn, itemlist)
            for j, item in enumerate(itemlist):
                try:
                    val, err, war=split_galfit_header(rawdata[j])
                    self.val[i][item]=val
                    self.err[i][item]=err
                    ## Sometimes, err is NaN
                    if(np.isfinite(self.val[i][item]) & np.isnan(self.err[i][item])): war=1
                    self.war[i][item]=war
                except: pass

        chi2_flex, self.is_file_exist=read_galfit_header_singlevalues(fnlist,
                                        itemlist=chi2_itemlist, return_success=True)
        self.Ndata=np.sum(self.is_file_exist)

        for item in chi2_itemlist:
            self.val[item]=np.copy(chi2_flex[item])
            self.err[item]=np.copy(chi2_flex[item])
            self.war[item]=np.copy(chi2_flex[item])

    ##============================ Save data ==============================
    def save_data(self, fn_output, add_header='', return_array=False):
        empty1=dharray.array_quickname(np.zeros((len(self.val), 1)), names='datatype', dtypes='i4')

        val_out=dharray.array_add_columns(self.val, empty1, add_front=True)
        val_out['datatype']=1 ## Data

        err_out=dharray.array_add_columns(self.err, empty1, add_front=True)
        err_out['datatype']=2 ## Error

        war_out=dharray.array_add_columns(self.war, empty1, add_front=True)
        war_out['datatype']=3 ## Warning

        print_array=np.append(val_out, err_out)
        print_array=np.append(print_array, war_out)
        header=' '.join(self.namelist.astype(str))
        header+='\n'
        header+=' '.join(self.fnlist.astype(str))
        header+='\n '
        header+=str(add_header)
        if(fn_output!=None): np.savetxt(fn_output, print_array, header=header)
        if(return_array==True): return print_array

    def swap_data(self, swaplist, swapitemlist1, swapitemlist2):
        imsi=np.copy(self.val[swapitemlist2][swaplist])
        self.val[swapitemlist2][swaplist]=np.copy(self.val[swapitemlist1][swaplist])
        self.val[swapitemlist1][swaplist]=np.copy(imsi)
        self.val['is_swap'][swaplist]=True

        imsi=np.copy(self.err[swapitemlist2][swaplist])
        self.err[swapitemlist2][swaplist]=np.copy(self.err[swapitemlist1][swaplist])
        self.err[swapitemlist1][swaplist]=np.copy(imsi)
        self.err['is_swap'][swaplist]=True

        imsi=np.copy(self.war[swapitemlist2][swaplist])
        self.war[swapitemlist2][swaplist]=np.copy(self.war[swapitemlist1][swaplist])
        self.war[swapitemlist1][swaplist]=np.copy(imsi)
        self.war['is_swap'][swaplist]=True




##=============== PostProcessing ====================
def convert_add_params_array(usingdat, is_mag=False):
    if(is_mag): comp=['XC', 'YC', 'MAG', 'RE', 'N', 'AR', 'PA']
    else: comp=['XC', 'YC', 'MU_E', 'RE', 'N', 'AR', 'PA']
    comp2=dharray.array_attach_string(comp, '2_', add_at_head=True)
    comp3=dharray.array_attach_string(comp, '3_', add_at_head=True)
    comp4=dharray.array_attach_string(comp, '4_', add_at_head=True)
    add_params1_array=np.full((len(usingdat),7), np.nan)
    add_params2_array=np.full((len(usingdat),7), np.nan)
    add_params3_array=np.full((len(usingdat),7), np.nan)
    is_3rd_comp=np.isin(['4_XC'], usingdat.dtype.names)[0]
    for i in range (len(comp)):
        add_params1_array[:,i]=usingdat[comp2[i]]
        add_params2_array[:,i]=usingdat[comp3[i]]
        if(is_3rd_comp): add_params3_array[:,i]=usingdat[comp4[i]]

    ## For 3rd comp, if the data has not N --> it might be PSF
    is_psf=np.isfinite(add_params2_array[:,0]) &  np.isnan(add_params2_array[:,4])
    is_sersic=np.isfinite(add_params2_array[:,0]) &  np.isfinite(add_params2_array[:,4])
    add_params2_array[is_psf,2]=usingdat[is_psf]['3_MAG'] # Mu_e = mag
    if(is_3rd_comp):
        is_psf2=np.isfinite(add_params3_array[:,0]) &  np.isnan(add_params3_array[:,4])
        add_params3_array[is_psf2,2]=usingdat[is_psf2]['4_MAG'] # Mu_e = mag
    return is_sersic, add_params1_array, add_params2_array, add_params3_array
