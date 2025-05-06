import DH_array as dharray
import DH_path as dhpath
import numpy as np
import os
import random
import time
import pickle
import copy
from astropy.io import fits
from dataclasses import dataclass
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd

from astropy import units as u
from astropy.modeling.functional_models import Sersic1D, Sersic2D
from astropy.convolution import Gaussian2DKernel
from astropy.utils.data import download_file, get_pkg_data_filename
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval,ImageNormalize,AsinhStretch

import subprocess
from multiprocessing import Pool
import warnings


def sersic_integral2D(mu_e, reff, n, ar, plate_scale, res=500):
    """
    Descr - measure total magnitude of a given galaxy
    INPUT
     - mu_e : surface brightness at Reff [mag/arc2]
     - reff : Reff
     - n : Sersic index
     - ar : axis ratio (b/a)
     - plate_scale : Plate scale [arcsec / pix]
     OUTPUT
     - Total magnitude [mag]

    """
    amp=10**((25-mu_e)/2.5)
    amp=amp*(plate_scale**2)  # amp : flux / pix2
    e=e=1-ar  # ar : b/a,   e = (a-b)/a
    model=Sersic2D(amplitude=amp, r_eff=reff, n=n, ellip=e)


    x,y = np.meshgrid(np.arange(res)-res/2, np.arange(res)-res/2)
    ## Multi-gals
    if(hasattr(mu_e, "__len__")):
        x=np.repeat(x.reshape(np.shape(x)[0], np.shape(x)[1], 1), len(mu_e), axis=2)
        y=np.repeat(y.reshape(np.shape(y)[0], np.shape(y)[1], 1), len(mu_e), axis=2)
        img=model(x,y)
        flux=np.sum(np.sum(img, axis=0), axis=0)
    else:
        img=model(x,y)
        flux=np.sum(img)
    return 25-2.5*np.log10(flux)



#=================== AUTO Galfit ==========================

@dataclass
class AutoGalfit:
    """
    Automatically run Galfit with input presets
    See also -> Runlist_Galfit (Generate presets)
    """
    dir_work : str  # Working directory. If it is not exist, make dir.
    outputname : str # Extension for name of configuration file
    prefix : str = '' # prefix for name of configuration file
    suffix : str = ''  # suffix for name of configuration file
    magz : float = 22.5 # magnitude zero point
    platescl : np.ndarray = np.array([0.262, 0.262])  # Thumbnail pixel scale in arcsec
    size_fitting : int = -1  # Fix the fitting area       ## -1 -> Full image
    size_conv : int = -1  # Convolution size.                               ## -1 --> Auto (size of the PSF image)
    silent : bool =False # Print message or not
    info_from_image: bool = True # Read info from the input image

    fn_galfit: str = dhpath.fn_galfit()  #Galfit direction
    fna_inputimage : str ='test.fits' # Name of fits file to analyze
    fna_inputmask : str ='testmask.fits' # Name of mask file for analysis

    fna_psf : str = 'psf_area_g.fits'
    fna_sigma : str = 'sigma_g.fits'

    est_sky : float = 10  # Estimate of sky level

    def __post_init__(self):
        os.makedirs(self.dir_work, exist_ok=True)

        fn, _ext = os.path.splitext(self.outputname)
        self.n_comp=int(0)
        self.comp_dtype=[('fitting_type', '<U10'),
                         ('est_xpos', '<f8'), ('est_ypos', '<f8'),
                         ('est_mag', '<f8'), ('est_reff', '<f8'), ('est_n', '<f8'),
                         ('est_axisratio', '<f8'), ('est_pa', '<f8')]

        #Filenames
        self._set_path_name()

        if(self.info_from_image==True): self.read_info() # Read info from the input image
        self.generate_galconfig()
        self.generate_constraints()

    def _set_path_name(self):
        self.fn_gal_conf = self.dir_work + 'galfit_param_' + self.prefix + self.outputname+ self.suffix + ".dat" # Configuration file
        self.fn_constraint=self.dir_work + 'constraints.dat'  ## We can use a single constraint file

        self.fn_psf = self.dir_work + self.fna_psf   # PSF file
        try: self.psf_size=np.shape(fits.getdata(self.fn_psf))[0]
        except: print(">> ♣♣♣ Warning! ♣♣♣ PSF file does not exist!")

        if(self.fna_sigma=='none'): self.fn_sigma = 'none'  # sigma file
        else:
            self.fn_sigma = self.dir_work + self.fna_sigma
            try: fits.open(self.fn_sigma)
            except: print(">> ♣♣♣ Warning! ♣♣♣ Sigma file does not exist!")

        if(self.fna_inputmask=='none'): self.fn_inputmask = 'none'  # Masking file
        else:
            self.fn_inputmask=self.dir_work+self.fna_inputmask
            try: fits.open(self.fn_inputmask)
            except: print(">> ♣♣♣ Warning! ♣♣♣ Masking file does not exist!")

        self.fn_output_imgblock=self.dir_work+'result_'+ self.prefix + self.outputname + self.suffix + ".fits"
        self.fn_inputimage=self.dir_work+self.fna_inputimage

    def read_info(self):
        if(self.silent==False):
            print("● Input data from the input image header ", self.fn_inputimage)
            # check_fits(self.fn_inputimage)
        hdu = fits.open(self.fn_inputimage)
        header=hdu[0].header
        # self.magz=header['MAGZERO']
        # self.platescl=np.array([header['PIXSCAL1'], header['PIXSCAL2']])
        self.image_size=np.shape(hdu[0].data)[0]


    def add_fitting_comp(self, fitting_type='sersic',
                        est_params=np.full(7, np.nan), is_allow_vary=True):
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
        default_params=np.array([self.image_size/2, self.image_size/2, 20, 20, 4, 1, 0])
        nans=np.isnan(est_params)
        est_params[nans]=np.copy(default_params[nans])

        ## setting
        for i, item in enumerate(self.comp.dtype.names):
            if(i==0): self.comp[index][item]=np.copy(fitting_type)
            else: self.comp[index][item]=np.copy(est_params[i-1])

        if(self.silent==False):
            print(">> Input Data : ")
            print(self.comp.dtype)
            print(self.comp)

        ## =============== INPUT values ==================
        fconf = open(os.path.join(self.fn_gal_conf), 'a')
        fconf.write('\n# Component number: %d\n'%int(self.n_comp+1))
        fconf.write(' 0) %s                     #  object type\n'%fitting_type)
        fconf.write(' 1) %f  %f   %d %d         #  position x, y\n' %(est_params[0], est_params[1], allow_vary_array[0], allow_vary_array[1]))
        if(fitting_type=='psf'):
            if(self.silent==False): print("PSF")
            fconf.write(' 3) %f      1              #  Total magnitude\n'  %est_params[2])
        else:
            fconf.write(' 3) %f      1              #  Integrated magnitude\n'  %est_params[2])
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
        if(self.size_conv>=0): size_conv=self.size_conv
        else: size_conv=self.psf_size

        ## Fitting box
        if(self.size_fitting>=0):
            half_size=np.around(self.size_fitting/2)
            size_min=np.around(self.image_size/2-half_size)
            size_max=np.around(self.image_size/2+half_size)
            if(size_min<1): size_min=1
            if(size_max>self.image_size): size_max=self.image_size
        else:
            size_min=1
            size_max=self.image_size

        # Create name and open configuration file for writing
        fconf = open(os.path.join(self.fn_gal_conf), 'w')
        fconf.write('# Automatically generated by Auto_galfit.ipynb _ DJ Khim\n')
        fconf.write('# IMAGE and GALFIT CONTROL PARAMETERS\n')

        fconf.write('A) %s                     # Input data image (FITS file)\n' %self.fn_inputimage)
        fconf.write('B) %s                     # Output data image block\n' %self.fn_output_imgblock)
        fconf.write('C) %s                   # Sigma image name (made from data if blank or "none")\n'%self.fn_sigma)
        fconf.write('D) %s                     # Input PSF image and (optional) diffusion kernel\n' %self.fn_psf)
        fconf.write('E) 1                      # PSF fine sampling factor relative to data\n')
        fconf.write('F) %s                     # Bad pixel mask (FITS image or ASCII coord list)\n' %self.fn_inputmask)
        fconf.write('G) %s                     # File with parameter constraints (ASCII file)\n' %self.fn_constraint)
        fconf.write('H) %d    %d   %d    %d      # Image region to fit (xmin xmax ymin ymax)\n' %(size_min, size_max, size_min, size_max))
        fconf.write('I) %d    %d               # Size of the convolution box (x y)\n'  %(size_conv, size_conv))
        fconf.write('J) %f                     # Magnitude photometric zeropoint\n' % self.magz)
        fconf.write('K) %f    %f               # Plate scale (dx dy)  (arcsec per pixel)\n'  %(self.platescl[0], self.platescl[1]))
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


    def generate_constraints(self):
        if(self.silent==False): print("● Generated constraints file to ", self.fn_constraint)

        # Create name and open configuration file for writing
        fconf = open(os.path.join(self.fn_constraint), 'w')
        fconf.write('# Automatically generated by Auto_galfit.ipynb _ DJ Khim\n')
        fconf.write('# GALFIT CONSTRAINT PARAMETERS\n')
        fconf.write('# Component     Parameter    Constraint range       Comment\n\n')


    def add_constraints(self, id_comp, poslim=[-10, 10], refflim=[5, 200], nlim=[0.2, 1.5]):
        id_comp=int(id_comp)
        if(self.silent==False): print("● Added constraint for ", id_comp)
        fconf = open(os.path.join(self.fn_constraint), 'a')

        if(hasattr(poslim, "__len__")==True):
            par_in='x'
            descr='# Soft constraint: %s-position to within %d and %d of the >>INPUT<< value.'%(par_in, poslim[0], poslim[1])
            fconf.write('  %d        %s        %d  %d     %s\n\n'%(id_comp, par_in, poslim[0], poslim[1], descr))
            par_in='y'
            descr='# Soft constraint: %s-position to within %d and %d of the >>INPUT<< value.'%(par_in, poslim[0], poslim[1])
            fconf.write('  %d        %s        %d  %d     %s\n\n'%(id_comp, par_in, poslim[0], poslim[1], descr))

        if(hasattr(refflim, "__len__")==True):
            par_in='re'
            descr='# Soft constraint: Constrains the effective radius to '
            descr+='within values from %.2f to %.2f.'%(refflim[0], refflim[1])
            fconf.write('  %d        %s        %.2f to %.2f     %s\n\n'%(id_comp, par_in, refflim[0], refflim[1], descr))

        if(hasattr(nlim, "__len__")==True):
            par_in='n'
            descr='# Soft constraint: Constrains the sersic index n to within values from %.2f to %.2f.'%(nlim[0], nlim[1])
            fconf.write('  %d        %s        %.2f to %.2f     %s\n\n'%(id_comp, par_in, nlim[0], nlim[1], descr))



        fconf.close()


    def run_galfit(self, print_galfit=False):
        '''
        Create and execute Galfit shell commands
        '> /dev/null' suppresses output
        '''
        if(self.silent==False): print("● Run galfit")
        cmd = self.fn_galfit
        cmd += ' -c %s ' % self.fn_gal_conf #+ ' > /dev/null'
        if(self.silent==False): print(">> Command : ", cmd)
        if(print_galfit==True): result=subprocess.run(cmd, shell=True, cwd=os.getcwd())
        else: result=subprocess.run(cmd, shell=True, cwd=os.getcwd(), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if(self.silent==False):
            print(">> Result : ", result)
            if(result.returncode==0): print(">> Galfit done!")
            else: print("\n>>♣♣♣ Warning! ♣♣♣ Galfit has a problem!")
        return result



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

def drawing_galfit(fn='./TestRun/result_test.fits', gs=None, data_index=0, silent=False,              #Basic
                   fn_masking=None, show_masking=True, show_masking_center=True, fill_masking=True,    #Masking
                   show_htitle=True, show_vtitle=True, descr='',  c_vtitle='k',                       #descr
                   show_ticks=True, scale_percentile=None):
    ## Gridspec Setting
    if(gs==None):
        fig=plt.figure(figsize=(13,4))
        gs=gridspec.GridSpec(1,3)

    ## open file
    hdu=fits.open(fn)
    imagedata=hdu[1].data  # Sample data
    chi2=hdu[2].header['CHI2NU']
    titlelist=['Original', 'Model', 'Residual']
    imgsize=np.shape(imagedata)
    if(silent==False):
        print("[Descr : ", descr, "]")
        print(">> Reduced Chi2 :", chi2)
        print(">> Orig - Max, Min, PTP :", hdu[1].data.max(), hdu[1].data.min(), hdu[1].data.ptp())
        print(">> Resi - Max, Min, PTP :", hdu[3].data.max(), hdu[3].data.min(), hdu[3].data.ptp())

    ## Masking file
    if(type(fn_masking)!=type(None)): masking=fits.getdata(fn_masking)
    else:
        show_masking=False
        show_masking_center=False
    # Masking center positions
    if(show_masking_center==True):
        maxmask=np.nanmax(masking)
        centerpos=np.full((maxmask, 2), np.nan)
        for i in range (maxmask): # loop for [1, maxmask]
            target=np.where(masking==(i+1))
            centerpos[i]=np.nanmean(target, axis=1)

    #=============== Galfit subfiles - 1 : Obs / 2 : Model / 3 : Residual ============
    for i in range (3):
        ax0=plt.subplot(gs[data_index*3+i])
        imagedata=hdu[i+1].data
        mean, median, std=np.nanmean(imagedata), np.nanmedian(imagedata), np.nanstd(imagedata)
        if(show_masking==True):
            maskedimage=np.ma.masked_array(imagedata, masking, fill_value=median) #median masking
            if(fill_masking==True): imagedata=maskedimage.filled()
            else: imagedata=maskedimage

        # Scales
        if(scale_percentile==None):
            ax0.imshow(imagedata, cmap='bone', origin='lower')
        else:
            newscale=[np.percentile(imagedata, 50-scale_percentile), np.percentile(imagedata, 50+scale_percentile)]
            ax0.imshow(imagedata, cmap='bone', origin='lower', vmin=newscale[0], vmax=newscale[1])

        # Masking centers
        if(show_masking_center==True): plt.scatter(centerpos[:,1], centerpos[:,0], c='r', s=20)

        # Show titles
        if(show_htitle==True): plt.title(titlelist[i], size=15, fontname='DejaVu Serif', fontweight='semibold')
        if(i==0):
            if(show_vtitle==True): plt.ylabel(descr+"\n"+r"$\chi^2$=%.4f"%chi2, size=20, c=c_vtitle, fontname='DejaVu Serif', fontweight='semibold')

        if(show_ticks==True):
            ax0.set_xticks([0, imgsize[0]/2, imgsize[0]])
            ax0.set_yticks([0, imgsize[1]/2, imgsize[1]])
            ax0.tick_params(labelsize=10, width=1, length=6, axis='both', direction='in', right=True, top=True)
#         ax0.xaxis.set_minor_locator(AutoMinorLocator(5))
#         ax0.yaxis.set_minor_locator(AutoMinorLocator(5))

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




#========================== RUN LIST ==========================
@dataclass
class Runlist_Galfit:
    """
    Descr - Runlist
    INPUT
     *compname: list of components (Ncomp = len(components))
         (Default=np.array(['sersic2', 'sersic2']))
     *namelist: Namelist (Nset = len(namelist))
         (Default=np.array(['ss_23', 'ss_24', 'ss_25', 'ss_26']))

     *est_params_list1: The list of estimated parameters for comp 1
                        The single value in the list will be duplicated.
                        For values nan, they will be default values.
                  index    [  0   |  1   | 2 |  3 | 4 |5 |6 ]
                  params   [ xpos | ypos |mag|reff| n |ar|pa]
                  default  [center|center|20 | 20 | 4 |1 |0 ]
         (Default=np.ndarray([np.nan, np.nan, np.array([23, 24, 25, 26]), 30, 1, 1, 0]))
         ** for values of np.nan --> It will be converted into default values (See AutoGalfit)
     *est_params_list2: The list of estimated parameters for comp 2
         (Default=np.ndarray([np.nan, np.nan, np.array([23, 24, 25, 26]), 30, 1, 1, 0]))

     *use_constraint_list: Whether use the constraint or not
                           The single value in the list will be duplicated.
                           (Default=False)
     *lim_pos1, lim_pos2, lim_reff1, lim_reff2, lim_n1, lim_n2: Constraints
    """
    compname : np.ndarray = np.array(['sersic2', 'sersic2'])  # list of components (Ncomp = len(components))
    namelist : np.ndarray = np.array(['ss_23', 'ss_24', 'ss_25', 'ss_26']) # Namelist (Nset = len(namelist))
    est_params_list1 : np.ndarray = np.array([np.nan, np.nan, np.array([23, 24, 25, 26]), 30, 1, 1, 0], dtype=object) # See note
    est_params_list2 : np.ndarray = np.array([np.nan, np.nan, np.array([23, 24, 25, 26]), 10, 4, 1, 0], dtype=object) # See note
    use_constraint_list : bool = False
    lim_pos1 : np.ndarray = np.array([-20, 20]) # position constraint for comp1
    lim_pos2 : np.ndarray = np.array([-20, 20]) # position constraint for comp2
    lim_reff1 : np.ndarray = np.array([5, 200]) # Effective radius constraint for comp1
    lim_reff2 : np.ndarray = np.array([5, 200]) # Effective radius constraint for comp2
    lim_n1 : np.ndarray = np.array([0.2, 5.5]) # sersic index (n) constraint for comp1
    lim_n2 : np.ndarray = np.array([0.2, 5.5]) # sersic index (n) constraint for comp2

    fn_galfit : str = dhpath.fn_galfit()

    group_id : int = -1
    size_conv : int = -1
    size_fitting : int = -1

    def __post_init__(self):
        if(type(self.compname)==str): print("Error! compname is not an array!")
        if(type(self.namelist)==str): print("Error! namelist is not an array!")
        self.Ncomp = len(self.compname)
        self.Nset = len(self.namelist)

        ## Repeat est_params except for array
        self.est_params_table1=np.full((self.Nset, len(self.est_params_list1)), np.nan)
        self.est_params_table2=np.full((self.Nset, len(self.est_params_list1)), np.nan)
        for i in range(len(self.est_params_list1)):
            self.est_params_table1[:,i]=dharray.repeat_except_array(self.est_params_list1[i], self.Nset)
            self.est_params_table2[:,i]=dharray.repeat_except_array(self.est_params_list2[i], self.Nset)
        self.use_constraint_list=dharray.repeat_except_array(self.use_constraint_list, self.Nset)

        # self.lim_pos1=np.repeat(self.lim_pos1, self.Nset, axis=0)
        # self.lim_pos2=np.repeat(self.lim_pos2, self.Nset, axis=0)
        # self.lim_n1=np.repeat(self.lim_n1, self.Nset, axis=0)
        # self.lim_n2=np.repeat(self.lim_n2, self.Nset, axis=0)

        ## Generate array
        runlist_dtype=[('name', '<U20'), ('compname', object),
                        ('est_params1', object), ('est_params2', object),
                        ('size_conv', '<i4'), ('size_fitting', '<i4'),
                       ('lim_pos1', object), ('lim_pos2', object),
                       ('lim_reff1', object), ('lim_reff2', object),
                       ('lim_n1', object), ('lim_n2', object),
                       ('use_lim', '<i4'), ('group_ID', '<i4'),
                        ('est_mag1', '<f8'), ('est_mag2', '<f8'),   #Showing only
                        ('est_reff1', '<f8'), ('est_reff2', '<f8'), #Showing only
                        ('est_n1', '<f8'), ('est_n2', '<f8'),       #Showing only
                      ]
        self.runlist=np.zeros(self.Nset, dtype=runlist_dtype)

        ## Input values
        for i in range(self.Nset):
            self.runlist['est_params1'][i]=self.est_params_table1[i]
            self.runlist['est_params2'][i]=self.est_params_table2[i]
        loop=list(['name',
                   'est_mag1', 'est_mag2',
                   'est_reff1', 'est_reff2', 'est_n1', 'est_n2',
                   'size_conv', 'size_fitting' , 'use_lim'])
        loopdata=[self.namelist,
                  self.est_params_table1[:,2], self.est_params_table2[:,2],
                  self.est_params_table1[:,3], self.est_params_table2[:,3],
                  self.est_params_table1[:,4], self.est_params_table2[:,4],
                  self.size_conv, self.size_fitting, self.use_constraint_list]

        for i, item in enumerate(loop):
            self.runlist[item]=loopdata[i]
        self.runlist['group_ID'] = self.group_id
        self.runlist['compname']=[self.compname]
        self.runlist['lim_pos1']=[self.lim_pos1]
        self.runlist['lim_pos2']=[self.lim_pos2]
        self.runlist['lim_reff1']=[self.lim_reff1]
        self.runlist['lim_reff2']=[self.lim_reff2]
        self.runlist['lim_n1']=[self.lim_n1]
        self.runlist['lim_n2']=[self.lim_n2]
        # for i in range (self.Nset):
        #     self.runlist[i]['compname']=self.compname
        #     self.runlist[i]['lim_pos1']=self.lim_pos1
        #     self.runlist[i]['lim_pos2']=self.lim_pos2
        #     self.runlist[i]['lim_n1']=self.lim_n1
        #     self.runlist[i]['lim_n2']=self.lim_n2


        ## make Nan for safety)
        if(self.Ncomp==1):
            items=['est_mag2', 'est_reff2', 'est_n2']
            for item in items:
                self.runlist[item]=np.nan
        if(self.Ncomp>1):
            if(self.compname[1]=='psf'):
                items=['est_reff2', 'est_n2']
                for item in items:
                    self.runlist[item]=np.nan

    def add_runlist(self, add_runlist):
        self.runlist=np.append(self.runlist, add_runlist.runlist)
        self.namelist=np.append(self.namelist, add_runlist.namelist)

    def show_runlist(self, show_full=False):
        if(show_full==True): display(pd.DataFrame(self.runlist))
        else:
            showlist=['name', 'compname', 'est_mag1', 'est_mag2', 'est_reff1', 'est_reff2',
                      'est_n1', 'est_n2', 'use_lim', 'group_ID']
            display(pd.DataFrame(self.runlist[showlist]))

    def generate_runlist_empty_sersic(self):
        runlist_empty_sersic=copy.deepcopy(self.runlist)
        for i in range (len(runlist_empty_sersic['compname'])):
            for j in range (len(runlist_empty_sersic['compname'][i])):
                if(runlist_empty_sersic['compname'][i][j]=='sersic2'):
                    runlist_empty_sersic['compname'][i][j]='sersic'
            killlist=np.isfinite(runlist_empty_sersic['est_params1'][i])
            runlist_empty_sersic['est_params1'][i][killlist]=0
            killlist=np.isfinite(runlist_empty_sersic['est_params2'][i])
            runlist_empty_sersic['est_params2'][i][killlist]=0
        self.runlist_empty_sersic=runlist_empty_sersic
        #return runlist_empty_sersic



    def run_galfit_with_runlist(self, dir_work, prefix='', suffix='',
                                base_params1_array=None, base_params2_array=None,
                                base_params1_mode='add', base_params2_mode='add',
                                use_empty_sersic=False,
                                is_2nd_mag_offset=False,
                                is_allow_vary1=True, is_allow_vary2=True,
                                fna_inputimage='image.fits', fna_inputmask='none',
                                fna_psf='psf_area_g.fits', fna_sigma='sigma_g.fits',
                                silent=False, overwrite=False, print_galfit=False, group_id=-1,
                                ):
        """
        Descr - Run galfit with Runlist
        INPUT
         - dir_work
         * base_params1_array: It will be added/replaced to the est_params. For None, it will be ignored.
                              It applies to the 1st component (except for the sky).
                              If length of array is 1, the array will be applied to the all runlists.
                              (Default=None)
                 index    [  0   |  1   | 2 |  3 | 4 |5 |6 ]
                 params   [ xpos | ypos |mag|reff| n |ar|pa]
         * base_params2_array: It will be added/replaced to the est_params. For None, it will be ignored.
                              It applies to the 2nd component (except for the sky).
                              (Default=None)
         * base_params1_mode: 'add' or 'replace' (Default='add')
         * base_params2_mode: 'add' or 'replace' (Default='add')
         * is_2nd_mag_offset: If True, the magnitude of the 2nd comp. would be respective to 1st comp.
                              The input values are being offsets. (2nd mag = 1st mag + input values)
                              (Default=False)
         * is_allow_vary1: If false, the value is fixed. Galfit will not change the parameter
                           It apply to the 1st component (except for the sky).
           1) True/False -> Apply the value for all parameters (Default=True)
           2) Array -> Apply the values for each parameter
                 index    [  0   |  1   | 2(X) |  3 | 4 |5 |6 ]
                 params   [ xpos | ypos |mag(X)|reff| n |ar|pa]  ## Mag always changes (True)!
         * is_allow_vary2: If false, the value is fixed. Galfit will not change the parameter
                           It apply to the 2nd component (except for the sky).
        """
        ## usingrunlist - based on group_ID
        if(use_empty_sersic==True): temp=copy.deepcopy(self.runlist_empty_sersic)
        else: temp=copy.deepcopy(self.runlist)
        if(group_id<0): usingrunlist=copy.deepcopy(temp)
        else: usingrunlist=copy.deepcopy(temp[np.isin(temp['group_ID'], group_id)])

        ## If base_params_array is input --> Add values to runlist
        if(hasattr(base_params1_array, "__len__")):
            #if(len(base_params1_array)==1): np.repeat(base_params1_array, len(usingrunlist), axis=0)
            if(hasattr(base_params1_array[0], "__len__")==False):
                base_params1_array=np.repeat([base_params1_array], len(usingrunlist), axis=0)
            if(silent==False): print(">> Base params1 array will be considered!")
        if(hasattr(base_params2_array, "__len__")):
            #if(len(base_params2_array)==1): np.repeat(base_params2_array, len(usingrunlist), axis=0)
            if(hasattr(base_params2_array[0], "__len__")==False):
                base_params2_array=np.repeat([base_params2_array], len(usingrunlist), axis=0)
            if(silent==False): print(">> Base params1 array will be considered!")


        ## ================Loop=================
        for i_run in range (len(usingrunlist)):
            thisrunlist=usingrunlist[i_run]
            if(hasattr(base_params1_array, "__len__")):
                if(base_params1_mode=='add'): thisrunlist['est_params1']+=base_params1_array[i_run]
                else: thisrunlist['est_params1']=base_params1_array[i_run]
            if(hasattr(base_params2_array, "__len__")):
                if(base_params2_mode=='add'): thisrunlist['est_params2']+=base_params2_array[i_run]
                else: thisrunlist['est_params2']=base_params2_array[i_run]
            if(is_2nd_mag_offset==True):
                thisrunlist['est_params2'][2]=thisrunlist['est_params2'][2]+thisrunlist['est_params1'][2]
            outputname=thisrunlist['name']

            ## Check whether the result is exist
            if(overwrite==False):
                if(os.path.exists(dir_work+'result_'+prefix+outputname+suffix+".fits")==True): continue  # Skip

            ## Main
            run=AutoGalfit(fn_galfit=self.fn_galfit, dir_work=dir_work,
                           fna_inputimage=fna_inputimage, outputname=outputname,
                            fna_psf=fna_psf, fna_sigma=fna_sigma, fna_inputmask=fna_inputmask,
                           size_conv=thisrunlist['size_conv'],  size_fitting=thisrunlist['size_fitting'],
                          silent=silent, prefix=prefix, suffix=suffix)

#         * est_params: Estimated parameters.
#                       For values nan, they will be default values.
#                 index    [  0   |  1   | 2 |  3 | 4 |5 |6 ]
#                 params   [ xpos | ypos |mag|reff| n |ar|pa]
#                 default  [center|center|20 | 20 | 4 |1 |0 ]

            ## Single comp
            run.add_fitting_comp(fitting_type=thisrunlist['compname'][0],
                                 est_params=thisrunlist['est_params1'],
                                 is_allow_vary=is_allow_vary1)
            if(thisrunlist['use_lim']==1):
                run.add_constraints(2, poslim=thisrunlist['lim_pos1'],
                                    nlim=thisrunlist['lim_n1'],
                                    refflim=thisrunlist['lim_reff1'])

            ## Double comp
            if(len(thisrunlist['compname'])>1):
                run.add_fitting_comp(fitting_type=thisrunlist['compname'][1],
                                     est_params=thisrunlist['est_params2'],
                                     is_allow_vary=is_allow_vary2)
                if(thisrunlist['use_lim']==1):
                    run.add_constraints(3, poslim=thisrunlist['lim_pos2'],
                                        nlim=thisrunlist['lim_n2'],
                                        refflim=thisrunlist['lim_reff2'])

            ## Run
            run.run_galfit(print_galfit=print_galfit)

        ## Calc chi2
    #     if(calc_chi2==True):
    #         if(silent==False): print("=========== Calculate Chi2 ===========\n")
    #         if(fna_inputmask==None or fna_inputmask=='none'): fn_masking=None
    #         else: fn_masking=dir_work+fna_inputmask
    #         if(fna_sigma==None or fna_sigma=='none'): fn_sigma=None
    #         else: fn_sigma=dir_work+fna_sigma
    #         self.cal_update_chi2_with_runlist(dir_work=dir_work, silent=silent,
    #                                           fn_sigma=fn_sigma, fn_masking=fn_masking,
    #                                           center_size=chi2_center_size, overwrite=chi2_overwrite)
    #
    # def cal_update_chi2_with_runlist(self, dir_work, fn_sigma, silent=False,
    #                                  fn_masking=None, center_size=-1, overwrite=False):
    #
    #     fnlist=dharray.array_attach_string(self.namelist, 'result_', add_at_head=True)
    #     fnlist=dharray.array_attach_string(fnlist, dir_work, add_at_head=True)
    #     fnlist=dharray.array_attach_string(fnlist, '.fits')
    #     if(silent==False):
    #         print("fnlist ", fnlist)
    #         print("fn_sigma ", fn_sigma)
    #         print("fn_masking ", fn_masking)
    #         print("center_size ", center_size)
    #
    #     cal_update_chi2(fnlist=fnlist, fn_sigma=fn_sigma,
    #                 fn_masking=fn_masking, center_size=center_size, overwrite=overwrite)


##======================== Reduced chi2 ========================
def cal_update_chi2(fnlist, fn_sigma, fn_masking=None, center_size=-1, repair_fits=True, image_size=200,
                    header_item='Chi_100', header_item_aic='AIC_100', overwrite=False, silent=True):
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
     * center_size : The region for calculating chi2.
       (Radius : center_size/2) If -1, calculate for all region (default : -1)
     * repair_fits : Repair the Galfit header if it has problem (Default : True)
     * header_item : Input chi2 as a new header.
       'R'+'header_item' will be added for the reduced chi2 values (default : 'Chi_100')
     * Overwrite : (default : False)
    """

    ## Input sigma
    if(fn_sigma==None): # If no sigma -> Make a flat image
        # sampledat=fits.getdata(fnlist[0], ext=3)
        sig=np.full((image_size, image_size), 1)
    else: sig=fits.getdata(fn_sigma)

    ## Input masking
    if(fn_masking==None): mask=None
    else: mask=fits.getdata(fn_masking)

    for fn in fnlist:
        try: dat=fits.getdata(fn, ext=3)
        except:
            if(silent==False): print(">> Cannot read the file:", fn)
            continue

        header=fits.getheader(fn, ext=2)
        chi2=cal_chi2(dat, sig, mask, center_size)
        if(overwrite==False):
            try:  # Header exists --> Next file
                header[header_item]
                continue
            except: # Do work!
                pass

        if(repair_fits==True):
            hdul = fits.open(fn)
            hdul.verify('silentfix')
            hdul.writeto(fn, overwrite=True)
        fits.setval(fn, header_item, value=chi2, ext=2)
        ndof=header['NDOF']
        p=header['NFREE']
        fits.setval(fn, 'R'+header_item, value=chi2/ndof, ext=2)

        ##AIC
        if(center_size<0): Npixel=image_size**2
        else: Npixel=center_size**2
        aic=chi2+2*p+(2*p*(p+1))/(Npixel-p-1)
        fits.setval(fn, header_item_aic, value=aic, ext=2)
        if(silent==False): print(">>", fn, "Done")


def cal_chi2(dat, sig, mask=None, center_size=-1):
    """
    Descr - calculate chi2 (See also. cal_update_chi2)
    INPUT
     - dat : data array
     - sig : sigma array
     - mask : masking array. It will mask all pixels having non-zero. If None, do not mask. (Default : None)
     - center_size : The region for calculating chi2.
       (Radius : center_size/2) If -1, calculate for all region (default : -1)
    """
    if(hasattr(mask, "__len__")): mdat=np.ma.masked_array(dat, mask!=0)
    else: mdat=dat
    if(center_size<0): return np.sum((mdat**2/sig**2))
    else:
        center=int(len(dat)/2)
        radius=int(center_size/2)
        imgrange=[center-radius, center+radius]
        return np.sum((mdat[imgrange[0]:imgrange[1],imgrange[0]:imgrange[1]]**2/sig[imgrange[0]:imgrange[1],imgrange[0]:imgrange[1]]**2))



#===================== Codes for ResultGalfit ==================
def read_galfit_header(fn, itemlist, ext=2):
    result=np.full(len(itemlist), np.NaN, dtype=object)
    try:
        header=fits.getheader(fn, ext=ext)
        for i, item in enumerate(itemlist):
            try: result[i]=header[item]
            except: continue
    except: None
    return result

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
    result_array=dharray.array_quickname(result_array, itemlist)
    read_success=np.zeros(len(fnlist)).astype(bool) ## Retuen whether file is exist
    errorcount=0
    for i in range (len(fnlist)):
        try:
            h=fits.getheader(fnlist[i], ext=2)
            read_success[i]=True
            for j, name in enumerate(itemlist):
                try:
                    result_array[i][name]=h[name]
                except:
                    errorcount+=1
                    continue
        except:
            read_success[i]=False
            continue
    if(errorcount>0): print(">> ♣♣♣ Warning! ♣♣♣ Header has problems!")
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
    #================== Will be input or made ===========
    val : np.ndarray = np.zeros(1) # Value
    err : np.ndarray = np.zeros(1) # Error
    war : np.ndarray = np.zeros(1) # Warning
    war_tot : np.ndarray = np.zeros(1) # Warning total
    is_file_exist : np.ndarray = np.zeros(1) # File exist or not
    Ndata : int = 0 # Total number of data = sum(is_file_exist)


    def __post_init__(self): # make empty tables
        self.namelist=np.array(self.namelist)
        self.fnlist=np.array(self.fnlist)
        itemlist=self.data_type
        datalen1=len(self.namelist)
        datalen2=len(self.fnlist)
        if(datalen1!=datalen2): print("Error! Check data size!")

        self.val=np.full((datalen1,len(itemlist)), np.NaN)
        self.val=dharray.array_quickname(self.val, itemlist)
        self.err=np.full((datalen1,len(itemlist)), np.NaN)
        self.err=dharray.array_quickname(self.err, itemlist)
        self.war=np.full((datalen1,len(itemlist)), np.NaN)
        self.war=dharray.array_quickname(self.war, itemlist)

    def bulk_input(self, val, err, war):
        self.val=dharray.array_quickname(val, names=self.val.dtype.names, dtypes=self.val.dtype)
        self.err=dharray.array_quickname(err, names=self.val.dtype.names, dtypes=self.val.dtype)
        self.war=dharray.array_quickname(war, names=self.val.dtype.names, dtypes=self.val.dtype)
        self.war_tot=self.val['war_tot']
        self.is_file_exist=np.zeros(len(self.val)).astype(bool)
        self.is_file_exist[np.isfinite(self.val['CHI2NU'])]=True
        self.Ndata=np.sum(self.is_file_exist)

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
        self.is_file_exist=self.is_file_exist[cutlist]
        self.Ndata=np.sum(self.is_file_exist)

    def read_data(self, itemlist):
        fnlist=self.fnlist
        chi2itemlist=['CHI2NU', 'RChi_100', 'AIC_100']

        ## Read data
        for i, fn in enumerate(fnlist):
            rawdata=read_galfit_header(fn, itemlist, ext=2)
            for j, item in enumerate(itemlist):
                try:
                    val, err, war=split_galfit_header(rawdata[j])
                    self.val[i][item]=val
                    self.err[i][item]=err
                    self.war[i][item]=war
                except: pass

        chi2_flex, self.is_file_exist=read_galfit_header_singlevalues(fnlist,
                                        itemlist=chi2itemlist, return_success=True)
        self.Ndata=np.sum(self.is_file_exist)

        for item in chi2itemlist:
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

class ResultGalfit:
    def __init__(self, runlist=None, dir_work=None,   ## 2) For input_runlist mode
                 fnlist=None, namelist=None, manual_limit=None, manual_check_constraint=None, ## 3) For input_manual mode
                 fn_load='', fna_load='submodels.dat', group_id=None,  ## 1) Load data
                 tolerance=1e-3, silent=False, auto_input=True, ignore_no_files=False,
                 suffix='',
                 comp_crit='RChi_100', input_only=False,
                 extended_itemlist=['CHI2NU', 'RChi_100', 'AIC_100', 'war_tot', 'group_ID']
                ):
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
          * fnlist, namelist, manual_limit, manual_check_constraint
        """
        self.runlist=runlist
        self.galfit_itemlist=['2_XC', '2_YC', '2_MU_E', '2_RE', '2_N', '2_AR', '2_PA', '2_MAG',
                              '3_XC', '3_YC', '3_MU_E', '3_RE', '3_N', '3_AR', '3_PA', '3_MAG']
        self.extended_itemlist=extended_itemlist
        self.tolerance=tolerance  # Chi2 tolerance for "Good results" (|Min chi2 - chi2 |/(Min chi2) < tolerance)
        self.silent=silent
        self.dir_work=dir_work
        self.ignore_no_files=ignore_no_files
        self.comp_crit=comp_crit
        self.suffix=suffix

        if(fna_load==None): fn_load=None
        elif(fn_load=='' and fna_load!=None):
            fn_load=dir_work+fna_load

        if(auto_input==True):
            if(self.silent==False): print("=========== Run ResultGalfit ===========\n")
            ## ========= INPUT ===================
            # INPUT 1) Load data
            if(fn_load!=None and fna_load!=None):
                try: self.input_load(fn_load, group_id=group_id)
                except: pass
            ## INPUT 2) Runlist
            elif(hasattr(runlist, "__len__")):  ## If runlist is not empty, use runlist
                self.input_runlist(runlist=runlist, group_id=group_id, dir_work=dir_work)
            ## INPUT 3) Manual
            elif(hasattr(fnlist, "__len__")):   ## If fnlist is not empty, use fnlist
                self.input_manual(fnlist=fnlist, namelist=namelist,
                                  manual_limit=manual_limit, manual_check_constraint=manual_check_constraint)

            if(input_only==True): return

            ## Find best
            if(self.Data.Ndata!=0):
                self.best=self.find_best(remove_warn=True)
                self.best_warn=self.find_best(remove_warn=False)
                self.best_near=self.find_near_best(self.Data.val[comp_crit][self.best],
                                                   tolerance=self.tolerance, comp_crit=comp_crit)
            else: ## No data
                self.best, self.best_warn, self.best_near = None, None, None

    ## ================== INPUT methods ======================
    def input_load(self, fn_load, group_id=None):
        """
        Descr - Manually input data
        ** See input_runlist as well
        INPUT
         - fnlist
         - namelist
         * manual_limit : It will be used for check_bound()
         * manual_check_constraint : It will be used for check_bound()
         ** if the class has 'runlist', manual_limit and manual_check_constraint will be gotten from the runlist
        """
        if(self.silent==False): print("\n● Input load data")
        try:
            with open(fn_load) as f:
                header1 = f.readline().split()
                header2 = f.readline().split()
                header3 = f.readline().split()
                header3 = header3[1]
            self.Data=ResGalData(namelist=header1[1:], fnlist=header2[1:],
                                data_type=self.galfit_itemlist+self.extended_itemlist)
            if(self.silent==False): print(">> Reading success!")
        except:
            if(self.silent==False): print(">> Fail to load the data!")
            raise Exception("Fail to load the data!")

        if(self.silent==False):
            try: print("\n>> Replace dir_work : ", self.dir_work, "to", header3)
            except: print("\n>> New dir_work : ", header3)
        self.dir_work = header3

        dat=np.loadtxt(fn_load)
        val=dat[dat[:,0]==1]
        err=dat[dat[:,0]==2]
        war=dat[dat[:,0]==3]
        self.Data.bulk_input(val[:,1:], err[:,1:], war[:,1:]) # Remove data type (0=data, 1=err, 2=war)
        if(group_id!=None):
            cutlist=np.where(self.Data.val['group_ID']==group_id)[0]
            if(len(cutlist)==0): print("Cannot cut data!")
            else: self.Data.cut_data(cutlist)
        self.safe_itemlist=self.Data.clean_data(self.galfit_itemlist)

    def input_manual(self, fnlist, namelist, runlist=None, manual_limit=None, manual_check_constraint=None):
        """
        Descr - Manually input data
        ** See input_runlist as well
        INPUT
         - fnlist
         - namelist
         * manual_limit : It will be used for check_bound()
         * manual_check_constraint : It will be used for check_bound()
         ** if the class has 'runlist', manual_limit and manual_check_constraint will be gotten from the runlist
        """
        if(self.silent==False):
            if(hasattr(runlist, "__len__")==False):
                print("\n● Input data with the manual mode")

        self.Data=ResGalData(namelist=namelist, fnlist=fnlist,
                            data_type=self.galfit_itemlist+self.extended_itemlist)

        if(self.silent==False): print(">> # of data : ", len(fnlist))
        if(len(self.Data.namelist)==0):
            if(self.ignore_no_files==False): raise Exception("Namelist is empty!")
            else: print(">> ♣♣♣ Warning! ♣♣♣ Namelist is empty!")
        self.Data.read_data(itemlist=self.galfit_itemlist)

        if(self.Data.Ndata==0):
            if(self.ignore_no_files==False): raise Exception("No files!")
            else: print(">> ♣♣♣ Warning! ♣♣♣ No files!")
        if(self.silent==False): print(">> # of read data : ", self.Data.Ndata)

        ## If it has runlist
        if(hasattr(runlist, "__len__")):
            self.Data.val['group_ID']=runlist['group_ID']
            self.Data.err['group_ID']=runlist['group_ID']
            self.Data.war['group_ID']=runlist['group_ID']
            self.check_bound(runlist=runlist)

        ## If not --> Manual check
        else:
            try: self.check_bound(manual_limit=manual_limit, manual_check_constraint=manual_check_constraint)
            except: self.check_bound(check_bound=False)
        self.safe_itemlist=self.Data.clean_data(self.galfit_itemlist)

    def input_runlist(self, runlist, group_id=None, dir_work=None):
        """
        Descr - Input data using the given runlist
        ** It will extract fnlist, namelist, manual_limit, manual_check_constraint and run input_manual
        ** See input_manual as well
        INPUT
         - runlist
         * group_id : (None : see all group_id)
         * dir_work : working dir for data
        """
        if(self.silent==False):
            print("\n● Input data with Runlist")
            print(">> Group ID : ", group_id)

        if(group_id==None): runlist=runlist
        else: runlist=runlist[np.isin(runlist['group_ID'], group_id)]
        namelist=runlist['name']
        fnlist=dharray.array_attach_string(namelist, 'result_', add_at_head=True)
        fnlist=dharray.array_attach_string(fnlist, dir_work, add_at_head=True)
        fnlist=dharray.array_attach_string(fnlist, '.fits')
        self.input_manual(fnlist, namelist, runlist=runlist)


    def check_bound(self, runlist=None, parlist=['2_XC', '2_YC', '2_RE', '2_N', '3_XC', '3_YC', '3_RE', '3_N'],
                    constlist=['lim_pos1', 'lim_pos1', 'lim_reff1', 'lim_n1',
                    'lim_pos2', 'lim_pos2', 'lim_reff2', 'lim_n2'],
                    centervallist=[100,100,0,0,100,100,0,0], check_bound=True,
                    manual_limit=[None]*8,
                    manual_check_constraint=None):

        """
        Descr - Check parameter at the limit
        INPUT
         ** runlist : Automatically get limits (default : None)
         ** parlist : Parameters will be checked
         ** constlist : Corresponding runlist parameters (runlist only)
         ** Centervallist : Central value.
         ** check_bound : If False --> Skip this function
         ** manual_limit : Manually input constraint (e.g., 100 +-20 --> Central value : 20, manual limit = [-20, 20])
                           if manual_limit is not None list, This will ignore runlist value
        """
        if(check_bound==False): # Ignore function
            if(self.silent==False): print(">> Skip check bound")
        if(check_bound==True): # If not, ignore function
            if(self.silent==False):
                print("\n◯ Check bound")
                if(hasattr(runlist, "__len__")): print(">>>> Runlist exists")
                else: print(">>>> Runlist does not exist")

            self.bound=np.copy(self.Data.val)
            self.bound.fill(0)
            for i, par in enumerate(parlist):  # for given parameters
                if(hasattr(manual_limit[i], "__len__")): # If given value exists
                    lims=np.repeat([manual_limit[i]], len(self.Data.val), axis=0)

                else: # If given value does Not exist  -> Auto (Use runlist)
                    lims=np.full((len(self.Data.val), 2), np.nan)
                    for j in range (len(self.Data.val)):
                        lims[j]=runlist[constlist[i]][j]  # Limit array  ## len(runlist) = len(self.val)
                lims=lims+centervallist[i]  # (e.g., 100 +-20 --> Central value : 20, manual limit = [-20, 20])
                lower=self.Data.val[parlist[i]]<=lims[:,0]
                higher=self.Data.val[parlist[i]]>=lims[:,1]
                bound_code=(lower+higher)*2   # Error code : 2
                self.bound[par]=bound_code

            ## Whether use constraint or not
            if(hasattr(manual_check_constraint, "__len__")):
                removelist=np.where(manual_check_constraint==0)
            elif(hasattr(runlist, "__len__")):
                removelist=np.where(runlist['use_lim']==0)
                self.bound[removelist]=0

        self.Data.war=dharray.array_quickname(dharray.array_flexible_to_simple(self.Data.war)
                                          + dharray.array_flexible_to_simple(self.bound),
                                          names=self.Data.war.dtype.names, dtypes=self.Data.war.dtype)

    def find_best(self, remove_warn=True, comp_crit='RChi_100'):
        if(self.silent==False): print("\n● Find the best submodel")
        chi2list=self.Data.val[comp_crit]
        usingarray=np.arange(len(chi2list))
        if(remove_warn==True):
            if(self.silent==False): print("\n>> Without warning")
            usingarray=usingarray[self.Data.war_tot==0]
            ## No array
            if(len(usingarray)==0): return self.find_best(remove_warn=False)
            ## All data is NaN
            checknan=np.sum(np.isnan(chi2list[usingarray]))
            if(checknan==len(usingarray)): return self.find_best(remove_warn=False)


        min_chi=np.nanmin(chi2list[usingarray])
        minpos=np.where(chi2list[usingarray]==min_chi)[0]
        return usingarray[minpos]

    def find_near_best(self, min_chi, tolerance=0, comp_crit='RChi_100'):
        chi2list=self.Data.val[comp_crit]
        if(hasattr(min_chi, "__len__")): min_chi=np.nanmin(min_chi)
        self.chi2_diff=np.abs(chi2list-min_chi)/min_chi
        nearpos=np.where(self.chi2_diff<tolerance)[0] ## Fraction
        return nearpos

    def save_data(self, fna_output='submodels.dat', fn_output='', return_array=False):
        if(fna_output==None): fn_output=None
        elif(fn_output==''): fn_output=self.dir_work+fna_output
        if(self.silent==False): print("\n● Save the result : ", fn_output)
        return self.Data.save_data(fn_output=fn_output, add_header=self.dir_work, return_array=return_array)

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

        newitemlist=np.append(['group_ID'], newitemlist)
        if(hide_fail_read==True): newfilelist=self.Data.is_file_exist
        else: newfilelist=np.full(len(self.Data.is_file_exist), True) ## All the files
        errorlist=self.Data.namelist[self.Data.war_tot.astype(bool)]

        newitemlist_chi2=np.append(newitemlist, ['CHI2NU', 'RChi_100', 'AIC_100'])
        newitemlist_chi2=np.append(newitemlist, ['CHI2NU', 'RChi_100', 'AIC_100'])
        self.df_display(self.Data.val[newfilelist][newitemlist_chi2], newfilelist, errorlist, caption='Values')

        if(only_val==False):
            self.df_display(self.Data.err[newfilelist][newitemlist], newfilelist, errorlist, caption='Error')
            self.df_display(self.Data.war[newfilelist][newitemlist], newfilelist, errorlist, caption='Warning', is_limit_format=True)

    def df_display(self, data, newfilelist, errorlist, caption='datatype', is_limit_format=False):
        df=pd.DataFrame(data, index=self.Data.namelist[newfilelist])
        if(is_limit_format==True): formatting='{:.0f}'
        else: formatting='{:f}'
        df=df.style.apply(lambda x: ['background: lightyellow' if x.name in self.Data.namelist[self.best_near]
                                      else '' for i in x],
                           axis=1)\
            .apply(lambda x: ['background: lightgreen' if x.name in self.Data.namelist[self.best]
                                      else '' for i in x],
                           axis=1)\
            .apply(lambda x: ['color: red' if x.name in errorlist
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


##=============== PostProcessing ====================
def convert_base_params_array(usingdat):
    comp=['XC', 'YC', 'MU_E', 'RE', 'N', 'AR', 'PA']
    comp2=dharray.array_attach_string(comp, '2_', add_at_head=True)
    comp3=dharray.array_attach_string(comp, '3_', add_at_head=True)
    base_params1_array=np.full((len(usingdat),7), np.nan)
    base_params2_array=np.full((len(usingdat),7), np.nan)
    for i in range (len(comp)):
        base_params1_array[:,i]=usingdat[comp2[i]]
        base_params2_array[:,i]=usingdat[comp3[i]]

    psfpos=np.isnan(base_params2_array[:,2]) ## For 3rd comp --> it might be PSF
    base_params2_array[psfpos,2]=usingdat[psfpos]['3_MAG'] # Mu_e = mag
    return psfpos, base_params1_array, base_params2_array

def postprocessing(Runlist, dir_work, fn_sigma='', fn_masking='', suffix='',
                   fna_sigma='sigma_lg.fits', fna_masking='masking_double2_g.fits',
                   fna_image='image_lg.fits', fna_psf='psf_area_lg.fits',
                   center_size=100, repair_fits=True, image_size=200,
                   header_item='Chi_100', header_item_aic='AIC_100',
                   centermag='sersic', sersic_res=500,
                   overwrite=False, silent=True, print_galfit=False):
    dir_work=str(dir_work)
    if(fna_sigma==None): fn_sigma=None
    if(fna_masking==None): fn_masking=None
    if((fn_sigma=='') & (fna_sigma!=None)): fn_sigma=dir_work+fna_sigma
    if((fn_masking=='') & (fna_masking!=None)): fn_masking=dir_work+fna_masking

    ##chi2
    fnlist=dharray.array_attach_string(Runlist.namelist, 'result_', add_at_head=True)
    fnlist=dharray.array_attach_string(fnlist, dir_work, add_at_head=True)
    fnlist=dharray.array_attach_string(fnlist, '.fits')
    fnlist=np.array(fnlist)
    cal_update_chi2(fnlist, fn_sigma, fn_masking, center_size, repair_fits, image_size,
                   header_item, header_item_aic, overwrite, silent=silent)

    ##centermag
    if(centermag!=None):
        ResThis=ResultGalfit(Runlist.runlist,
                                dir_work=dir_work, group_id=None, auto_input=True,
                                fna_load=None,
                                silent=True, ignore_no_files=True)
        ###================ Mode 1 : Sersic calc. =================
        if(centermag=='sersic'):
            tmag2, sersiclist2=add_totalmag(ResThis.Data.val, comp=2, sersic_res=sersic_res)
            tmag3, sersiclist3=add_totalmag(ResThis.Data.val, comp=3, sersic_res=sersic_res)
            ## INPUT
            for i, fn in enumerate (fnlist[sersiclist2]):
                inputvalue="%.4f"%tmag2[i]
                inputvalue+=" +/- 0.0000"
                fits.setval(fn, '2_MAG', value=inputvalue, ext=2, after='2_MU_E')
            for i, fn in enumerate (fnlist[sersiclist3]):
                inputvalue="%.4f"%tmag3[i]
                inputvalue+=" +/- 0.0000"
                fits.setval(fn, '3_MAG', value=inputvalue, ext=2, after='3_MU_E')

        ###================ Mode 2 : Galfit =================
        if(centermag=='galfit'):
            fnlist_center=dharray.array_attach_string(Runlist.namelist, 'result_cimsi_', add_at_head=True)
            fnlist_center=dharray.array_attach_string(fnlist_center, dir_work, add_at_head=True)
            fnlist_center=dharray.array_attach_string(fnlist_center, '.fits')
            fnlist_center=np.array(fnlist_center)

            psfpos, base_params1_array, base_params2_array=convert_base_params_array(ResThis.Data.val)
            sersiclist2=np.arange(len(base_params1_array))
            sersiclist3=np.delete(sersiclist2, psfpos)

            Runlist.run_galfit_with_runlist(dir_work=dir_work, suffix=suffix,
                            fna_inputimage=fna_image, fna_inputmask=fna_masking, ## Masking -> same region
                            fna_psf=fna_psf, fna_sigma=fna_sigma,
                            use_empty_sersic=True,
                            base_params1_array=base_params1_array,
                            base_params2_array=base_params2_array,
                            base_params1_mode='replace',
                            base_params2_mode='replace',
                            is_allow_vary1=False, is_allow_vary2=False,
                            is_2nd_mag_offset=None,
                            overwrite=overwrite, prefix='cimsi_', silent=silent,
                            print_galfit=print_galfit)


            ## It will use fna_ not fn_!
            for i, fn in enumerate (fnlist):
                if(ResThis.Data.is_file_exist[i]==False): continue

                try:
                    head=fits.getheader(fnlist_center[i], ext=2)
                except:
                    print(">> File read error! ", fnlist_center[i])
                    continue

                tmag2=head['2_MAG']
                fits.setval(fn, '2_MAG', value=tmag2, ext=2, after='2_MU_E')
                if(np.isin(i, sersiclist3)==True):
                    tmag3=head['3_MAG']
                    fits.setval(fn, '3_MAG', value=tmag3, ext=2, after='3_MU_E')


    if(silent==False): print("Done")

def add_totalmag(datarray, comp=2, sersic_res=500):
    ## Integral mag
    if(comp==2):
        sersiclist=np.where(np.isfinite(datarray['2_MU_E']))
        usingdata=datarray[sersiclist]
        intmag=sersic_integral2D(mu_e=usingdata['2_MU_E'], reff=usingdata['2_RE'],
                           n=usingdata['2_N'], ar=usingdata['2_AR'], plate_scale=plate_scale, res=sersic_res)
    elif(comp==3):
        sersiclist=np.where(np.isfinite(datarray['3_MU_E']))
        usingdata=datarray[sersiclist]
        intmag=sersic_integral2D(mu_e=usingdata['3_MU_E'], reff=usingdata['3_RE'],
                           n=usingdata['3_N'], ar=usingdata['3_AR'], plate_scale=plate_scale, res=sersic_res)
    return intmag, sersiclist[0]





class ResultGalfitBest(ResultGalfit):
    def __init__(self, MomRes, silent=True, include_warn=False, comp_crit='RChi_100'):
        self.silent=silent
        self.MomRes=MomRes
        self.galfit_itemlist=MomRes.galfit_itemlist
        self.dir_work=MomRes.dir_work
        val_ori=MomRes.Data.val

        self.grouplist=np.unique(val_ori['group_ID'])
        self.grouplist=self.grouplist[np.isfinite(self.grouplist)]
        self.group_non_empty=np.full(len(self.grouplist), True)
        self.comp_crit=comp_crit
        index_total=np.arange(len(val_ori))

        ## Best models & removefnlist
        self.bestsubmodels=np.zeros(len(self.grouplist), dtype=object) # Best models (w/o warning)
        self.bestsubmodels_warn=np.zeros(len(self.grouplist), dtype=object) #Best models w/ w/o warning
        self.chi2_submodels=np.zeros(len(self.grouplist), dtype=object) # Chi2 (or AIC) for submodels
        self.chi2_bestsubmodels=np.zeros(len(self.grouplist), dtype=object) # Chi2 (or AIC) for bestsubmodels
        self.removefnlist=np.zeros(len(self.grouplist), dtype=object) # bad results fnlist
        for i, group_id in enumerate(self.grouplist):
            selected=np.where(val_ori['group_ID']==group_id)[0]
            self.Data=copy.deepcopy(MomRes.Data)
            self.Data.cut_data(selected)
            index_this=index_total[selected]  # Index w.r.t. total data

            ## If Ndata==0
            if(self.Data.Ndata==0):
                self.group_non_empty[i]=False
            else:
                # if the best submodel does not have error -> remove all except for the best submodel
                # if the best submodel has error -> find the best submodel without error (worse than the best)
                # after that, remove submodels worse than the best submodel w/o error
                index=self.find_best(remove_warn=True)
                self.bestsubmodels[i]=index_this[index]
                self.chi2_submodels[i]=np.copy(self.Data.val[comp_crit])
                self.chi2_bestsubmodels[i]=np.copy(self.Data.val[comp_crit][index])

                ## remove list
                badresults=np.where(self.chi2_submodels[i]>np.nanmin(self.chi2_bestsubmodels[i]))
                self.removefnlist[i]=self.Data.fnlist[badresults]

                #Best models w/ w/o warning
                index=self.find_best(remove_warn=False)
                self.bestsubmodels_warn[i]=index_this[index]


        ## clean up results

        if(np.sum(self.group_non_empty)==0): ## No data
            self.bestsubmodels_flat=None
            self.bestsubmodels_warn_flat=None
            self.Data=copy.deepcopy(MomRes.Data)
            self.safe_itemlist=self.Data.clean_data(self.galfit_itemlist)
        else:
            self.bestsubmodels_flat=dharray.array_flatten(self.bestsubmodels)
            self.bestsubmodels_warn_flat=dharray.array_flatten(self.bestsubmodels_warn)
            self.Data=copy.deepcopy(MomRes.Data)
            if(include_warn==True): self.Data.cut_data(self.bestsubmodels_warn_flat)
            else: self.Data.cut_data(self.bestsubmodels_flat)

            ## Best of the Best
            self.safe_itemlist=self.Data.clean_data(self.galfit_itemlist)
            self.best=self.find_best(remove_warn=True)
            self.best_warn=self.find_best(remove_warn=False)
            self.best_near=self.find_near_best(self.Data.val[comp_crit][self.best], tolerance=MomRes.tolerance)

    def remove_bad_results(self, save_backup=True):
        if(save_backup==True): self.MomRes.save_data(fna_output='autosave.dat')
        for thisrmlist in self.removefnlist:
            for fn in thisrmlist:
                try: os.remove(fn)
                except: continue


##=================== SWAP band ===============================
def galfit_swap_band(runlist, ResBest, silent=False, print_galfit=False):
    ## For loop for the best models
    using_best_submodels=ResBest.bestsubmodels[ResBest.group_non_empty] ## No data -> Skip
    for i in range (len(using_best_submodels)):
        print("\n>> Progress : ", i, "/", len(using_best_submodels))
        thismodel=using_best_submodels[i]
        thisresult=ResBest.Data.val[i]
        thisrunlist=runlist[thismodel][0]
        dir_work=ResBest.dir_work
        outputname=thisrunlist['name']+'_r'
        fitsname=dir_work+'result_'+outputname+'.fits'

        swap=AutoGalfit(fn_galfit=fn_galfit, dir_work=dir_work,
                       fna_inputimage='image_r.fits', outputname=outputname,
                        fna_psf='psf_area_r.fits', fna_sigma='sigma_r.fits', fna_inputmask='masking.fits',
                       size_conv=thisrunlist['size_conv'],  size_fitting=thisrunlist['size_fitting'],
                      silent=silent, prefix='', suffix='')

        swap.add_fitting_comp(fitting_type=thisrunlist['compname'][0],
                              est_xpos=thisresult['2_XC'], est_ypos=thisresult['2_YC'],
                              est_mag=thisresult['2_MU_E'], est_reff=thisresult['2_RE'],
                              est_n=thisresult['2_N'],
                              est_axisratio=thisresult['2_AR'], est_pa=thisresult['2_PA'],
                              is_allow_vary=False)

        ## Double comp
        if(len(thisrunlist['compname'])!=1):
            maglist=np.array([thisresult['3_MU_E'], thisresult['3_MAG']])
            try: usingmag=np.nanmax(maglist)
            except: usingmag=thisresult['2_MU_E']
            print(">>>> using mag : ", usingmag)

            swap.add_fitting_comp(fitting_type=thisrunlist['compname'][1],
                                  est_xpos=thisresult['3_XC'], est_ypos=thisresult['3_YC'],
                                  est_mag=usingmag, est_reff=thisresult['3_RE'],
                                  est_n=thisresult['3_N'],
                                  est_axisratio=thisresult['3_AR'], est_pa=thisresult['3_PA'],
                                  is_allow_vary=False)
        swap.run_galfit(print_galfit=print_galfit)
        cal_update_chi2([fitsname], dir_work+'sigma_r.fits', center_size=100,
                                 fn_masking=dir_work+'masking.fits', repair_fits=True, overwrite=True)
        print(fitsname)
        print(">> Update Chi2")

class ResultGalfitSwap(ResultGalfit):
    def __init__(self, OriResBest=None, dir_work=None,  ## For input_runlist mode
             fn_load=None, fna_load='submodels_swap.dat',  ## Load data
             tolerance=1e-3, silent=False, auto_input=True, ignore_no_files=False):
        #self.runlist=runlist
        self.OriResBest=OriResBest
        self.galfit_itemlist=['2_XC', '2_YC', '2_MU_E', '2_RE', '2_N', '2_AR', '2_PA']
        self.galfit_itemlist+=['3_XC', '3_YC', '3_MU_E', '3_RE', '3_N', '3_AR', '3_PA']
        self.galfit_itemlist+=['3_MAG']
        self.extended_itemlist=['CHI2NU', 'RChi_100', 'AIC_100', 'war_tot', 'group_ID']
        self.ignore_no_files=ignore_no_files
        self.tolerance=tolerance  # Chi2 tolerance for "Good results" (|Min chi2 - chi2 |/(Min chi2) < tolerance)
        self.silent=silent
        self.dir_work=dir_work


        if(fn_load==None and fna_load!=None and dir_work!=None):
            fn_load=dir_work+fna_load

        if(auto_input==True):
            if(self.silent==False): print("=========== Run ResultGalfit ===========\n")
            ## ========= INPUT ===================
            if(fn_load!=None):
                try: self.input_load(fn_load, group_id=group_id)
                except: pass
            else:  ## If runlist is not empty, use runlist
                self.input_copy()

            if(self.Data.Ndata!=0):
                self.best=self.find_best(remove_warn=True)
                self.best_warn=self.find_best(remove_warn=False)
                self.best_near=self.find_near_best(self.Data.val[comp_crit][self.best], tolerance=self.tolerance)
            else: ## No data
                self.best, self.best_warn, self.best_near = None, None, None

    def input_copy(self):
        if(self.silent==False): print("● Copy previous results\n")
        self.Data=copy.deepcopy(self.OriResBest.Data)
        using_best_submodels=self.OriResBest.bestsubmodels[self.OriResBest.group_non_empty] ## No data -> Skip
        for i in range (len(using_best_submodels)):
            if(self.silent==False): print(">>", i, "/", len(using_best_submodels))

            print(self.OriResBest.dir_work+'result_'+self.OriResBest.Data.namelist[i]+'_r.fits')

            ImsiRes=ResultGalfit(runlist=None, fna_load=None, dir_work=self.OriResBest.dir_work, group_id=None,
                                          fnlist=[self.OriResBest.dir_work+'result_'+self.OriResBest.Data.namelist[i]+'_r.fits'], namelist=['b1'],
                                         silent=False, auto_input=True, ignore_no_files=True)
            using_data_type=np.copy(self.OriResBest.Data.data_type[:-4])
            for item in using_data_type:
                if(ImsiRes.Data.err[0][item]>0):  # Fixed : err = -999. Select non fixed
                    self.Data.val[i][item]=np.copy(ImsiRes.Data.val[0][item])
                    self.Data.err[i][item]=np.copy(ImsiRes.Data.err[0][item])
                    self.Data.war[i][item]=np.copy(ImsiRes.Data.war[0][item])
        self.safe_itemlist=self.Data.clean_data(self.galfit_itemlist)
