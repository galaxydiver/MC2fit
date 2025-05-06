import DH_array as dharray
import numpy as np
import os
import random
import time
import pickle
from astropy.io import fits
from dataclasses import dataclass
from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import pandas as pd

from astropy import units as u
from astropy.convolution import Gaussian2DKernel
from astropy.utils.data import download_file, get_pkg_data_filename
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval,ImageNormalize,AsinhStretch

import subprocess
from multiprocessing import Pool
import warnings

#=================== AUTO Galfit ==========================

@dataclass
class AutoGalfit:
    dir_base : str  # Base directory. Input files will be imported from here.
    dir_work : str  # Working directory. Output files will be generated to here. If it is not exist, make dir.
    outputname : str # Extension for name of configuration file
    suffix : str = ''  # suffix for name of configuration file
    magz : float = 22.5 # magnitude zero point
    platescl : np.ndarray = np.array([0.262, 0.262])  # Thumbnail pixel scale in arcsec
    size_fitting : int = -1  # Fix the fitting area       ## -1 -> Full image
    size_conv : int = -1  # Convolution size.                               ## -1 --> Auto (size of the PSF image)
    silent : bool =False # Print message or not
    info_from_image: bool = True # Read info from the input image
    
    fn_galfit: str = '/home/donghyeon/Downloads/galfit3.0.7b/galfit3.0.7b/galfit'  #Galfit direction    
    fn_inputimage : str ='test.fits' # Name of fits file to analyze
    fn_inputmask : str ='testmask.fits' # Name of mask file for analysis

    fn_psf : str = 'psf_center_g.fits'
    fn_sigma : str = 'invvar_g.fits'
        
    est_sky : float = 10  # Estimate of sky level        
        
    def __post_init__(self):
        self.dir_workfull=self.dir_base+self.dir_work
        os.makedirs(self.dir_workfull, exist_ok=True)
        
        fn, _ext = os.path.splitext(self.outputname)
        self.n_comp=int(0)
        
        self.comp_dtype=[('fitting_type', '<U10'), ('est_xpos', '<f8'), ('est_ypos', '<f8'), 
                         ('est_mag', '<f8'), ('est_reff', '<f8'), ('est_n', '<f8'), 
                         ('est_axisratio', '<f8'), ('est_pa', '<f8')]
        
        #Filenames
        self._set_path_name()

        if(self.info_from_image==True): self.read_info() # Read info from the input image
        self.generate_galconfig()
        self.generate_constraints()
    
    def _set_path_name(self):
        self.fn_gal_conf = self.dir_workfull + 'galfit_param_' + self.outputname+ self.suffix + ".dat" # Configuration file
        self.fn_constraint=self.dir_workfull + 'constraints.dat'  ## We can use a single constraint file
        
        self.fn_psf = self.dir_workfull + self.fn_psf   # PSF file
        try: self.psf_size=np.shape(fits.getdata(self.fn_psf))[0]
        except: print(">> PSF file does not exist!")
        
        if(self.fn_sigma=='none'): pass
        else: self.fn_sigma = self.dir_workfull + self.fn_sigma   # sigma file
        if(self.fn_inputmask=='none'): pass
        else: self.fn_inputmask=self.dir_workfull+self.fn_inputmask
        
        self.fn_output_imgblock=self.dir_workfull+'result_'+self.outputname + self.suffix + ".fits"
        self.fn_inputimage=self.dir_workfull+self.fn_inputimage
        
    def read_info(self):
        if(self.silent==False): 
            print("● Input data from the input image header ", self.fn_inputimage)
            # check_fits(self.fn_inputimage)
        hdu = fits.open(self.fn_inputimage)
        header=hdu[0].header
        # self.magz=header['MAGZERO']
        # self.platescl=np.array([header['PIXSCAL1'], header['PIXSCAL2']])
        self.image_size=np.shape(hdu[0].data)[0]
    
            
    def add_fitting_comp(self, fitting_type='sersic', est_xpos=None, est_ypos=None, est_mag=20, est_reff=20, est_n=4,
                        est_axisratio=1, est_pa=0, is_allow_vary=True):
        self.n_comp+=1
        if(self.silent==False): print("● Added a fitting component ", self.n_comp)
        index=int(self.n_comp-1)
        
        ## Generate Array
        if(self.n_comp!=1):
            temp_comparray=np.zeros(self.n_comp, dtype=self.comp_dtype)
            temp_comparray[:-1]=np.copy(self.comp)
            self.comp=np.copy(temp_comparray)
        else:
            self.comp=np.zeros(self.n_comp, dtype=self.comp_dtype)
            
        ## default position
        if(type(est_xpos)==type(None)): est_xpos=self.image_size/2
        if(type(est_ypos)==type(None)): est_ypos=self.image_size/2
            
        self.comp[index]['fitting_type']=np.copy(fitting_type)
        self.comp[index]['est_xpos']=est_xpos
        self.comp[index]['est_ypos']=est_ypos
        self.comp[index]['est_mag']=est_mag
        self.comp[index]['est_reff']=est_reff
        self.comp[index]['est_n']=est_n
        self.comp[index]['est_axisratio']=est_axisratio
        self.comp[index]['est_pa']=est_pa
        
        if(self.silent==False): 
            print(">> Input Data : ")
            print(self.comp.dtype)
            print(self.comp)
        
        fconf = open(os.path.join(self.fn_gal_conf), 'a')
        fconf.write('\n# Component number: %d\n'%int(self.n_comp+1))
        fconf.write(' 0) %s                     #  object type\n'%fitting_type)
        fconf.write(' 1) %f  %f   %d %d         #  position x, y\n' %(est_xpos, est_ypos, is_allow_vary, is_allow_vary))
        if(fitting_type=='psf'):
            if(self.silent==False): print("PSF")
            fconf.write(' 3) %f      1              #  Total magnitude\n'  %est_mag) 
        else:
            fconf.write(' 3) %f      1              #  Integrated magnitude\n'  %est_mag) 
            fconf.write(' 4) %f      %d             #  R_e (half-light radius)   [pix]\n'  %(est_reff, is_allow_vary))
            fconf.write(' 5) %f      %d             #  Sersic index n (exponential n=1)\n'  %(est_n, is_allow_vary))
            fconf.write(' 6) 0.0         0          #     -----\n') 
            fconf.write(' 7) 0.0         0          #     -----\n') 
            fconf.write(' 8) 0.0         0          #     -----\n') 
            fconf.write(' 9) %f      %d             #  axis ratio (b/a)\n' %(est_axisratio, is_allow_vary))
            fconf.write('10) %f      %d             #  position angle (PA) [deg: Up=0, Left=90]\n'  %(est_pa, is_allow_vary))
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
        
        
    def add_constraints(self, id_comp, poslim=[-10, 10], nlim=[0.2, 1.5]):
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
                 scale_percentile=None, fix_raw=False, dpi=100, fn_save_image='UDG_example.png'): 
    
    # Check chi^2
    chi_list=np.full(3, np.nan)
    for i in range (len(fnlist)):
        hdu=fits.open(fnlist[i])
        chi_list[i]=hdu[2].header['CHI2NU']
                    
    # Drawing
    fig=plt.figure(figsize=(13,13))
    gs=gridspec.GridSpec(3,3)
    gs.update(hspace=0.05)
    
    if(fix_raw==True): Nraw=3
    else: Nraw=len(fnlist)
        
    for i in range (Nraw):
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
    compname : np.ndarray = np.array(['sersic2', 'sersic2'])  # list of components (Ncomp = len(components))
    namelist : np.ndarray = np.array(['ss_23', 'ss_24', 'ss_25', 'ss_26']) # Namelist (Nset = len(namelist)
    est_maglist1 : np.ndarray = np.array([23, 24, 25, 26]) # est mag list for comp1
    est_maglist2 : np.ndarray = np.array([23, 24, 25, 26]) # est mag list for comp2
    est_refflist1 : float = 30 # est Reff list for comp1
    est_refflist2 : float = 30 # est Reff list for comp2
    est_nlist1 : float  = 1 # est sersic index (n) list for comp1
    est_nlist2 : float  = 3 # est sersic index (n) list for comp2
    
    use_constraint : bool = False
    lim_pos1 : np.ndarray = np.array([-20, 20]) # position constraint for comp1
    lim_pos2 : np.ndarray = np.array([-20, 20]) # position constraint for comp2
    lim_n1 : np.ndarray = np.array([0.2, 5.5]) # sersic index (n) constraint for comp1
    lim_n2 : np.ndarray = np.array([0.2, 5.5]) # sersic index (n) constraint for comp1
    fn_galfit = '/home/donghyeon/Downloads/galfit3.0.7b/galfit3.0.7b/galfit'
    
    group_id : int = -1
    size_conv : int = -1
    size_fitting : int = -1
        
    def __post_init__(self):
        if(type(self.compname)==str): print("Error! compname is not an array!")
        if(type(self.namelist)==str): print("Error! namelist is not an array!")
        self.Ncomp = len(self.compname)
        self.Nset = len(self.namelist)
        
        ## Repeat except array
        self.est_maglist1=dharray.repeat_except_array(self.est_maglist1, self.Nset)
        self.est_maglist2=dharray.repeat_except_array(self.est_maglist2, self.Nset)
        self.est_refflist1=dharray.repeat_except_array(self.est_refflist1, self.Nset)
        self.est_refflist2=dharray.repeat_except_array(self.est_refflist2, self.Nset)
        self.est_nlist1=dharray.repeat_except_array(self.est_nlist1, self.Nset)
        self.est_nlist2=dharray.repeat_except_array(self.est_nlist2, self.Nset)
        self.use_constraint=dharray.repeat_except_array(self.use_constraint, self.Nset)
        # self.lim_pos1=np.repeat(self.lim_pos1, self.Nset, axis=0)
        # self.lim_pos2=np.repeat(self.lim_pos2, self.Nset, axis=0)
        # self.lim_n1=np.repeat(self.lim_n1, self.Nset, axis=0)
        # self.lim_n2=np.repeat(self.lim_n2, self.Nset, axis=0)
        
        ## Generate array
        runlist_dtype=[('name', '<U20'), ('compname', object), ('est_mag1', '<f8'), ('est_mag2', '<f8'), 
                        ('est_reff1', '<f8'), ('est_reff2', '<f8'), ('est_n1', '<f8'), ('est_n2', '<f8'), 
                       ('size_conv', '<i4'), ('size_fitting', '<i4'), 
                       ('lim_pos1', object), ('lim_pos2', object), ('lim_n1', object), ('lim_n2', object),
                       ('use_lim', '<i4'), ('group_ID', '<i4'),
                      ]
        
        self.runlist=np.zeros(self.Nset, dtype=runlist_dtype)
        loop=list(['name', 'est_mag1', 'est_mag2', 
                   'est_reff1', 'est_reff2', 'est_n1', 'est_n2', 
                   'size_conv', 'size_fitting' , 'use_lim'])
        loopdata=[self.namelist, self.est_maglist1, self.est_maglist2, 
                  self.est_refflist1, self.est_refflist2, self.est_nlist1, self.est_nlist2,
                 self.size_conv, self.size_fitting, self.use_constraint
                 ]
        
        for i, item in enumerate(loop):
            self.runlist[item]=loopdata[i]
        self.runlist['group_ID'] = self.group_id
        self.runlist['compname']=[self.compname]
        self.runlist['lim_pos1']=[self.lim_pos1]
        self.runlist['lim_pos2']=[self.lim_pos2]
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
        
    def show_runlist(self, show_full=False):
        if(show_full==True): display(pd.DataFrame(self.runlist))
        else:
            showlist=['name', 'compname', 'est_mag1', 'est_mag2', 'est_reff1', 'est_reff2', 
                      'est_n1', 'est_n2', 'use_lim', 'group_ID']
            display(pd.DataFrame(self.runlist[showlist]))
        
    def run_galfit_with_runlist(self, dir_base, dir_work, fn_inputmask='none', suffix='', 
                                silent=False, overwrite=False, print_galfit=False, group_id=-1):
        ## usingrunlist - based on group_ID
        if(group_id<0): usingrunlist=self.runlist
        else: usingrunlist=self.runlist[np.isin(self.runlist['group_ID'], group_id)]
        
        ## ================Loop=================
        for i_run in range (len(usingrunlist)):
            thisrunlist=usingrunlist[i_run]
            outputname=thisrunlist['name']

            ## Check whether the result is exist
            if(overwrite==False): 
                if(os.path.exists(dir_base+dir_work+'result_'+outputname+suffix+".fits")==True): continue  # Skip

            ## Main
            run=AutoGalfit(fn_galfit=self.fn_galfit, dir_base=dir_base, dir_work=dir_work, 
                           fn_inputimage='image.fits', outputname=outputname,
                            fn_psf='psf_area_g.fits', fn_sigma='invvar_g.fits', fn_inputmask=fn_inputmask, 
                           size_conv=thisrunlist['size_conv'], size_fitting=thisrunlist['size_fitting'],
                          silent=silent, suffix=suffix)

            ## Single comp
            run.add_fitting_comp(fitting_type=thisrunlist['compname'][0], 
                                 est_mag=thisrunlist['est_mag1'], est_n=thisrunlist['est_n1'], 
                                 est_reff=thisrunlist['est_reff1'])
            if(thisrunlist['use_lim']==1): run.add_constraints(2, poslim=thisrunlist['lim_pos1'], 
                                                                       nlim=thisrunlist['lim_n1'])
            
            ## Double comp
            if(len(thisrunlist['compname'])!=1):
                run.add_fitting_comp(fitting_type=thisrunlist['compname'][1], 
                                     est_mag=thisrunlist['est_mag2'], est_n=thisrunlist['est_n2'], 
                                     est_reff=thisrunlist['est_reff2'])
                if(thisrunlist['use_lim']==1): run.add_constraints(3, poslim=thisrunlist['lim_pos2'], 
                                                                           nlim=thisrunlist['lim_n2'])
             
            ## Run
            run.run_galfit(print_galfit=print_galfit)
            
            
            
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
    dum=rawdata.split('*') # When fitting is strange, it has *
    warning=0
    if(len(dum)==1):  # the value does not have '*'
        imsi=rawdata.split()
    else:  # the value has '*'
        imsi=rawdata.replace('*', '').split()
        warning=1
        
    value=imsi[0]
    err=imsi[2]
    return value, err, warning

def read_galfit_header_singlevalues(fnlist, itemlist=['CHI2NU'], return_success=True):
    result_array=np.full((len(fnlist),len(itemlist)), np.NaN)
    result_array=dharray.array_quickname(result_array, itemlist)
    read_success=np.zeros(len(fnlist)).astype(bool) ## Retuen whether file is exist
    for i in range (len(fnlist)):
        try: 
            h=fits.getheader(fnlist[i], ext=2)
            for j, name in enumerate(itemlist):
                try: 
                    result_array[i][name]=h[name]
                    read_success[i]=True
                except: 
                    read_success[i]=False
                    continue
        except:
            continue
    if(return_success==False): return result_array
    else: return result_array, read_success
    
    
#============================== Result Galfit =====================
runlist_itemlist=['2_XC', '2_YC', '2_MU_E', '2_RE', '2_N']
runlist_itemlist+=['3_XC', '3_YC', '3_MU_E', '3_RE', '3_N']
runlist_itemlist+=['3_MAG']

class ResultGalfit:
    def __init__(self, runlist=None, group_id=None, dir_data='./NewTest/test2/',
                 inputmode='auto', fnlist=[], allow=1e-4, namelist=None, fn_load=None, getbest=False):
        
        self.runlist=runlist
        self.itemlist=runlist_itemlist
        self.allow=allow
        self.dir_data=dir_data        
        if(getbest==True): self.getbest(runlist)
        elif(hasattr(runlist, "__len__")):
            self.input_runlist(runlist=runlist, group_id=group_id, dir_data=dir_data)

#         if(self.fn_load==None): self._read_data()
#         else: self._load_data(fn_load)

        self._make_war_tot()
        self.best=self.find_best(remove_warn=True)
        self.best_warn=self.find_best(remove_warn=False)
        self.best_near=self.find_near_best(self.chi2[self.best], allow=self.allow)
        
    def getbest(self, runlist):
        groupmax=np.unique(runlist['group_ID'])
        groupmax=groupmax[np.isfinite(groupmax)]
        best_fnlist=np.zeros(len(groupmax), dtype=object)
        best_namelist=np.zeros(len(groupmax), dtype=object)
        best_runlist=None
        for i, group_id in enumerate(groupmax):
            imsiResGal=ResultGalfit(runlist=runlist, group_id=group_id, dir_data=self.dir_data)
            best_fnlist[i]=np.array(imsiResGal.fnlist)[imsiResGal.best]
            best_namelist[i]=np.array(imsiResGal.namelist)[imsiResGal.best]
            if(hasattr(best_runlist, "__len__")==False): # First trial
                best_runlist=imsiResGal.runlist[imsiResGal.best]
            else: best_runlist=np.append(best_runlist, imsiResGal.runlist[imsiResGal.best])
            
        best_fnlist=dharray.list_flatten(best_fnlist)
        best_namelist=dharray.list_flatten(best_namelist)
        self.runlist=best_runlist
        self.input_manual(best_fnlist, best_namelist)
        
        
    def input_manual(self, fnlist, namelist, manual_limit=None, manual_check_constraint=None):
        self.fnlist=fnlist
        self.namelist=np.array(namelist)
        if(len(self.namelist)==0): 
            raise Exception("Namelist is empty!")
        self._set_data()
        self._read_data()
        if(np.sum(self.is_file_exist)==0): 
            raise Exception("No files!")
        if(hasattr(self.runlist, "__len__")): self.check_bound(runlist=self.runlist)
        else: 
            try: self.check_bound(manual_limit=manual_limit, manual_check_constraint=manual_check_constraint)
            except: self.check_bound(check_bound=False)
        self._clean_data()
        self._make_war_tot()

        
    def input_runlist(self, runlist, group_id=None, dir_data='./NewTest/test2/'):
        if(group_id==None): self.runlist=runlist
        else: self.runlist=runlist[np.isin(runlist['group_ID'], group_id)]
        namelist=self.runlist['name']
        fnlist=dharray.array_attach_string(namelist, 'result_', add_at_head=True)
        fnlist=dharray.array_attach_string(fnlist, dir_data, add_at_head=True)
        fnlist=dharray.array_attach_string(fnlist, '.fits')
        self.input_manual(fnlist, namelist)
        

    ## ===================== Data Sheet setting -> Reading ===================
    def _set_data(self): # make empty tables
        fnlist=self.fnlist
        itemlist=self.itemlist
        
        self.val=np.full((len(fnlist),len(itemlist)), np.NaN)
        self.val=dharray.array_quickname(self.val, itemlist)
        self.err=np.full((len(fnlist),len(itemlist)), np.NaN)
        self.err=dharray.array_quickname(self.err, itemlist)
        self.war=np.full((len(fnlist),len(itemlist)), np.NaN)
        self.war=dharray.array_quickname(self.war, itemlist)
        self.bound=np.full((len(fnlist),len(itemlist)), 0)  # check_bound will not cover all params --> not NaN
        self.bound=dharray.array_quickname(self.bound, itemlist)
        
    def _read_data(self): 
        fnlist=self.fnlist
        itemlist=self.itemlist
        
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
        self.chi2_flex, self.is_file_exist=read_galfit_header_singlevalues(fnlist, return_success=True)
        self.chi2=self.chi2_flex.view(float) ## Remove flexible type
        
    def _clean_data(self):
        ## Remove void fields
        self.val=dharray.array_remove_void(self.val)
        self.err=dharray.array_remove_void(self.err)
        self.war=dharray.array_remove_void(self.war)
        try: self.bwar=dharray.array_remove_void(self.bwar)
        except: pass
        self.chi2_flex=dharray.array_remove_void(self.chi2_flex)
        
        ## Combine val and chi2
        self.val_chi2=dharray.array_add_columns(self.val, self.chi2_flex)
        
        ## Check save itemlist
        newitemlist=[]  # Save list
        for item in self.itemlist:
            check=np.sum(np.isfinite(self.val[item]))
            if(check!=0): 
                newitemlist+=[item] # If any of data is not NaN -> save (discard the data if all var are NaN)
        self.safe_itemlist=np.array(newitemlist)
        
    def check_bound(self, runlist=None, parlist=['2_XC', '2_YC', '2_N', '3_XC', '3_YC', '3_N'],
                    constlist=['lim_pos1', 'lim_pos1', 'lim_n1', 'lim_pos2', 'lim_pos2', 'lim_n2'],
                    centervallist=[100,100,0,100,100,0], check_bound=True,
                    manual_limit=[None]*6,
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
        if(check_bound==True): # If not, ignore function
            for i, par in enumerate(parlist):  # for given parameters
                if(hasattr(manual_limit[i], "__len__")): # If given value exists
                    lims=np.repeat([manual_limit[i]], len(self.val), axis=0) 
                
                else: # If given value does Not exist  -> Auto (Use runlist)
                    lims=np.full((len(self.val), 2), np.nan)
                    for j in range (len(self.val)):
                        lims[j]=runlist[constlist[i]][j]  # Limit array  ## len(runlist) = len(self.val)
                
                lims=lims+centervallist[i]  # (e.g., 100 +-20 --> Central value : 20, manual limit = [-20, 20])
                lower=self.val[parlist[i]]<=lims[:,0]
                higher=self.val[parlist[i]]>=lims[:,1]
                bound=(lower+higher)*2   # Error code : 2
                self.bound[par]=bound
                
            ## Whether use constraint or not
            if(hasattr(manual_check_constraint, "__len__")):
                removelist=np.where(manual_check_constraint==0)
            elif(hasattr(runlist, "__len__")):
                removelist=np.where(runlist['use_lim']==0)
                self.bound[removelist]=0
                
        self.bwar=dharray.array_quickname(dharray.array_flexible_to_simple(self.war) 
                                          + dharray.array_flexible_to_simple(self.bound), 
                                          names=self.war.dtype.names, dtypes=self.war.dtype)
        
    def _make_war_tot(self, is_use_bwar=True):
        if(is_use_bwar==True): imsidat=np.copy(self.bwar[self.safe_itemlist])
        else: imsidat=np.copy(self.war[self.safe_itemlist])
        imsidat=dharray.array_flexible_to_simple(imsidat, remove_void=True)
        with warnings.catch_warnings():  ## Ignore warning. Nan warning is natural in this case.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.war_tot=np.nanmax(imsidat, axis=1)
        
        
    def find_best(self, remove_warn=True):
        usingarray=np.arange(len(self.chi2))
        if(remove_warn==True): 
            usingarray=usingarray[self.war_tot==0]
            ## No array
            if(len(usingarray)==0): return self.find_best(remove_warn=False)
            ## All data is NaN
            checknan=np.sum(np.isnan(self.chi2[usingarray]))
            if(checknan==len(usingarray)): return self.find_best(remove_warn=False)
        
        min_chi=np.nanmin(self.chi2[usingarray])
        minpos=np.where(self.chi2[usingarray]==min_chi)[0]
        return usingarray[minpos]
        
    def find_near_best(self, min_chi, allow=0):
        if(hasattr(min_chi, "__len__")): min_chi=min_chi[0]
        self.chi2_diff=np.abs(self.chi2-min_chi)/min_chi
        nearpos=np.where(self.chi2_diff<allow)[0] ## Fraction
        return nearpos
        
        
    ##============================ Save data ==============================
    def save_data(self, fns_output='result', fn_output=None, return_array=False):
        if(fn_output==None): fn_output=fns_output+"_result.dat"
        naming=dharray.array_quickname(np.zeros((len(self.val_chi2), 1)), names='data', dtypes='i4')
        warning_tot=dharray.array_quickname(self.war_tot[None].T, names='war_tot', dtypes='f8')
        
        val_out=dharray.array_add_columns(self.val_chi2, warning_tot)
        val_out=dharray.array_add_columns(val_out, naming, add_front=True)
        val_out['data']=1 ## Data

        err_out=dharray.array_add_columns(self.err, self.chi2_flex)
        err_out=dharray.array_add_columns(err_out, warning_tot)
        err_out=dharray.array_add_columns(err_out, naming, add_front=True)
        err_out['data']=2 ## Error
        
        war_out=dharray.array_add_columns(self.bwar, self.chi2_flex)
        war_out=dharray.array_add_columns(war_out, warning_tot)
        war_out=dharray.array_add_columns(war_out, naming, add_front=True)
        war_out['data']=3 ## Warning
        
        print_array=np.append(val_out, err_out)
        print_array=np.append(print_array, war_out)
        np.savetxt(fn_output, print_array)
        if(return_array==True): return print_array

#     def _load_data(self, fn=''): 
#         try: 
#             rawdata=np.loadtxt(fn)
#             test=rawdata[rawdata[:,0]==0]
#             if(len(test)!=0): print("File is strange!")
#         except: print("Loading file failed!")
#         val=rawdata[rawdata[:,0]==1][:,1:]
#         err=rawdata[rawdata[:,0]==2][:,1:]
#         war=rawdata[rawdata[:,0]==3][:,1:]   
#         self.chi2=np.copy(val[:,-2])
#         self.chi2_flex=dharray.array_quickname(self.chi2[None].T, names='CHI2NU', dtypes='f8')
#         self.is_file_exist=self.chi2.astype(bool)
        
#         self.val=dharray.array_quickname(val[:,:-2], self.itemlist)
#         self.err=dharray.array_quickname(err[:,:-2], self.itemlist)
#         self.war=dharray.array_quickname(war[:,:-2], self.itemlist)
    
    ##========================== Show data & display ============================
    def show_data(self, only_val=False, hide_nan=True, hide_fail_read=True):
        if(hide_nan==True): newitemlist=self.safe_itemlist
        else: newitemlist=np.array(self.itemlist) ## All the items
        if(hide_fail_read==True): newfilelist=self.is_file_exist
        else: newfilelist=np.full(len(self.is_file_exist), True) ## All the files
        errorlist=self.namelist[self.war_tot.astype(bool)]

        
        newitemlist_chi2=np.append(newitemlist, 'CHI2NU')
        self.df_display(self.val_chi2[newfilelist][newitemlist_chi2], newfilelist, errorlist, caption='Values')
        
        if(only_val==False):
            self.df_display(self.err[newfilelist][newitemlist], newfilelist, errorlist, caption='Error')            
            self.df_display(self.bwar[newfilelist][newitemlist], newfilelist, errorlist, caption='Warning', is_limit_format=True)

    def df_display(self, data, newfilelist, errorlist, caption='data', is_limit_format=False):
        df=pd.DataFrame(data, index=self.namelist[newfilelist])
        if(is_limit_format==True): formatting='{:.0f}'
        else: formatting='{:f}'
        df=df.style.apply(lambda x: ['background: lightyellow' if x.name in self.namelist[self.best_near]
                                      else '' for i in x], 
                           axis=1)\
            .apply(lambda x: ['background: lightgreen' if x.name in self.namelist[self.best]
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
