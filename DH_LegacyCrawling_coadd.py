import DH_array as dharray
import DH_path as dhpath
import numpy as np
import os
import random
import pickle
import copy
import shutil
from dataclasses import dataclass, field
import warnings
import urllib.error, socket
import requests
import os, glob
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator
from skimage.transform import resize


from astropy.modeling.functional_models import Sersic1D
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval,ImageNormalize,AsinhStretch
from astropy.convolution import Gaussian2DKernel


##=================== Functions for LegacyCrawling ============================
def download_data(url, fn_save, overwrite=False, silent=False):
    # Download brick data
    if(silent==False):
        print("\n○ Get the data file from the server")
        print(">> Download from : ", url)
    if(os.path.exists(fn_save)==True):
        if(silent==False): print(">> The file already exists : ", fn_save)
        if(overwrite==False): return
        else:
            if(silent==False): print(">> Overwrite the file")

    try: tmp_path=download_file(url, show_progress=True)
    except:
        if(silent==False): print(">> Download failed")
        return 1

    if(silent==False):
        print(">> Temp path : ", tmp_path)
    shutil.move(tmp_path, fn_save)
    if(silent==False): print(">> Saved to : ", fn_save)
    return 0

def crop_fits(filename, coord, size=(100, 100), ext_img=1, ext_wcs=1, fill_value=np.nan):
    """
    Descr : Crop fits image around the coordinate
    INPUT
     - filename : Input fits
     - coord : Center Coordinate (list)
     - size : Size of the crop image
     * ext_img (default - 1) : Extension for the image
     * ext_wcs (default - 1) : Extension for the wcs

    """
    image_data = fits.getdata(filename, ext=ext_img)
    image_header = fits.getheader(filename, ext=ext_wcs)
    wcs = WCS(image_header)
    coord_pixel=wcs.all_world2pix(coord[0], coord[1], 0)
    cutout = Cutout2D(image_data, position=coord_pixel, size=size, wcs=wcs, fill_value=fill_value, mode='partial')
    return cutout

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


##========================== Legacy Crawling ============================

@dataclass
class LegacyCrawling():
    coord : np.ndarray = field(default_factory=lambda: np.array([185.7119, 13.5928])) # Coordinate
    fn_ccd_fits : str = 'image' # Output CCD image
    dir_brick : str = dhpath.dir_brick()  # Brick restoring directory
    dir_work : str = './Data/'   # Working directory. Input files will be imported from here. If it is not exist, make dir.
    silent : bool = False     # Ignore printing message or not
    silent_error : bool = False     # Ignore printing message or not
    band : str = 'g'
    image_size : int = 200          # Size of the image
    plate_scale : float = 0.262       # Plate scale [arcsec/pixel]
    psf_mode : str = 'area'  # PSF generate method
    add_offset : float = 10  # Add an offset for all pixel (to avoid galfit error)
    is_psf_fwhm : bool = True # whether psf is fwhm or just psf (pixel) size (Default : True)
    is_overwrite_download : bool = False # whether update the plate images
    is_ignore_err : bool = False # whether ignore previous errors
    is_overwrite_data : bool = False # whether update the output files
    region : str = 's' # s - South, n - North
    save_error : bool = False # whether save the error file
    sleep : int = 2 # sleep time when it get HTTP 429 error (Too much requests) [seconds]
    retry : int = 5 # The number of trials for retrying download when it get HTTP 429 error (Too much requests)
    use_waittime_rand : bool = True # sleep is not a fixed number, it will be used the maximum of the random time.

    def __post_init__(self):
        self.coord=np.array(self.coord)
        os.makedirs(self.dir_work, exist_ok=True)  #Working dir
        os.makedirs(self.dir_brick+self.region+'_psfsize/', exist_ok=True) #PSF dir
        os.makedirs(self.dir_brick+self.region+'_invvar/', exist_ok=True) #invvar dir
        os.makedirs(self.dir_brick+self.region+'_chi2/', exist_ok=True) #chi2 dir
        os.makedirs(self.dir_brick+self.region+'_maskbits/', exist_ok=True) #maskbits dir
        self.cutout_url='https://www.legacysurvey.org/viewer/fits-cutout?ra=%.4f&dec=%.4f'%(self.coord[0], self.coord[1])
        self.cutout_url+='&size=%03d&layer=ls-dr9-'%self.image_size
        if(self.region=='s'): self.cutout_url+='south'
        else: self.cutout_url+='north'
        self.cutout_url+='&pixscale=0.262'
        self.cutout_url+='&bands='+self.band

        self.fn_err=self.dir_work+'err_'+self.region+self.band+'.dat'
        self.fn_http_err=self.dir_work+'err_http_'+self.region+self.band+'.dat'
        self.fn_http_err_bricks=self.dir_work+'err_bricks_http_'+self.region+self.band+'.dat'
        self.fn_image=self.dir_work+self.fn_ccd_fits+"_"+self.region+self.band+".fits"
        self._download_FITS_control(retry=self.retry)

    def _download_FITS_control(self, retry=3, current_try=0):
        if(os.path.exists(self.fn_err)==True):
            if(self.silent_error==False): print(">> The coord has error files!", self.fn_err)
            if(self.is_ignore_err==False): return
            else:
                if(self.silent_error==False): print(">> Ignore the error")
        try:
            result=self._download_FITS(overwrite=self.is_overwrite_download)
            if(result==1):
                if(self.silent_error==False): print(">> No data error! Error saved : ", self.fn_err)
                if(self.save_error): np.savetxt(self.fn_err, np.array(['fits', 'nodata']), fmt="%s")
                return
            self._find_bricks()

        except urllib.error.HTTPError as e:
            #self.err=e
            if(self.silent_error==False): print(">> HTTP Error! Error saved : ", self.dir_work)
            if(self.save_error):
                if(e.code==429): ## Too much requests --> save error separately
                    if(current_try<self.retry):
                        if(self.use_waittime_rand): waiting=time.sleep(random.uniform(self.sleep/2, self.sleep))
                        else: time.sleep(self.sleep)
                        result=self._download_FITS_control(retry=self.retry, current_try=current_try+1)
                    else: np.savetxt(self.fn_http_err, np.array(['http', e]), fmt="%s")
                else: np.savetxt(self.fn_err, np.array(['http', e]), fmt="%s")  ## Other errors

            #raise(e)
        except socket.timeout as e:
            if(self.silent_error==False): print(">> Time out error!", self.dir_work)
            #raise(e)


    def _download_FITS(self, overwrite=True):
        if(self.silent==False):
            print("\n● Get the CCD plate FITS file from the server")
            print(">> Download from : ", self.cutout_url)
        if(os.path.exists(self.fn_image)==True):
            if(self.silent==False): print(">> The file already exists : ", self.fn_image)
            if(overwrite==False): return
            else:
                if(self.silent==False): print(">> Overwrite the file")

        # Download FITS
        temp_fn=download_file(self.cutout_url, show_progress=True)
        if(self.silent==False):
            print(">> Temp path : ", temp_fn)

        #hdu = fits.open(image_fn, mode='update')
        hdu = fits.open(temp_fn, mode='update')
        hdu[0].data = hdu[0].data+self.add_offset
        hdu.writeto(self.fn_image, overwrite=True)
        ptp=np.ptp(hdu[0].data)
        hdu.close()
        if(self.silent==False): print(">> Saved to : ", self.fn_image)
        if(ptp<1e-9): return 1
        else: return 0

    def _find_brick_id(self, ra, dec):
        ra, dec=normalize_coord(ra, dec)
        if(self.region=='s'): using_sum_data=sum_south
        else: using_sum_data=sum_north
        return np.where((using_sum_data['ra1']<=ra) & (using_sum_data['ra2']>=ra) & (using_sum_data['dec1']<=dec) & (using_sum_data['dec2']>=dec))[0]

    def _find_bricks(self):
        """
        Descr
         - Find the brick(s) based on the coordinates of the center & boundary of the region
         - It will make bricklist (the list of the bricks)
        """
        if(self.silent==False): print("○ Find bricks contains the coordinate")

        #===================== Bricks finding =========================
        ## Brick contains the center position
        self.bricks_center=self._find_brick_id(self.coord[0], self.coord[1])

        ## Brick contains 4 boundaries (and center position)
        image_coord_boundary = np.zeros((2,2))
        image_coord_boundary[:,0] = self.coord-self.plate_scale/3600*self.image_size/2
        image_coord_boundary[:,1] = self.coord+self.plate_scale/3600*self.image_size/2

        bricks0=self._find_brick_id(self.coord[0], self.coord[1])
        bricks1=self._find_brick_id(image_coord_boundary[0,0], image_coord_boundary[1,0])
        bricks2=self._find_brick_id(image_coord_boundary[0,0], image_coord_boundary[1,1])
        bricks3=self._find_brick_id(image_coord_boundary[0,1], image_coord_boundary[1,0])
        bricks4=self._find_brick_id(image_coord_boundary[0,1], image_coord_boundary[1,1])
        bricks=np.concatenate((bricks0, bricks1, bricks2, bricks3, bricks4))
        self.bricks_boundary=np.unique(bricks)
        if(self.silent==False):
            if(self.region=='s'): print(">> Region : South")
            else: print(">> Region : North")
            print(">> Center Brick : ", self.bricks_center)
            print(">> Boundary Bricks : ", self.bricks_boundary)

    def _get_median_from_multi_fits(self, fnlist, is_mean=False, is_return_2d=False, center_crop=None):
        """
        Descr
         1) Crop the fits files and make a combined map of psf or sigma map
         2) Note that if the galaxy is located at the edge of the brick, it need several bricks
            (it made this code complicated)
        INPUT
         - fnlist : brick files namelist
         * is_mean : (default : False) If true, get mean instead of median
         * is_return_2d : (default : False) If true, output is 2D map. If False, output is scalar.
        """
        tot_data=np.zeros(len(fnlist), dtype='object')
        tot_data_flat=np.zeros((len(fnlist), self.image_size, self.image_size))
        dat_avail=np.full(len(fnlist), True)
        for i in range (len(fnlist)):
            try:
                imsidat=crop_fits(fnlist[i], coord=self.coord, size=(self.image_size, self.image_size), ext_img=1, ext_wcs=1)
            #if(is_zero_remove==True): imsidat.data[imsidat.data==0]=1e-6
                tot_data[i]=imsidat
                tot_data_flat[i]=np.copy(tot_data[i].data)
            except: dat_avail[i]=False
        if(self.silent==False): print(">> Data avail : ", dat_avail)
        if(np.sum(dat_avail)==0): return np.nan
        tot_data=tot_data[dat_avail]
        tot_data_flat=tot_data_flat[dat_avail]

        if(center_crop!=None):
            halfsize=int(self.image_size/2)
            crop_lower=halfsize-int(center_crop)
            crop_upper=halfsize+int(center_crop)
            tot_data_flat=tot_data_flat[:,crop_lower:crop_upper,crop_lower:crop_upper]

        if(is_return_2d==True):  # Return 2D array
            if(is_mean==True): result=np.nanmean(tot_data_flat, axis=0)
            else: result=np.nanmedian(tot_data_flat, axis=0)
        else:  # Return single value
            if(is_mean==True): result=np.nanmean(tot_data_flat)
            else: result=np.nanmedian(tot_data_flat)

        return result


    def _get_binarysum_from_multi_fits(self, fnlist):
        """
        Descr
         1) Crop the fits files and make a combined map
         2) Note that if the galaxy is located at the edge of the brick, it need several bricks
            (it made this code complicated)
        INPUT
         - fnlist : brick files namelist
        """
        tot_data=np.zeros(len(fnlist), dtype='object')
        tot_data_flat=np.zeros((len(fnlist), self.image_size, self.image_size), dtype='int16')
        binarysmum=np.zeros((self.image_size, self.image_size), dtype='int16')
        for i in range (len(fnlist)):
            imsidat=crop_fits(fnlist[i], coord=self.coord, size=(self.image_size, self.image_size),
                                  ext_img=1, ext_wcs=1, fill_value=0) # Empty fill value = 0
            #if(is_zero_remove==True): imsidat.data[imsidat.data==0]=1e-6
            tot_data[i]=imsidat
            tot_data_flat[i]=np.copy(tot_data[i].data)
            binarysmum = binarysmum | tot_data_flat[i] #Binary or operator

        return binarysmum


    def psf(self, mode='default', manual_seeing_psf=None, overwrite=None, is_ignore_err=None, return_2dmap=False):
        """
        Descr : Get the PSF-size and generate the PSF profile
        INPUT
         - mode
           * default : self.psf_mode (follows the setting)
           * brick : median value of the brick
           * area : median value of the given area (the region of interest)
           * center : value of the center (galaxy)
           *
         - manual_seeing_psf : Generate PSF profile with this value (ignore the mode)
         - overwrite : whether overwrite the previous file
        """

        if(type(overwrite)==type(None)): overwrite=self.is_overwrite_data
        if(type(is_ignore_err)==type(None)): is_ignore_err=self.is_ignore_err
        if(mode=='default'): mode=self.psf_mode  ## Default : self.psf_mode
        if(self.silent==False):
            print("\n● Generate the PSF profile based on bricks")
            print(">> PSF mode : ", mode)
            print(">> Filter band : ", self.band)
        item='psfsize_'+self.band
        self.fn_psf=self.dir_work+'psf_'+mode+"_"+self.region+self.band+'.fits'
        if(os.path.exists(self.fn_err)==True):  ## Error check
            if(self.silent_error==False): print(">> The file has 0 psf size", self.fn_err)
            if(is_ignore_err==False): return
            else:
                if(self.silent_error==False): print(">> Ignore the error")

        if(os.path.exists(self.fn_psf)==True):
            if(self.silent==False): print(">> The file already exists : ", self.fn_psf)
            if(overwrite==False): return
            else:
                if(self.silent==False): print(">> Overwrite the file")


        #===================== Get PSF Seeing =========================
        if(type(manual_seeing_psf)!=type(None)):
            if(self.silent==False):
                print(">> Manual PSF Seeing : %.5f [arcsec]"%manual_seeing_psf)
            self.seeing_psf=manual_seeing_psf

        else:
            if(mode=='brick'): ## median psf of the brick
                if(self.region=='s'): using_sum_data=sum_south
                else: using_sum_data=sum_north
                self.seeing_psf = using_sum_data[self.bricks_center[0]][item]
            else:
                if(mode=='area'): ## median psf of the area
                    fnlist=self._download_brick_data(dataname='psfsize', bricklist=self.bricks_boundary, retry=self.retry)
                    center_crop=None
                elif(mode=='single_brick'): ## median psf of the area (single brick)
                    fnlist=self._download_brick_data(dataname='psfsize', bricklist=self.bricks_center, retry=self.retry)
                    center_crop=None
                elif(mode=='target'): ## central psf
                    fnlist=self._download_brick_data(dataname='psfsize', bricklist=self.bricks_boundary, retry=self.retry)
                    center_crop=10
                else:
                    if(self.silent==False):
                        print(">> PSF Mode Error!")
                        return
                if(return_2dmap):
                    return self._get_median_from_multi_fits(fnlist, center_crop=center_crop, is_return_2d=True)
                else:
                    self.seeing_psf=self._get_median_from_multi_fits(fnlist, center_crop=center_crop)

        #===================== PSF generating =========================
        if(self.silent==False): print(">> PSF Seeing : %.5f [arcsec]"%self.seeing_psf)
        if((self.seeing_psf<1e-3) | (np.isnan(self.seeing_psf))):
            if(self.silent_error==False): print(">> PSF Size is Zero!")
            if(self.save_error): np.savetxt(self.fn_err, np.array(['psf', self.seeing_psf]), fmt="%s")
            return

        generate_psf(self.fn_psf, psf_size=self.seeing_psf, plate_scale=self.plate_scale, is_fwhm=self.is_psf_fwhm, silent=self.silent)
        if(self.silent==False): print(">> PSF dir : ", self.fn_psf)


    def maskbits(self, overwrite=None, is_ignore_err=None):
        if(type(overwrite)==type(None)): overwrite=self.is_overwrite_data
        if(type(is_ignore_err)==type(None)): is_ignore_err=self.is_ignore_err
        if(self.silent==False):
            print("\n● Get the maskbits data")

        self.fn_maskbits=self.dir_work+"maskbits_"+self.region+'x.fits'

        if(os.path.exists(self.fn_err)==True):
            if(self.silent_error==False): print(">> The file has 0 psf size", self.fn_err)
            if(is_ignore_err==False): return
            else:
                if(self.silent_error==False): print(">> Ignore the error")
        if(os.path.exists(self.fn_maskbits)==True):
            if(self.silent==False): print(">> The file already exists : ", self.fn_maskbits)
            if(overwrite==False): return
            else:
                if(self.silent==False): print(">> Overwrite the file")


        fnlist=self._download_brick_data(dataname='maskbits', bricklist=self.bricks_boundary, is_link_without_band=True, retry=self.retry)
        maskbits_map=self._get_binarysum_from_multi_fits(fnlist)

        maskbits = fits.PrimaryHDU()
        maskbits.data = maskbits_map
        maskbits.writeto(self.fn_maskbits, overwrite = True)
        if(self.silent==False): print(">> maskbits map generated : ", self.fn_maskbits)


    def _download_brick_data(self, dataname='psfsize', bricklist=None, is_link_without_band=False, retry=3, current_try=0):
        """
        Descr :
         1) Find the name of the brick using Richard's catalog
         2) Download the brick PSF data
        INPUT
         * dataname : 'psfsize', 'chi2', 'invvar'
         * bricklist : if None, it will use bricks_center
        """
        if(hasattr(bricklist, "__len__")!=True):
            bricklist=self.bricks_center
            if(self.silent==False): print("Warning! Bricklist is empty!")

        fnlist=[]
        for i in range (len(bricklist)):
            thisbrick=bricklist[i]

            if(self.region=='s'): using_sum_data=sum_south
            else: using_sum_data=sum_north
            brickname=using_sum_data[thisbrick]['brickname']
            coaddurl='https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/'
            if(self.region=='s'): coaddurl+='south/coadd/'
            else: coaddurl+='north/coadd/'
            coaddurl+=brickname[:3]+"/"+brickname+"/"+'legacysurvey-'
            coaddurl+=brickname+"-"+dataname

            if(is_link_without_band):
                coaddurl+=".fits.fz"
                folder=self.dir_brick+self.region+"_"+dataname+'/'+brickname[:4]+'/' ## Group folder
                os.makedirs(folder, exist_ok=True)
                fn_save=folder+brickname+"-"+dataname+"-x.fits.fz"
            else:
                coaddurl+="-"+self.band+".fits.fz"
                folder=self.dir_brick+self.region+"_"+dataname+'/'+brickname[:4]+'/' ## Group folder
                os.makedirs(folder, exist_ok=True)
                fn_save=folder+brickname+"-"+dataname+"-"+self.band+".fits.fz"

            try:
                fnlist=fnlist+[fn_save]
                download_data(coaddurl, fn_save, overwrite=self.is_overwrite_download, silent=self.silent)

            except urllib.error.HTTPError as e:
                if(self.silent_error==False): print(">> HTTP Error!")
                if(e.code==429): ## Too much requests --> save error separately
                    if(current_try<self.retry):
                        time.sleep(self.sleep)
                        result=self._download_brick_data(dataname=dataname, bricklist=bricklist,
                                                         is_link_without_band=is_link_without_band,
                                                         retry=self.retry, current_try=current_try+1)
                    else: np.savetxt(self.fn_http_err_bricks, np.array(['http', e]), fmt="%s")

            except:
                print(">> Time out error!")
                raise(e)
        return fnlist



def normalize_coord(ra, dec):
    if(ra<0): ra=ra+360
    if(ra>360): ra=ra-360
    if(dec>90): dec=180-dec
    if(dec<-90): dec=-180-dec
    return ra, dec
