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
from astropy.utils.data import conf
from pathlib import Path

from astropy.modeling.functional_models import Sersic1D
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval,ImageNormalize,AsinhStretch
from astropy.convolution import Gaussian2DKernel

import DH_multicore as mulcore

##========= FITS info =======================
fn_legacy_sum_south = dhpath.fn_legacy_sum_south() ## Summary data
fn_legacy_sum_south_dr10 = dhpath.fn_legacy_sum_south_dr10() ## Summary data
fn_legacy_sum_north = dhpath.fn_legacy_sum_north() ## Summary data
#'/home/donghyeon/ua/udg/galfit/galfit/RawData/survey-bricks-dr9-south.fits' # Fits info data
print("LegacyCrawling - Loading summary fits south : ", fn_legacy_sum_south)
sum_south=fits.getdata(fn_legacy_sum_south, ext=1)
print("LegacyCrawling - Loading summary fits south (DR10) : ", fn_legacy_sum_south_dr10)
sum_south_dr10=fits.getdata(fn_legacy_sum_south_dr10, ext=1)
print("LegacyCrawling - Loading summary fits north : ", fn_legacy_sum_north)
sum_north=fits.getdata(fn_legacy_sum_north, ext=1)

##=================== Functions for LegacyCrawling ============================
def download_manager(url, fn_save, overwrite=False, silent=True,
                     Ntry=3, sleep=2, timeout=10, use_waittime_rand=True,
                     allow_insecure=True,
                     silent_error=False,
                     silent_timeerror=True,
                     current_try=0,
                     modify_fits=False, add_offset=0, ext_add_offset_list=None
                    ):
    """
    Descr - dhcrawl.download_data with Additional functions
    Ntry: # of try of downloading
    sleep: sleep time when it get HTTP 429 error (Too much requests) [seconds]
    use_waittime_rand : sleep is not a fixed number, it will be used the maximum of the random
    """
    if(current_try==0):
        if(conf.remote_timeout!=timeout):
            # print("Remote_timeout modified to ", timeout)
            conf.remote_timeout=timeout

    ## error
    fn_err = str(Path(fn_save).parent)+"/"+"err_"+Path(fn_save).stem+".dat"
    ## temp error (too much request, Temporarily Unavailable)
    fn_err_tmp = str(Path(fn_save).parent)+"/"+"err_temp_"+Path(fn_save).stem+".dat"

    if(overwrite==False):
        if(os.path.exists(fn_save)==True):
            if(silent==False): print(">> File exist:", fn_save)
            return 0
        if(os.path.exists(fn_err)==True):
            if(silent==False): print(">> Error log exist:", fn_err)
            return 0

    try:
        download_data(url, fn_save, overwrite=overwrite, is_try=False,
                              silent=silent, allow_insecure=allow_insecure,
                              modify_fits=modify_fits, add_offset=add_offset,
                              ext_add_offset_list=ext_add_offset_list)
        return 0

    except urllib.error.HTTPError as e: ## HTTP error
        # if(silent_timeerror==False): print(">> HTTP Error!")
        if(e.code==429): ## Too much requests --> save error separately
            if(current_try<Ntry):
                if(use_waittime_rand): time.sleep(random.uniform(sleep/2, sleep))
                else: time.sleep(sleep)
                return download_manager(url, fn_save, overwrite=overwrite, silent=silent, silent_error=silent_error,
                                Ntry=Ntry, sleep=sleep, timeout=timeout, use_waittime_rand=use_waittime_rand,
                                allow_insecure=allow_insecure,
                                silent_timeerror=True,
                                current_try=current_try+1,
                                modify_fits=modify_fits, add_offset=add_offset,
                                ext_add_offset_list=ext_add_offset_list)
                                 ## Restart

            else:
                np.savetxt(fn_err_tmp, np.array(['http', e]), fmt="%s") ## Ntry cut
                if(silent_error==False): print(">>>> Too much request error after N tries!", fn_save)
                return 1
        elif(e.code==503): ## Temporarily Unavailable --> save error separately
            np.savetxt(fn_err_tmp, np.array(['http', e]), fmt="%s") ## Ntry cut
            if(silent_error==False): print(">>>> Temporarily Unavailable!", fn_save)
            return 1
        else:
            np.savetxt(fn_err, np.array(['http', e]), fmt="%s")  ## Other errors
            if(silent_error==False): print(">>>> Error! code: ", e.code, fn_save)
            return 1

    except socket.timeout as e: ## Timeout error -> Just quit
        if(silent_timeerror==False): print(">> Time out error!", fn_save)
        if(current_try<Ntry):
            if(use_waittime_rand): time.sleep(random.uniform(sleep/2, sleep))
            else: time.sleep(sleep)
            return download_manager(url, fn_save, overwrite=overwrite, silent=silent, silent_error=silent_error,
                            Ntry=Ntry, sleep=sleep, timeout=timeout, use_waittime_rand=use_waittime_rand,
                            allow_insecure=allow_insecure,
                            silent_timeerror=True,
                            current_try=current_try+1,
                                modify_fits=modify_fits, add_offset=add_offset,
                                ext_add_offset_list=ext_add_offset_list)

                             ## Restart
        else:
            if(silent_error==False): print(">>>> Time out error after N tries!", fn_save)
            return 1

    except:
        if(silent_error==False): print(">> Other Error!", fn_save)
        return 1



def download_data(url, fn_save, overwrite=False,
                  silent=False, allow_insecure=True, is_try=True,
                  modify_fits=False, add_offset=0, ext_add_offset_list=None):
    # Download brick data
    if(silent==False):
        print("\n○ Get the data file from the server")
        print(">> Download from : ", url)
    if(os.path.exists(fn_save)==True):
        if(silent==False): print(">> The file already exists : ", fn_save)
        if(overwrite==False): return
        else:
            if(silent==False): print(">> Overwrite the file")

    if(is_try):
        try: tmp_path=download_file(url, show_progress=True, allow_insecure=allow_insecure)
        except:
            if(silent==False): print(">> Download failed")
            return 1
    else: tmp_path=download_file(url, show_progress=True, allow_insecure=allow_insecure)
    if(silent==False):
        print(">> Temp path : ", tmp_path)
        print(modify_fits)
    if(modify_fits):
        try:
            hdu = fits.open(tmp_path, mode='update')
            if((add_offset!=0) & (hasattr(ext_add_offset_list, "__len__"))): ## add offset
                if(silent==False):
                    print(">> Add offset ", add_offset, "to extensions", ext_add_offset_list)

                for ext in range(len(hdu)):
                    if(np.isin(ext, ext_add_offset_list)):
                        hdu[ext].data = hdu[ext].data+add_offset
        #hdu.writeto(fn_save, overwrite=True)
            hdu.close()
            shutil.move(tmp_path, fn_save)
            if(silent==False): print(">> Saved to : ", fn_save)
        except:
            if(silent==False): print(">> fits file has problems!")
    else:
        shutil.move(tmp_path, fn_save)
        if(silent==False): print(">> Saved to : ", fn_save)
    return 0

## ========================== Cutout image and PSF models ==============
def download_cutout(folder, coord,
                    Ntry=5, sleep=2, timeout=10, use_waittime_rand=True,
                    fnkey_cutout_south_dr9='cutout_s_dr9.fits',
                    fnkey_cutout_north_dr9='cutout_n_dr9.fits',
                    fnkey_cutout_south_dr10='cutout_s_dr10.fits',
                    imgsize=200, overwrite=False, silent=True, silent_error=False, silent_timeerror=True):


    os.makedirs(folder, exist_ok=True)  #Working dir
    fn_cutout_south_dr9=folder+fnkey_cutout_south_dr9
    fn_cutout_north_dr9=folder+fnkey_cutout_north_dr9
    fn_cutout_south_dr10=folder+fnkey_cutout_south_dr10 ## DR10 does not have northern part

    cutout_url='https://www.legacysurvey.org/viewer/cutout.fits?ra=%.4f&dec=%.4f'%(coord[0], coord[1])
    cutout_url_south_dr9=cutout_url+'&size=%d&layer=ls-dr9'%(imgsize)+'-south&subimage'
    cutout_url_north_dr9=cutout_url+'&size=%d&layer=ls-dr9'%(imgsize)+'-north&subimage'
    cutout_url_south_dr10=cutout_url+'&size=%d&layer=ls-dr10'%(imgsize)+'-south&subimage'

    result=download_manager(cutout_url_south_dr9, fn_cutout_south_dr9, overwrite=overwrite,
                     Ntry=Ntry, sleep=sleep, timeout=timeout, use_waittime_rand=use_waittime_rand,
                     silent=silent, silent_error=silent_error, silent_timeerror=silent_timeerror,
                     )
    result+=download_manager(cutout_url_north_dr9, fn_cutout_north_dr9, overwrite=overwrite,
                     Ntry=Ntry, sleep=sleep, timeout=timeout, use_waittime_rand=use_waittime_rand,
                     silent=silent, silent_error=silent_error, silent_timeerror=silent_timeerror,
                     )
    result+=download_manager(cutout_url_south_dr10, fn_cutout_south_dr10, overwrite=overwrite,
                     Ntry=Ntry, sleep=sleep, timeout=timeout, use_waittime_rand=use_waittime_rand,
                     silent=silent, silent_error=silent_error, silent_timeerror=silent_timeerror,
                     )
    return result

def download_psf(folder, coord,
                 Ntry=5, sleep=2, timeout=10, use_waittime_rand=True,
                 fnkey_psf_south_dr9='psf_model_dr9_s.fits',
                 fnkey_psf_north_dr9='psf_model_dr9_n.fits',
                 fnkey_psf_south_dr10='psf_model_dr10_s.fits',
                 skip_dr10=False,
                 overwrite=False, silent=True, silent_error=False, silent_timeerror=True):

    os.makedirs(folder, exist_ok=True)  #Working dir

    fn_psf_south_dr9=folder+fnkey_psf_south_dr9
    fn_psf_north_dr9=folder+fnkey_psf_north_dr9
    fn_psf_south_dr10=folder+fnkey_psf_south_dr10 ## DR10 does not have northern part

    psf_url='https://www.legacysurvey.org/viewer/coadd-psf/?ra=%.4f&dec=%.4f'%(coord[0], coord[1])
    psf_url_south_dr9=psf_url+'&layer=ls-dr9'+'-south'
    psf_url_north_dr9=psf_url+'&layer=ls-dr9'+'-north'
    psf_url_south_dr10=psf_url+'&layer=ls-dr10'+'-south'


    result=download_manager(psf_url_south_dr9, fn_psf_south_dr9, overwrite=overwrite,
                     Ntry=Ntry, sleep=sleep, timeout=timeout, use_waittime_rand=use_waittime_rand,
                     silent=silent, silent_error=silent_error, silent_timeerror=silent_timeerror,
                     )
    result+=download_manager(psf_url_north_dr9, fn_psf_north_dr9, overwrite=overwrite,
                     Ntry=Ntry, sleep=sleep, timeout=timeout, use_waittime_rand=use_waittime_rand,
                     silent=silent, silent_error=silent_error, silent_timeerror=silent_timeerror,
                     )
    if(skip_dr10==False):
        result+=download_manager(psf_url_south_dr10, fn_psf_south_dr10, overwrite=overwrite,
                         Ntry=Ntry, sleep=sleep, timeout=timeout, use_waittime_rand=use_waittime_rand,
                         silent=silent, silent_error=silent_error, silent_timeerror=silent_timeerror,
                         )
    return result

def multicore_download_cutout(DirInfo, silent=True, silent_error=False, silent_timeerror=True,
                              Ntry=5, sleep=2, timeout=10, use_waittime_rand=True,
                              overwrite=False, psf_only=True, skip_dr10=False,
                          Ncore=10, show_progress=10000, use_try=False,
                          **kwargs):
    global sub_download_cutout
    print("Start Multicore run")

    def sub_image_stat(i):
        result=0
        if(psf_only==False):
            result+=download_cutout(DirInfo.dir_work_list[i], DirInfo.coord_array[i], overwrite=overwrite,
                            Ntry=Ntry, sleep=sleep, timeout=timeout, use_waittime_rand=use_waittime_rand,
                            silent=silent, silent_error=silent_error, silent_timeerror=silent_timeerror)
        result+=download_psf(DirInfo.dir_work_list[i], DirInfo.coord_array[i], overwrite=overwrite,
                     Ntry=Ntry, sleep=sleep, timeout=timeout, use_waittime_rand=use_waittime_rand,
                     silent=silent, silent_error=silent_error, silent_timeerror=silent_timeerror,
                     skip_dr10=skip_dr10)
        return result

    return mulcore.multicore_run(sub_image_stat, len(DirInfo.dir_work_list),
                                 Ncore=Ncore, show_progress=show_progress, use_try=use_try,
                                 **kwargs)


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
    silent_timeerror: bool = False
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
    timeout: int = 5
    use_waittime_rand : bool = True # sleep is not a fixed number, it will be used the maximum of the random time.
    dr : str = 'dr9' ## DR 9 or DR 10

    def __post_init__(self):
        self.coord=np.array(self.coord)
        os.makedirs(self.dir_work, exist_ok=True)  #Working dir
        if(self.dr=='dr9'): self.dir_brick_dr=self.dir_brick+'dr9/'
        elif(self.dr=='dr10'): self.dir_brick_dr=self.dir_brick+'dr10/'
        else: self.dir_brick_dr=self.dir_brick+'dr_none/'
        os.makedirs(self.dir_brick_dr+self.region+'_psfsize/', exist_ok=True) #PSF dir
        os.makedirs(self.dir_brick_dr+self.region+'_invvar/', exist_ok=True) #invvar dir
        #os.makedirs(self.dir_brick_dr+self.region+'_chi2/', exist_ok=True) #chi2 dir
        os.makedirs(self.dir_brick_dr+self.region+'_maskbits/', exist_ok=True) #maskbits dir

        self.cutout_url='https://www.legacysurvey.org/viewer/fits-cutout?ra=%.4f&dec=%.4f'%(self.coord[0], self.coord[1])
        self.cutout_url+='&size=%03d&layer=ls-'%self.image_size
        self.cutout_url+= self.dr+'-'
        if(self.region=='s'): self.cutout_url+='south'
        else: self.cutout_url+='north'
        self.cutout_url+='&pixscale=%0.3f'%self.plate_scale
        self.cutout_url+='&bands='+self.band

        self.fn_err=self.dir_work+'err_'+self.region+self.band+'.dat'
        self.fn_http_err=self.dir_work+'err_http_'+self.region+self.band+'.dat'
        self.fn_http_err_bricks=self.dir_work+'err_bricks_http_'+self.region+self.band+'.dat'
        self.fn_image=self.dir_work+self.fn_ccd_fits+"_"+self.dr+"_"+self.region+self.band+".fits"

        download_manager(self.cutout_url, self.fn_image, overwrite=self.is_overwrite_data,
                         Ntry=self.retry, sleep=self.sleep, timeout=self.timeout, use_waittime_rand=self.use_waittime_rand,
                         silent=self.silent, silent_error=self.silent_error,
                         silent_timeerror=self.silent_timeerror,
                         modify_fits=True, add_offset=self.add_offset, ext_add_offset_list=[0]
                         )
        self._find_bricks()


    def _find_brick_id(self, ra, dec):
        ra, dec=normalize_coord(ra, dec)
        if(self.region=='s'):
            if(self.dr=='dr9'): using_sum_data=sum_south
            else: using_sum_data=sum_south_dr10
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


        self.fn_psf=self.dir_work+'psf_'+mode+"_"+self.dr+"_"+self.region+self.band+'.fits'
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
                if(self.region=='s'):
                    if(self.dr=='dr9'): using_sum_data=sum_south
                    else: using_sum_data=sum_south_dr10
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


    def sigma(self, mode='invvar', overwrite=None, is_ignore_err=None):
        if(type(overwrite)==type(None)): overwrite=self.is_overwrite_data
        if(type(is_ignore_err)==type(None)): is_ignore_err=self.is_ignore_err
        if(self.silent==False):
            print("\n● Generate the sigma profile based on bricks")
            print(">> Mode : ", mode)
            print(">> Filter band : ", self.band)

        self.fn_sigma=self.dir_work+"sigma_"+self.dr+"_"+self.region+self.band+'.fits'

        if(os.path.exists(self.fn_err)==True): ## Error check
            if(self.silent_error==False): print(">> The file has 0 psf size", self.fn_err)
            if(is_ignore_err==False): return
            else:
                if(self.silent_error==False): print(">> Ignore the error")
        if(os.path.exists(self.fn_sigma)==True):
            if(self.silent==False): print(">> The file already exists : ", self.fn_sigma)
            if(overwrite==False): return
            else:
                if(self.silent==False): print(">> Overwrite the file")


        fnlist=self._download_brick_data(dataname=mode, bricklist=self.bricks_boundary, retry=self.retry)
        invvar_map=self._get_median_from_multi_fits(fnlist, is_return_2d=True)
        ## invvar -> sigma ==> Sigma = 1/sqrt(invvar)

        invvar_map[invvar_map==0]=np.unique(invvar_map)[1]*0.1 # 0.1 * second smallest data (first smallest except 0)
        sigma_map=1/np.sqrt(invvar_map)

        sigma = fits.PrimaryHDU()
        sigma.data = sigma_map.astype('float32')
        sigma.writeto(self.fn_sigma, overwrite = True)
        if(self.silent==False): print(">> Sigma map generated : ", self.fn_sigma)

    def maskbits(self, overwrite=None, is_ignore_err=None):
        if(type(overwrite)==type(None)): overwrite=self.is_overwrite_data
        if(type(is_ignore_err)==type(None)): is_ignore_err=self.is_ignore_err
        if(self.silent==False):
            print("\n● Get the maskbits data")

        self.fn_maskbits=self.dir_work+"maskbits_"+self.dr+"_"+self.region+'x.fits'

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

            if(self.region=='s'):
                if(self.dr=='dr9'): using_sum_data=sum_south
                else: using_sum_data=sum_south_dr10
            else: using_sum_data=sum_north
            brickname=using_sum_data[thisbrick]['brickname']
            coaddurl='https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/'
            coaddurl+=self.dr+"/"
            if(self.region=='s'): coaddurl+='south/coadd/'
            else: coaddurl+='north/coadd/'
            coaddurl+=brickname[:3]+"/"+brickname+"/"+'legacysurvey-'
            coaddurl+=brickname+"-"+dataname

            if(is_link_without_band):
                coaddurl+=".fits.fz"
                folder=self.dir_brick_dr+self.region+"_"+dataname+'/'+brickname[:4]+'/' ## Group folder
                os.makedirs(folder, exist_ok=True)
                fn_save=folder+brickname+"-"+dataname+"-x.fits.fz"
            else:
                coaddurl+="-"+self.band+".fits.fz"
                folder=self.dir_brick_dr+self.region+"_"+dataname+'/'+brickname[:4]+'/' ## Group folder
                os.makedirs(folder, exist_ok=True)
                fn_save=folder+brickname+"-"+dataname+"-"+self.band+".fits.fz"

            fnlist=fnlist+[fn_save]
            download_manager(coaddurl, fn_save, overwrite=self.is_overwrite_download,
                             Ntry=self.retry, sleep=self.sleep, timeout=self.timeout, use_waittime_rand=self.use_waittime_rand,
                             silent=self.silent, silent_error=self.silent_error,
                             silent_timeerror=self.silent_timeerror,
                             modify_fits=False
                             )

        return fnlist


def normalize_coord(ra, dec):
    if(ra<0): ra=ra+360
    if(ra>360): ra=ra-360
    if(dec>90): dec=180-dec
    if(dec<-90): dec=-180-dec
    return ra, dec

def generate_psf(fn_psf='psf.fits', psf_size=1.2, plate_scale=0.262, is_fwhm=True, silent=False, return_psf=False):
    ## psf_size : arcsec unit
    if(is_fwhm==True):
        FWHMpsf = psf_size/plate_scale   # Median seeing / # pixel scale
        sig2fwhm = 2 * np.sqrt(2 * np.log(2))   # converting FWHM -> sigma
        ## FWHM = 2*sqrt(2 ln 2) * sigma (standard deviation)
        mod_sig = FWHMpsf/sig2fwhm
        gaussian_2D_kernel = Gaussian2DKernel(mod_sig, mode = 'oversample')
        if(silent==False): print("PSF size : ", mod_sig)
    else: ## Just use psf_size
        gaussian_2D_kernel = Gaussian2DKernel(psf_size, mode = 'oversample')
        if(silent==False): print("PSF size : ", psf_size)
    ker = gaussian_2D_kernel.array
    #conv = ker.shape[1] + 2

    # Output
    if(fn_psf!=None):
        PSFfile = fits.PrimaryHDU()
        PSFfile.data = ker.astype('float32')
        PSFfile.writeto(fn_psf, overwrite = True)
        fits.setval(fn_psf, 'psfsize', value=psf_size, comment='arcsec')
    if(return_psf==True): return ker
