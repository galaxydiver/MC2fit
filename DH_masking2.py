import numpy as np
import sep
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel ## For a kernel
import copy

class Masking2():
    def __init__(self,
                 imagedata=None, fn_image=None, mask_prev=None, fn_mask_prev=None, ## Load
                 silent=True,
                 ## Main Masking parameters
                 thresh=1, minarea=5, is_relative=True,
                 suppress_center=3, is_center_circle=True, centermasking_is_touch=True,
                 ## Superbright parameters
                 superbright=False, superbright_thresh=3,
                 superbright_minarea=1, superbright_suppress_center=0,
                 ## cig parameters
                 cig=True, cig_thresh=2, cig_minarea=1, cig_maxarea=80, ## 5 pixel radius
                 ## nb parameters
                 nb=True, nb_blur=0.2, nb_thresh=1, nb_minarea=1,
                 nb_suppress_center=50

                 ):

        self.silent=silent
        self._load(imagedata, fn_image, mask_prev, fn_mask_prev) ## Load image
        mask_base=self.mask_prev


        if(superbright):
            if(self.silent==False): print("\n● Superbright mask")
            obj, mask=self._main_masking(self.imagedata, thresh=superbright_thresh,
                                         subtract_bkg=True, is_relative=False,
                                         mask_prev=self.mask_prev,
                                         minarea=superbright_minarea, filter_kernel='ker')

            mask=suppress_center_mask(mask, size_suppress=superbright_suppress_center, obj=obj,
                                      is_center_circle=is_center_circle, is_touch=centermasking_is_touch,
                                      silent=silent)
            self.obj_superbright, self.mask_superbright=clean_mask(self.imagedata, mask+mask_base, silent=silent)
            mask_base=copy.deepcopy(self.mask_superbright)


        ## Main Mask
        if(self.silent==False): print("\n● Main Mask")
        obj, mask=self._main_masking(self.imagedata, thresh=thresh,
                                     subtract_bkg=True, is_relative=is_relative,
                                     mask_prev=mask_base,
                                     minarea=minarea, filter_kernel='default')


        self.obj_main, self.mask_main=clean_mask(self.imagedata, mask+mask_base, silent=silent) ## Clean before suppressing
        ## supress the mask near the center -> remove primary target
        mask=suppress_center_mask(self.mask_main, size_suppress=suppress_center, obj=self.obj_main,
                                  is_center_circle=is_center_circle, is_touch=centermasking_is_touch,
                                  silent=silent)
        self.obj_main_sup, self.mask_main_sup=clean_mask(self.imagedata, mask+mask_base, silent=silent) ## Clean after suppressing

        ## detect the primary target
        imsi_mask=np.zeros_like(self.mask_main_sup)
        imsi_mask[(self.mask_main_sup==0) & (self.mask_main!=0)]=1 ## Supressed ones
        self.obj_pri, self.mask_pri=clean_mask(self.imagedata, imsi_mask, silent=silent)

        if(cig):
            ## Clumps inside the target
            if(self.silent==False): print("\n● Clumps in the galaxy")
            imsi_mask=np.zeros_like(self.mask_pri)
            imsi_mask[self.mask_pri==0]=1
            self.imsi_mask=imsi_mask
            obj, mask=self._main_masking(self.imagedata, thresh=cig_thresh,
                                         subtract_bkg=True, is_relative=is_relative,
                                         mask_prev=imsi_mask,
                                         minarea=cig_minarea, filter_kernel='default',
                                         bw=10, bh=10) ## Small region
            toolarge=np.where(obj['npix']>cig_maxarea)[0] + 1 ## Obj starts from 1
            mask[np.isin(mask, toolarge)]=0
            self.obj_cig, self.mask_cig=clean_mask(self.imagedata, mask, silent=silent) ## Clean after suppressing
            self.obj_main_supcig, self.mask_main_supcig=clean_mask(self.imagedata, self.mask_cig+self.mask_main_sup, silent=silent)

        if(nb):
            ## Clumps inside the target
            if(self.silent==False): print("\n● Additional harsh masking for nb galaxies")

            obj, mask=self._main_masking(self.imagedata, thresh=nb_thresh,
                                         subtract_bkg=True, is_relative=is_relative,
                                         mask_prev=self.mask_superbright+self.mask_pri,
                                         minarea=nb_minarea, filter_kernel='default',
                                         ) ## Small region
            self.obj_nb, self.mask_nb=clean_mask(self.imagedata, mask+self.mask_superbright+self.mask_pri, silent=silent) ## Clean before suppressing
            mask=masking_blur(self.mask_nb, kernel_value=nb_blur)
            self.obj_nb, self.mask_nb=clean_mask(self.imagedata, mask, silent=silent) ## Clean before suppressing

            mask=suppress_center_mask(self.mask_nb, size_suppress=nb_suppress_center, obj=self.obj_nb,
                                      is_center_circle=is_center_circle, is_touch=centermasking_is_touch,
                                      silent=silent)
            self.test=copy.deepcopy(mask)
            self.obj_nb_sup, self.mask_nb_sup=clean_mask(self.imagedata, mask+mask_base, silent=silent) ## Clean after suppressing
            if(cig):
                self.obj_nb_supcig, self.mask_nb_supcig=clean_mask(self.imagedata, self.mask_main_supcig+self.mask_nb_sup, silent=silent)



    def _load(self, imagedata=None, fn_image=None, mask_prev=None, fn_mask_prev=None):
        ## Load image
        if(self.silent==False): print("● Load data")
        if(fn_image!=None):
            dat=fits.open(fn_image)
            imagedata=dat[0].data
            if(self.silent==False): print(">> Image: ", fn_image)
        elif(hasattr(imagedata, "__len__")):
            if(self.silent==False): print(">> Image: input data")
            pass
        else: print("Error! No data!")
        self.imagedata=np.copy(imagedata)

        ## Load Mask
        empty_mask=np.zeros_like(imagedata).astype(int)
        if(fn_mask_prev!=None):
            dat=fits.open(fn_mask_prev)
            mask_prev=dat[0].data
            if(self.silent==False): print(">> Mask: ", fn_mask_prev)
        elif(hasattr(mask_prev, "__len__")):
            if(self.silent==False): print(">> Mask: input data")
            pass
        else:
            mask_prev=np.copy(empty_mask)
            if(self.silent==False): print(">> Mask: empty mask")
        self.mask_prev=mask_prev.astype('int32')


    def _main_masking(self, imagedata, thresh, subtract_bkg=True, is_relative=True,
                    mask_prev=None, minarea=0, bw=64, bh=64,
                   filter_kernel='default'):
        """
        Descr - Mask
        INPUT
         - imagedata: Ndarray
         - thresh: threshold value (pixel value or sigma)
         * subtract_bkg: If true, subtract the background (Default: True)
         * minarea: Minimum masking area (Default: 0)
         * is_relative: If true, thresh value = thresh * bkg.globalrms.  (Default: True)
         * filter_kernel: 'default', 'ker', or manual kernel
        OUTPUT
         - mask_output
        """
        if(self.silent==False):
            print("○ Mask")
            print(">> subtract background:", subtract_bkg)
            print(">> Masking min area:", minarea)
            if(is_relative): print(">> Threshold: %f * background rms"%thresh)
            else: print(">> Threshold: %f [pixel value]"%thresh)
            print(">> filter_kernel:", filter_kernel)

        this_imagedata = imagedata.byteswap().newbyteorder()

        ## ======Preprosessing==============
        # Measure a spatially varying background on the image
        # It requires byteswap.newbyteorder. (Data value is the same)
        bkg = sep.Background(this_imagedata, mask=mask_prev, bw=bw, bh=bh)
        # fwhm = bkg.globalrms * np.sqrt(8 * np.log(2))
        # smoothimage=gaussian_filter(this_imagedata, self.fwhm/(2*np.sqrt(2*np.log(2))))
        smoothimage=this_imagedata

        if(subtract_bkg): usingimg=smoothimage - bkg
        else: usingimg=smoothimage
        self.usingimg=usingimg

        ## Setting
        if(is_relative==True): err=bkg.globalrms
        else: err=None

        if(self.silent==False): print(">> Backgrond rms:", bkg.globalrms)

        ##
        if(filter_kernel=='default'):
            return sep.extract(usingimg, thresh=thresh, minarea=minarea, err=err,
                                        segmentation_map=True, mask=mask_prev)
        elif(filter_kernel=='ker'):
            gaussian_2D_kernel = Gaussian2DKernel(2, mode = 'oversample')
            ker = gaussian_2D_kernel.array ## 9*9 Kernel
            return sep.extract(usingimg, thresh=thresh, minarea=minarea, err=err,
                                        filter_type='conv', filter_kernel=ker,
                                        segmentation_map=True, mask=mask_prev)


        else:
            return sep.extract(usingimg, thresh=thresh, minarea=minarea, err=err,
                                        filter_kernel=filter_kernel,
                                        segmentation_map=True, mask=mask_prev)

def suppress_center_mask(mask, size_suppress,
                    obj=None, ## Optional
                    is_center_circle=False, is_touch=False, silent=True):
    """
    Descr - Remove masks near the center
    INPUT
     - mask: Ndarray
     - size_suppress: radius or half-size of the box (if size_suppress<=0, skip)
     * obj: obj from sep
     * is_center_circle: If False, use a box. (Default: False)
     * is_touch: If True, remove the masks where they touch the central region (Default: False)
    OUTPUT
     - mask_output
    """
    if(silent==False):
        print("○ Supress masks near center")
        print(">> is_center_circle:", is_center_circle)
        print(">> is_touch:", is_touch)
        print(">> size_suppress:", size_suppress)


    if(size_suppress<=0): return mask
    mask_output=np.copy(mask)
    halfpos=int(len(mask)/2) ## center location

    if(is_touch):
        ## Remove the masks where they touch the central region
        if(is_center_circle):
            aperture=make_aperture_mask(imgsize=len(mask), radius=size_suppress, invert=False)
            groups_in_center=mask_output*aperture
        else:
            groups_in_center=mask_output[halfpos-size_suppress:halfpos+size_suppress,halfpos-size_suppress:halfpos+size_suppress]

    else:
        ## Remove the masks where their centers are inside the central region
        grouplist=np.unique(mask[mask!=0])
        if(hasattr(obj, "__len__")):
            groupcenter=np.vstack((obj['x'], obj['y'])).T
        else:
            groupcenter=np.zeros((len(grouplist), 2))
            for i, gn in enumerate(grouplist):
                groupcenter[i]=np.mean(np.where(mask==gn), axis=1)

        pos_diff=np.abs(groupcenter-halfpos)

        if(is_center_circle):
            dist=np.sum(pos_diff**2, axis=1)**0.5
            groups_in_center = grouplist[dist<=size_suppress]
        else:
            groups_in_center = grouplist[(pos_diff[:,0]<=size_suppress) & (pos_diff[:,1]<=size_suppress)]

    mask_output[np.isin(mask_output, groups_in_center)]=0
    return mask_output

def clean_mask(image, mask, silent=True):
    if(silent==False):
        print("○ Cleaning...")
        print(">> Max number:", np.nanmax(mask))

    reverse_mask=1-mask.astype(bool).astype('int32')
    imgdata=image.byteswap().newbyteorder()

    obj, mask = sep.extract(imgdata, thresh=0, mask=reverse_mask, segmentation_map=True, filter_kernel=None, clean=True)
    if(silent==False): print(">> New Max number:", np.nanmax(mask))
    return obj, mask


def masking_blur(mask, kernel_value=0.2):
    blur=gaussian_filter(mask.astype(bool).astype(float), kernel_value)
    blur[blur!=0]=1
    return blur


def make_aperture_mask(imgsize, radius, yrad=None, center=-1, xpos=None, ypos=None, invert=True):
    """
    make an aperture mask with a given radius.
    If center<0, center is half of the image size.
    The manual center (xpos, ypos) can be given.
    If invert is true, mask outside (inner part is 0)
    """
    if(hasattr(imgsize, "__len__")==False):
        imgsize=(imgsize, imgsize)
    mask=np.zeros(imgsize, dtype=int)
    mx, my=np.mgrid[0:imgsize[0],0:imgsize[1]]
    if((xpos==None) & (ypos==None)): ## Image center
        if(center<0): center=(imgsize[0]/2, imgsize[1]/2)
        xpos=center[0]
        ypos=center[1]
    if(yrad==None):  ## circle
        yrad=radius

    #check=np.where((mx-xpos)**2 + (my-ypos)**2 <= radius**2)
    check=np.where( ((mx-xpos)**2)/radius**2 + ((my-ypos)**2)/yrad**2 <= 1)


    # else:
        # mask=np.zeros((imgsize, imgsize), dtype=int)
        # mx, my=np.mgrid[0:imgsize,0:imgsize]
        # if((xpos==None) & (ypos==None)):
        #     if(center<0): center=int(imgsize/2)
        #     check=np.where((mx-center)**2 + (my-center)**2 <= radius**2)
        # else:
        #     check=np.where((mx-xpos)**2 + (my-ypos)**2 <= radius**2)

    mask[check]=1
    if(invert): return 1-mask
    else: return mask

def save_mask(fn, maskdata):
    maskfile = fits.PrimaryHDU()
    maskfile.data = maskdata.astype('int16')
    maskfile.writeto(fn, overwrite = True)
