import numpy as np
import sep
from astropy.io import fits
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel ## For a kernel


class Masking():
    def __init__(self,
                 imagedata=None, previous_mask=None, minarea=5,
                 fn_image=None, fn_save_masking='masking.fits', fn_save_obj=None,
                 suppress_center=True, center_size=3, is_center_circle=True, centermasking_is_touch=False,
                 thresh=1.5, is_abs_cut=False,
                 show_masking=False,
                 ## Superbright parameters
                 superbright=False, superbright_thresh=3, superbright_area_frac=0.5, superbright_suppress_center=None,
                 extreme_lower=1, extreme_upper=5,
                 img_size=200,
                 zero_data=10, zero_area_crit=None, skip_main_mask=False,
                 filter_kernel='default'
                 ):
        """
        ADD DESCR
        Descr - Masking
        INPUT
         * imagedata: data (Default: None)
         * fn_image: image file when imagedata=none (Default: None)
         * fn_save_masking: save fn (Default: 'masking.fits')
         * previous_mask: previous mask image (Default: None)
         * suppress_center: Remove the masking at the center (with radius center_size) (Default: True)
         * center_size: The radius criterion for the center (Default: 3)
         * thresh: threshold pixel value for detection. See is_abs_cut (Default: 1.5)
         * is_abs_cut: If True, thresh will be the absolute value for detection.
                       If False, detection limit = thresh * bkg.globalrms
                       (Default: False)

         INPUT - Return
         * show_masking

        SUPERBRIGHT_INPUT
         * superbright: do superbright (Default: False)
         * superbright_thresh: threshold for superbright
         * superbright_suppress_center: Remove the masking at the center.
           When the value is None, it follows 'suppress_center' (Default: None)

        EXTREMEPIXEL_INPUT
         * extreme_lower: mask where the pixel value is less than median(image)-extreme_lower (Default: 1)
         * extreme_upper: mask where the pixel value is larger than median(image)+extreme_upper (Default: 5)

        ZERODARA_INPUT
         * zero_data: masking pixels where the pixels have the zero CCD value (Default: 10)
         * zero_area_crit: masking pixels when the number of pixel is larger than this value
           (Default: None = no masking)

        """
        ## Load image
        if(fn_image!=None):
            dat=fits.open(fn_image)
            imagedata=dat[0].data
        elif(hasattr(imagedata, "__len__")): pass
        else: print("No data!")
        self.imagedata=np.copy(imagedata)

        self.center_size=center_size
        self.is_center_circle=is_center_circle
        self.previous_mask=previous_mask
        self.minarea=minarea
        self.filter_kernel=filter_kernel
        self.is_abs_cut=is_abs_cut
        self.centermasking_is_touch=centermasking_is_touch
        self.img_size=img_size

        ## Empty mask setting
        empty_mask=np.zeros(np.shape(imagedata)).astype(int)
        if(hasattr(previous_mask, "__len__")==False): previous_mask=np.copy(empty_mask)
        self.superbright_mask=np.copy(empty_mask)
        self.extreme_mask=np.copy(empty_mask)
        self.zero_mask=np.copy(empty_mask)
        self.main_mask=np.copy(empty_mask)


        ## ===========  Masking  ==================
        if(superbright==True):
            self.__superbright(imagedata, previous_mask, superbright_thresh,
                               superbright_area_frac, superbright_suppress_center)
        self.usingmask=masking_sum(previous_mask, self.superbright_mask)
        if(extreme_lower!=None and extreme_upper!=None):
            self.__extreme_pixels(imagedata, extreme_lower, extreme_upper)
        self.usingmask=masking_sum(self.usingmask, self.extreme_mask)
        if(zero_area_crit!=None):
            self.__zero_masking(imagedata, zero_data, zero_area_crit)
        self.usingmask=masking_sum(self.usingmask, self.zero_mask)
        self.usingmask_bool=masking_bool(self.usingmask)

        if(skip_main_mask==False):
            self.__main_mask(imagedata, thresh, suppress_center)
        self.mask=masking_sum(self.usingmask, self.main_mask)
        if(hasattr(self.mask, "__len__")==False): self.mask=np.copy(empty_mask)

        ## ======================= Print results ========================
        if(show_masking==True):
            plt.imshow(self.mask, cmap='jet', origin='lower')
            plt.grid(None)

        # Make files
        if(fn_save_masking!=None):
            maskfile = fits.PrimaryHDU()
            maskfile.data = self.mask.astype('int16')
            maskfile.writeto(fn_save_masking, overwrite = True)
        if(fn_save_obj!=None):
            np.save(fn_save_obj, self.main_obj)

        self.resultimage=np.ma.masked_array(self.imagedata, self.mask)
        self.mask_area=len(np.where(self.mask!=0)[0])


    def __main_mask(self, imagedata, thresh, suppress_center):

        this_imagedata = imagedata.byteswap().newbyteorder()

        # measure a spatially varying background on the image
        # It requires byteswap.newbyteorder. (Data value is the same)
        bkg = sep.Background(this_imagedata, mask=self.usingmask_bool)
        # try: bkg = sep.Background(this_imagedata)
        # except:
        #     this_imagedata = this_imagedata.byteswap().newbyteorder()
        #     bkg = sep.Background(this_imagedata, mask=self.usingmask_bool)
        self.sigma = bkg.globalrms
        self.fwhm = self.sigma * np.sqrt(8 * np.log(2))
        #smoothimage=gaussian_filter(this_imagedata, self.fwhm/(2*np.sqrt(2*np.log(2))))
        smoothimage=this_imagedata
        datasub = smoothimage - bkg
        # if(superbright==False): return datasub

        if(self.is_abs_cut==True): err=None
        else: err=bkg.globalrms

        if(self.filter_kernel=='default'):
            objects, mask = sep.extract(datasub, thresh=thresh, minarea=self.minarea,
                                        err=err, segmentation_map=True, mask=self.usingmask_bool)
        else:
            objects, mask = sep.extract(datasub, thresh=thresh, minarea=self.minarea, filter_kernel=self.filter_kernel,
                                        err=err, segmentation_map=True, mask=self.usingmask_bool)

        self.main_mask=self.__do_suppress_center(objects, mask, self.center_size, self.is_center_circle, self.centermasking_is_touch, suppress_center)
        self.main_obj=objects

    ##============================ Ftns =====================
    def __do_suppress_center(self, obj, mask, center_size, is_center_circle=False, is_touch=False, suppress_center=True):
        if(suppress_center==False): return mask
        if(len(obj)==0): return mask

        mask_output=np.copy(mask)
        halfpos=int(len(mask_output)/2)

        if(is_touch): ## Masking objects when the boundaries of the masking regions are inside the central region
            if(is_center_circle):
                center_group=masking_sum(mask_output, make_aperture_mask(self.img_size, center_size)) ## Count from 1 (0 is background)
            else:
                center_group=mask_output[halfpos-center_size:halfpos+center_size,halfpos-center_size:halfpos+center_size]
        else: ## Masking objects when the centers of the masking regions are inside the central region
            xdiff=np.abs(obj['x']-halfpos)
            ydiff=np.abs(obj['y']-halfpos)
            if(is_center_circle):
                dist=(xdiff**2 + ydiff**2)**0.5
                center_group_mask=np.where(dist<=center_size)[0] ## Count from 0
            else:
                center_group_mask=np.where((xdiff<=center_size) & (ydiff<=center_size))[0]
            center_group=center_group_mask+1 ## Count from 1 (0 is background)
        center_group=np.unique(center_group)
        mask_output[np.isin(mask_output, center_group)]=0
        return mask_output

    def __superbright(self, imagedata, previous_mask,
                      superbright_thresh=3, superbright_area_frac=0.5, superbright_suppress_center=None):
        # To kill super bright (cosmic ray or briht stars)
        # high threshold, large kernel

        this_imagedata=imagedata.byteswap().newbyteorder()
        gaussian_2D_kernel = Gaussian2DKernel(2, mode = 'oversample')
        ker = gaussian_2D_kernel.array ## 9*9 Kernel
        objects, mask = sep.extract(this_imagedata, thresh=superbright_thresh, minarea=self.minarea,
                                     filter_type='conv', filter_kernel=ker,
                                     segmentation_map=True, mask=masking_bool(previous_mask))

        mask=self.__do_suppress_center(objects, mask, self.center_size, self.is_center_circle, False, superbright_suppress_center)

        ## Area Fraction cut check (Too large mask -> remove)
        area=len(this_imagedata)**2
        areacrit=area*superbright_area_frac

        if(np.sum(mask.astype(bool))>areacrit): ## RollBack
            mask.fill(0)
            this_imagedata=np.copy(imagedata)

        ## return mask
        self.superbright_mask=mask.astype(int)

    def __extreme_pixels(self, imagedata, extreme_lower=1, extreme_upper=5):
        ## Additional masking for extreme pixels
        masked_image=np.ma.masked_array(imagedata, self.usingmask)
        median_value=np.ma.median(masked_image)
        strange=np.where((imagedata>(median_value+extreme_upper))
                         | (imagedata<(median_value-extreme_upper)))
        self.extreme_mask[strange]=1

    def __zero_masking(self, imagedata, zero_data=10, zero_area_crit=40000):
        ## Additional masking for extreme pixels
        zerodata=np.where(imagedata==zero_data)
        if(len(zerodata[0])>=zero_area_crit):
            self.zero_mask[zerodata]=1
        else: return

def masking_sum(prev_mask=None, new_mask=None):
    if(hasattr(prev_mask, "__len__")): prev_mask=prev_mask.astype('int16')
    if(hasattr(new_mask, "__len__")): new_mask=new_mask.astype('int16')

    if(hasattr(prev_mask, "__len__")):
        if(hasattr(new_mask, "__len__")):
            previous_mask_max=np.nanmax(prev_mask) ## New mask id = start from max prev_mask
            new_mask[new_mask!=0]+=previous_mask_max
            return new_mask+prev_mask
        else: return prev_mask
    else:
        if(hasattr(new_mask, "__len__")): return new_mask
        else: return None

def masking_compress(mask):
    if(hasattr(mask, "__len__")):
        mask_uniq=np.unique(mask)
        for newmasknum, masknum in enumerate(mask_uniq):
            if(masknum==0): continue
            else:
                target=np.where(mask==masknum)
                mask[target]=newmasknum
        return mask
    else: return None

def masking_bool(mask):
    if(hasattr(mask, "__len__")): return mask.astype(bool)
    else: return None


def masking_blur(mask, kernel_value=0.2):
    blur=gaussian_filter(mask.astype(bool).astype(float), kernel_value)
    blur[blur!=0]=1
    return blur


def make_aperture_mask(imgsize, radius, center=-1, xpos=None, ypos=None):
    mask=np.ones((imgsize, imgsize), dtype=int)
    mx, my=np.mgrid[0:imgsize,0:imgsize]
    if((xpos==None) & (ypos==None)):
        if(center<0): center=int(imgsize/2)
        check=np.where((mx-center)**2 + (my-center)**2 <= radius**2)
    else:
        check=np.where((mx-xpos)**2 + (my-ypos)**2 <= radius**2)

    mask[check]=0
    return mask

def mask_numbering(mask):
    """
    Bool (or 1) mask -> mask with numbers
    """
    objects, mask = sep.extract(mask, thresh=0.5, err=None, segmentation_map=True, )
    return mask




# def masking(imagedata=None, previous_mask=None,
#             fn_image=None, fn_save_masking='masking.fits',
#             suppress_center=True, center_size=3, thresh=1.5, is_abs_cut=False,
#             return_mask=False, return_image=False,
#             show_masking=False, imagedata_applied=None,
#             ## Superbright parameters
#             superbright=False, thresh_superbright=3, superbright_min=1, superbright_max=5, superbright_area_frac=0.5, superbright_suppress_center=None,
#             ):
#     """
#     ADD DESCR
#     Descr - Masking
#     INPUT
#      * imagedata: data (Default: None)
#      * fn_image: image file when imagedata=none (Default: None)
#      * fn_save_masking: save fn (Default: 'masking.fits')
#      * previous_mask: previous mask image (Default: None)
#      * suppress_center: Remove the masking at the center (with radius center_size) (Default: True)
#      * center_size: The radius criterion for the center (Default: 3)
#      * thresh: threshold pixel value for detection. See is_abs_cut (Default: 1.5)
#      * is_abs_cut: If True, thresh will be the absolute value for detection.
#                    If False, detection limit = thresh * bkg.globalrms
#                    (Default: False)
#
#      INPUT - Return
#      * return_mask
#      * return_image
#      * show_masking
#      * imagedata_applied
#
#     SUPERBRIGHT_INPUT
#      * superbright: do superbright (Default: False)
#      * thresh_superbright: threshold for superbright
#      * superbright_min: mask where the pixel value is less than median(image)-superbright_min (Default: 1)
#      * superbright_max: mask where the pixel value is larger than median(image)+superbright_max (Default: 5)
#      * superbright_suppress_center: Remove the masking at the center.
#        When the value is None, it follows 'suppress_center' (Default: None)
#
#     """
#
#     def do_suppress_center(mask, center_size, suppress_center=True):
#         if(suppress_center==False): return mask
#         mask_output=np.copy(mask)
#         halfpos=int(len(mask_output)/2)
#         center_group=mask_output[halfpos-center_size:halfpos+center_size,halfpos-center_size:halfpos+center_size]
#         center_group=np.unique(center_group)
#         mask_output[np.isin(mask_output, center_group)]=0
#         return mask_output
#
#     if(fn_image!=None):
#         dat=fits.open(fn_image)
#         imagedata=dat[0].data
#     elif(hasattr(imagedata, "__len__")): pass
#     else: print("No data!")
#     imagedata = imagedata.byteswap().newbyteorder()
#
#     # measure a spatially varying background on the image
#     # It requires byteswap.newbyteorder. (Data value is the same)
#     bkg = sep.Background(imagedata)
#
#     sigma = bkg.globalrms
#     fwhm = sigma * np.sqrt(8 * np.log(2))
#     smoothimage=gaussian_filter(imagedata, fwhm/(2*np.sqrt(2*np.log(2))))
#     datasub = smoothimage - bkg
#     # if(superbright==False): return datasub
#
#     #==============construct a mask==================
#     if(superbright==True):
#         # To kill super bright (cosmic ray or briht stars)
#         # Run mask twice
#         # First : high threshold, large kernel
#         # Second : Normal
#
#         ## ========== Find masks with a high threshold & large kernel =========
#         gaussian_2D_kernel = Gaussian2DKernel(2, mode = 'oversample')
#         ker = gaussian_2D_kernel.array ## 9*9 Kernel
#         objects, mask = sep.extract(datasub, thresh=thresh_superbright,
#                                                      filter_type='conv', filter_kernel=ker,
#                                                      segmentation_map=True)
#
#         mask=do_suppress_center(mask, center_size, suppress_center)
#
#         ## Additional masking (superbright_min and max)
#         newimagedata=np.copy(imagedata)
#         newimagedata[mask!=0]=np.nanmedian(newimagedata)
#         strange=np.where((newimagedata>(np.nanmedian(newimagedata)+superbright_max))
#                          | (newimagedata<(np.nanmedian(newimagedata)-superbright_min)))
#
#
#         ## Area Fraction cut check (Too large mask -> remove)
#         area=len(imagedata)**2
#         areacrit=area*superbright_area_frac
#
#         if(np.sum(mask.astype(bool))>areacrit): ## RollBack
#             mask.fill(0)
#             newimagedata=np.copy(imagedata)
#
#         newimagedata = newimagedata.byteswap().newbyteorder()
#         return masking(imagedata=newimagedata, previous_mask=mask,
#                        fn_image=None, fn_save_masking=fn_save_masking,
#                        suppress_center=suppress_center,
#                        center_size=center_size, thresh=thresh, is_abs_cut=is_abs_cut,
#                        return_mask=return_mask, return_image=return_image,
#                        show_masking=show_masking,
#                        imagedata_applied=imagedata,
#                        superbright=False)
#
#
#     ##===================== Non superbright ========================
#     else:
#         if(hasattr(previous_mask, "__len__")==True):
#             sep_mask=previous_mask.astype(bool)
#         else: sep_mask=None
#         if(is_abs_cut==True):
#             objects, mask = sep.extract(datasub, thresh=thresh,
#                                         segmentation_map=True, mask=sep_mask)
#         else: objects, mask = sep.extract(datasub, thresh=thresh,
#                                           err=bkg.globalrms, segmentation_map=True, mask=sep_mask)
#     mask=do_suppress_center(mask, center_size, suppress_center)
#
#     ## If it is second term, add previous mask
#     if(hasattr(previous_mask, "__len__")==True):
#         previous_mask_max=np.nanmax(previous_mask)
#         mask[mask!=0]+=previous_mask_max ## New mask id = start from max prev_mask
#         mask+=previous_mask
#
#     ## ======================= Print results ========================
#     if(show_masking==True):
#         plt.imshow(mask, cmap='jet', origin='lower')
#         plt.grid(None)
#
#     # Make files
#     if(fn_save_masking!=None):
#         maskfile = fits.PrimaryHDU()
#         maskfile.data = mask
#         maskfile.writeto(fn_save_masking, overwrite = True)
#     if(return_image==True):
#         if(hasattr(imagedata_applied, "__len__")):
#             returnimage=np.ma.masked_array(imagedata_applied, mask)
#         else: returnimage=np.ma.masked_array(imagedata, mask)
#         if(return_mask==True): return mask, returnimage
#         else: return returnimage
#     if(return_mask==True): return mask




def masking_simple(fn_image='./Data/image_crop.fits', fn_save_masking='masking.fits', check_masking=False, suppress_center=True, return_mask=False):

    dat=fits.open(fn_image)
    imagedata=dat[0].data
    imagedata = imagedata.byteswap().newbyteorder()

    # measure a spatially varying background on the image
    # It requires byteswap.newbyteorder. (Data value is the same)
    bkg = sep.Background(imagedata)

    sigma = bkg.globalrms
    fwhm = sigma * np.sqrt(8 * np.log(2))
    smoothimage=gaussian_filter(imagedata, fwhm/(2*np.sqrt(2*np.log(2))))
    datasub = smoothimage - bkg

    # construct a mask
    objects, mask = sep.extract(datasub, 1.5, err=bkg.globalrms, segmentation_map=True)
    center=mask[int(len(mask)/2), int(len(mask)/2)]
    newmask=np.copy(mask)

    if(suppress_center==True): newmask[newmask==center]=0
    if(check_masking==True):
        plt.imshow(newmask, cmap='jet', origin='lower')
        plt.grid(None)

    # construct mask
    if(fn_save_masking!=None):
        maskfile = fits.PrimaryHDU()
        maskfile.data = newmask
        maskfile.writeto(fn_save_masking, overwrite = True)
    if(return_mask==True): return newmask
