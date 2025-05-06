from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
import time
from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator
import copy
import os
import pandas as pd

from skimage.transform import resize
from astropy.modeling.functional_models import Sersic1D
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval,ImageNormalize,AsinhStretch

import requests
from bs4 import BeautifulSoup
from PIL import Image

class Legacy_crawling:
    def __init__(self, coord, dir_base='./Data/', dir_work='./', fn_ccd_fits='image', fn_crop_fits='image_crop', silent=False, save_ccd_fits=False):
        self.coord=coord         # list : Coordinate  
        self.dir_base=dir_base   # Base directory. Input files will be imported from here.
        self.dir_work=dir_work   # Working directory. Output files will be generated to here. If it is not exist, make dir.
        self.dir_workfull=dir_base+dir_work
        os.makedirs(self.dir_workfull, exist_ok=True)
        
        self.fn_ccd_fits = fn_ccd_fits    # Output CCD image
        self.fn_crop_fits = fn_crop_fits  # Output crop image
        self.silent=silent 
        self.save_ccd_fits=save_ccd_fits  # Bool. whether save CCD image?
        self.baseurl='https://www.legacysurvey.org/viewer/exposures/?ra=%.4f&dec=%.4f&layer=ls-dr9'%(self.coord[0], self.coord[1])
        if(silent==False):
            print("========== Legacy Crawling ===========")
            print(">> Coordinate - RA : %f, Dec : %f"%(self.coord[0], self.coord[1]))
            print(">> FITS path : ", self.dir_workfull+self.fn_crop_fits+".fits")
        self.i_get_websource()
        self.i_get_ccdlist()
        
    def i_get_websource(self):
        """
        Descr : Get web source from baseurl
        ** __Init__ will run this function!
        """
        web_source=requests.get(url=self.baseurl)
        if(self.silent==False): print("\n● Get web source from : ", self.baseurl)
        self.web_source=BeautifulSoup(web_source.content, "html.parser")
    
    def i_get_ccdlist(self):
        """
        Descr : Get CCD plate list
        ** __Init__ will run this function!
        ** Run get_websource first.
        """
        tdlist=self.web_source.findAll("td") # Find all <td>
        hreflist=np.zeros(len(tdlist), dtype='object')  # Hyperlink (fits file)
        imglist=np.zeros(len(tdlist), dtype='object')   # Sample images list
        for i in range (len(tdlist)):
            try: hreflist[i]=tdlist[i].find("a")['href'] 
            except: pass
            try: imglist[i]=tdlist[i].find("img")['src']
            except: pass

        imglist=imglist[np.nonzero(imglist)[0].astype(int)]

        ## Get CCD info using hreflist
        data_index=np.nonzero(hreflist)[0].astype(int)
        hreflist=hreflist[data_index]
        ccd_info_raw=np.zeros(len(data_index), dtype='object')
        ccd_info_raw1=np.zeros(len(data_index), dtype='object')
        for i in range (len(data_index)):
            ccd_info_raw[i]=tdlist[data_index[i]].text.split()
            ccd_info_raw1[i]=tdlist[data_index[i]].findAll("small")[1]

        ccd_info_dtype=[('idnum', '<i4'), ('proj', '<U6'), ('filter', '<U6'), ('plate1', '<i4'), ('plate2', '<U6'), ('exp', '<f8'), ('MJD', '<f8')]
        ccd_info=np.zeros(len(data_index), dtype=ccd_info_dtype)
        for i in range (len(data_index)):
            ccd_info[i]['idnum']=i
            ccd_info[i]['proj']=ccd_info_raw[i][1]
            ccd_info[i]['filter']=ccd_info_raw[i][2]
            ccd_info[i]['plate1']=ccd_info_raw[i][3]
            ccd_info[i]['plate2']=ccd_info_raw[i][4][:-1]
            ccd_info[i]['exp']=ccd_info_raw[i][5]
            ccd_info[i]['MJD']=ccd_info_raw1[i].text.split()[-1][:-1]
            
        self.tdlist=tdlist
        self.hreflist=hreflist
        self.imglist=imglist
        self.ccd_info=ccd_info
        
        self.ccdfitslist=np.zeros(len(self.ccd_info), dtype='object')
        self.ccdfits_imglist=np.zeros(len(self.ccd_info), dtype='object')
        for i in range (len(self.ccd_info)):
            self.ccdfitslist[i]='https://www.legacysurvey.org/viewer/image-data/ls-dr9/'\
            +str(self.ccd_info[i]['proj'])+'-'+str(self.ccd_info[i]['plate1'])+'-'+str(self.ccd_info[i]['plate2'])
            self.ccdfits_imglist[i]='https://www.legacysurvey.org/viewer/image-stamp/ls-dr9/'\
            +str(self.ccd_info[i]['proj'])+'-'+str(self.ccd_info[i]['plate1'])+'-'+str(self.ccd_info[i]['plate2'])
        if(self.silent==False): 
            print("\n● Get data from the server")
            print(">> CCD list : \n")
            display(pd.DataFrame(self.ccd_info))
        
    def show_quickimage(self, ccdid=0):
        """
        Descr : Show the quick image
        INPUT : CCD ID
        """
        imgurl=self.imglist[ccdid]
        image_fn=download_file(imgurl)
        image = Image.open(image_fn)
        display(image)
        
    def show_ccdimage(self, ccdid=0):
        """
        Descr : Show the full CCD plate image
        INPUT : CCD ID
        """
        subweb_source=requests.get(url='https://www.legacysurvey.org/'+self.hreflist[ccdid])
        subweb_source=BeautifulSoup(subweb_source.content, "html.parser")
        
        imgurl=self.ccdfits_imglist[ccdid]+'.jpg'  # subweb_source.findAll("image")
        image_fn=download_file(imgurl)
        image = Image.open(image_fn)
        im_array = np.asarray(image)
        
        target=subweb_source.findAll("rect")[0] #Rectangular
        target_coord=[float(target['x'])+float(target['width'])/2, 
                      np.shape(im_array)[0]-float(target['y'])-float(target['height'])/2]
        
        fig=plt.figure(figsize=(10.73*1.5,5.61*1.5))
        plt.imshow(im_array)
        plt.scatter(target_coord[0], target_coord[1], ec='orange', fc='none', s=500, linewidth=2)
                
    def get_FITS(self, ccdid=0, save_fits=False):
        """
        Descr : Download CCD FITS.
        INPUT : 
         - CCD ID
         * save_fits (default - False) : whether save the FITS image (at dir_workfull)
        """
        ccd_path=self.ccdfitslist[ccdid]
        if(self.silent==False): 
            print("\n● Get the CCD plate FITS file from the server")
            print(">> Download from : ", ccd_path)
        ccd_fn=download_file(ccd_path, show_progress=True)
        if(save_fits==True):
            hdu = fits.open(ccd_fn)
            fn=self.dir_workfull+self.fn_ccd_fits+".fits"
            hdu.writeto(fn, overwrite=True)
            if(self.silent==False): print(">> Saved to : ", fn)
        else:
            if(self.silent==False): print(">> Did not save the full FITS file (If you want, save_fits -> True)")
        
        return ccd_fn
    
    def auto_get_FITS(self, select_filter='g', crop_size=200, save_ccd_fits=False):
        """
        Descr : Automatically get FITS image.
        ** 1) Select filter
        ** 2) Sort by exposure time (longer first)
        ** 3) Download
        ** 4) Header check
        ** 5) Try to crop
        ** 6) Try to fix the header
        Input
         * select_filter (default - 'g') : Select filter
         * crop_size (default - 200) : Size of the crop image
         * save_ccd_fits (default - False) : Save the CCD plate image
        """
        usingccd=self.ccd_info[np.where(self.ccd_info['filter']==select_filter)]
        cropname=self.dir_workfull+self.fn_crop_fits+".fits"
        
        ## Sort - Long exp first
        newid=usingccd['exp'].argsort()
        usingccd=usingccd[newid][::-1] 
        
        if(self.silent==False): 
            print("\n● Select -> Get -> Crop -> Edit header -> Save FITS file")
            print(">> Selected FITS : \n")
            display(pd.DataFrame(usingccd))
        for i in range (len(usingccd)):
            if(self.silent==False): print("\n[CCD Images Trial : %d/%d]"%(i, len(usingccd)))
            try: 
                ccd_fn=self.get_FITS(usingccd['idnum'][i], save_fits=save_ccd_fits)
                print(">> Downloaded completed")
                result=check_header(ccd_fn, header_item_list=[['MAGZERO', 'MAGZPT']])
                if(np.isnan(np.sum(result))): raise Exception('>>ERROR! Header does not have the zero point!')
                print(">> Check header completed")
                crop_fits(ccd_fn, self.coord, size=crop_size, cropname=cropname, silent=self.silent)
                print(">> Crop successed!")
                header_find_copy(ccd_fn, cropname, ['EXPTIME', 'GAIN'], [None, ['ARAWGAIN']])
                print(">> Header fixed!")
                self.selected_CCD=usingccd['idnum'][i]  #Return selected CCD ID
                
                ## Add NCOMBINE
                fits.setval(cropname, 'NCOMBINE', value=1)

                return 
            except: continue
        print("\n\n>> |Warning!| Crop failed!")    
        
        
        
def list_modi_2layers(items, silent=False):
    """
    Descr : Change the format of list
      * e.g.) 'asdf' -> [['asdf']]
      * e.g.) ['asdf'] -> [['asdf']]
      * e.g.) ['a1', 'b1'] -> [['a1'], ['b1']]
      * e.g.) [['a1', 'a2'], 'b1'] -> [['a1', 'a2'], ['b1']]
    INPUT
     - items
    """
    if(type(items)==type(None)): return None
    elif(type(items)==str):
        return [[items]]
    elif(hasattr(items, "__len__")!=True): 
        if(silent==False): print("** Input value is not an array or list!")
        return np.nan
    else:
        # For each sub-lists
        for i in range (len(items)):
            if(type(items[i])==str): 
                items[i]=[items[i]]
    return items
    
def check_header(filename, header_item_list=None, silent=False):
    """
    Descr: Find items in header
    INPUT
     - filename : FITS file name
     - header_item_list : list of the items wanted to check  
       * e.g.) ['data'] -> Find header 'data'
       * e.g.) [['gainA', 'gainB'], ['data']] : Find either 'gainA' or 'gainB', and find 'data'
    OUTPUT
     - the list of #ext that includes each of the item
    """
    if(silent==False): print("\n● Check headers\n>> Checking list : ", header_item_list)
    header_item_list=list_modi_2layers(header_item_list, silent=silent)  # list in list (1st list : # of items wanted to find / 2nd list : possible items)
    if(silent==False): print(">> Modified Checking list : ", header_item_list)
    if(type(header_item_list)==type(None)):
        print(">> Checking list is empty!")
        return
       
    hdu = fits.open(filename)
    extpos=np.full(len(header_item_list), np.nan)  # return the # of ext for each item
    for i, item in enumerate(header_item_list): # 1st list : items wanted to find
        if(silent==False): print(">>>> Item : ", item)
        is_found=False
        for ext in range (len(hdu)): # For each ext.
            header_item=list(hdu[ext].header)
            
            result=np.zeros(len(item))  # 2nd list : possible items
            for j in range(len(item)):
                result[j]=np.isin(item[j], header_item)
            result=np.sum(result) # 0 : failed to found
            if(result!=0): 
                extpos[i]=ext
                is_found=True
                break
        if(is_found==False): 
            print(">> FITS does not have the item :", item)
            return
    print(">> Checking FITS header completed!")
    return extpos
    
    
def crop_fits(filename, coord, size=100, ext_img=1, ext_wcs=1, silent=False, ignore_size_error=False,
              cropname='/home/donghyeon/ua/udg/spectrum/galfit/image/crop/example_cutout.fits'):
    """
    Descr : Crop fits image around the coordinate
    INPUT
     - filename : Input fits
     - cropname : Output (cropped) fits 
     - coord : Center Coordinate (list)
     - size : Size of the crop image
     * ext_img (default - 1) : Extension for the image
     * ext_wcs (default - 1) : Extension for the wcs
     * ignore_size_error (default - False) : Save the image even if the size is incorrect.
    """
    if(silent==False): print("\n● Crop images\n>> Get data...")
    image_data = fits.getdata(filename, ext=ext_img)
    image_header = fits.getheader(filename, ext=ext_wcs)
    
    if(silent==False): print(">> Open FITS...")
    hdu = fits.open(filename)[0]
    wcs = WCS(image_header)
    coord_pixel=wcs.all_world2pix(coord[0], coord[1], 0) 
    
    # Make the cutout, including the WCS
    cutout = Cutout2D(image_data, position=coord_pixel, size=[size, size], wcs=wcs)
    if(silent==False): print(">> FITS shape : ", np.shape(cutout.data))
    if(ignore_size_error==False):
        if((np.shape(cutout.data)[0]!=size) | (np.shape(cutout.data)[1]!=size)): 
            if(silent==False): print(">> Size does not correct")
            raise Exception('>> ERROR! Size does not correct')
    
    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data
    
    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())
   
    if(silent==False): print(">> Save FITS...")
    # Write the cutout to a new FITS file
    hdu.writeto(cropname, overwrite=True)
    if(silent==False): print(">> Done!")
    
    
def header_find_copy(fits1, fits2, headerlist, headerlist_prevname=None, silent=False):
    """
    Descr: Find items from the header of fits2. If items do not exist, copy the items from fits1.
    INPUT
     - fits1 : Original fits path
     - fits2 : Final fits path
     - headerlist : Item list
     - headerlist_prevname (default : None) : If the name of items are different from fits1 and fits2.
          e.g.1) fits1 : "ARAWGAIN" or "TOTGAIN", fits2 : "GAIN"  --> headerlist=['GAIN'], headerlist_prevname=[['ARAWGAIN', 'TOTGAIN']]
          e.g.2) headerlist=['GAIN', 'EXPTIME'], headerlist_prevname=[['ARAWGAIN', 'TOTGAIN'], None]
    """
    if(silent==False): 
        print("\n● Header Edit")
        print(">> Header list : ", headerlist)
        print(">> Header previous name list : ", headerlist_prevname)
    if(type(headerlist_prevname)==type(None)): # Default input
        headerlist_prevname=np.full(len(headerlist), None)
    if(len(headerlist)!=len(headerlist_prevname)): print("** |Warning!| headerlist and headerlist_prevname have differnt length!")
        
    prev=fits.open(fits1)
    post=fits.open(fits2)[0]  #EXT 0 only
    header_item=list(post.header)
    
    ## ================ Find =====================
    for i, itemname in enumerate(headerlist): 
        # Find the item in the fits2 -> if yes, pass
        if (np.sum(np.isin(itemname, header_item))!=0):
            if(silent==False): print(">> - Item : ", itemname, "is exist")
                
        # If it does not find item,
        # Find the item in the fits1
        else:
            if(silent==False): print(">> - Item : ", itemname, "does not exist")
            found=False # check whether the code finds the item
            
            # Loop for Ext
            for ext in range (len(prev)):
                temp=list(prev[ext].header)
                # ====== possiblenames ====
                possiblenames=headerlist_prevname[i]
                if(type(possiblenames)==type(None)): possiblenames=[headerlist[i]]  # None -> copy headerlist
                elif(hasattr(possiblenames, "__len__")):
                    if(possiblenames[0]==None): possiblenames=[headerlist[i]]  # [None] -> copy headerlist

                # Find the item in the fits1 - ext i
                for name in possiblenames:
                    if (np.sum(np.isin(name, temp))!=0):               
                        if(silent==False): print(">>>> - Ext ", ext, "has the item")
                        found=True
                        fits.setval(fits2, itemname, value=prev[i].header[name])
                        break
                    else:
                        if(silent==False): print(">>>> - Ext ", ext, "does not have the item")
            if(found==False): print(">>>> |Warning!| Fail to found the Item : ", itemname, "\n")
            
            
def show_FITSimage(fn, wcs_header=0, use_zscale=True):
    image_data = fits.getdata(fn, ext=0)
    image_header = fits.getheader(fn, ext=wcs_header)
    wcs = WCS(image_header)
    ax = plt.subplot(projection=wcs)
    
    if(use_zscale==True): 
        norm = ImageNormalize(image_data, interval = ZScaleInterval())
        im = ax.imshow(image_data, origin='lower', norm=norm, cmap='bone')
    else:
        image_data=np.log10(image_data)
        im = ax.imshow(image_data, origin='lower', cmap='bone')
