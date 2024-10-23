import DH_array as dharray
import DH_path as dhpath
import numpy as np
import os
import random
import pickle
import copy
import time
import shutil
import os, glob
from multiprocessing import Pool
from astropy.io import fits
from pathlib import Path

## For sigma masking
from astropy.stats import sigma_clip, sigma_clipped_stats

## Importing DH packages
import DH_array as dharray
import DH_path as dhpath
import DH_LegacyCrawling as dhcrawl
import DH_Galfit as dhgalfit
import DH_masking as dhmask
import DH_FileCheck as dhfc
import DH_multicore as mulcore

from astropy.modeling.functional_models import Sersic1D, Sersic2D


def multicore_extract_psf_fits(dir_work_list,
                           fna='', band_need=['g', 'r'],
                           Ncore=10, show_progress=10000, use_try=False, show_multicore=True):
    global sub_extract_fits
    def sub_extract_fits(i):
        fn=dir_work_list[i]+fna

        if(os.path.exists(fn)):
            head=fits.getheader(fn)
            Nband=len(head.get("bands")) ## Number of bands in the file
            for ext in range (Nband): ## For each extension
                thisband=head.get("band%d"%ext)
                if(np.isin(thisband, band_need)):
                    newfn=dir_work_list[i]+Path(fn).stem+thisband+'.fits'
                    fits.writeto(newfn, fits.getdata(fn, ext), fits.getheader(fn, ext),
                                 overwrite=True, output_verify='silentfix')
            return 1
        else:
            #print("No data at ", i)
            return 0

    return mulcore.multicore_run(sub_extract_fits, len(dir_work_list), Ncore=Ncore,
                          use_try=use_try, show_progress=show_progress, debug=show_multicore)


def multicore_crawling(DirInfo=None, dir_group=None, region='s', dr='dr9',
                       bandlist=['g', 'r'],
                       timeout=None,
                       psf_mode='area', silent=True, silent_error=True, use_maskbit=True,
                       is_overwrite_data=False, skip_search=False, save_error=True,
                       is_ignore_err=False, sleep=1, retry=5, image_size=200,
                       silent_timeerror=True,
                       download_image_only=False,
                       is_overwrite_download=False,
                       dir_brick=None,
                       Ncore=2, show_progress=10000, use_try=True, show_multicore=False, ## Multi-core
                       ):
    if(dir_brick==None): dir_brick=dhpath.dir_brick()
    global sub_crawling
    print("Start Multicore run")

    def sub_crawling(i):
        coord=DirInfo.coord_array[i]
        dir_work=DirInfo.dir_work_list[i]
        for band in bandlist:
            Crawl_VIR=dhcrawl.LegacyCrawling(dir_work=dir_work, band=band, region=region,
                              dr=dr,
                              dir_brick=dir_brick,
                              coord=coord, psf_mode=psf_mode, silent=silent,
                              silent_error=silent_error, save_error=save_error,
                              silent_timeerror=silent_timeerror,
                              image_size=image_size,
                              is_overwrite_data=is_overwrite_data,
                              is_overwrite_download=is_overwrite_download,
                              is_ignore_err=is_ignore_err, sleep=sleep, retry=retry)
            if(download_image_only==False):
                Crawl_VIR.psf()
                Crawl_VIR.psf(mode='target')
                Crawl_VIR.sigma()
                Crawl_VIR.maskbits()

    mulcore.multicore_run(sub_crawling, len(DirInfo.dir_work_list), Ncore=Ncore,
                          use_try=use_try, show_progress=show_progress, debug=show_multicore)

    res=np.zeros(len(bandlist), dtype=object)
    for i in range (len(bandlist)):
        res[i]=dhfc.FileGroupSearch(DirInfo=DirInfo, fn_set=['image_'+dr+'_'+region+bandlist[i]+'.fits',
                                                        'sigma_'+dr+'_'+region+bandlist[i]+'.fits',
                                                       ])
    return res



# def multicore_crawling(DirInfo=None, dir_group=None, region='s',
#                        timeout=None, Ncore=2, show_progress=100, use_try=True,## Multi-core
#                        psf_mode='area', silent=True, silent_error=True, use_maskbit=True,
#                        is_overwrite_data=False, skip_search=False, save_error=True):
#
#     FG=dhfc.FileGroupSearch(DirInfo=DirInfo, region=region, skip_search=skip_search,
#                             keys_band=['psf_target_', 'err_'],
#                              use_maskbit=use_maskbit, bandlist=['g', 'r'], dir_group=dir_group)
#     FGr=copy.deepcopy(FG)
#     if(skip_search==False):
#         donelist=(FG.donelist[0][:,0] | FG.donelist[1][:,0]) & FG.donelist[2]
#         FG.get_todo(donelist)
#         donelist=(FG.donelist[0][:,1] | FG.donelist[1][:,1]) & FG.donelist[2]
#         FGr.get_todo(donelist)
#
#
#     global sub_crawling_g, sub_crawling_r
#     def sub_crawling_g(i):
#         coord=FG.coord_array_todo[i]
#         dir_work=FG.dir_work_list_todo[i]
#         try:
#             Crawl_VIR_g=dhcrawl.LegacyCrawling(dir_work=dir_work, band='g', region=region,
#                               coord=coord, psf_mode=psf_mode, silent=silent, silent_error=silent_error, save_error=save_error,
#                                                is_overwrite_data=is_overwrite_data)
#             Crawl_VIR_g.psf()
#             Crawl_VIR_g.psf(mode='target')
#             Crawl_VIR_g.sigma()
#             Crawl_VIR_g.maskbits()
#         except: pass
#         if(show_progress>0):
#             if(i%show_progress==0): print(i, "/", len(FG.coord_array_todo), "| Time : ", time.time()-start)
#
#     def sub_crawling_r(i):
#         coord=FGr.coord_array_todo[i]
#         dir_work=FGr.dir_work_list_todo[i]
#         try:
#             Crawl_VIR_r=dhcrawl.LegacyCrawling(dir_work=dir_work, band='r', region=region,
#                               coord=coord, psf_mode=psf_mode, silent=silent, silent_error=silent_error,save_error=save_error,
#                                                is_overwrite_data=is_overwrite_data)
#             Crawl_VIR_r.psf()
#             Crawl_VIR_r.psf(mode='target')
#             Crawl_VIR_r.sigma()
#             Crawl_VIR_r.maskbits()
#         except: pass
#
#         if(show_progress>0):
#             if(i%show_progress==0): print(i, "/", len(FGr.coord_array_todo), "| Time : ", time.time()-start)
#
#     start=time.time()
#     if(Ncore>1):
#         if(timeout==None): ## Normal multi-core
#             pool = Pool(processes=int(Ncore))
#             dummy=pool.map(sub_crawling_g, np.arange(len(FG.coord_array_todo)))
#             pool.close()
#             pool.join()
#             pool = Pool(processes=int(Ncore))
#             dummy=pool.map(sub_crawling_r, np.arange(len(FGr.coord_array_todo)))
#             pool.close()
#             pool.join()
#
#         else: ## Pebble multi-core
#     #         with ProcessPool(int(Ncore)) as pool:
#     #             future = pool.map(sub_crawling, np.arange(len(coord_array_todo)), timeout=timeout)
#             pool = ProcessPool(int(Ncore))
#             future = pool.map(sub_crawling_g, np.arange(len(FG.coord_array_todo)), timeout=timeout)
#             pool.close()
#             pool.join()
#             pool = ProcessPool(int(Ncore))
#             future = pool.map(sub_crawling_r, np.arange(len(FGr.coord_array_todo)), timeout=timeout)
#             pool.close()
#             pool.join()
#     else:
#         for i in range (len(FG.coord_array_todo)):
#             sub_crawling_g(i)
#             #sub_crawling_r(i)
#
#     print("Done! | Time : ", time.time()-start)

def multicore_masking_bit(dir_work_list, Ncore=2, masking_index_list=[2,3,4,5,6,7], silent=True, region='s',
                         fna_masking='ext_maskbits_0.fits', is_overwrite=False, show_progress=1000):
    global sub_masking_bit

    def sub_masking_bit(i):
        dir_work=dir_work_list[i]
        fn_save=dir_work+fna_masking
        fn_error1=dir_work+"err_"+region+"g.dat"
        fn_error2=dir_work+"err_"+region+"r.dat"
        if(os.path.exists(fn_error1)):
            if(silent==False): print(">> The target has an error file : ", fn_error1)
            return 0
        if(os.path.exists(fn_error2)):
            if(silent==False): print(">> The target has an error file : ", fn_error2)
            return 0
        if((is_overwrite==False) & (os.path.exists(fn_save)==True)):
            if(silent==False): print(">> The file already exists : ", fn_save)
            return 0

        fn_bitmask=dir_work_list[i]+'maskbits_'+region+'x.fits'
        dat=fits.getdata(fn_bitmask).astype('int64')
        udat=dharray.unpackbits(dat, 14)

        for mii, mi in enumerate(masking_index_list):
            if(mii==0): masking=np.copy(udat[:,:,mi])
            else: masking = masking | udat[:,:,mi]

        masking=masking.astype('int16')
        maskfile = fits.PrimaryHDU()
        maskfile.data = masking
        maskfile.writeto(fn_save, overwrite = True)

        if(show_progress>0):
            if(i%show_progress==0): print(i, "/", len(dir_work_list), "| Time : ", time.time()-start)
        return 0

    start=time.time()
    pool = Pool(processes=int(Ncore))
    dummy=pool.map(sub_masking_bit, np.arange(len(dir_work_list)))
    pool.close()
    pool.join()
    print("Done! | Time : ", time.time()-start)

def multicore_image_stat(DirInfo, band='g',
                         show_progress=1000, Ncore=2):
    global sub_image_stat
    def sub_image_stat(i):
        dir_work=DirInfo.dir_work_list[i]
        try:
            dat=fits.getdata(dir_work+'image_l'+band+".fits")
#             if(fn_masking!=None):
#                 masking=fits.getdata(dir_work+fn_masking)
#             else:
#                 masking=False
#             mdat=np.ma.masked_array(dat, masking)
#             mdat=mdat[100-centersize:100+centersize,100-centersize:100+centersize]
#             mdat=mdat-10

#             if(fn_masking!=None):  ## check frac_masking
#                 tot_masking=np.sum(mdat.mask.astype(bool))
#                 #if(tot_masking!=0): print(i)
#                 tot_area=np.shape(mdat)[0] * np.shape(mdat)[1]
#                 f_mask=tot_masking/tot_area
#                 if((f_mask)>frac_masking):
#                     return np.nan, np.nan, np.nan, f_mask

            if(show_progress>0):
                if(i%show_progress==0): print(i, "/", len(DirInfo.dir_work_list), "| Time : ", time.time()-start)
            return np.nanmin(dat), np.nanmax(dat), np.nanmedian(dat), np.nanstd(dat)
        except:
            print("Error! Files do not exist!", i)
            return np.nan, np.nan, np.nan, np.nan


    start=time.time()
    pool = Pool(processes=int(Ncore))
    #results=pool.map(sub_measure_centermag, np.arange(len(DirInfo.dir_work_list)))
    mmin, mmax, mmedian, mstd = zip(*pool.map(sub_image_stat, np.arange(len(DirInfo.dir_work_list))))
    print("Done! | Time : ", time.time()-start)
    pool.close()
    pool.join()
    return mmin, mmax, mmedian, mstd

def multicore_masking(DirInfo, Ncore=2, show_progress=1000, band='g',
                      centermasking_is_touch=False, centermasking_is_touch_nb=True,
                      silent=True, fna_prev_masking=None, is_overwrite=True, suffix='', debug=False):
    magz=22.5 #zero mag
    mu0=24 # g-band
    pixscl = 0.262
    global sub_masking
    def sub_masking(i):
        dir_work=DirInfo.dir_work_list[i]

        fn_image=dir_work+'image_l'+band+'.fits'
        fn_masking=dir_work+'masking_l'+band+suffix+'.fits'
        fn_masking_blur=dir_work+'masking_l'+band+suffix+'_bl.fits'

        fn_masking_nb=dir_work+'masking_l'+band+suffix+'_nb.fits' #nearby
        fn_masking_nbb=dir_work+'masking_l'+band+suffix+'_nbb.fits' #nearby + blur
        fn_masking_d=dir_work+'masking_l'+band+suffix+'_d.fits'   #double
        fn_masking_nd=dir_work+'masking_l'+band+suffix+'_nd.fits' #nearby - double

        fn_masking_object=dir_work+'masking_l'+band+suffix+'_obj'   #double
        fn_masking_nb_object=dir_work+'masking_l'+band+suffix+'_nb_obj'   #double
        fn_masking_d_object=dir_work+'masking_l'+band+suffix+'_d1_obj'   #double
        fn_masking_nd_object=dir_work+'masking_l'+band+suffix+'_nd1_obj'   #double
        fn_masking_d2_object=dir_work+'masking_l'+band+suffix+'_d2_obj'   #double
        fn_masking_nd2_object=dir_work+'masking_l'+band+suffix+'_nd2_obj'   #double
        if(show_progress>0):
            if(i%show_progress==0): print(i, "/", len(DirInfo.dir_work_list), "| Time : ", time.time()-start)

        ##========= Find cosmic ray & too bright stars ===========
#         dat=fits.getdata(fn_image)
#         if(np.nanmax(dat)>15):
#             print(">>", i, " has too bright sources!")

        ## Input previous masking (ext_bitmask)
        if(fna_prev_masking!=None):
            fn_prev_masking=dir_work+fna_prev_masking
            prev=fits.getdata(fn_prev_masking)
        else: prev=None
        if(is_overwrite==False):
            if(os.path.exists(fn_masking_nd)): return
        try:
            ## First masking
            Mask1=dhmask.Masking(fn_image=fn_image, fn_save_masking=fn_masking,
                                 centermasking_is_touch=centermasking_is_touch,
                                 superbright=True, superbright_thresh=0.5, superbright_area_frac=0.3,
                                 extreme_upper=5, extreme_lower=0.3, suppress_center=True, zero_area_crit=100,
                                 fn_save_obj=fn_masking_object,
                                 center_size=20, previous_mask=prev)

            suppress_center_mask=dhmask.make_aperture_mask(200, 50)
            blurmask=dhmask.masking_blur(Mask1.mask, kernel_value=0.5)
            blurmask=dhmask.mask_numbering(blurmask)
            blurmask_sup_center=dhmask.mask_numbering((blurmask*suppress_center_mask + Mask1.mask).astype('int32'))
            Mask1_blur=dhmask.Masking(fn_image=fn_image, fn_save_masking=fn_masking_blur,
                               centermasking_is_touch=False,
                               suppress_center=False, skip_main_mask=True,
                               previous_mask=blurmask_sup_center)

            ## Second masking
            thval=1.5* (10**((magz - mu0)/2.5) * (pixscl**2))
            Mask2=dhmask.Masking(fn_image=fn_image, fn_save_masking=fn_masking_d,
                                 centermasking_is_touch=centermasking_is_touch,
                                 suppress_center=False,
                                 fn_save_obj=fn_masking_d_object,
                                 thresh=thval, is_abs_cut=True, previous_mask=Mask1.mask)

            # Except for Full masked images
            if(Mask2.mask_area!=40000):

                newmask=dhmask.masking_sum(Mask2.mask, dhmask.make_aperture_mask(200, 50))
                Mask3_aper=dhmask.Masking(fn_image=fn_image, fn_save_masking=None,
                                 centermasking_is_touch=centermasking_is_touch,
                               suppress_center=False,
                               fn_save_obj=fn_masking_d2_object,
                            thresh=5, is_abs_cut=False, previous_mask=newmask)

                Mask3=dhmask.Masking(fn_image=fn_image, fn_save_masking=None,
                                 centermasking_is_touch=centermasking_is_touch,
                               suppress_center=False, skip_main_mask=True,
                            previous_mask=dhmask.masking_sum(Mask2.mask, Mask3_aper.main_mask))


                ## Fourth masking
                # Using sigma clipping


                sigdata = sigma_clip(Mask3.resultimage, sigma=8, maxiters=1, sigma_lower=100)
                #Newtrash : sigma clipping - sky values
                Mask4=dhmask.Masking(fn_image=fn_image, fn_save_masking=None,
                                     suppress_center=False,
                                     centermasking_is_touch=centermasking_is_touch,
                                     minarea=5, extreme_upper=sigdata.max()-np.nanmean(Mask3.resultimage),
                                     extreme_lower=-10,
                                     thresh=100, is_abs_cut=True,
                                     previous_mask=Mask3.mask, filter_kernel=None)

                blurmask=dhmask.masking_blur(Mask4.mask)
                blurmask=dhmask.mask_numbering(blurmask)
                Mask5=dhmask.Masking(fn_image=fn_image, fn_save_masking=fn_masking_d,
                               centermasking_is_touch=centermasking_is_touch,
                               suppress_center=False, skip_main_mask=True,
                            previous_mask=blurmask)

        ##====================== For NB ========================
            ## Nearby masking
            thval=0.5* (10**((magz - mu0)/2.5) * (pixscl**2))
            Mask_nb=dhmask.Masking(fn_image=fn_image, fn_save_masking=fn_masking_nb,
                           centermasking_is_touch=centermasking_is_touch_nb,
                           suppress_center=True, minarea=400,
                           fn_save_obj=fn_masking_nb_object,
                        thresh=thval, is_abs_cut=True, previous_mask=Mask1.mask)


            suppress_center_mask=dhmask.make_aperture_mask(200, 50)
            blurmask=dhmask.masking_blur(Mask1.mask, kernel_value=0.5)
            blurmask=dhmask.mask_numbering(blurmask)
            blurmask_sup_center=dhmask.mask_numbering((blurmask*suppress_center_mask + Mask1.mask).astype('int32'))
            Mask1_nbb=dhmask.Masking(fn_image=fn_image, fn_save_masking=fn_masking_nbb,
                               centermasking_is_touch=False,
                               suppress_center=False, skip_main_mask=True,
                               previous_mask=blurmask_sup_center)


            ## Second masking
            thval=1.5* (10**((magz - mu0)/2.5) * (pixscl**2))
            #thval=5* (10**((magz - mu0)/2.5) * (pixscl**2))
            Mask2_nb=dhmask.Masking(fn_image=fn_image, fn_save_masking=fn_masking_nd,
                                 suppress_center=False,
                               centermasking_is_touch=centermasking_is_touch_nb,
                               fn_save_obj=fn_masking_nd_object,
                                 thresh=thval, is_abs_cut=True, previous_mask=Mask_nb.mask)



            # Except for Full masked images
            if(Mask2_nb.mask_area!=40000):

                newmask=dhmask.masking_sum(Mask2_nb.mask, dhmask.make_aperture_mask(200, 50))
                Mask3_nb_aper=dhmask.Masking(fn_image=fn_image, fn_save_masking=None,
                               suppress_center=False,
                               centermasking_is_touch=centermasking_is_touch_nb,
                               fn_save_obj=fn_masking_nd2_object,
                            thresh=5, is_abs_cut=False, previous_mask=newmask)

                Mask3_nb=dhmask.Masking(fn_image=fn_image, fn_save_masking=None,
                               suppress_center=False, skip_main_mask=True,
                               centermasking_is_touch=centermasking_is_touch_nb,
                            previous_mask=dhmask.masking_sum(Mask2_nb.mask, Mask3_nb_aper.main_mask))

                ## Third masking
                # Using sigma clipping
                sigdata = sigma_clip(Mask3_nb.resultimage, sigma=8, maxiters=1, sigma_lower=100)
                #Newtrash : sigma clipping - sky values
                Mask4_nb=dhmask.Masking(fn_image=fn_image, fn_save_masking=None,
                                     suppress_center=False,
                               centermasking_is_touch=centermasking_is_touch_nb,
                                     minarea=5, extreme_upper=sigdata.max()-np.nanmean(Mask3_nb.resultimage),
                                     extreme_lower=-10,
                                     thresh=100, is_abs_cut=True,
                                     previous_mask=Mask3_nb.mask, filter_kernel=None)

                blurmask=dhmask.masking_blur(Mask4_nb.mask)
                blurmask=dhmask.mask_numbering(blurmask)
                Mask5_nb=dhmask.Masking(fn_image=fn_image, fn_save_masking=fn_masking_nd,
                               centermasking_is_touch=centermasking_is_touch_nb,
                               suppress_center=False, skip_main_mask=True,
                            previous_mask=blurmask)
                if(debug==True):
                    return Mask1, Mask2, Mask3, Mask4, Mask5, Mask_nb, Mask2_nb, Mask3_nb, Mask5_nb, Mask5_nb

        except:
            print("Error >> ", i)

    start=time.time()
    if(Ncore>1):
        pool = Pool(processes=int(Ncore))
        pool.map(sub_masking, np.arange(len(DirInfo.dir_work_list)))
        pool.close()
        pool.join()
        print("Done! | Time : ", time.time()-start)
    else:
        if(debug==True):
            return sub_masking(0)
        else:
            for i in range (len(DirInfo.dir_work_list)):
                sub_masking(i)
            print("Done! | Time : ", time.time()-start)



def multicore_run_galfit_with_runlist(Runlist, proj_folder, DirInfo,
                                      fna_image='image_lg.fits',
                                      fna_masking=None,
                                      fna_psf='psf_target_lg.fits',
                                      fna_sigma='sigma_lg.fits',
                                      output_block=True, use_try=True,
                                      Ncore=2, show_multicore=True,
                                      silent=True, show_progress=10, remove_galfit_interval=300,
                                      add_params_array=None,
                                      Ncomp_add_params_array=0,
                                      add_params_mode='add',
                                      is_nth_mag_offset=False, print_galfit=False,
                                      suffix='', overwrite=False, fast_skip=True,
                                      is_allow_vary=True,
                                      fix_psf_mag=False,
                                      est_sky=10,
                                      skiplist=None,
                                      re_cut_list=None,
                                      n_ratio=None,
                                      debug=False,
                                      plate_scale=0.262,
                                      zeromag=22.5,
                                      is_run_galfit_dir_work=False,
                                      ):
    if(hasattr(add_params_array, "__len__")==False): add_params_array=np.full(len(DirInfo.dir_work_list), None)
    if(hasattr(skiplist, "__len__")==False): skiplist=np.full(len(DirInfo.dir_work_list), False)
    if(hasattr(re_cut_list, "__len__")==False): re_cut_list=np.full(len(DirInfo.dir_work_list), None)
    if(len(add_params_array)!=len(DirInfo.dir_work_list)):
        print("Warning! Check add_params_array!")
        print("Stop the function")
        return

    global sub_run_galfit

    def sub_run_galfit(i):
        dir_work=DirInfo.dir_work_list[i]
        fn_done=dir_work+proj_folder+"done_galfit.dat"
        if(fast_skip==True):
            if(os.path.exists(fn_done)): return 0
        if(skiplist[i]==True): return 0

        if(i%remove_galfit_interval==0):
            existlist1=glob.glob('./galfit.*')
            for fb in existlist1:
                try: os.remove(fb)
                except: pass
            try: os.remove('./fit.log')
            except: pass

        Runlist.run_galfit_with_runlist(dir_work=dir_work, proj_folder=proj_folder,
                                fna_image=fna_image, fna_masking=fna_masking,
                                fna_psf=fna_psf, fna_sigma=fna_sigma,
                                add_params_array=add_params_array[i],
                                Ncomp_add_params_array=Ncomp_add_params_array,
                                add_params_mode=add_params_mode,
                                is_nth_mag_offset=is_nth_mag_offset,
                                debug=debug,
                                is_allow_vary=is_allow_vary,
                                fix_psf_mag=fix_psf_mag,
                                overwrite=overwrite, suffix=suffix, silent=silent,
                                print_galfit=print_galfit,
                                output_block=output_block,
                                re_cut=re_cut_list[i],
                                n_ratio=n_ratio,
                                est_sky=est_sky,
                                plate_scale=plate_scale,
                                zeromag=zeromag,
                                is_run_galfit_dir_work=is_run_galfit_dir_work,
                                )
        np.savetxt(fn_done, np.array(['done']), fmt="%s")
        return 0

    mulcore.multicore_run(sub_run_galfit, len(DirInfo.dir_work_list), Ncore=Ncore,
                          use_try=use_try, show_progress=show_progress, debug=show_multicore)

def multicore_galfit_todo(Runlist, proj_folder, DirInfo,
                          Ncore=10, show_progress=1000, use_try=False, show_multicore=True):
    global sub_galfit_todo
    print("Start Multicore run")

    def sub_galfit_todo(i):
        dir_work=DirInfo.dir_work_list[i]
        fn_done=dir_work+proj_folder+"done_galfit.dat"
        if(os.path.exists(fn_done)==True):
            return 0
        else: return 1

    return mulcore.multicore_run(sub_galfit_todo, len(DirInfo.dir_work_list), Ncore=Ncore,
                          use_try=use_try, show_progress=show_progress, debug=show_multicore)


def multicore_galfit_check(Runlist, proj_folder, DirInfo, names="output_*.fits",
                          Ncore=10, show_progress=1000, use_try=False, show_multicore=True):
    global sub_galfit_check
    print("Start Multicore run")

    def sub_galfit_check(i):
        dir_work=DirInfo.dir_work_list[i]
        existlist1=glob.glob(dir_work+proj_folder+names)

        for fb in existlist1:
            try:
                dat=fits.getdata(fb, ext=2)
                dat=fits.getdata(fb, ext=3)
            except:
                print(">>Error!", fb)
                return 1
        return 0

    return mulcore.multicore_run(sub_galfit_check, len(DirInfo.dir_work_list), Ncore=Ncore,
                          use_try=use_try, show_progress=show_progress, debug=show_multicore)


def multicore_extract_fits(dir_work_list,
                           fna='', prev_pattern='', new_pattern='', ext=2,
                           Ncore=10, show_progress=1000, use_try=False, show_multicore=True):
    global sub_extract_fits

    def sub_extract_fits(i):
        dir_work=dir_work_list[i]
        existlist1=glob.glob(dir_work_list[i]+fna)
        ext_int=int(ext)
        if(len(existlist1)!=0):
            newfnlist=np.core.defchararray.replace(existlist1, prev_pattern, new_pattern)
            for j, fn in enumerate(existlist1):
                newfn=newfnlist[j]
                fits.writeto(newfn, fits.getdata(fn, ext_int), fits.getheader(fn, ext_int),
                overwrite=True, output_verify='silentfix')
            return 0
        else:
            #print("No data at ", i)
            return 1

    return mulcore.multicore_run(sub_extract_fits, len(dir_work_list), Ncore=Ncore,
                          use_try=use_try, show_progress=show_progress, debug=show_multicore)



def multicore_postprocessing(Runlist, proj_folder, DirInfo,
                             fna_sigma='sigma_lg.fits', fna_masking='masking_double2_g.fits',
                             fna_image='image_lg.fits', fna_psf='psf_target_lg.fits',
                             chi_radius=50, repair_fits=True, image_size=200,
                             chi_item_name='50',
                             centermag='sersic',
                             plate_scale=0.262, zeromag=22.5,
                             fn_base_noext='', chi2_full_image=True,
                             sersic_res=500, remove_galfit_interval=300,
                             overwrite=False, silent=True, print_galfit=False, fast_skip=True,
                             Ncore=10, show_progress=1000, use_try=False, show_multicore=True
                             ):
    global sub_postprocessing
    if(hasattr(chi_radius, "__len__")==False): ## Const
        chi_radius=np.repeat(chi_radius, len(DirInfo.dir_work_list)) ## 1D array
    if(hasattr(chi_radius, "__len__")):
        if(np.ndim(chi_radius)==1): ## 1D array
            chi_radius=np.array([chi_radius]).T ## 2D array (N*1)

    def sub_postprocessing(i):
        dir_work=DirInfo.dir_work_list[i]
        fn_done=dir_work+proj_folder+"done_post.dat"
        if(fast_skip==True):
            if(os.path.exists(fn_done)): return
        if(i%remove_galfit_interval==0):
            existlist1=glob.glob('./galfit.*')
            for fb in existlist1:
                try: os.remove(fb)
                except: pass

        dum=dhgalfit.PostProcessing(proj_folder=proj_folder, Runlist=Runlist, dir_work=dir_work,
                                fna_sigma=fna_sigma, fna_masking=fna_masking,
                                fna_image=fna_image, fna_psf=fna_psf,
                                chi_radius=chi_radius[i], repair_fits=repair_fits, image_size=image_size,
                                chi_item_name=chi_item_name,
                                fn_base_noext=fn_base_noext,
                                plate_scale=plate_scale, zeromag=zeromag,
                                chi2_full_image=chi2_full_image,
                                centermag=centermag, sersic_res=500,
                                overwrite=overwrite, silent=silent, print_galfit=print_galfit)
        np.savetxt(fn_done, np.array(['done']), fmt="%s")
        return 0

    return mulcore.multicore_run(sub_postprocessing, len(DirInfo.dir_work_list), Ncore=Ncore,
                          use_try=use_try, show_progress=show_progress, debug=show_multicore)



def multicore_check_postprocessing(Runlist, proj_folder, DirInfo, comp_crit='AIC_50',
                                   Ncore=10, show_progress=1000, use_try=False, show_multicore=True, reff_snr_crit=None,
                                   ):
    global sub_check_postprocessing
    print("Start Multicore run")

    def sub_check_postprocessing(i):
        dir_work=DirInfo.dir_work_list[i]
        #existlist1=glob.glob(dir_work+proj_folder+"result_*.fits")
        try:
            Res=dhgalfit.ResultGalfit(proj_folder=proj_folder, runlist=Runlist.runlist,
                                      dir_work=dir_work, reff_snr_crit=reff_snr_crit,
                                      group_id=None, comp_crit=comp_crit,
                                      silent=True, ignore_no_files=False)
            ResBest=dhgalfit.ResultGalfitBest(Res, include_warn=False, comp_crit=comp_crit)
            return 0

        except:
            print(">>Error!", dir_work)
            return 1


    return mulcore.multicore_run(sub_check_postprocessing, len(DirInfo.dir_work_list), Ncore=Ncore,
                          use_try=use_try, show_progress=show_progress, debug=show_multicore)


class ResPack:
    def __init__(self, Runlist, proj_folder, DirInfo, comp_autoswap=True, swapmode='N',
                  comp_4th_nsc_autoswap=True, comp_4th_ss_autoswap=False,
                  chi2_itemlist= ['CHI2NU', 'AIC_F', 'RChi_50', 'AIC_50'],
                  comp_crit='RChi_50', show_progress=100,
                  reff_snr_crit=2, reff_snr_n_cut=2, re_cut_list=None,
                  Ncore=1, use_try=True, show_multicore=True):

        self.comp_crit=comp_crit
        self.chi2_itemlist=chi2_itemlist
        self.comp_autoswap=comp_autoswap
        self.swapmode=swapmode
        self.comp_4th_nsc_autoswap=comp_4th_nsc_autoswap
        self.comp_4th_ss_autoswap=comp_4th_ss_autoswap
        self.reff_snr_crit=reff_snr_crit
        self.reff_snr_n_cut=reff_snr_n_cut
        self.Runlist=Runlist
        if(hasattr(re_cut_list, "__len__")==False): self.re_cut_list=np.full(len(DirInfo.dir_work_list), np.nan)
        else: self.re_cut_list=np.copy(re_cut_list)

        self._getdata(proj_folder, Runlist.runlist, DirInfo.dir_work_list,
                      show_progress=show_progress, Ncore=Ncore,
                      use_try=use_try, show_multicore=show_multicore)

    def _getdata(self, proj_folder, runlist, dir_work_list,
                 show_progress=1000, Ncore=1,
                 use_try=True, show_multicore=True):
        global sub_getdata
        def sub_getdata(i):
            Res=dhgalfit.ResultGalfit(proj_folder=proj_folder, runlist=runlist,
                                      dir_work=dir_work_list[i],
                                      reff_snr_crit=self.reff_snr_crit,
                                      reff_snr_n_cut=self.reff_snr_n_cut,
                                      group_id=None, comp_crit=self.comp_crit,
                                      chi2_itemlist=self.chi2_itemlist,
                                      silent=True, ignore_no_files=True,
                                     comp_autoswap=self.comp_autoswap,
                                     swapmode=self.swapmode,
                                     comp_4th_nsc_autoswap=self.comp_4th_nsc_autoswap,
                                     comp_4th_ss_autoswap=self.comp_4th_ss_autoswap,
                                     re_cut=self.re_cut_list[i]

                                     )
            fulldat=Res.save_data(fna_output=None, return_array=True)
            Nfile=np.sum(Res.Data.is_file_exist)
            if(Nfile==0):
                bob_index=0
                bobwarn_index=0
            else:
                bob_index=Res.best[0]
                bobwarn_index=Res.best_warn[0]

            ResBest=dhgalfit.ResultGalfitBest(Res, include_warn=False, get_single=True, comp_crit=self.comp_crit)
            best_index=ResBest.bestsubmodels_flat

            ResBestWarn=dhgalfit.ResultGalfitBest(Res, include_warn=True, get_single=True, comp_crit=self.comp_crit)
            bestwarn_index=ResBestWarn.bestsubmodels_warn_flat

            return fulldat, int(bob_index), int(bobwarn_index), best_index, bestwarn_index, int(Nfile)


        fulldat, bob_index, bobwarn_index, best_index, bestwarn_index, Nfile = mulcore.multicore_run(
                              sub_getdata, len(dir_work_list), Ncore=Ncore,
                              use_try=use_try, use_zip=True, show_progress=show_progress, debug=show_multicore,
                              errorcode=[0,0,0,0,0,0])

        self.fulldat=fulldat #Full data
        self.groupid_list=np.unique(self.fulldat[0]['group_ID']).astype(int)
        self.bob_index=np.array(bob_index) #Best of Best - index
        self.bobwarn_index=np.array(bobwarn_index) #Best of Best - index
        self.best_index=np.copy(best_index) #Best submodels  - index
        self.bestwarn_index=np.copy(bestwarn_index) #Best submodels  - index

        self.Nfile=np.array(Nfile) #Number of available files


    def generate_onlyval(self, dataarray, index=None):
        onlyval=np.full(len(dataarray), -1, dtype=dataarray[0].dtype)
        onlyerr=np.full(len(dataarray), -1, dtype=dataarray[0].dtype)

        for i in range (len(dataarray)):
            try:
                thisarray=dataarray[i]
                vallist=np.where(thisarray['datatype']==1)[0]
                errlist=np.where(thisarray['datatype']==2)[0]

                if(hasattr(index, "__len__")):
                    if(np.isnan(index[i])): pass
                    else:
                        onlyval[i]=dataarray[i][vallist][int(index[i])]
                        onlyerr[i]=dataarray[i][errlist][int(index[i])]
                else:
                    onlyval[i]=dataarray[i][vallist]
                    onlyerr[i]=dataarray[i][errlist]
            except:
                print("Error ", i)
        return onlyval, onlyerr
        ## fulldat, bestdat, bob_index, bob_onlyval, bob_onlyerr, Nfile

    def preset_generate_onlyval(self, submodel=-1, silent=False, is_mag=False):
        if(submodel==-1): ##best of best
            if(silent==False): print(">> Get all submodels")
            self.bob_onlyval, self.bob_onlyerr=self.generate_onlyval(self.fulldat, index=self.bob_index)
            self.bobwarn_onlyval, self.bobwarn_onlyerr=self.generate_onlyval(self.fulldat, index=self.bobwarn_index)
        else:
            if(silent==False): print(">> Get submodels : ", self.groupid_list[submodel])
            temp1, temp2=self.generate_onlyval(self.fulldat, index=self.best_index[:,submodel])
            setattr(self, 'sub'+str(submodel)+"_onlyval", temp1)
            setattr(self, 'sub'+str(submodel)+"_onlyerr", temp2)
            temp3, temp4=self.generate_onlyval(self.fulldat, index=self.bestwarn_index[:,submodel])
            setattr(self, 'sub'+str(submodel)+"warn_onlyval", temp3)
            setattr(self, 'sub'+str(submodel)+"warn_onlyerr", temp4)
            self.thismodel_onlyval=temp1
            self.thismodel_onlyerr=temp2
            self.thismodelwarn_onlyval=temp3
            self.thismodelwarn_onlyerr=temp4
            #self.thismodel_onlyval, self.thismodel_onlyerr=self.generate_onlyval(self.fulldat, index=self.best_index[:,submodel])
            #self.thismodelwarn_onlyval, self.thismodelwarn_onlyerr=self.generate_onlyval(self.fulldat, index=self.bestwarn_index[:,submodel])
        self._get_add_params_array(submodel, is_mag=is_mag)

    def _get_add_params_array(self, submodel=-1, is_mag=False):
        if(submodel==-1): ##best of best
            self.is_sersic, self.add_params1_array, self.add_params2_array, self.add_params3_array\
            = dhgalfit.convert_add_params_array(self.bob_onlyval, is_mag=is_mag)
            self.add_params_array=np.stack((self.add_params1_array, self.add_params2_array, self.add_params3_array), axis=1)

            self.is_sersic_warn, self.addwarn_params1_array, self.addwarn_params2_array, self.addwarn_params3_array\
            = dhgalfit.convert_add_params_array(self.bobwarn_onlyval, is_mag=is_mag)
            self.addwarn_params_array=np.stack((self.addwarn_params1_array, self.addwarn_params2_array, self.addwarn_params3_array), axis=1)

        else:
            self.thismodel_is_sersic, self.thismodel_add_params1_array, self.thismodel_add_params2_array, self.thismodel_add_params3_array\
            = dhgalfit.convert_add_params_array(self.thismodel_onlyval, is_mag=is_mag)
            self.thismodel_add_params_array=np.stack((self.thismodel_add_params1_array, self.thismodel_add_params2_array, self.thismodel_add_params3_array), axis=1)

            self.thismodelwarn_is_sersic, self.thismodelwarn_add_params1_array, self.thismodelwarn_add_params2_array, self.thismodelwarn_add_params3_array\
            = dhgalfit.convert_add_params_array(self.thismodelwarn_onlyval, is_mag=is_mag)
            self.thismodelwarn_add_params_array=np.stack((self.thismodelwarn_add_params1_array, self.thismodelwarn_add_params2_array, self.thismodelwarn_add_params3_array), axis=1)


class ClassResult():
    def __init__(self, ResPack, use_warn=False, chi2_name='hRe',
                 aic_crit=28.7, mag_crit=24, likelihood_crit=0.1, dist_crit=10):
        """
        INPUT
         - ResPack : Result package
         * use_warn : If true, select the best result regardless of the warning (default: False)
         * chi2_name : The region for chi2 values (default: 'hRe')
           - hRe: half Reff (based on the best simple sersic results)
           - hRt: half Reff (based on the best multi-component results)
           - Re: Reff
           - 50: 50-pixel-radius circle
         * aic_crit : AIC difference criterion. (default: 28.7)
           - if you want to use 3 sigma, use 6.18
         * mag_crit : [mag] darker NSCs will be classified as a new group. (default: 24)
         * likelihood_crit : Lower likelihood results will be classified as a new group (default: 0.1)
         * dist_crit : [pixel] Distance criterion.
                       If the distance between the NSC and galaxy center is larger than this value,
                       That image will be classified as a new group (default: 10)

        """

        if(use_warn==False): self.bob_onlyval=ResPack.bob_onlyval
        else: self.bob_onlyval=ResPack.bobwarn_onlyval
        self.classification=np.copy(self.bob_onlyval['group_ID'])
        self.aic_best=np.copy(self.bob_onlyval['AIC_'+chi2_name])

        ResPack.preset_generate_onlyval(submodel=0)
        if(use_warn==False): thismodel_onlyval=ResPack.thismodel_onlyval
        else: thismodel_onlyval=ResPack.thismodelwarn_onlyval
        self.aic_s=np.copy(thismodel_onlyval['AIC_'+chi2_name])

        ## AIC_best does not have 5 sig confidence -> simple sersic
        target=np.where(self.aic_s-self.aic_best<aic_crit)[0]
        self.classification[target]=0

        ## 2-comp. composite
        ResPack.preset_generate_onlyval(submodel=1)
        if(use_warn==False): thismodel_onlyval=ResPack.thismodel_onlyval
        else: thismodel_onlyval=ResPack.thismodelwarn_onlyval
        self.aic_ss=np.copy(thismodel_onlyval['AIC_'+chi2_name])

        target=np.where((self.classification==2) & (self.aic_ss-self.aic_best<aic_crit))[0]
        self.classification[target]=4

        ## dim psf
        ResPack.preset_generate_onlyval(submodel=2)
        if(use_warn==False): thismodel_onlyval=ResPack.thismodel_onlyval
        else: thismodel_onlyval=ResPack.thismodelwarn_onlyval
        self.aic_spsf=np.copy(thismodel_onlyval['AIC_'+chi2_name])

        target=np.where((self.classification==2) & (thismodel_onlyval['3_MAG']>mag_crit))[0]
        self.classification[target]=5

        ## Far NSC
        dist=np.array([thismodel_onlyval['2_XC']-thismodel_onlyval['3_XC'],
                       thismodel_onlyval['2_YC']-thismodel_onlyval['3_YC']])
        dist=np.sum(dist**2, axis=0)**0.5
        target=np.where((self.classification==2) & (dist>dist_crit))[0]
        self.classification[target]=6

        ## Bad likelihood -> New def
        self.likelihood=self.bob_onlyval['LIK_'+chi2_name]
        target=np.where(self.bob_onlyval['LIK_'+chi2_name] <likelihood_crit)[0]
        self.classification[target]=7

        dmu0_main=Sersic1D(amplitude=1, r_eff=1, n=self.bob_onlyval['2_N'])
        dmu0_main=-2.5*np.log10(dmu0_main(0))
        self.mu_0=self.bob_onlyval['2_MU_E']+dmu0_main
