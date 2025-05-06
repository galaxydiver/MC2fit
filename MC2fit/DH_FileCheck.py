import DH_array as dharray
import DH_path as dhpath
import DH_masking as dhmask
import numpy as np
import random
import pickle
import copy
import time
import shutil
from astropy.io import fits
from dataclasses import dataclass
import os, glob
from multiprocessing import Pool
from astropy.io import fits


def multicore_checkpixel(DirInfo, region='s', band='g', Ncore=2,
                         fna_image='image_', silent=False,
                                show_progress=1000, zero_value=10):
    global sub_checkpixel
    def sub_checkpixel(i):
        dir_work=DirInfo.dir_work_list[i]
        fn_error=dir_work+'err_'+region+band+".dat"
        if(os.path.exists(fn_error)):
            return np.nan, np.nan, np.nan, np.nan
        try:
            dat=fits.getdata(dir_work+fna_image+region+band+".fits")
            total=len(np.where(np.isfinite(dat))[0])
            empty=len(np.where(dat==zero_value)[0])
            minus=len(np.where(dat<=zero_value)[0])
            median=np.nanmedian(dat)
            if(show_progress>0):
                if(i%show_progress==0): print(i, "/", len(DirInfo.dir_work_list), "| Time : ", time.time()-start)
            return total, empty, minus, median
        except:
            if(silent==False): print("Error! Files do not exist!", i)
            return np.nan, np.nan, np.nan, np.nan


    start=time.time()
    pool = Pool(processes=int(Ncore))
    #results=pool.map(sub_measure_centermag, np.arange(len(DirInfo.dir_work_list)))
    total, empty, minus, median = zip(*pool.map(sub_checkpixel, np.arange(len(DirInfo.dir_work_list))))
    print("Done! | Time : ", time.time()-start)
    pool.close()
    pool.join()
    return total, empty, minus, median

def generate_checkpixel(DirInfo, regionlist=['s', 'n'], bandlist=['g', 'r'], suffix='',
                        fna_image='image_', silent=False,
                              Ncore=2, show_progress=1000, zero_value=10):
    for region in regionlist:
        for band in bandlist:
            print(region, band)
            dat=multicore_checkpixel(DirInfo, region=region, band=band,         fna_image=fna_image,
                                      show_progress=show_progress, Ncore=Ncore, zero_value=zero_value,
                                      silent=silent)
            pack_dat=np.vstack((dat[0], dat[1], dat[2], dat[3])).T ## total, empty, minus, median
            np.savetxt(DirInfo.dir_base+'checkpixel_'+region+band+suffix+'.dat', pack_dat)

class CheckPixel:
    def __init__(self, DirInfo, regionlist=['s', 'n'], bandlist=['g', 'r'], bad_pixel_crit=2000, suffix=''):
        self.DirInfo=DirInfo
        self.regionlist=regionlist
        self.bandlist=bandlist
        self.suffix=suffix
        self.load_data()
        self.extract_data()
        self.briefing()
        self.select_survey(bad_pixel_crit)

    def load_data(self):
        count=0
        for i, region in enumerate(self.regionlist):
            for j, band in enumerate(self.bandlist):
                count+=1
                dat=np.loadtxt(self.DirInfo.dir_base+'checkpixel_'+region+band+self.suffix+'.dat')
                if(count==1):
                    self.pixel_total=np.zeros((len(dat), len(self.regionlist), len(self.bandlist)))
                    self.pixel_empty=np.zeros((len(dat), len(self.regionlist), len(self.bandlist)))
                    self.pixel_under=np.zeros((len(dat), len(self.regionlist), len(self.bandlist)))
                    self.pixel_median=np.zeros((len(dat), len(self.regionlist), len(self.bandlist)))
                self.pixel_total[:,i,j]=dat[:,0]
                self.pixel_empty[:,i,j]=dat[:,1]
                self.pixel_under[:,i,j]=dat[:,2]
                self.pixel_median[:,i,j]=dat[:,3]

    def extract_data(self):
        self.exist = np.isfinite(self.pixel_empty)
        self.exist_region = np.all(self.exist, axis=2)
        self.pixel_empty_region=np.max(self.pixel_empty, axis=2)

    def briefing(self):
        for i, region in enumerate(self.regionlist):
            for j, band in enumerate(self.bandlist):
                print("Region -", region, "Band -", band, " exist :", np.sum(self.exist[:,i,j]))
            print("Region -", region, "Total exist :", np.sum(self.exist_region[:,i]))

    def select_survey(self, bad_pixel_crit=4000): ## Only for 's', 'n' / 'g', 'r'
        self.list_s=(self.pixel_empty_region[:,0]<bad_pixel_crit)
        self.list_n=(self.pixel_empty_region[:,1]<bad_pixel_crit) & np.invert(self.pixel_empty_region[:,0]<bad_pixel_crit)
        self.list_bad=np.invert(self.pixel_empty_region[:,0]<bad_pixel_crit) & np.invert(self.pixel_empty_region[:,1]<bad_pixel_crit)
        print("List South :", np.sum(self.list_s))
        print("List North :", np.sum(self.list_n))
        print("List Bad :", np.sum(self.list_bad))




def multicore_checkmasking(DirInfo, band='g', Ncore=2, suffix='', center_crop=None,
                           is_center_circle=True,
                           show_progress=1000, zero_value=10, is_nearby=False):
    global sub_checkmasking
    def sub_checkmasking(i):
        dir_work=DirInfo.dir_work_list[i]

        fn_masking=dir_work+'masking_l'+band+suffix+'.fits'
        fn_masking_d=dir_work+'masking_l'+band+suffix+'_d.fits'   #double
        fn_masking_nb=dir_work+'masking_l'+band+suffix+'_nb.fits' #nearby
        fn_masking_nd=dir_work+'masking_l'+band+suffix+'_nd.fits' #nearby - double

        if(is_nearby==False): m3area, m4area = np.nan, np.nan

        try:
            mask=fits.getdata(fn_masking)
            if(center_crop!=None):
                image_size=len(mask)
                halfsize=int(image_size/2)
                if(is_center_circle):
                    filter_kernel=dhmask.make_aperture_mask(image_size, center_crop)
                else:
                    crop_lower=halfsize-int(center_crop)
                    crop_upper=halfsize+int(center_crop)
                    filter_kernel=np.ones(np.shape(mask), dtype=int)
                    filter_kernel[crop_lower:crop_upper,crop_lower:crop_upper]=0
                mask[filter_kernel==1]=0
            m1area=np.sum(mask.astype(bool))

            if(os.path.exists(fn_masking_d)):
                mask=fits.getdata(fn_masking_d)
                if(center_crop!=None): mask[filter_kernel==1]=0
                m2area=np.sum(mask.astype(bool))
            else: m2area=np.nan
            ## ==================== Nearby ================
            if(is_nearby==True):
                mask=fits.getdata(fn_masking_nb)
                if(center_crop!=None): mask[filter_kernel==1]=0
                m3area=np.sum(mask.astype(bool))

                if(os.path.exists(fn_masking_nd)):
                    mask=fits.getdata(fn_masking_nd)
                    if(center_crop!=None): mask[filter_kernel==1]=0
                    m4area=np.sum(mask.astype(bool))
                else: m4area=np.nan

        except:
            print(">>ERROR! :", i)

        if(show_progress>0):
            if(i%show_progress==0): print(i, "/", len(DirInfo.dir_work_list), "| Time : ", time.time()-start)
        return m1area, m2area, m3area, m4area

    start=time.time()
    pool = Pool(processes=int(Ncore))
    #results=pool.map(sub_measure_centermag, np.arange(len(DirInfo.dir_work_list)))
    m1area, m2area, m3area, m4area= zip(*pool.map(sub_checkmasking, np.arange(len(DirInfo.dir_work_list))))
    print("Done! | Time : ", time.time()-start)
    pool.close()
    pool.join()
    pack_dat=np.vstack((m1area, m2area, m3area, m4area)).T
    return pack_dat

def force_symlink(file1, file2):
    if(os.path.islink(file2)):
        os.remove(file2)
    os.symlink(file1, file2)


def multicore_rename_folder(dir_work_list, dir_work_list2, Ncore=2, fna='', show_progress=1000,
                            ignore_errors=True):
    global sub_rename_folder

    def sub_rename_folder(i):
        dir_work_pre=dir_work_list[i]
        dir_work_new=dir_work_list2[i]
        shutil.move(dir_work_pre, dir_work_new)
        if(show_progress>0):
            if(i%show_progress==0): print(i, "/", len(dir_work_list), "| Time : ", time.time()-start)

    start=time.time()
    pool = Pool(processes=int(Ncore))
    pool.map(sub_rename_folder, np.arange(len(dir_work_list)))
    pool.close()
    pool.join()
    print("Done! | Time : ", time.time()-start)



def multicore_rename(dir_work_list, Ncore=2, fna='', silent=False,
                     prev_pattern='', new_pattern='', show_progress=1000):
    global sub_rename

    def sub_rename(i):
        dir_work=dir_work_list[i]
        existlist1=glob.glob(dir_work_list[i]+fna)
        if(len(existlist1)!=0):
            newfnlist=np.core.defchararray.replace(existlist1, prev_pattern, new_pattern)
            for j, fb in enumerate(existlist1):
                os.replace(fb, newfnlist[j])
        else:
            if(silent==False): print("No data : ", i)
        if(show_progress>0):
            if(i%show_progress==0): print(i, "/", len(dir_work_list), "| Time : ", time.time()-start)

    start=time.time()
    pool = Pool(processes=int(Ncore))
    pool.map(sub_rename, np.arange(len(dir_work_list)))
    pool.close()
    pool.join()
    print("Done! | Time : ", time.time()-start)

def multicore_checkname(dir_work_list, Ncore=2, fna='', show_progress=1000):
    global sub_checkname

    def sub_checkname(i):
        dir_work=dir_work_list[i]
        existlist1=glob.glob(dir_work_list[i]+fna)
        if(len(existlist1)!=0):
            print("File exist", i)
        if(show_progress>0):
            if(i%show_progress==0): print(i, "/", len(dir_work_list), "| Time : ", time.time()-start)

    start=time.time()
    pool = Pool(processes=int(Ncore))
    pool.map(sub_checkname, np.arange(len(dir_work_list)))
    pool.close()
    pool.join()
    print("Done! | Time : ", time.time()-start)




def multicore_compare_name(dir_work_list, Ncore=2, fna1='', fna2='', show_progress=1000):
    global sub_compare_name

    def sub_compare_name(i):
        dir_work=dir_work_list[i]
        existlist1=glob.glob(dir_work_list[i]+fna1)
        existlist2=glob.glob(dir_work_list[i]+fna2)
        if(len(existlist1)!=len(existlist2)):
            print(">> Problems", i)
        if(show_progress>0):
            if(i%show_progress==0): print(i, "/", len(dir_work_list), "| Time : ", time.time()-start)

    start=time.time()
    pool = Pool(processes=int(Ncore))
    pool.map(sub_compare_name, np.arange(len(dir_work_list)))
    pool.close()
    pool.join()
    print("Done! | Time : ", time.time()-start)


def multicore_remove(dir_work_list, Ncore=2, fna='', show_progress=1000):
    global sub_remove

    def sub_remove(i):
        dir_work=dir_work_list[i]
        existlist1=glob.glob(dir_work_list[i]+fna)
        for fb in existlist1:
            os.remove(fb)
        if(show_progress>0):
            if(i%show_progress==0): print(i, "/", len(dir_work_list), "| Time : ", time.time()-start)

    start=time.time()
    pool = Pool(processes=int(Ncore))
    pool.map(sub_remove, np.arange(len(dir_work_list)))
    pool.close()
    pool.join()
    print("Done! | Time : ", time.time()-start)

def multicore_remove_folder(dir_work_list, Ncore=2, fna='', show_progress=1000,
                            ignore_errors=True):
    global sub_remove_folder

    def sub_remove_folder(i):
        dir_work=dir_work_list[i]
        shutil.rmtree(dir_work+fna, ignore_errors=ignore_errors)
        if(show_progress>0):
            if(i%show_progress==0): print(i, "/", len(dir_work_list), "| Time : ", time.time()-start)

    start=time.time()
    pool = Pool(processes=int(Ncore))
    pool.map(sub_remove_folder, np.arange(len(dir_work_list)))
    pool.close()
    pool.join()
    print("Done! | Time : ", time.time()-start)

def multicore_copy_folder(dir_work_list, Ncore=2, fna_prev='', fna_new='', show_progress=1000,
                            dirs_exist_ok=True):
    global sub_copy_folder

    def sub_copy_folder(i):
        dir_work=dir_work_list[i]
        try: shutil.copytree(dir_work+fna_prev, dir_work+fna_new,
                        dirs_exist_ok=dirs_exist_ok)
        except: pass
        if(show_progress>0):
            if(i%show_progress==0): print(i, "/", len(dir_work_list), "| Time : ", time.time()-start)

    start=time.time()
    pool = Pool(processes=int(Ncore))
    pool.map(sub_copy_folder, np.arange(len(dir_work_list)))
    pool.close()
    pool.join()
    print("Done! | Time : ", time.time()-start)


def multicore_generate_link(DirInfo, DirInfo2, CheckPixel, show_progress=10000, Ncore=2,
                            bandlist=['g', 'r'],
                            dr_north='dr9', dr_south='dr10',
                            is_psf_others=False,
                            is_maskbits=False):
    global sub_generate_link
    def sub_generate_link(i):
        dir_work_src=os.path.abspath(DirInfo.dir_work_list[i])+"/"
        dir_work_dst=os.path.abspath(DirInfo2.dir_work_list[i])+"/"
        if(CheckPixel==None):
            dr=''
            region=''
        else:
            if(CheckPixel.list_bad[i]==True): return
            elif(CheckPixel.list_s[i]==True):
                region='s'
                dr=dr_south
            elif(CheckPixel.list_n[i]==True):
                region='n'
                dr=dr_north
            else: print("Error! ", i)

        for band in ['g', 'r']:
            src=dir_work_src+"image_"+dr+"_"+region+band+".fits"
            dst=dir_work_dst+"image_l"+band+".fits" #link
            force_symlink(src, dst)

            src=dir_work_src+"sigma_"+dr+"_"+region+band+".fits"
            dst=dir_work_dst+"sigma_l"+band+".fits"
            force_symlink(src, dst)

            src=dir_work_src+"psf_model_"+dr+"_"+region+band+".fits"
            dst=dir_work_dst+"psf_model_l"+band+".fits"
            force_symlink(src, dst)

            if(is_psf_others):
                src=dir_work_src+"psf_area_"+dr+"_"+region+band+".fits"
                dst=dir_work_dst+"psf_area_l"+band+".fits"
                force_symlink(src, dst)

                src=dir_work_src+"psf_target_"+dr+"_"+region+band+".fits"
                dst=dir_work_dst+"psf_target_l"+band+".fits"
                force_symlink(src, dst)

        if(is_maskbits):
            for mask in ['0', '1', '2']:
                src=dir_work_src+"ext_maskbits_"+region+mask+".fits"
                dst=dir_work_dst+"ext_maskbits_l"+mask+".fits"
                force_symlink(src, dst)

            src=dir_work_src+"maskbits_"+region+"x.fits"
            dst=dir_work_dst+"maskbits_lx.fits"
            force_symlink(src, dst)


        if(show_progress>0):
            if(i%show_progress==0): print(i, "/", len(DirInfo.dir_work_list), "| Time : ", time.time()-start)

    start=time.time()
    pool = Pool(processes=int(Ncore))
    pool.map(sub_generate_link, np.arange(len(DirInfo.dir_work_list)))
    print("Done! | Time : ", time.time()-start)
    pool.close()
    pool.join()



class DirInfo():
    """
    Descr - Generate Dir Info.
            It contains directory locations with coordinates, name, and other informations
    INPUT
     - dir_base: Base directory for Galfit
     * coord_array: coordinates e.g., [[ra1,dec1],[ra2,dec2], ...]
     * catalogname_list : Name of galaxies.
     * idlist : id of galaxies - the folder name will be based on this ID.
     * idxlist : idx is defined as the sorted numbers used only here (0,1,2...)
         It will be automatically generated, but you can input the idx manually.
         (Default: None - automatically generated.)
     * use_group : If True, the galaxy folder will be grouped. (Default: True)
     * group_rule : Group folder naming rule (Default: 'G%03d/')
     * group_cut_digit: Group ID = int(galid / 10 ** group_cut_digit)
     * folder_rule : Individual galaxy folder name rule (Default: "sgal%07d/")
    """
    def __init__(self, dir_base, coord_array=None, catalogname_list=None,
                 idlist=[], idxlist=None,
                 dir_work_base='gals/',
                 use_group=True, group_rule='G%03d/', group_cut_digit=4,
                 folder_rule="sgal%07d/",
                 ):
        self.dir_base=dir_base  # Base dir
        self.coord_array=coord_array
        self.dir_work_base=dir_work_base

        self.idlist=idlist      # SDSS ID
        self.catalogname_list=catalogname_list
        self.use_group=use_group
        self.group_rule=group_rule
        self.folder_rule=folder_rule
        group_cut=10**group_cut_digit
        self.groupid_list=(self.idlist/group_cut).astype(int) # Group ID
        self.groupid_list_unique=np.unique(self.groupid_list) # Unique Group ID

        self.dir_work_group=self._get_group_name(self.groupid_list)
        self.dir_work_group_uniq=np.unique(self.dir_work_group)
        self.foldername_list=np.char.mod(self.folder_rule, self.idlist)
        self.dir_work_list=np.char.add(self.dir_work_group, self.foldername_list)

        if(hasattr(idxlist, "__len__")): self.idxlist=idxlist
        else: self.idxlist=np.arange(len(coord_array))

    def _get_group_name(self, groupid_list):
        if(self.use_group==False): return np.char.mod(self.dir_base+self.dir_work_base, groupid_list)
        # if(hasattr(groupid_list, "__len__")==False):
        #     groupid_list=[groupid_list] #int
        #     return np.char.mod(self.dir_base+self.dir_work_base+self.group_rule, groupid_list)[0]
        else: return np.char.mod(self.dir_base+self.dir_work_base+self.group_rule, groupid_list)

    def cut_data(self, cutlist, new_idx=False):
        self.idlist=self.idlist[cutlist]

        if(hasattr(self.catalogname_list, "__len__")):
            self.catalogname_list=self.catalogname_list[cutlist]
        if(hasattr(self.coord_array, "__len__")):
            self.coord_array=self.coord_array[cutlist]
        self.groupid_list=self.groupid_list[cutlist]
        self.groupid_list_unique=np.unique(self.groupid_list)
        self.dir_work_group=self._get_group_name(self.groupid_list)
        self.dir_work_group_uniq=np.unique(self.dir_work_group)
        self.foldername_list=np.char.mod(self.folder_rule, self.idlist)
        self.dir_work_list=np.char.add(self.dir_work_group, self.foldername_list)

        ## Give idx
        if(new_idx):self.idxlist=np.arange(len(coord_array))
        else: self.idxlist=self.idxlist[cutlist]

    def find_ids(self, galid_array, is_idx=False):
        ids=np.full(len(galid_array), np.nan)
        for i in range (len(galid_array)):
            if(is_idx): ids[i]=np.where(self.idxlist==galid_array[i])[0]
            else: ids[i]=np.where(self.idlist==galid_array[i])[0]
        #check nan
        nan_count=np.sum(np.isnan(ids))
        if(nan_count==0): return ids.astype(int)
        else: return ids

    def change_work_base(self, dir_work_base='gals/', return_new=True):
        if(return_new):
            newDI=copy.deepcopy(self)
            newDI.dir_work_base=dir_work_base
            newDI.dir_work_group=newDI._get_group_name(newDI.groupid_list)
            newDI.dir_work_group_uniq=np.unique(newDI.dir_work_group)
            newDI.foldername_list=np.char.mod(newDI.folder_rule, newDI.idlist)
            newDI.dir_work_list=np.char.add(newDI.dir_work_group, newDI.foldername_list)
            return newDI
        else:
            self.dir_work_base=dir_work_base
            self.dir_work_group=self._get_group_name(self.groupid_list)
            self.dir_work_group_uniq=np.unique(self.dir_work_group)
            self.foldername_list=np.char.mod(self.folder_rule, self.idlist)
            self.dir_work_list=np.char.add(self.dir_work_group, self.foldername_list)

    def make_dir_work(self):
        for i in range (len(self.dir_work_list)):
            os.makedirs(self.dir_work_list[i], exist_ok=True)


def cut_DirInfo(DirInfoPrev, cutlist, new_idx=False):
    """
    Descr - cut DirInfo. (See class DirInfo)
    INPUT
     - DirInfoPrev: Previous DirInfo
     - cutlist: cut list (e.g., [0,1,2,3], [True, False, True])
     * new_idx : If give new idx for the new DirInfo (Default: False)
    """
    newDI=copy.deepcopy(DirInfoPrev)
    newDI.cut_data(cutlist, new_idx=new_idx)
    if(len(DirInfoPrev.idlist)!=len(cutlist)): print("Warning! Cutlist size is different from the DirInfo")
    print("Previous :", len(DirInfoPrev.idlist), " |  New : ", len(newDI.idlist))
    return newDI

class FileGroupSearch:
    def __init__(self, DirInfo=None, dir_work_base='gal_img/',
                 fn_set=['psf_model_dr9_s.fits', 'psf_model_dr9_n.fits', 'psf_model_dr10_s.fits'],
                 dir_group=None, skip_search=False):

        if(skip_search):
            ## Return the entire lists
            self.dir_work_list_todo=DirInfo.dir_work_list
            self.coord_array_todo=np.copy(DirInfo.coord_array)

        else:
            self.DirInfo=DirInfo
            self.fn_set=fn_set
            self.dir_group=dir_group
            self.find()
            self.print_summary()

    def find(self):
        if(self.DirInfo.use_group==False):
            # DI that does not have group
            base_folder=self.DirInfo.dir_base+self.DirInfo.dir_work_base+"*/" #gals/gals
            self.groupgal=np.arange(len(self.DirInfo.dir_work_list))
        elif(self.dir_group==None):
            # non group
            base_folder=self.DirInfo.dir_base+self.DirInfo.dir_work_base+"*/*/" #gals/Group/sgals
            self.groupgal=np.arange(len(self.DirInfo.dir_work_list))
        else:
            #group
            self.dir_group=int(self.dir_group)
            target_group_name=str(self.DirInfo._get_group_name(self.dir_group))       #gals/Group/
            print(target_group_name)
            base_folder=target_group_name+"*/"                        #sgals
            self.groupgal=np.where(self.DirInfo.dir_work_group==target_group_name)[0]


        self.dir_work_list_todo=self.DirInfo.dir_work_list[self.groupgal]
        try: print("Gal Index: ", self.groupgal[0], "-", self.groupgal[-1])
        except: print("Warning! No target!")
        self.coord_array_todo=self.DirInfo.coord_array[self.groupgal]
        print("Total NGal : ", len(self.dir_work_list_todo))


        ## Finding setting
        self.filelist_fileonly_keys=np.zeros(len(self.fn_set), dtype='object') ## glob keys
        self.filelist_err_keys=np.zeros(len(self.fn_set), dtype='object')

        self.filelist_fileonly=np.zeros(len(self.fn_set), dtype='object') ## found list
        self.filelist_err=np.zeros(len(self.fn_set), dtype='object')

        self.filelist_fileonly_exact=np.zeros(len(self.fn_set), dtype='object') ## supposed to be
        self.filelist_err_exact=np.zeros(len(self.fn_set), dtype='object')

        self.donelist=np.zeros(len(self.fn_set), dtype='object')
        self.donelist_err=np.zeros(len(self.fn_set), dtype='object')
        self.donelist_w_err=np.zeros(len(self.fn_set), dtype='object')


        for i in range (len(self.fn_set)):
            thiskey=self.fn_set[i].split('.')[0]
            self.filelist_err_keys[i]=base_folder+"err_"+thiskey+".dat"
            self.filelist_fileonly_keys[i]=base_folder+self.fn_set[i]

        ## Main finding
            self.filelist_fileonly[i]=glob.glob(self.filelist_fileonly_keys[i]) ## find files
            self.filelist_err[i]=glob.glob(self.filelist_err_keys[i])

            self.filelist_fileonly_exact[i]=np.char.add(self.dir_work_list_todo, self.fn_set[i])
            self.filelist_err_exact[i]=np.char.add(self.dir_work_list_todo, "err_"+thiskey+".dat")

            self.donelist[i]=np.isin(self.filelist_fileonly_exact[i], self.filelist_fileonly[i])
            self.donelist_err[i]=np.isin(self.filelist_err_exact[i], self.filelist_err[i])
            self.donelist_w_err[i]=(self.donelist[i] | self.donelist_err[i])


    def print_summary(self):

        for i, fn in enumerate(self.fn_set):
            print(">> File:", fn, " - ", np.sum(self.donelist[i]), "/", len(self.dir_work_list_todo))
            thiskey=self.fn_set[i].split('.')[0]
            err="err_"+thiskey+".dat"
            print(">> Err file:", err, " - ", np.sum(self.donelist_err[i]), "/", len(self.dir_work_list_todo))
            print(">> Total:", fn, " - ", np.sum(self.donelist_w_err[i]), "/", len(self.dir_work_list_todo))
            print("\n")


#     def get_todo(self, donelist, invert=False):
#         if(invert): self.todo_index=np.where(donelist!=0)[0]
#         else: self.todo_index=np.where(donelist==0)[0]
#         self.dir_work_list_todo=self.dir_work_list_todo[self.todo_index]
#         self.coord_array_todo=self.coord_array_todo[self.todo_index]

#         print("Remaining Ngals :", len(self.dir_work_list_todo))

# class FileGroupSearch:
#     def __init__(self, DirInfo=None, use_maskbit=True, bandlist=['g', 'r'],
#                  keys_band=['sigma_', 'err_', 'err_http_'], keys_univ=['maskbits_'],
#                  formats_band=['.fits', '.dat', '.dat'], formats_univ=['.fits'],
#                  region='s', dir_group=None, skip_search=False):
#
#         if(skip_search):
#             ## Return the entire lists
#             self.dir_work_list_todo=DirInfo.dir_work_list
#             self.coord_array_todo=np.copy(DirInfo.coord_array)
#
#         else:
#             if(DirInfo.use_group==False):
#                 # do not use group
#                 base_folder=DirInfo.dir_base+"gals/*/" #gals/gals
#                 self.groupgal=np.arange(len(DirInfo.dir_work_list))
#             elif(dir_group==None):
#                 # non group
#                 base_folder=DirInfo.dir_base+"gals/*/*/" #gals/Group/sgals
#                 self.groupgal=np.arange(len(DirInfo.dir_work_list))
#             else:
#                 #group
#                 dir_group=int(dir_group)
#                 target_group_name=DirInfo._get_group_name(dir_group)       #gals/Group/
#                 base_folder=target_group_name+"*/"                        #sgals
#                 self.groupgal=np.where(DirInfo.dir_work_group==target_group_name)[0]
#
#
#             self.dir_work_list_todo=DirInfo.dir_work_list[self.groupgal]
#             try: print("Gal Index: ", self.groupgal[0], "-", self.groupgal[-1])
#             except: print("Warning! No target!")
#             self.coord_array_todo=DirInfo.coord_array[self.groupgal]
#             print("Total NGal : ", len(self.dir_work_list_todo))
#
#             ## Finding setting
#
#             keys=keys_band+keys_univ
#             formats=formats_band+formats_univ
#             self.filelist_keys=np.zeros(len(keys), dtype='object')
#             self.filelist_exact=np.zeros(len(keys), dtype='object')
#
#             for i in range (len(keys)):
#                 self.filelist_keys[i]=base_folder+keys[i]+region+"*"+formats[i]
#                 if(np.isin(keys[i], keys_band)):
#                     self.filelist_exact[i]=np.zeros(len(bandlist), dtype='object')
#                     for j in range (len(bandlist)):
#                         self.filelist_exact[i][j]=np.char.add(self.dir_work_list_todo,
#                         keys[i]+region+bandlist[j]+formats[i])
#                 else:
#                     self.filelist_exact[i]=np.char.add(self.dir_work_list_todo, keys[i]+region+"x"+formats[i])
#
#             ## Main finding
#             self.filelist_exist=np.zeros(len(keys), dtype='object')
#             self.donelist=np.zeros(len(keys), dtype='object')
#             for i in range (len(keys)):
#                 self.filelist_exist[i]=glob.glob(self.filelist_keys[i])
#                 if(np.isin(keys[i], keys_band)):
#                     self.donelist[i]=np.zeros((len(self.groupgal), len(bandlist)), dtype=int)
#                     for j in range (len(bandlist)):
#                         self.donelist[i][:,j]=np.isin(self.filelist_exact[i][j], self.filelist_exist[i])
#                         print(">> Keys:", keys[i], "Band:", bandlist[j], " - ", np.sum(self.donelist[i][:,j]), "/", len(self.dir_work_list_todo))
#                 else:
#                     self.donelist[i]=np.isin(self.filelist_exact[i], self.filelist_exist[i])
#                     print(">> Keys:", keys[i], " - ", np.sum(self.donelist[i]), "/", len(self.dir_work_list_todo))
#
#
#     def get_todo(self, donelist, invert=False):
#         if(invert): self.todo_index=np.where(donelist!=0)[0]
#         else: self.todo_index=np.where(donelist==0)[0]
#         self.dir_work_list_todo=self.dir_work_list_todo[self.todo_index]
#         self.coord_array_todo=self.coord_array_todo[self.todo_index]
#
#         print("Remaining Ngals :", len(self.dir_work_list_todo))
