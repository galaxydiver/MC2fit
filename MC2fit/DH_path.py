import DH_array as dharray
import numpy as np
import os
import random
import pickle
import copy
import shutil
from dataclasses import dataclass
from multiprocessing import Pool
import warnings

home=os.path.expanduser('~')
try: this_machine=np.loadtxt(home+"/this_machine.dat")
except: this_machine=-1

if(this_machine==1):
    print("This machine : SO desktop computer")
elif(this_machine==2):
    print("This machine : Mac")
elif(this_machine==3):
    print("This machine : HPC")
else:
    print("Unknown machine, ", this_machine)
    print("You may manually open DH_path.py and change the path")

def fn_legacy_sum_south(): ## DR9 south fits file
    if(this_machine==1): return '/home/donghyeon/ua/udg/galfit/galfit/RawData/survey-bricks-dr9-south.fits' # Fits info data
    if(this_machine==2): return home+'/ua/udg/galfit/RawData/survey-bricks-dr9-south.fits' # Fits info data
    if(this_machine==3): return '/home/u14/galaxydiver/xdisk/udg/galfit/RawData/survey-bricks-dr9-south.fits' # Fits info data
    if(this_machine==-1): return '/home/u14/galaxydiver/xdisk/udg/galfit/RawData/survey-bricks-dr9-south.fits' # Fits info data

def fn_legacy_sum_north(): ## DR9 north fits file
    if(this_machine==1): return '/home/donghyeon/ua/udg/galfit/galfit/RawData/survey-bricks-dr9-north.fits' # Fits info data
    if(this_machine==2): return home+'/ua/udg/galfit/RawData/survey-bricks-dr9-north.fits' # Fits info data
    if(this_machine==3): return '/home/u14/galaxydiver/xdisk/udg/galfit/RawData/survey-bricks-dr9-north.fits' # Fits info data
    if(this_machine==-1): return '/home/u14/galaxydiver/xdisk/udg/galfit/RawData/survey-bricks-dr9-north.fits' # Fits info data

def fn_legacy_sum_south_dr10(): ## DR10 south fits file
    if(this_machine==1): return '/home/donghyeon/ua/udg/galfit/galfit/RawData/survey-bricks-dr10-south.fits' # Fits info data
    if(this_machine==2): return home+'/ua/udg/galfit/RawData/survey-bricks-dr10-south.fits' # Fits info data
    if(this_machine==3): return '/home/u14/galaxydiver/xdisk/udg/galfit/RawData/survey-bricks-dr10-south.fits' # Fits info data
    if(this_machine==-1): return '/home/u14/galaxydiver/xdisk/udg/galfit/RawData/survey-bricks-dr10-south.fits' # Fits info data


def dir_brick(is_test=False): ## Folder for Legacy_bricks
    if(this_machine==1):
        if(is_test): return '/home/donghyeon/Legacy_bricks_test/'
        else: return '/home/donghyeon/Legacy_bricks/'

    if(this_machine==2): return home+'/ua/Legacy_bricks_test/' # Fits info data
    if(this_machine==3): return '/home/u14/galaxydiver/xdisk/Legacy_bricks/' # Fits info data
    if(this_machine==-1): return '/home/u14/galaxydiver/xdisk/Legacy_bricks/' # Fits info data


def fn_galfit(): ## Galfit folder
    if(this_machine==1): return '/home/donghyeon/galfit3.0.7b/galfit'
    if(this_machine==2): return home+'/ua/galfit/galfit3.0.7b/galfit'
    if(this_machine==3): return '/home/u14/galaxydiver/galfit'
    if(this_machine==-1): return '/home/u14/galaxydiver/galfit'
