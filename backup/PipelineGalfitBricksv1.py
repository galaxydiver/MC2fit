'''
Module:
    PipelineGalfit

Contains the functions required to fit Galfit Sersic model to UDG candidates

Module initially named FigureMEFHPC_MP
May 23, 2017: V2_0
    * added documentation
    * deleted unnecessary code

Oct 21, 2017 V3_0
    * Fixed problems with finding centers of fits
    * Made compatible with latest version of UDG_Sersic_Screen

Nov 14, 2017 V4_0
    * Converted to a Class

Dec 15, 2017 V4_1
    * Fixed error caused by Galfit crashes
        Now keeps track of Galfit log to make sure that candidate was fully processed

Dec 20, 2017 V4_2
    * Fixed error causing by not identifying all Galfit crashes
    * Added option to show residuals rather than models

Dec 26, 2017 V5_0
    * Adjusted Galfit masking to better detect overlying objects

Mar 11, 2018 HPC
    * Adapted GalfitComa_5_0 for HPC
        Made many changes.  Because of lockout problems
        with simultaneous connections, results saved as cvs files
        rather than being directly added to database.
        Also eliminated usage of ds9


Mar 20, 2018 HPCv1_0
    * Switched to using SExtractor for Galfit detections in order to get FWHM
        Used for detecting unresolved objects overlying galaxy

Mar 30, 2018 HPCv1_0
    * Using results of screening Sersic to populate Galfit starting parameters
        Corrected many Galfit crashes caused by bad starting parameters

Apr 01, 2018 HPCv1_0
    * Added Galfit central coordinates to database

Apr 15, 2018 HPCv1_1
    * Save data as csv file rather than insert into database
        This avoided locking and I/O errors

Apr 16, 2018
    * Adapted from GalfitComaHPCv1_1
        Does not create figures for every candidate.
        Only creates csv files with data

Apr 20, 2018 create_figures
    * Only creates figures from candidates that have passed Galfit criteria

Apr 23, 2018 create_figuresV1_1
    * Redid scaling of thumbnails to show individual coadds more clearly and
        account for noise properties in the 3-band stack

May 9, 2018 FigureMEFHPC
    * Changed outputs so that Multiframe fits files are generated rather than figures

May 22, 2018 FigureMEFHPC1_1
    * Mask bright objects that are more than 40 pixels from center

May 24, 2018 FigureMEFHPC_MP
    * Adapted to use Python multiprocessing rather than array job

May 27, 2018 FigureMEFHPC_MP
    * Added option to use '/tmp' on HPC node for RAM_Disk
        Cannot create true RAM Disk on HPC. Default is to use 'rsgrps/dfz' hard drive

Jan 29, 2019 PipelineGalfit
    * Changed name signifying it is part of the pipeline
    * Made significant changes to allow routines to run as part of an automated
        pipeline rather than individually.

Aug 3, 2019 switched name to PipelineGalfitv2_5
    * Changed masking of central objects to use ellipsoids extending to threshold values
        of 2D Gaussian profile
    * Cleaned up a lot of code and added documentation

Jun 2020, PipelineGalfitv8_0
    * Ported to Python 3 as of version 7.0
    * Added documentation
    * Added code to allow saving masks
    * Made changes to process reprojected (North up) images

Oct 2021, ver Bricks
    * Made various changes to use Legacy Bricks rather than than NOAO observations

Dec 2021, ver v1
    * Modified to use Legacy Survey inverse variance to generate Galfit sigma images

Jan 2022, ver v1
    * Modified to use Legacy Survey psf images to obtain psf

Mar 2022, ver v1
    * Added capability to provide a list of bricks for analysis

Apr 2022, ver v1
    * Added option of excluding z for Galfit structural estimates

May 2022, ver v1
    * Adjusted masking for better fitting in crowded fields


Galfit errors:
    0: Galfit successfully ran
    1: Galfit crashed
    2: Galfit gave bad result
    3: Sersic screen failed
    4: No image in band
    5: Full stack failed either Sersic screen or Galfit
    6: No adequate thumbnails in band
'''
from PipelineConfigBricksv1 import DO_SPECIAL, GF_ONLY, DOPRINT, SHOWIMAGES, \
    CHECK_WARNINGS, USE27, DOFILTER, BANDTODO, NO_Z
import sqlite3
from DatabaseUtilitiesBricksv1 import gen_GFtables
from math import log10, pi, sqrt, log, cos
from os import path, mkdir, makedirs, listdir, remove
from numpy import std, where, array, max as npmax, float32, uint8,concatenate, \
    median, zeros, ones, int16, float64, zeros_like, mgrid, random, argsort, \
    savez_compressed, load as npload, sum as npsum, isnan, \
    array_split, flatnonzero, asscalar, searchsorted, nonzero, append as npappend, \
    sort as npsort, lib
from cv2 import GaussianBlur, BORDER_REFLECT
from PipelineUtilitiesBricksv1 import myopenfits, mysavefits, showArray, coords_on_image
import sep
from time import time, sleep
from datetime import datetime
from SercicFitBricksv1 import mySercicFit, tofitfuncexplmfitlinls
from GalfitUtilitiesBricksv1 import deg2hms, deg2dms, GalfitSE, get_SEdata, hms2deg, \
     getgalfit, getmodel, get_mu0_thresh
from shutil import copy2
import csv
from astropy.io import fits
from multiprocessing import Process
from uncertainties import ufloat, wrap # @UnresolvedImport
from uncertainties.unumpy import log10 as unlog10, gamma as ungamma # @UnresolvedImport
from scipy.special import gammainccinv
from glob import glob
from PipelineCatalogBricksv1 import simple_ellipse
from astropy.coordinates import SkyCoord # @UnresolvedImport
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic # @UnresolvedImport
#from PipelineAstromaticBricksv1 import SW
from PipelineSersicScreenBricksv1 import sky2pix
#from PipelineSersicScreenv8_0 import get_thumbs_SWarp
if CHECK_WARNINGS:
    from numpy import seterr
    seterr(all='warn')
from astropy.convolution import Gaussian2DKernel
if CHECK_WARNINGS:
    seterr(all='raise')
import warnings

# GLOBALS

# Flags

# Shuffle galaxies to analyze among job cores
doshuffle = True
# Save the results.  Would be set to False when testing and wish to avoid overwriting
savedata = True
# Print outputs and show images when debugging
#===============================================================================
#    dotest (Boolean): Use testing mode if set.
#    debug (Boolean): Display images, print intermediate results when debugging
#    save_masks (Boolean): Save masks used for Galfit
#===============================================================================
dotest = False
debug = False
save_masks = False

# Constants

# Folder where Galfit results will be saved
data_save_loc = 'GF_data/'
# File name prefixes for Galfit results using a fixed and variable Sersic index
GF_fixed_name = 'GF_nfixed_'
GF_float_name = 'GF_nvar_'


def get_sigmas(flux, majaxis, minaxis, amplitude ):
    '''
    Calculates standard deviations of a 2D Gaussian profile from SExtractor outputs.
    Volume of Gaussian profile:
        V = 2 * pi * A * sigma_x * sigma_y
        where A is peak (central) amplitude and sigma_x * sigma_y are the
            standard deviations along the x and y axes
        Assume x is along major axis
        V = 2 * pi * A * sigma_x * (minor_axis/major_axis) sigma_x
        V = 2 * pi * A * (minor_axis/major_axis) * sigma_maj^2
        sigma_maj^2 = V/(2 * pi *A) * major_axis/minor_axis
        sigma_maj = sqrt(V/(2 * pi *A) * major_axis/minor_axis)
        sigma_min = minor_axis/major_axis * sigma_maj
    inputs:
        All are SExtractor parameters
        flux (float): Total flux (FLUX_AUTO)
        majaxis (float): Major axis (A_IMAGE)
        minaxis (float): Minor axis (B_IMAGE)
        amplitude (float): Peak flux (FLUX_MAX)
    Outputs:
        sig_maj, sig_min (floats): Standard deviations along
            major and minor axes
    '''
    sqval = majaxis/minaxis  * flux / (2* pi* amplitude)
    if sqval < 0:
        return 0, 0
    else:
        sig_maj = sqrt(sqval)
    sig_min = minaxis/majaxis * sig_maj
    return sig_maj, sig_min

def name2coords(UDGname):
    RAdec = UDGname[4:].split('_')
    if len(RAdec) != 2:
        RA = RAdec[0][:7]
        dec = RAdec[0][7:]
    else:
        RA = RAdec[0]
        dec = RAdec[1]
    RAhms = RA[:2] + ':' +  RA[2:4] + ':' +  RA[4:6] + '.' + RA[-1]
    decdms = dec[:-4] + ':' +  dec[-4:-2] + ':' +  dec[-2:]
    RAdeg, decdeg = hms2deg(RAhms, decdms)
    return RAdeg, decdeg

def get_gauss_thresh(sigthresh, amplitude, sig_maj, sig_min):
    '''
    Purpose:
        Calculates distance along major and minor axes for Gaussian profile
            to reach a threshold level
        2D Gaussian profile with 0 means:
            f(x, y) = A * exp[-(x^2/(2 * sigma_X^2) + y^2/(2 * sigma_Y^2))]
            ln(f(x,y)/A) = -[x^2/(2 * sigma_X^2) + y^2/(2 * sigma_Y^2)]
            Treat x and y independently:
            thresh_x ^2 = -[ln(f(x,y)/A) * 2] * sigma_X^2
            thresh_x = sigma_X * sqrt(-2 * (ln(f(x,y)/A))
            thresh_y = sigma_Y * sqrt(-2 * (ln(f(x,y)/A))
    inputs:
        sigthresh (float): threshold value
        amplitude (float): Peak flux (FLUX_MAX)
        sig_maj, sig_min (floats): Standard deviations along
            major and minor axes
    Outputs:
        thresh_maj, thresh_min (floats): Distances along
            major and minor axes to reach thresholds
    '''
    logval = sigthresh/amplitude
    if logval <= 0:
        return 0, 0
    sqval =  -2* log(logval)
    if sqval < 0:
        return 0, 0
    thresh_fac = sqrt(sqval)
    thresh_maj = sig_maj * thresh_fac
    thresh_min = sig_min * thresh_fac
    return thresh_maj, thresh_min

def create_coadd_tbl(cur, conn):
    '''
    Creates an Sqlite3 table in the current database that will contain initial Galfit results.

    inputs:
        cur: Sqlite3 cursor to current database
        conn: Sqlite3 connection to curent database
    '''

    tbl = 'COADDSvar'

    SQL = """ CREATE TABLE IF NOT EXISTS """ + tbl + """( brick TEXT, galnum INTEGER, Filename TEXT, RA REAL, dec REAL,
    Simulation INTEGER, Ncombine_g REAL, Ncombine_r REAL, Ncombine_z REAL, Ncombine_all REAL,
    x0 REAL, y0 REAL, Rearcsec REAL, Rearcsecerr REAL, AR REAL, ARerr REAL, PA REAL, PAerr REAL,
    n REAL, nerr REAL, mag_g REAL, mag_r REAL, mag_z REAL,
    magerr_g REAL, magerr_r REAL, magerr_z REAL,
    mu_0_g REAL, mu_0_r REAL, mu_0_z REAL, mu_0_gerr REAL, mu_0_rerr REAL, mu_0_zerr REAL,
    sky_g REAL, sky_r REAL, sky_z REAL, skyerr_g REAL, skyerr_r REAL, skyerr_z REAL,
    chi2nu_g REAL, chi2nu_r REAL, chi2nu_z REAL, chi2nu_all REAL,
    Gal_res_g INTEGER, Gal_res_r INTEGER, Gal_res_z INTEGER, Gal_res_all INTEGER,
    good INTEGER, EntryDate TEXT)
    """
    cur.execute(SQL)
    conn.commit()



def coadd_tbl_2_sql(info_folder, db, do_n= False, start_folder = ''):
    '''
    Imports csv files with Galfit results into database.  Data is initially save in csv format
    because of Sqlite3 locking problems.  Also flags detections that are simulations.  This table
    will contain data from all galaxies processed with Galfit
    inputs:
        info_folder (string):  Location of folder containing database
        db (string): Name of database
        do_n (Boolean): If set, data was generated with floating Sersic index.  Otherwise, index
            was fixed at one.  Galfir is run with and without a
            fixed Sersic index and results are stored in two separate tables.

    '''
    # degrees to radians factor
    deg2rad = pi/180
    # pixel scale
    pixscl = 0.262
    # Connect to database
    conn = sqlite3.connect(info_folder + db)
    conn.text_factory = bytes
    cur = conn.cursor()
    tblcoadd = 'COADDSvar'

    # If Sersic index part of fit create separate table and get list of csv files
    if do_n:
        tbl = 'Galfitfloat_Passvar'

        wiseloc = start_folder + 'wssa_sample_8192-bintable.fits'
        hdulist1 = fits.open(wiseloc)
        nside1 = hdulist1[1].header['NSIDE']
        order1 = hdulist1[1].header['ORDERING']
        hp1 = HEALPix(nside=nside1, order=order1, frame=Galactic())
        wise = hdulist1[1].data['I12']
        tauloc = start_folder + 'HFI_CompMap_ThermalDustModel_2048_R1.20.fits'
        hdulist2 = fits.open(tauloc)
        nside2 = hdulist2[1].header['NSIDE']
        order2 = hdulist2[1].header['ORDERING']
        hp2 = HEALPix(nside=nside2, order=order2, frame=Galactic())
        tau = hdulist2[1].data['TAU353']
        if GF_ONLY:
            create_coadd_tbl(cur, conn)
        SQL = 'CREATE TABLE IF NOT EXISTS ' + tbl + ' AS SELECT * FROM ' + tblcoadd + ' WHERE 0'
        cur.execute(SQL)
        conn.commit()
        cur.execute("PRAGMA table_info(" + tbl + ")")
        data =  array(cur.fetchall())
        colnames = data[:,1].astype(str)
        changed = False
        if  'tau' not in colnames:
            SQL = 'ALTER TABLE ' + tbl + ' ADD COLUMN tau REAL'
            cur.execute(SQL)
            changed = True
        if  'WISE' not in colnames:
            SQL = 'ALTER TABLE ' + tbl + ' ADD COLUMN WISE REAL'
            cur.execute(SQL)
            changed = True
        if  'RA_corr' not in colnames:
            SQL = 'ALTER TABLE ' + tbl + ' ADD COLUMN RA_corr REAL'
            cur.execute(SQL)
            changed = True
        if  'dec_corr' not in colnames:
            SQL = 'ALTER TABLE ' + tbl + ' ADD COLUMN dec_corr REAL'
            cur.execute(SQL)
            changed = True
        if  'Filename_corr' not in colnames:
            SQL = 'ALTER TABLE ' + tbl + ' ADD COLUMN Filename_corr REAL'
            cur.execute(SQL)
            changed = True
        if changed:
            conn.commit()
        allfiles = glob(info_folder + data_save_loc + GF_float_name + '*')
    # Otherwise create table for holding values with fixed Sersic index
    else:
        create_coadd_tbl(cur, conn)
        allfiles = glob(info_folder + data_save_loc + GF_fixed_name + '*')
    # Extract detection number, RA and dec from compressed file (matches.npz) containing
    #    detections that satisfied initial criteria
    if path.exists(info_folder + 'matches.npz'):
        npzfile = npload(info_folder + 'matches.npz')
        output = array(npzfile['results'])
        matchgalnums = output[:,0].astype(int)
        RAs = output[:,4].astype(float)
        decs = output[:,5].astype(float)
        gal_sort_index = argsort(matchgalnums)
        matchgalnums = matchgalnums[gal_sort_index]
        RAs = RAs[gal_sort_index]
        decs = decs[gal_sort_index]
    else:
        print(info_folder + 'matches.npz does not exist. Cannot test for simulations in Galfit result table')
    # Keep track of number galaxies entered into database
    num_galaxies_entered = 0
    # Iterate through files containing data
    for tfile in allfiles:
        with  open(tfile, 'rU') as csvfile:
            rows = csv.reader(csvfile, delimiter=',', quotechar="'")
            # Iterate through rows
            for row in rows:
                # Ignore any comment lines
                if row[0][0] != '#':
                    # Extract galaxy number, RA and dec
                    galnum = int(float(row[1]))
                    index  = searchsorted(matchgalnums, galnum)
                    RA = RAs[index]
                    dec = decs[index]
                    simnum = int(row[3])
                    if GF_ONLY:
                        fname = row[2]
                        RA,dec = name2coords(fname)
                        simnum = -1

                    #===========================================================
                    # Convert appropriate values from string to integers
                    #    combinfo: Number of bricks stacked for each band
                    #    fitinfo: Results from Galfit
                    #    resinfo: Value designates whether Galfit was successful or had problems
                    #===========================================================
                    combinfo = [int(x) for x in row[4:8]]
                    fitinfo = [float(x) for x in row[8:-6]]

                    resinfo = [int(x) for x in row[-6:-1]]
                    # Create row to save in database
                    allinfo = row[0:3]
                    allinfo[1] = int(round(float(allinfo[1])))
                    allinfo.append(RA)
                    allinfo.append(dec)
                    allinfo.append(simnum)
                    allinfo.extend(combinfo)
                    allinfo.extend(fitinfo)
                    allinfo.extend(resinfo)
                    allinfo.append(row[-1])
                    vals = '?, ' * (len(row) + 2)
                    if do_n:
                        x0 = fitinfo[0]
                        y0 = fitinfo[1]
                        newdec  = dec + ((y0-100) * pixscl)/3600.
                        #=======================================================
                        # Cosine correction for declination
                        # Can be significant for high declinations in MzLS_BASS
                        #=======================================================
                        corrfact = cos(newdec * deg2rad)
                        newRA = RA - (((x0-100) * pixscl)/3600.)/corrfact
                        RAstr = deg2hms(newRA,separator = '')
                        decstr = deg2dms(newdec,separator = '')
                        newname = 'SMDG'  + RAstr + decstr
                        l =  [newRA]
                        b = [newdec]
                        coords = SkyCoord(l, b, unit='deg', frame='icrs')
                        wiseval = hp1.interpolate_bilinear_skycoord(coords, wise)[0]
                        tauval = 1.49e4 * hp2.interpolate_bilinear_skycoord(coords, tau)[0]
                        allinfo.append(round(tauval,5))
                        allinfo.append(round(wiseval, 5))
                        allinfo.append(round(newRA, 5))
                        allinfo.append(round(newdec, 5))
                        allinfo.append(newname)
                        vals = '?, ' * (len(row) + 7)
                    num_galaxies_entered += 1
                    if num_galaxies_entered %100 == 0:
                        print('number of galaxies entered into database', num_galaxies_entered)
                    # Create query to enter data and store in appropriate table
                    vals = vals[:-2]
                    if do_n:
                        SQL  = 'INSERT OR IGNORE INTO ' + tbl  + ' VALUES (' + vals + ')'
                    else:
                        SQL  = 'INSERT OR IGNORE INTO ' + tblcoadd  + ' VALUES (' + vals + ')'
                    cur.execute(SQL, tuple(allinfo))
    conn.commit()

def get_processed(info_folder):
    '''
    Purpose:
        Retrieves sorted array of thumbnails already created
        Inputs:
            info_folder (string):  Location of folder containing individual
                lists of thumbnails processed by each job during Sersic screen
        Output:
            Sorted list of all thumbnails created during Sersic screen
            Sorted list used to facilitate later searching

    '''
    # List of all saved numpy files containing lists of processed thumbnails
    allfiles = glob(info_folder +  'Processed*')
    # Initialize numpy integer array to hold results
    processed = array([]).astype(int)
    for afile in allfiles:
        # Load data and append to array
        data = npload(afile)
        processed = npappend(processed, data)
    # sort array for later searching
    processed = npsort(processed)
    return processed

def GalfitPassTbl(info_folder, db, criteria_dict):
    '''
    Creates table in database containing only those Galfit results that pass specific criteria.
    If screening all candidates, criteria will be those for r_e (arcsec), axis ratio, mu0(g) and mu0(z)
    If only processing specific UDGs or only simulations, accept regardless of Galfit results
    inputs:
        info_folder (string):  Location of folder containing database
        db (string): Name of database
        criteria_dict (dictionary): Criteria for r_e (arcsec), axis ratio, mu0(g) and mu0(z)
    '''
    GFtbl = 'Galfit_Passvar'
    coaddtbl = 'Coaddsvar'
    # Query for creating table based upon Galfit results
    SQL = """CREATE TABLE IF NOT EXISTS """ + GFtbl + """ AS SELECT * FROM """ + coaddtbl + """ WHERE (((Rearcsec > ?) and (AR  > ?)
            and ((mu_0_g > ?) or ((mu_0_g = 0) and (mu_0_z > ?) ) and Simulation = -1) or Simulation >= 0))  """
    # Set criteria for accepting results and create table
    if DO_SPECIAL or DOFILTER:
        criteria = (0, 0, 0 ,0)
    else:
        criteria = (criteria_dict['r_e'],criteria_dict['AR'],criteria_dict['mu0g'],criteria_dict['mu0z'])
    conn = sqlite3.connect(info_folder + db)
    conn.text_factory = bytes
    cur = conn.cursor()
    cur.execute(SQL, criteria)
    conn.commit()



def calc_mu_uncertainties(mag, magerr, re_arcsec, re_arcsecerr, AR, ARerr, n, nerr):
    '''
    Purpose:
        Calculates mu0 and mu0 errors from other Galfit parameters (mag, r_e and b/a) and their errors
    mu0 = mag + 2.5 * log(gamma(2 * n + 1) * pi * a * b) where a and b are major/minor axes
    mu0 = mag + 2.5 * log(gamma(2 * n + 1) * pi * a * a * b/a)
    scale length, h, = a
    mu0 = mag + 2.5 * log(gamma(2 * n + 1) * pi * h^2 * b/a)
    Note:  For n = 1, gamma(2 * n + 1) = 2
    Input parameters:
        mag (float): magnitude
        magerr (float): magnitude error
        re_arcsec (float): effective radius (arcsec)
        re_arcsecerr (float): effective radius error(arcsec)
        AR (float): axis ratio (b/a)
        ARerr (float): axis ratio (b/a) error
        n (float) Sersic Index
        nerr (float) Sersic Index error
    Output:
        mu0 (float): central surface brightness
        mu0err (float): central surface brightness error
    '''
    # Convert inverse incomplete gamma function to uncertainties format
    gammainccinva = wrap(gammainccinv)
    # Use values of 0 if Galfit wasn't successful
    if mag == 0:
        return 0.0, 0.0
    if re_arcsec ==0:
        return 0.0, 0.0
    # Convert values and errors into uncertainty formats
    re = ufloat(re_arcsec,re_arcsecerr)
    n = ufloat(n, nerr)
    mag = ufloat(mag,magerr)
    AR = ufloat(AR,ARerr)
    #===========================================================================
    # Convert r_e to scale length
    # Scale length, h, = re/b_n^n
    # b_n = gammainccinv(2. * n, 0.5)
    # where gammainccinv is the inverse of the incomplete gamma function
    #===========================================================================
    b_n = gammainccinva(2. * n, 0.5)
    h2r_e =  b_n**n
    h = re/h2r_e
    mu = mag + 2.5 * unlog10(ungamma(2 * n + 1) * pi * h ** 2 * AR)
    # Format and extract results
    mu= mu.format('10.3f')
    mu0, mu0err = float(str(mu).split('+/-')[0]), float(str(mu).split('+/-')[1])
    return mu0, mu0err

class dogalfit():
    """
    Purpose:
        Supplies thumbnails to Galfits and gets results.
        Also creates fits files showing results.
    Input parameters:
        core_num (integer): Core number being used for processing
        galnums (integer list): List of candidate numbers to be analyzed by core
        brickinfo (array): Brick names and psf's
        osplatform: System platform (Mac, HPC, or Unix)
        info_folder: Folder containing results of computations, including database
        image_folder: Location of folder containing original fits files
        RAM_dir: Work directory.  Uses RAM on Mac for fast I/O.  Regular storage on HPC
        processedIDs (integer array): sorted array containing unique IDs for each thumbnail
            created during Sersic screen
        do_n (Boolean): Let Sersic index vary during Galfit procedure
        half_edge (Boolean): Thumbnail edge = 2 * half_edge + 1.  An odd number has a defined integer center
        UDGs (string list): List of UDGs being analyzed
    Outputs:
        Multiframe fits files containing Coadd images and data
        csv files conating Galfit res
    """
    def __init__(self, core_num, brick_list, gf_info, osplatform, info_folder, image_folder, RAM_dir, processedIDs,
            do_n=True, half_edge=100, UDGs=[]):
        self.gf_info = gf_info
        print('Core', core_num, 'Number of bricks', len(brick_list))
        # Zero point for nanomaggies
        self.nanomaggie_zpt = 22.5
        # Legacy Survey brick pixel scale
        if not USE27:
            self.pixscl = 0.262
        else:
            self.pixscl = 0.27
        # Get names of bricks and associated psf's
        self.bricks = brick_list

        self.half_edge = half_edge
        self.core_num = core_num
        self.processedIDs = processedIDs
        if DOPRINT:
            print(core_num)
        # If set, let Sersic index vary
        self.fitsersic = do_n
        # Set up folders for Class use
        self.osplatform = osplatform
        self.info_folder = info_folder
        self.image_folder = image_folder
        # Location to save Galfit results
        self.save_folder = info_folder + data_save_loc
        if not path.exists(self.save_folder):
            mkdir(self.save_folder)
        self.image_folder = image_folder
        self.RAM_dir = RAM_dir
        # parameters to save with thumbnails
        self.brickkeys = ['band',  'padfx', 'skymed']

        if DOPRINT:
            print(len(self.galnums))
            print(self.galnums[:max(10, len(self.galnums))])
        self.UDGs = UDGs
        #=======================================================================
        # Cores will use unique names for storing SExtractor and Galfit setup files
        # goodfiles: List of files that will kept when erasing temporary
        #    SExtractor Galfit, and SWarp files.  These include executables (sex, galfit),
        #    files required by SExtractor ('sex.nnw', 'gauss_3.0_7x7.conv') and
        #    configurations (gfname + '_sextractor.conf', gfname + '_sextractor.param')
        #=======================================================================
        self.gfname = 'Gfrel' + str(core_num)
        self.goodfiles = ['sex', 'galfit', 'sex.nnw', 'gauss_3.0_7x7.conv',
                          self.gfname + '_sextractor.conf', self.gfname + '_sextractor.param' ]
        # Create file and headers to hold analysis results
        if savedata:
            # Use appropriate Global file names for runs with and without a floating Sersic index
            if not do_n:
                self.gf_file_name = GF_fixed_name
            else:
                self.gf_file_name = GF_float_name
            # Open csv file to save results
            self.Sorterfile = open(self.save_folder + self.gf_file_name +str(core_num) + '.csv', 'w', newline='')
            #===============================================================
            # keysCSV: Column headers for saved Galfit results
            #     galnum (integer): Galaxy number
            #     Filename (string): galaxy name
            #     Ncombine_(g, r, z, all) (integer):  Number of bricks used to create stack in each band
            #     x0, y0 (floats): x and y coordinates on thumbnail
            #     Rearcsec, Rearcsecerr (floats): Effective radius (arcsec) and error
            #     AR, ARerr (floats): Axis ratio (b/a) and error
            #     theta, thetaerr (floats): Position angle and error
            #     n, nerr (floats): Sersic Index and error
            #     mag_(g, r, z), mag_(g, r, z)err (floats): magnitudes and errors
            #     mu0_(g, r, z), mu0_(g, r, z)err (floats): Central surface brightness and error
            #     sky_(g, r, z), sky_(g, r, z)err (floats): Sky level and error
            #     chi2nu__(g, r, z, all) (floats): Chi^2/nu produced by Galfit
            #     Gal_res_(g, r, z, all) (integers): Failure codes for Galfit.  If success then equals 0
            #     good (Boolean): Classified as UDG after visual or automatic screening
            #     EntryDate (string Date): Date/time of Galfit analysis
            #===============================================================
            keysCSV = ['#brick', 'galnum', 'Filename', 'Ncombine_g', 'Ncombine_r', 'Ncombine_z', 'Ncombine_all',
                 'x0', 'y0', 'Rearcsec', 'Rearcsecerr', 'AR', 'ARerr', 'theta', 'thetaerr', 'n', 'nerr',
                 'mag_g', 'mag_r', 'mag_z', 'mag_gerr', 'mag_rerr', 'mag_zerr',
                 'mu0_g', 'mu0_r', 'mu0_z', 'mu0_gerr', 'mu0_rerr', 'mu0_zerr',
                 'sky_g', 'sky_r', 'sky_z', 'sky_gerr', 'sky_rerr', 'sky_zerr',
                 'chi2nu_g', 'chi2nu_r', 'chi2nu_z', 'chi2nu_all',
                 'Gal_res_g', 'Gal_res_r', 'Gal_res_z', 'Gal_res_all', 'good', 'EntryDate' ]
            # Create csv file for Galfit results
            self.Sorterwriter = csv.writer(self.Sorterfile)
            self.Sorterwriter.writerow(keysCSV)
            self.Sorterfile.close()
        self.setup()

    def setup(self):
        """
        Purpose:
            Creates folders, files, and parameter lists that
                will be used by other Class routines
        """
        if debug:
            # Clear ds9 frames
            showArray([], newframe = 1, clear = 1)
        # Create a directory to hold coadded multiframe fits files
        self.Coadd_dir = self.info_folder + 'Coadds/'
        if not path.exists(self.Coadd_dir):
            mkdir(self.Coadd_dir)
        # Create temporary directory to hold Galfit data and images
        self.GF_dir = self.RAM_dir + 'GalfitDataMP' + str(self.core_num) + '/'
        if not path.exists(self.GF_dir):
            mkdir(self.GF_dir)
        # Copy required files to temporary directory
        if not path.exists(self.GF_dir + 'galfit'):
            copy2(self.RAM_dir + 'sex.nnw', self.GF_dir + 'sex.nnw')
            copy2(self.RAM_dir + 'gauss_3.0_7x7.conv', self.GF_dir + 'gauss_3.0_7x7.conv')
            copy2(self.RAM_dir +'sex', self.GF_dir + 'sex')
            copy2(self.RAM_dir +'galfit', self.GF_dir + 'galfit')
        # Clear the directory holding Galfit results to reduce file searching
        self.clear_gf_dir()
        #=======================================================================
        # Keep track of Galfit log numbers.
        #    Used to detect crashes since a new number
        #        will not be generated if it crashes
        #=======================================================================
        self.lognums = set()
        # Initialize log numbers
        self.lognums.add(0)
        # Extract latest Galfit log number. Returns 0 if no previous Galfit analyses
        _mu_model, _REkpc, _im_model, _fitinfo, lognum = getmodel(self.GF_dir, self.nanomaggie_zpt,
                                                                  pixscl=self.pixscl, dosky = 1)
        self.lognums.add(lognum)
        if not path.exists(self.image_folder):
            print(("The folder with brick files does not exist. It should have the following path:\n" +
               self.image_folder))
            exit()
        #=======================================================================
        # Retrieve tables containing pertinent data.
        # Tables are used rather than a database because of SQLite locking problems
        #=======================================================================
        self.get_table_data()
        #=======================================================================
        #  Define acceptable final mu_0 for each band
        #    g requires a minimum of 24.0
        #    Others are determined relative to theoretical g-band differences
        #=======================================================================
        self.band_mu0_dict = dict()
        self.band_mu0_dict['g'] = 24.0
        self.band_mu0_dict['r'] = 23.6
        self.band_mu0_dict['z'] = 23.0
        if DOFILTER and (BANDTODO=='z'):
            self.band_mu0_dict['all'] = 22.6
        else:
            self.band_mu0_dict['all'] = 23.6
        self.band_mu0_thresh = dict()
        self.band_mu0_thresh['g'] = get_mu0_thresh(self.nanomaggie_zpt, self.band_mu0_dict['g'], pixscl = self.pixscl)
        self.band_mu0_thresh['r'] = get_mu0_thresh(self.nanomaggie_zpt, self.band_mu0_dict['r'], pixscl = self.pixscl)
        self.band_mu0_thresh['z'] = get_mu0_thresh(self.nanomaggie_zpt, self.band_mu0_dict['z'], pixscl = self.pixscl)
        if NO_Z:
            self.band_mu0_thresh['all'] = self.band_mu0_thresh['g'] + self.band_mu0_thresh['r']
        else:
            self.band_mu0_thresh['all'] = self.band_mu0_thresh['g'] + self.band_mu0_thresh['r'] + self.band_mu0_thresh['z']
        # Conversion factor for Gaussian sigma to FWHM
        self.sig2fwhm = 2 * sqrt(2 * log(2))
        # MzLS_BASS filters are g, z, and r.  'all' is image with all bricks coadded (3-band stack)
        if not DOFILTER:
            self.filters = ['g', 'r', 'z', 'all']
            self.bands = ['g', 'r', 'z']
        else:
            self.filters = [BANDTODO, BANDTODO, BANDTODO, 'all']
            self.bands = [BANDTODO, BANDTODO, BANDTODO]
        #=======================================================================
        # Edge size is odd so that center falls on integer pixel
        # Since Python arrays are zero-based, self.half_edge will also be the center pixel location
        #    (self.edge-1)/2
        #=======================================================================
        self.edge = 2 * self.half_edge + 1
        self.imshape = (self.edge,self.edge)
        # Total thumbnail area
        self.thumbarea = float(self.edge * self.edge)
        # Create the xy grid for fitting
        self.Xin, self.Yin = array(mgrid[0:self.edge, 0:self.edge], dtype = float64)
        # Create masking region for bright object detection
        self.make_peripheral_mask()
        # Flatten the mesh grid for curve fitting (routines only accept flat arrays)
        self.Xin = self.Xin.ravel()
        self.Yin = self.Yin.ravel()
        # Keep track of time and number of processed candidates
        self.start_time = time()
        self.num_processed  = 0
        #=======================================================================
        #  Set up SExtractor parameters
        #    NUMBER;  Detection number
        #    X_IMAGE,Y_IMAGE: x and y locations of detection center on thumbnail
        #        Note: Corner is 1, 1 rather than Python 0,0
        #    A_IMAGE,B_IMAGE: Major and minor axes of detection
        #    FWHM_IMAGE: FWHM of detection
        #    FLUX_MAX: Peak flux of detection
        #    FLUX_AUTO: Total flux of detection
        #    THETA_IMAGE: Position angle of detection
        #=======================================================================
        SEKEYS = ('NUMBER','X_IMAGE','Y_IMAGE', 'A_IMAGE','B_IMAGE', 'FWHM_IMAGE',
                  'FLUX_MAX', 'FLUX_AUTO', 'THETA_IMAGE')
        #=======================================================================
        # Create SExtractor class  and configuration file for Galfit thumbnails
        #    gfname: Base name of configuration and parameter files
        #    GF_dir: Working directory where intermediate and final data will be saved
        #    detect_thresh: SExtractor DETECT_THRESH
        #    min_area: SExtractor DETECT_MINAREA
        #    max_area: SExtractor DETECT_MAXAREA
        #    thresh_type: SExtractor THRESH_TYPE
        #    back_size: SExtractor BACK_SIZE
        #    sekeys: SExtractor Catalog PArameters
        #    deblend_mincount : SExtractor DEBLEND_MINCONT
        #=======================================================================
        self.gfse_rel = GalfitSE(self.gfname, self.GF_dir, detect_thresh = 2.5,
                min_area = 10, max_area = 10000000, thresh_type='RELATIVE',
                back_size = 20, sekeys = SEKEYS, deblend_mincount = .1)

    def make_peripheral_mask(self):
        '''
        PURPOSE:
            Creates a circular mask of 40 pixels from center.  Inner values are 0.0 and outer are 1.0
            This is used to mask bright objects away from the center
        '''
        # Define radius
        radius = 40
        # Create array of ones the same size of thumbnails
        self.peripheral_mask = ones((self.edge, self.edge), dtype = float32)
        cx, cy = self.half_edge, self.half_edge # The center of circle
        # Set values of pixels within radius to 0
        index = (self.Xin -cx)**2 + (self.Yin-cy)**2 <= radius**2
        self.peripheral_mask[index] = 0

    def clear_gf_dir(self):
        '''
        Purpose:
            Clear the Galfit directory.
            Speeds up Galfit processing on HPC
        '''
        gf_dir = self.GF_dir
        gf_files = listdir(gf_dir)
        for gf_file in gf_files:
            #continue
            #===================================================================
            # Only delete files created during SExtractor and Galfit processing
            #    Configuration and other files require for running (goodfiles) are not deleted
            #===================================================================
            if gf_file not in self.goodfiles:
                remove(gf_dir + gf_file)

    def get_table_data(self):
        '''
        Purpose:
            Load tables containing information needed for analysis.
            Tables were derived from SQLite database.
            Tables are used because of locking problems with SQLite
            Tables can also be loaded into RAM for faster searching
        '''

        #=======================================================================
        # Load table containing information about detection matches.
        # These include the galaxy number, detection number, brick name, brick band and central coordinates
        #    match_galnums (integer): galaxy (cluster) number assigned when matching
        #    match_detnums (integer): wavelet detection number
        #    match_bricks (string): Brick name
        #    match_filters (char): Brick filter
        #    match_coords_RA (float): Right ascension of cluster centroid
        #    match_coords_dec (float): declination ascension of cluster centroid
        #    match_simnums (integer): Simulation number if assigned to simulation.
        #        Otherwise, -1
        #=======================================================================
        if path.exists(self.info_folder + 'matches.npz'):
            npzfile = npload(self.info_folder + 'matches.npz')
            output = array(npzfile['results'])
            self.match_galnums = output[:,0].astype(int)
            gal_sort_index = argsort(self.match_galnums)
            output = output[gal_sort_index]
            self.match_galnums = output[:,0].astype(int)
            self.match_detnums = output[:,1].astype(int)
            self.match_bricks = output[:,2].astype(str)
            self.match_filters = output[:,3].astype(str)
            self.match_coords_RA  = output[:,4].astype(float)
            self.match_coords_dec  = output[:,5].astype(float)
            self.match_simnums = output[:,6].astype(int)
        else:
            print(self.info_folder + 'matches.npz does not exist')
            print('Exiting')
            exit()

    def process_galaxies(self):
        """
        Purpose:
            Entry point for processing galaxies
        """
        gfgalnums, gfbricks  = self.gf_info
        # Iterate through candidates
        for self.brick in self.bricks:
            bricklocs = where(gfbricks == self.brick)
            galnums = gfgalnums[bricklocs]
            if DOFILTER:
                bands = [BANDTODO, BANDTODO, BANDTODO]
            else:
                bands = ['g', 'r', 'z']
            self.psfimage = []
            self.psfmedians = []
            for band in bands:
                psffilename = self.brick + '_psfsize_' + band + '.fits.fz'
                psfbrick_path = self.image_folder + self.brick + '_' + band + '_res/' + psffilename
                if path.exists(psfbrick_path):
                    hdul = fits.open(psfbrick_path)
                    for hdx in range(len(hdul)):
                        hdu = hdul[hdx]
                        # Not using primary header
                        if hdu.name != 'PRIMARY':
                            psfdata = hdu.data
                            self.psfimage.append(psfdata)
                            self.psfmedians.append(median(psfdata[where(psfdata > 0) ]))
                            hdul.close()
            self.brickimage = []
            self.bricksubimage = []
            self.invvarimage = []
            for band in bands:
                brickfilename = self.brick + '_' + band + '.fits.fz'
                brick_path = self.image_folder + self.brick + '_' + band + '_res/' + brickfilename
                if path.exists(brick_path):
                    hdul = fits.open(brick_path)
                    self.brickimage.append(hdul[1].data)
                    self.head = hdul[1].header
                    self.bricksubimage.append(hdul[2].data)
                    hdul.close()
                invvarfilename = self.brick + '_invvar_' + band + '.fits.fz'
                invvar_path = self.image_folder + self.brick + '_' + band + '_res/' + invvarfilename
                if path.exists(invvar_path):
                    hdul = fits.open(invvar_path)
                    for hdx in range(len(hdul)):
                        hdu = hdul[hdx]
                        # Not using primary header
                        if hdu.name != 'PRIMARY':
                            self.invvarimage.append(hdu.data)
                            hdul.close()
        # Candidates to process
            self.galnums = galnums
            lengalnums = len(self.galnums)
            for cdx in range(lengalnums):
                self.num_processed +=1
                # Candidate to process
                self.galnum =  self.galnums[cdx]
                print(self.core_num, 'Processing Galaxy number', self.galnum, 'on brick', self.brick, cdx, 'of', lengalnums)
                # Get coordinates of galaxy
                index  = searchsorted(self.match_galnums, self.galnum)
                RAdeg, decdeg = self.match_coords_RA[index], self.match_coords_dec[index]
                # Get simulation number of match (-1 if no associated simulation)
                self.match_simnum = self.match_simnums[index]
                if GF_ONLY:
                    # If running Galfit only on specific UDG, extract coordinates from UDG name
                    GFUDG = self.UDGs[cdx]
                    # Extract coordinates from SMDG name in hms/dms format
                    RAdec = GFUDG[4:].split('_')
                    if len(RAdec) != 2:
                        RA = RAdec[0][:7]
                        dec = RAdec[0][7:]
                    else:
                        RA = RAdec[0]
                        dec = RAdec[1]
                    RAhms = RA[:2] + ':' +  RA[2:4] + ':' +  RA[4:6] + '.' + RA[-1]
                    decdms = dec[:-4] + ':' +  dec[-4:-2] + ':' +  dec[-2:]
                    # Convert to decimal degrees
                    RAdeg, decdeg = hms2deg(RAhms, decdms)
                #===================================================================
                # Generate SMUDGe name (also name of multiframe fits file containing thumbnails):
                #     'SMDG' + 7 digit RA in hms + 6 digit declination in dms with sign
                #===================================================================
                self.RAcoadd = deg2hms(RAdeg,separator = '')
                self.deccoadd = deg2dms(decdeg,separator = '')
                self.MEFname = 'SMDG'  + self.RAcoadd + self.deccoadd
                # Path to saved multiframe fits file
                self.fileloc = self.Coadd_dir + self.MEFname + '.fits'
                # Process candidate
                self.fit_galaxies(RAdeg, decdeg)
            print('Time for job ' + str(self.core_num) + ': ',time() - self.start_time)

    def fit_galaxies(self, RAdeg, decdeg):
        """
        Purpose:
            Derives and saves Galfit results for candidate in csv file
            Creates individual multiframe fits files for candidates
                Frames contain thumbnail images of all bands (g, r, z and 3-band stack)
                Frame headers contain parameters needed to create ds9 images
            Inputs:
                RAdeg, decdeg (floats):  Coordinates of galaxy being processed
        """

        #=======================================================================
        # Length of sorted ID list
        # Needed to prevent errors when searching for value > maximum
        #=======================================================================
        #===============================================================================
        # List that will hold output data for csv file
        # Initialize with galaxy number, name, and simulation number (-1 if not simulation)
        #===============================================================================
        coadd_results = [self.brick]
        coadd_results.append(str(self.galnum))
        coadd_results.append(str(self.MEFname))
        coadd_results.append(self.match_simnum)
        #=======================================================================
        # Create zero arrays of non-structural parameters and their errors for output table
        # Each array has space for each band and the three band stack (four entries)
        # Structural parameters are only evaluated for the three band stack
        #    mu0_arr (float list): central surface brightness (mag/arcsec^2)
        #    mag_arr (float list): magnitude (mag)
        #    sky_arr (float list): sky level (ADU)
        #    chi2nu_arr (float list): Chi^2/nu computed by Galfit
        #    modmax_arr float list): Peak amplitude of the Galfit model (ADU)
        #=======================================================================
        (mu0_arr, mu0err_arr, mag_arr, magerr_arr, sky_arr, skyerr_arr, chi2nu_arr, modmax_arr) = (array([0., 0, 0, 0]) for i in range(8))
        #=======================================================================
        # Create zero arrays of non-UDG related parameters and their errors for output table
        #    ncombine_arr (integer list): Number of observations coadded for band
        #    gf_result_arr (integer list): Error result of entire fitting procedure.
        #        0: Galfit successfully ran
        #        1: Galfit crashed
        #        2: Galfit gave bad result
        #        3: Sersic screen failed
        #        4: No image in band
        #        5: Full stack failed either Sersic screen or Galfit
        #        6: No adequate thumbnails in band
        #=======================================================================
        ncombine_arr = array([0, 0, 0, 0])
        gf_result_arr = array([0, 0, 0, 0])
        #=======================================================================
        # Create zero arrays of non-UDG related parameters and their errors for output table
        #    band_success (boolean array):  Galfit Sersic estimate success for band (Successful = 1)
        #        Used for keeping track of number of coadds in 3-band stack
        #    success_arr (integer array): Result of scipy leastsq termination for Sersic estimate
        #=======================================================================
        band_success = array([0, 0, 0, 0])
        success_arr = array([0, 0, 0, 0])
        # Associate detections with filter
        if not DOFILTER:
            self.sersic_filter_dict = {'g':[], 'r':[], 'z':[], 'all':[]}
        else:
            self.sersic_filter_dict = {BANDTODO:[], 'all':[]}

        #===================================================================
        # Create thumbnails for psf centered on detection
        #    half_edge (integer): Half - 1/2 of thumbnail side (full edge = 2 * half_edge + 1)
        #===================================================================
        coords = (RAdeg, decdeg)
        x_cen, y_cen = (sky2pix(coords, self.head))
        x_cen = round(float(x_cen))
        y_cen = round(float(y_cen))
        xint = int(x_cen)
        yint = int(y_cen)
        try:
            psf_g = median(self.psfimage[0][xint-4:xint+5, yint-4:yint+5])
        except:
            print()
            print()
            print(self.core_num, 'Error at Galfit 1035', self.brick)
            print('RAdeg, decdeg, x_cen, y_cen',RAdeg, decdeg, x_cen, y_cen)
            print('Will now crash')
            print()
            psf_g = median(self.psfimage[0][xint-4:xint+5, yint-4:yint+5])
        if psf_g == 0:
            psf_g = self.psfmedians[0]
        psf_r = median(self.psfimage[1][xint-4:xint+5, yint-4:yint+5])
        if psf_r == 0:
            psf_r = self.psfmedians[1]
        psf_z = median(self.psfimage[2][xint-4:xint+5, yint-4:yint+5])
        if psf_z == 0:
            psf_z = self.psfmedians[2]
        FWHMg = psf_g * 1/self.pixscl
        FWHMr = psf_r * 1/self.pixscl
        FWHMz = psf_z * 1/self.pixscl

        #=======================================================================
        # Has thumbnail of candidate on brick band already been generated during Sersic screening?
        # If yes, reuse it
        # If not, generate it
        #=======================================================================
        for cdx in range(len(self.bands)):
            brickband = self.bands[cdx]
            #===================================================================
            # Create a unique name for processed detection on brick.
            #    Used to avoid recreating thumbnails
            # Name created from brick name, filter and candidate ID
            #===================================================================
            processID_str= self.brick + '_' + brickband + str(int(self.galnum)).zfill(6)
            savename = 'det' + processID_str

            # Create thumbnails
            if GF_ONLY:
                # processedIDs will not exist so see if file exists
                if path.exists(self.info_folder + 'Thumbnails/' + savename + '.npz'):
                    self.sersic_filter_dict[brickband].append(savename)
                    continue
            if self.fitsersic:
                #===========================================================
                # Thumbnails not created during Sersic screen should have been created
                #    during initial Galfit pass with Sersic index fixed at one
                #===========================================================
                if path.exists(self.info_folder + 'Thumbnails/' + savename + '.npz'):
                    self.sersic_filter_dict[brickband].append(savename)
                    continue
            brick_name = self.brick + '_' + brickband
            imbrick = self.brickimage[cdx]
            subimbrick = self.bricksubimage[cdx]
            imhead = self.head

            # Get inverse variance image
            iminvvarim = self.invvarimage[cdx]

            # Maximum dimensions needed to verify that detection is actually on brick
            imbrickshape = imbrick.shape
            ymax = imbrickshape[0]
            xmax = imbrickshape[1]
            # Verify that detection center is actually on brick
            if not coords_on_image((RAdeg, decdeg), imhead, xmax, ymax):
                print('Candidate coordinates are not on brick', brick_name, (RAdeg, decdeg))

            #===================================================================
            # Find edges of thumbnail (center +/- half edge)
            # Crop at edges of brick if the thumbnail extends past (0, xmax, ymax)
            #===================================================================
            ystart = max(y_cen-self.half_edge, 0)
            yend = min(y_cen+self.half_edge + 1, ymax)
            xstart = max(x_cen-self.half_edge, 0)
            xend = min(x_cen+self.half_edge + 1, xmax)
            self.imthumb = imbrick[ystart:yend,xstart:xend]
            #===============================================================
            # Values used for padding image thumbnail regions outside of brick boundaries
            # Sigma squared will use 10000000
            #===============================================================
            med = median(subimbrick)
            tempim = self.imthumb.copy()
            invvarthumb = iminvvarim[ystart:yend,xstart:xend]
            #===============================================================
            # Create sigma squared image as inverse of inverse variance
            # First avoid division by zero
            #===============================================================
            invvarthumb[where(invvarthumb == 0)]= 0.000001
            sigmasqthumb = 1/invvarthumb
            # Pad regions falling off of brick.  This should not occur for bricks
            if ystart == 0:
                    # Thumbnail extends past lower edge
                    self.imthumb = lib.pad(self.imthumb, (( self.half_edge - y_cen,0), (0, 0)),
                                        'constant',  constant_values=((med, med), (med,med)))
                    tempim = lib.pad(tempim, (( self.half_edge - y_cen,0), (0, 0)),
                                        'constant',  constant_values=((-1000, -1000), (-1000,-1000)))
                    sigmasqthumb = lib.pad(sigmasqthumb, (( self.half_edge - y_cen,0), (0, 0)),
                                        'constant',  constant_values=((10000000, 10000000), (10000000,10000000)))
            if yend == ymax:
                # Thumbnail extends past upper edge
                self.imthumb = lib.pad(self.imthumb, ((0, y_cen + self.half_edge + 1 - ymax), (0, 0)),
                                    'constant', constant_values=((med, med), (med,med)))
                tempim = lib.pad(tempim, ((0, y_cen + self.half_edge + 1 - ymax), (0, 0)),
                                    'constant', constant_values=((-1000, -1000), (-1000,-1000)))
                sigmasqthumb = lib.pad(sigmasqthumb, ((0, y_cen + self.half_edge + 1 - ymax), (0, 0)),
                                    'constant', constant_values=((10000000, 10000000), (10000000,10000000)))
            if xstart == 0:
                # Thumbnail extends past left edge
                self.imthumb = lib.pad(self.imthumb, ((0, 0), (self.half_edge - x_cen, 0)),
                                    'constant', constant_values=((med, med), (med,med)))
                tempim = lib.pad(tempim, ((0, 0), (self.half_edge - x_cen, 0)),
                                    'constant', constant_values=((-1000, -1000), (-1000,-1000)))
                sigmasqthumb = lib.pad(sigmasqthumb, ((0, 0), (self.half_edge - x_cen, 0)),
                                    'constant', constant_values=((10000000, 10000000), (10000000,10000000)))
            if xend == xmax:
                # Thumbnail extends past right edge
                self.imthumb = lib.pad(self.imthumb, ((0, 0), (0, x_cen + self.half_edge + 1 - xmax)),
                                    'constant', constant_values=((med, med), (med,med)))
                tempim = lib.pad(tempim, ((0, 0), (0, x_cen + self.half_edge + 1 - xmax)),
                                    'constant', constant_values=((-1000, -1000), (-1000,-1000)))
                sigmasqthumb = lib.pad(sigmasqthumb, ((0, 0), (0, x_cen + self.half_edge + 1 - xmax)),
                                    'constant', constant_values=((10000000, 10000000), (10000000,10000000)))
            # Calculate fraction of thumbnail pixels outside brick borders
            padlocs =  where(tempim == -1000)
            padfx = round(len(padlocs[0])/self.thumbarea, 4)
            brickvals = array([brickband, padfx, med])
            # Save image thumbnail and pertinent associated information
            outfile = self.info_folder + 'Thumbnails/' + savename + '.npz'
            savez_compressed(outfile, tempim, self.brickkeys, brickvals)
            # Save sigma squared thumbnail and pertinent associated information
            outfile = self.info_folder + 'Thumbnails/' + savename + 'sigmasq.npz'
            savez_compressed(outfile, sigmasqthumb, self.brickkeys, brickvals)
            # Associate the thumbnail with the brick band
            self.sersic_filter_dict[brickband].append(savename)
        #=======================================================================
        # Number of thumbnails in each filter band
        # Should always be 1 for Legacy Survey bricks
        #=======================================================================
        if not DOFILTER:
            leng = len(self.sersic_filter_dict.get('g',[]))
            lenr = len(self.sersic_filter_dict.get('r',[]))
            lenz = len(self.sersic_filter_dict.get('z',[]))
            num_thumbs = leng + lenr + lenz
        else:
            num_thumbs = len(self.sersic_filter_dict.get(BANDTODO,[]))

        #=======================================================================
        # Arrays to hold the individual coadded thumbnails (g, r, z and 3-band stack)
        # Band images are stacked to create a final coadded thumbnail
        #=======================================================================
        coaddims = zeros((4,self.edge, self.edge), dtype = float32)
        # Create an image with all zeros to use as default if none available in band
        self.blank_im = zeros((self.edge, self.edge), dtype = float32)
        # Noise level for coadded image
        stdarr = [0., 0., 0., 0.]
        # Create empty lists to hold information needed for each band in Galfit

        imaddarr = []
        sigmasqimaddarr = []
        segmaparr = []
        sky_medarr = []
        sersic_magarr = []
        # Number of thumbnails in band
        num_thumbsarr = []

        # Number of images making up stack (1 for each band, 3 for 3-band stack)
        if not NO_Z:
            NCombinearr = [1, 1, 1, 3]
        else:
            NCombinearr = [1, 1, 1, 2]
        #===============================================================
        # Create and process coadded thumbnails.
        # Also retrieve individual thumbnails and parameters needed to
        #     create files
        #===============================================================
        for idx in range(len(self.filters)):
            # filter being processed will be g, r, z, or all
            self.coadd_band = self.filters[idx]
            # Create name for coadded image
            self.coaddname = 'COADD'  + self.RAcoadd + self.deccoadd + self.coadd_band
            #===================================================================
            # gf_result
            #    0: Galfit successfully ran
            #    1: Galfit crashed
            #    2: Galfit gave bad result
            #    3: Sersic screen failed
            #    4: No image in band
            #    5: Full stack failed either Sersic screen or Galfit
            #    6: No adequate thumbnails in band
            #    -1: Error somewhere
            #===================================================================
            gf_result = -1
            # Number of thumbnails in band
            num_thumbs = len(self.sersic_filter_dict[self.coadd_band])
            num_thumbsarr.append(num_thumbs)
            #===================================================================
            # Only process filters that have at least one associated image
            # There should always be exactly one when  using bricks
            #===================================================================
            if not DOFILTER:
                if (idx < 3) and (num_thumbs != 1):
                    print('line', 1200, ' in PipelineGalfit.  Wrong number of bricks in band.', self.coaddname, self.coadd_band  )
                    #exit()
                else:
                    if NO_Z:
                        numband2use = 2
                    else:
                        numband2use = 3
                    if (idx == 3) and (num_thumbs != numband2use):
                        print('line', 1204, ' in PipelineGalfit.  Wrong number of bricks in 3 band stack.' , self.coaddname )
                        #exit()
            if num_thumbs > 0:
                # Create the coadded thumbnails containing image and sigma squared.
                self.imadd, self.sigmasqimadd, self.padcoadd, brickkeys, thedata = self.create_coadds_corr()
                # Extract fitting parameters and calculate average values when needed
                [_band, padfx, brickmedian] = thedata
                if self.coadd_band  != 'all':
                    NCombine = 1
                    self.magzadd = self.nanomaggie_zpt
                else:
                    #===========================================================
                    # Stack of 3 bands
                    # zpt = 22.5 - 2.5 * log10(3) ~ 22.5 - 1.2 or
                    # zpt = 22.5 - 2.5 * log10(2) ~ 22.5 - 0.75
                    #===========================================================
                    if not NO_Z:
                        NCombine = 3
                        self.magzadd = self.nanomaggie_zpt - 1.2
                    else:
                        NCombine = 2
                        self.magzadd = self.nanomaggie_zpt - 0.75
                padfx, brickmedian = float(padfx), float(brickmedian)
                #if self.fitsersic:
                # Select value to use as FWHM in each band
                if self.coadd_band == 'g':
                    FWHMpsf = FWHMg
                elif self.coadd_band == 'r':
                    FWHMpsf = FWHMr
                elif self.coadd_band == 'z':
                    FWHMpsf = FWHMz
                else:
                    if self.coadd_band == 'all':
                        # For 3-band stack, use Max FWHM of bands since it correspond to worst
                        if not DOFILTER:
                            if not NO_Z:
                                FWHMpsf = max(array([FWHMg,FWHMr,FWHMz]))
                            else:
                                FWHMpsf = max(array([FWHMg,FWHMr]))
                #===============================================================
                # If using psf during Galfit, use band FWHMs to estimate seeing
                # Approximate psf as a Gaussian
                #===============================================================
                #===============================================================
                # Convert FWHM to Gaussian sigma
                # Create Gaussian kernel
                # Save for Galfit use
                #===============================================================
                mod_sig = FWHMpsf/self.sig2fwhm
                gaussian_2D_kernel = Gaussian2DKernel(mod_sig, mode = 'oversample')
                ker = gaussian_2D_kernel.array
                conv = ker.shape[1] + 2
                mysavefits(ker, self.GF_dir + 'GFpsf.fits', None)
                if DOPRINT:
                    print(self.sersic_filter_dict[self.coadd_band])
                    print(self.coadd_band)
                # Save coadded images and fitting parameters as compressed numpy file
                thedata = array([NCombine, self.magzadd, padfx, brickmedian])
                outfile = self.info_folder + 'Thumbnails/' + self.coaddname + '.npz'
                savez_compressed(outfile, self.imadd, self.padcoadd, brickkeys, thedata)
                outfile = self.info_folder + 'Thumbnails/' + self.coaddname + 'sigmasq.npz'
                savez_compressed(outfile, self.sigmasqimadd, self.padcoadd, brickkeys, thedata)

                if SHOWIMAGES:
                    showArray(self.padcoadd, newframe = 1, txt = self.coaddname)
                    showArray(self.imadd, newframe = 1, txt = self.coaddname)

                # Location of pixels that are not padded (inside of brick edges)
                self.addnonpadlocs = where(self.padcoadd == 0)
                if SHOWIMAGES:
                    showArray(self.padcoadd, newframe = 1, txt = 'padcoadd')
                    showArray(self.imadd, newframe = 1, txt = 'coadd im'+ str(self.galnum))

                # Get global background level and noise
                bkg = sep.Background(array(self.imadd))
                imstd = bkg.globalrms
                imback = bkg.globalback
                #===============================================================
                # Do initial Sersic fit of data
                # This will be used to detect objects overlying the candidate
                #     Global background level (imback) provided to prevent abrupt changes at padded regions
                #        when doing smoothing
                # Success values < 5 from curve fit indicates successful fit
                # If successful, continue processing
                #     fitall (float array): Model using results of Sersic fit
                #     thumbgauadd (float array): Gaussian smoothed image of non-padded portion of coadd
                #     segmap: Detection mask (segmentation map) from smoothed image
                #     nonmask_locs: Unmasked locations of Gaussian smoothed image (thumbgauadd)
                #     success (integer): Cause of scipy.optimize.leastsq termination
                #     sersic_AR: Axis ratio from fit
                #     sersic_mag: magnitude from fit
                #     sersic_Re: Effective radius from fit
                #===============================================================
                fitall, thumbgauadd, segmap, nonmask_locs, success, sersic_AR, sersic_mag, sersic_Re = self.Sersic_fit(idx, imback)
                if debug:
                    showArray(thumbgauadd, newframe = 1, txt = self.coadd_band)
                gf_result = 0
                success_arr[idx] = success
                if success < 5: # Successful Sersic screen
                    band_success[idx] = 1
                    # Get median background from aggressively masked, object-subtracted image
                    medianthumbgau = median(self.imadd[nonmask_locs])
                    if self.coadd_band == 'all':
                        medianthumbgauall = medianthumbgau
                    # Create zero-based residual image
                    imadd_res = self.imadd - fitall -medianthumbgau
                    #===========================================================
                    # Save results for SExtractor
                    # Run SExtractor on residual image
                    # Extract parameters needed for further masking:
                    #     numbers: SExtractor detection numbers
                    #     fwhms: SExtractor FWHMs
                    #     peaks: Peak amplitudes
                    #     xvals, yvals:  x, y locations on thumbnail
                    #     SEflux: Total fluxes
                    #     SEtheta: Rotation angles
                    #     SEmaj, SEmin: Major, minor axes
                    # Retrieve segmentation map
                    #===========================================================
                    mysavefits(imadd_res, self.GF_dir + self.gfname + '.fits', None)
                    self.gfse_rel.extract_sources()
                    numbers, fwhms,peaks, xvals, yvals, SEflux, SEtheta, SEmaj, SEmin= get_SEdata(self.GF_dir, self.gfname + '.cat')
                    galsegmap, _h = myopenfits(self.GF_dir + self.gfname + '_det.fits')
                    if SHOWIMAGES:
                        showArray(galsegmap, newframe = 1,  txt = 'galsegmap pre' + str(idx)+ str(self.galnum))
                    #===========================================================
                    # Get unique ID number of detections in footprint of candidate
                    # ctrgal_locs: Locations of candidate in smoothed image
                    # If object is large (>200 pix) or has large FWHMpsf (> 2.0 value permitted for band)
                    #     remove it from SExtractor segmentation image
                    #     This will result in it being left in the fitting locations provided to Galfit
                    # Otherwise mask it with a an elliptical mask representing the border of a
                    #     2D elliptical Gaussian profile having a threshold of 1/2 background noise
                    #===========================================================
                    galvals = set(galsegmap[self.ctrgal_locs].ravel())
                    if dotest:
                        print('self.coadd_band,  FWHM', self.coadd_band, FWHMpsf)
                    # Python set always includes 0, so need length > 1
                    if len(galvals) > 1:
                        # Find threshold for elliptical mask, if needed
                        sigstd = std(imadd_res[nonmask_locs])
                        sigthresh = sigstd/2.0
                        for galval in galvals:
                            if galval > 0:  # Ignore 0 value
                                # Find index of the detection numbers to extract other parameters
                                galindex  =flatnonzero(numbers == galval)
                                galindex = galindex[0]
                                # Location of object on detection map
                                galobjlocs =where(galsegmap == galval)
                                if DOPRINT:
                                    print('self.coadd_band,  fwhms[galindex], FWHM', self.coadd_band,  fwhms[galindex], FWHMpsf)
                                #===============================================
                                # Criteria for removing object mask or expanding it with elliptical mask
                                # If larger than criteria, remove it from masking which will include area in fit
                                # If smaller than criteria, generate elliptical mask
                                #===============================================
                                if ((len(galobjlocs[0]) >200) or (float(fwhms[galindex]) >(2.0 * FWHMpsf))):
                                    galsegmap[galobjlocs] = 0
                                else:
                                    # Get 2D Gaussian sigmas from SExtractor results
                                    amplitude = peaks[galindex]
                                    sig_maj, sig_min = get_sigmas(SEflux[galindex], SEmaj[galindex], SEmin[galindex], amplitude)
                                    if sig_maj > 0:
                                        # Get distance along major and minor axes where 2D Gaussian profile crosses threshold
                                        thresh_maj, thresh_min = get_gauss_thresh(sigthresh, amplitude, sig_maj, sig_min)
                                        if thresh_maj > 0:
                                            # Create an elliptical mask on the segmentation map
                                            simple_ellipse(galsegmap, xvals[galindex], yvals[galindex], thresh_maj,
                                                           thresh_min, SEtheta[galindex], galval)
                    mu0_thresh = self.band_mu0_thresh[self.coadd_band]
                    bkg = sep.Background(imadd_res)
                    # ignore numpy deprecation warning_
                    warnings.simplefilter('ignore')
                    _objects, segmapbright = sep.extract(imadd_res - bkg, 2 * mu0_thresh, segmentation_map = True,
                                   minarea = 10, deblend_cont = .005, filter_kernel = None)
                    warnings.simplefilter('always')
                    if SHOWIMAGES:
                        showArray(imadd_res, newframe = 1,  txt = 'imadd_res' + str(idx)+ str(self.galnum))
                        showArray(thumbgauadd, newframe = 1,  txt = 'coadd' + str(idx)+ str(self.galnum))
                        showArray(galsegmap, newframe = 1,  txt = 'galsegmap' + str(idx)+ str(self.galnum))
                        showArray(self.imadd - fitall -medianthumbgau, newframe = 1,  txt = 'imadd_res' + str(idx)+ str(self.galnum))
                    # Add residual detections from SExtractor to the original segmentation mask
                    segmap[::] += galsegmap
                    segmap[::] += segmapbright
                    if SHOWIMAGES:
                        showArray(segmap, newframe = 1,  txt = 'pre galfit')
                        showArray(self.imadd, newframe = 1, txt = 'imadd')
                    if dotest and DOPRINT:
                        print('imadd.shape', self.imadd.shape)
                    if dotest:
                        print(self.magzadd, self.pixscl, self.edge, medianthumbgau)
                    # Set boundaries for initial Galfit estimates which are based on the screening Sersic results
                    if sersic_AR < 0.2:
                        sersic_AR = 0.2
                    if sersic_Re > 150:
                        sersic_Re = 150
                    if debug:
                        showArray(segmap, newframe = 1, txt = self.coadd_band + ' galfit segmap')
                    #===========================================================
                    # Save pertinent variables for each band in an array
                    # This has to be done because the 3-band stack is processed before the individual bands
                    #    but requires information from those bands before processing
                    #===========================================================
                    imaddarr.append(self.imadd)
                    sigmasqimaddarr.append(self.sigmasqimadd)
                    segmaparr.append(segmap)
                    sky_medarr.append(medianthumbgau)
                    sersic_magarr.append(sersic_mag)
                else:
                    # Unsuccessful screening Sersic fit.  Set results to 0
                    imaddarr.append(self.blank_im)
                    segmaparr.append(self.blank_im)
                    sky_medarr.append(0)
                    sersic_magarr.append(0)

                    sigmasqimaddarr.append(self.blank_im)
                    im_model = self.blank_im.copy()
                    medianthumbgau = 0
                    mu_model = 0
                    gf_result = 3
                #===============================================================
                # Standard deviation of the object-subtracted image, which will contain the galaxy,
                #     is used to set limits on the scale of the saved thumbnails
                #===============================================================
                stdarr[idx] = imstd
                # Save coadded images since models for bands will be done after 3-band stack
                coaddims[idx,:,:] = self.imadd
                if idx < 3:
                    coaddfilter =  self.coaddname[-1]
                else:
                    coaddfilter = 'all'
            if (num_thumbs == 0):
                print(1377, 'pipelineGalfit. No image in band.  Should not have reached here.')
                print()
                print()
                print()
                print()
                #exit()
                # No image in band.
                if num_thumbs == 0:
                    gf_result = 4
                else:
                    # Images in band but no adequate thumbnails
                    gf_result = 6
                #Set default values
                success = 6
                success_arr[idx] = success
                sky_medarr.append(0)
                mu_model, REkpc, NCombine,  = -1000, 0, 0
                coaddfilter = self.filters[idx]
                self.coaddname = "No image in " + coaddfilter
                im_model = self.blank_im.copy()
                segmap = self.blank_im.copy()
                self.imadd = self.blank_im.copy()
                imaddarr.append(self.blank_im)

                sigmasqimaddarr.append(self.blank_im)
                segmaparr.append(self.blank_im)
            #===================================================================
            #  Start by analyzing 3-band stack to get structural parameters
            #    These include r_e, center location, b/a, n, and theta
            #    These values will not be allowed to vary when estimating photometry
            #        parameters for individual bands
            #===================================================================
            if self.coadd_band == 'all' and success < 5:
                gf_result = 0
                # Set number of images in stack and magnitude zeropoint for 3-band or 2-band stack
                if not NO_Z:
                    Ncombine = 3
                    magzpt = 21.3
                else:
                    Ncombine = 2
                    magzpt = 21.75
                #=======================================================================
                # Runs Galfit on thumbnail
                # Input parameters:
                #     osplatform: System platform (Mac, HPC, or Unix)
                #     GF_dir: Folder holding Galfit data, images and executables (unique for each HPC job)
                #     Ncombine (integer): Number of coadded thumbnails
                #     imadd (float array): Thumbnail containing image to process
                #     segmap (integer array): Mask for regions not fitted (Values > 0 are not fitted)
                #     magzadd (float): magnitude zero point
                #     pixscl (float): Thumbnail pixel scale in arcsec
                #     edge_len (integer): Length of thumbnail edge
                #     sky_med (float):  Estimate of sky level
                #     sersic_mag (float):  Estimate of magnitude
                #     sersic_AR( float):  Estimate of sky axis ratio
                #     sersic_Re (float):  Estimate of effective radius
                #     PAest (float):  Estimate of position angle
                #     fitsersic (Boolean): Allow Sersic index to vary
                #     usefixed (Boolean): Fix structural parameters derived for 3-band stack while fitting
                #     half_edge (integer): Length of thumbnail (edge - 1)/2.  This will be thumbnail center.
                #     conv (integer): Size of convolution box if using psf
                #     sigmasqim (float array): Thumbnail sigma squared image to process
                #=======================================================================
                getgalfit(self.osplatform, self.GF_dir, NCombine,
                          self.imadd, segmap, magzpt, self.pixscl, edge_len=self.edge, sky_med=medianthumbgauall,
                          sersic_mag=sersic_mag, sersic_AR=sersic_AR, sersic_Re=sersic_Re, PAest=0, fitsersic = self.fitsersic,
                          usefixed=False, half_edge=self.half_edge, conv=conv, sigmasqim = self.sigmasqimadd)
                #=========================================================
                # Extract Galfit information
                #    mu_model (float): central surface brightness
                #    Rekpc (float): Effective radius in kpc at distance if Coma
                #    im_model (float array): Model of fit produced by Galfit
                #    fitinfo (tuple): Fit results produced by Galfit
                #    lognum (integer): Log number produced by Galfit.  This should be unique.
                #=========================================================
                mu_model, REkpc, im_model, fitinfo, lognum = getmodel(self.GF_dir, self.magzadd, pixscl=self.pixscl, dosky=1, getmodelmax=True)
                if DOPRINT:
                    print(self.coaddname, REkpc)
                #===========================================================
                # Check to see if Galfit log number has already been used.
                # If it has, then Galfit crashed during processing
                #===========================================================
                if lognum in self.lognums:
                    if dotest and SHOWIMAGES:
                        showArray(im_model, newframe = 1)
                    # Designate Galfit crash
                    gf_result = 1
                    mu_model, REkpc = 0, 0
                else:
                    self.lognums.add(lognum)
                    # If Galfit unable to fit, then REkpc will be zero
                    if REkpc == 0:
                        # Designate bad Galfit results
                        gf_result = 2
                if dotest and SHOWIMAGES:
                    showArray(imadd_res, newframe = 1, txt = 'galft thumbgau')
                    showArray(segmap, newframe = 1, txt = 'galft segmap')
                    showArray(im_model, newframe = 1, txt = 'galft model')
                # Convert the galfit results from strings to floats
                fitinfo = [round(float(x),3) for x in fitinfo]
                if gf_result == 0:
                    #===========================================================
                    # Galfit success
                    # Structural results from 3-band stack.  Photometry results ignored
                    #===========================================================
                    (XC_all, YC_all,Re_all, Reerr_all, _mag_all, _magerr_all, AR_all, ARerr_all,
                        PA_all, PAerr_all, _sky_all, _skyerr_all, chi2nu_all, modmax_all, n_all, nerr_all ) = fitinfo
                # Round results for saving
                mu0_arr[idx] = round(mu_model,3)
                # Use higher resolution to allow better binning for completeness studies
                AR_all = round(AR_all, 3)
                ARerr_all = round(ARerr_all, 3)
                REarcsec_all= round(float(Re_all) * float(self.pixscl),2)
                REarcsecerr_all = round(float(Reerr_all) * float(self.pixscl),2)
                # Standard deviation is used to set scale limits on saved thumbnails
                stdarr[idx] = imstd
                # Set unsubtracted  and object-subtracted thumbnails to zero background
                skyval = medianthumbgau
                self.imadd = self.imadd - skyval
                if gf_result == 0:
                    modmax_arr[idx] =  modmax_all - skyval
                # Fill data arrays
                coaddims[idx,:,:] = self.imadd
                gf_result_arr[idx] = gf_result
                gf_result_all = gf_result
                chi2nu_arr[idx] = chi2nu_all
                if SHOWIMAGES:
                    showArray(im_model, newframe = 1)
                    showArray(self.imadd, newframe = 1,  txt = 'imadd')
            else:
                # Sersic screen was not successful.  Save results as default values
                [XC_all, YC_all,Re_all, Reerr_all, _mag_all, _magerr_all, AR_all, ARerr_all,
                        PA_all, PAerr_all, _sky_all, _skyerr_all, chi2nu_all, modmax_all, n_all, nerr_all] = [0.0] * 16
                REarcsec_all, REarcsecerr_all = 0.0, 0.0
                coaddims[idx,:,:] = self.imadd
                gf_result_all = gf_result
                gf_result_arr[idx] = gf_result
                modmax_arr[idx] = modmax_all
        for idx in range(len(self.filters) - 1):
            #===================================================================
            # Run Galfit on the 3 bands using the structural parameters (r_e, theta, b/a, n, center)
            #    from the 3-band stack
            # Only proceed if Galfit ran successfully on the 3 band stack
            #===================================================================
            if (gf_result_all == 0) and (success_arr[idx] < 5):
                gf_result = 0
                # Set number of images in stack and magnitude zeropoint for individual filters
                Ncombine = 1
                magzpt = self.nanomaggie_zpt
                #=======================================================================
                # Runs Galfit on thumbnail
                # Input parameters:
                #     osplatform: System platform (Mac, HPC, or Unix)
                #     GF_dir: Folder holding Galfit data, images and executables (unique for each HPC job)
                #     Ncombine (integer): Number of coadded thumbnails
                #     imadd (float array): Thumbnail containing image to process
                #     segmap (integer array): Mask for regions not fitted (Values > 0 are not fitted)
                #     magzpt (float): magnitude zero point for nanomaggies
                #     pixscl (float): Thumbnail pixel scale in arcsec/pixel
                #     edge_len (integer): Length of thumbnail edge
                #     sky_med (float):  Estimate of sky level
                #     xest (float):  Estimate of x center
                #     yest (float):  Estimate of y center
                #     PAest (float):  Position angle  (fixed from 3-band stack)
                #     nest (float):  Sersic index  (fixed from 3-band stack)
                #     sersic_AR (float):  Axis ratio  (fixed from 3-band stack)
                #     sersic_mag (float):  Estimate of magnitude
                #     sersic_Re (float):  Effective radius (fixed from 3-band stack)
                #     fitsersic (Boolean): Allow Sersic index to to vary
                #     usefixed (Boolean): Fix structural parameters derived for 3-band stack while fitting
                #     half_edge (integer): Length of thumbnail (edge - 1)/2.  This will be thumbnail center.
                #     conv (integer): Size of convolution box for psf kernel
                #     sigmasqim Thumbnail sigma squared image to process
                # In this case, usefixed is set to True so structural parameters (r_e, b/a, theta, n) will
                #      be fixed at the results obtained from the 3-band stack
                #=======================================================================
                getgalfit(self.osplatform, self.GF_dir, Ncombine, imaddarr[idx], segmaparr[idx], magzpt, self.pixscl,
                                      edge_len=self.edge, sky_med=sky_medarr[idx], xest=XC_all, yest=YC_all,  PAest=PA_all, nest = n_all,
                                      sersic_AR=sersic_AR, sersic_mag=sersic_mag, sersic_Re=Re_all, fitsersic = self.fitsersic,
                                      usefixed=True, half_edge=self.half_edge, conv=conv, sigmasqim = sigmasqimaddarr[idx])
                #=========================================================
                # Extract Galfit information
                #    mu_model (float): central surface brightness calculated from model
                #    im_model (float array): Model of fit produced by Galfit
                #    fitinfo (tuple): Fit results produced by Galfit
                #    lognum (integer): Log number produced by Galfit.  This should be unique.
                #=========================================================
                mu_model, im_model, fitinfo, lognum = getmodel(self.GF_dir, magzpt, pixscl=self.pixscl, dosky=1, getmodelmax=True, usefixed = True)
                if isnan(mu_model):
                    return
                if DOPRINT:
                    print(self.coaddname, REkpc)
                #===========================================================
                # Check to see if Galfit log number has already been used.
                # If it has, then Galfit crashed during processing
                #===========================================================
                if lognum in self.lognums:
                    if dotest and SHOWIMAGES:
                        showArray(im_model, newframe = 1)
                    # Designate Galfit crash
                    gf_result = 1
                    mu_model, REkpc = 0, 0
                else:
                    self.lognums.add(lognum)
                    # If Galfit unable to fit, then REkpc will be zero
                    if REkpc == 0:
                        gf_result = 2
                if dotest and SHOWIMAGES:
                    showArray(imadd_res, newframe = 1, txt = 'galft thumbgau')
                    showArray(segmap, newframe = 1, txt = 'galft segmap')
                    showArray(im_model, newframe = 1, txt = 'galft model')
                # Convert the galfit results from strings to floats
                fitinfo = [round(float(x),3) for x in fitinfo]
                if gf_result == 0:
                    (mag_arr[idx], magerr_arr[idx], sky_arr[idx], skyerr_arr[idx], chi2nu_arr[idx], modmax_arr[idx]) = fitinfo
                mu0_arr[idx] = round(mu_model, 3)
            else:
                #===============================================================
                # # If bad Galfit result from 3-band stack or no image in band
                #    or bad Sersic screen for band, set results to default values
                # gf_result
                #    3: Sersic screen failed
                #    4: No image in band
                #    5: Galfit failed
                #    6: No adequate thumbnails in band
                #===============================================================
                if num_thumbsarr[idx] == 0:
                    gf_result = 4
                elif ncombine_arr[idx] ==0:
                    gf_result = 6
                elif gf_result_all > 0:
                    gf_result = 5
                elif success_arr[idx] > 4:
                    gf_result = 3
                [mag_arr[idx], magerr_arr[idx], sky_arr[idx], skyerr_arr[idx], chi2nu_arr[idx], modmax_arr[idx]] =   [0] * 6
            gf_result_arr[idx] = gf_result
            if NCombinearr[idx] > 0:
                imstdnan = std(imaddarr[idx])
                if isnan(imstdnan):
                    return
                skyval = sky_medarr[idx]
                self.imadd = self.imadd - skyval
                if gf_result == 0:
                    modmax_arr[idx] =  modmax_arr[idx] - skyval
                stdarr[idx] = imstd
                coaddims[idx,:,:] = imaddarr[idx]- skyval
        #  Calculate uncertainties mu0 and its uncertainties for all three bands
        for mudx in range(3):
            mudata = calc_mu_uncertainties(mag_arr[mudx], magerr_arr[mudx], REarcsec_all, REarcsecerr_all, AR_all, ARerr_all, n_all, nerr_all)
            mu0_arr[mudx] = mudata[0]
            mu0err_arr[mudx] = mudata[1]
        # Create multi-extension frame containing all 4 thumbnails (3 bands and 3-band stack) with pertinent results in header
        self.saveMEF(coaddims, stdarr, ncombine_arr, modmax_arr, mu0_arr, mag_arr, [REarcsec_all,REarcsec_all,REarcsec_all,REarcsec_all], gf_result_arr)
        if save_masks:
            self.fileloc2 = self.Coadd_dir + self.MEFname + 'mask.fits'
            self.saveMasks(segmaparr, stdarr, ncombine_arr, modmax_arr, mu0_arr, mag_arr, [REarcsec_all,REarcsec_all,REarcsec_all,REarcsec_all], gf_result_arr)
        # Create time stamp for data
        time_local = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # Create list for saving results in text files for later import into database
        data_list =  ncombine_arr.astype(str)
        data_list =  concatenate((coadd_results, data_list, array([XC_all, YC_all, REarcsec_all, REarcsecerr_all,
                                AR_all, ARerr_all, PA_all, PAerr_all, n_all, nerr_all]), mag_arr[:3], magerr_arr[:3],
                                mu0_arr[:3], mu0err_arr[:3], sky_arr[:3], skyerr_arr[:3], chi2nu_arr[:3], chi2nu_arr[3],
                                gf_result_arr[:3], gf_result_arr[3], '0', time_local), axis=None)#, [str(AR_all), str(ARerr_all)]])#,
        if not savedata:
            print(data_list)
        if savedata:
            # Save Galfit results as csv
            with open(self.save_folder + self.gf_file_name +str(self.core_num) + '.csv', 'a') as Sorterfile:
                Sorterwriter = csv.writer(Sorterfile)
                Sorterwriter.writerow(data_list)
        # Clear Galfit data and reinitialize log number
        self.clear_gf_dir()
        self.lognums = set([0])
        if DOPRINT:
            for idx in range(len(mag_arr)):
                print(mag_arr[idx])

    def saveMEF(self, coaddims, stdarr, ncombine_arr, modmax_arr, mu0_arr, mag_arr, Re_arr, gf_result_arr):
        """
        Purpose:
            Save thumbnails in all three band as well as 3-band stack as Multi-frame fits file
            Pertinent Galfit results will be in the header
        Input parameters (arrays containing information for all coadded 3 bands plus full stack):
            coaddims (3D float array):  unsubtracted thumbnails
            stdarr (float list): standard deviations
            ncombine_arr (integer list): Number of observations coadded for band
            modmax_arr (float list): Peak amplitude of the Galfit model
            mu0_arr (float list): central surface brightness
            mag_arr (float list): magnitude
            Re_arr (float list): r_e (arcsec)
            gf_result_arr (integer list): Galfit result.  Success = 0
        Output:
            Multi-frame fits
        """
        if not DOFILTER:
            filters = ['g', 'r', 'z', 'all']
        else:
            filters = [BANDTODO, BANDTODO, BANDTODO, 'all']
        # Header keys
        keys = ['FILTER', 'STD', 'NCOMBINE', 'MODMAX', 'MU0', 'MAG', 'Re', 'GF_RES']
        # Header values
        vals =  [filters, stdarr, ncombine_arr, modmax_arr, mu0_arr, mag_arr, Re_arr, gf_result_arr]
        # Create the multi-frame fits file
        new_hdul = fits.HDUList()
        for idx in range(4):
            newim = fits.ImageHDU(coaddims[idx])
            # Create the header for each frame
            head = newim.header
            for jdx in range(len(keys)):
                head[keys[jdx]] = vals[jdx][idx]
            new_hdul.append(newim)
        new_hdul.writeto(self.fileloc, overwrite=True)

    def saveMasks(self, masks, stdarr, ncombine_arr, modmax_arr, mu0_arr, mag_arr, Re_arr, gf_result_arr):
        """
        Purpose:
            Save thumbnails in all three band as well as 3-band stack as Multi-frame fits file
            Pertinent Galfit results will be in the header
        Input parameters (arrays containing information for all coadded 3 bands plus full stack):
            coaddims (3D float array):  untracted thumbnails
            stdarr (float list): standard deviations
            ncombine_arr (integer list): Number of observations coadded for band
            modmax_arr (float list): Peak amplitude of the Galfit model
            mu0_arr (float list): central surface brightness
            mag_arr (float list): magnitude
            Re_arr (float list): r_e (arcsec)
            gf_result_arr (integer list): Galfit result.  Success = 0
        Output:
            Multi-frame fits
        """
        if not DOFILTER:
            filters = ['g', 'r', 'z', 'all']
        else:
            filters = [BANDTODO, BANDTODO, BANDTODO, 'all']
        # Header keys
        keys = ['FILTER', 'STD', 'NCOMBINE', 'MODMAX', 'MU0', 'MAG', 'Re', 'GF_RES']
        # Header values
        vals =  [filters, stdarr, ncombine_arr, modmax_arr, mu0_arr, mag_arr, Re_arr, gf_result_arr]
        # Create the multi-frame fits file
        new_hdul = fits.HDUList()
        for idx in range(4):
            newim = fits.ImageHDU(masks[idx])
            # Create the header for each frame
            head = newim.header
            for jdx in range(len(keys)):
                head[keys[jdx]] = vals[jdx][idx]
            new_hdul.append(newim)
        new_hdul.writeto(self.fileloc2, overwrite=True)



    def Sersic_fit(self, idx, background):
        '''
        Purpose:
            Performs an initial Sersic exponential (n = 1) screen on thumbnail
        Input:
            idx (integer): Index of band that is being analyzed
                Corresponds to bands (g, r, z, and 3-band stack)
            background (float):  Global image background level
        Outputs:
            fitall (float array): Model using results of Sersic fit
            thumbgauadd (float array): Gaussian smoothed image of non-padded portion of coadd
            segmap: Detection mask from smoothed image
            nonmask_locs: Unmasked locations of Gaussian smoothed image (thumbgauadd)
            success (integer): Cause of scipy.optimize.leastsq termination
            sersic_AR: Axis ratio from fit
            sersic_mag: magnitude from fit
            sersic_Re: Effective radius from fit
        '''
        # Conversion from Sersic major axis to effective radius
        #if self.coadd_band == 'all':
            #print()
        maj_axis2Re_pix = 1.67835
        if dotest and DOPRINT:
            print('magz ', self.magzadd)
        if dotest:
            print('idx, self.coadd_band', idx, self.coadd_band)
        #=======================================================================
        # Calculate the maximum value in ADU required to meet a mu0 threshold
        #    The threshold varies with band (band_mu0_dict)
        #=======================================================================
        mu0_thresh = self.band_mu0_thresh[self.coadd_band]
        #=======================================================================
        # Check to see if image has been padded
        #     addnonpadlocs: Locations of pixels that are not padded (inside of brick edges)
        # thumbgauadd: Gaussian smoothed image of non-padded portion of coadd
        #=======================================================================
        sigmas = 3
        if len(self.addnonpadlocs[0])< self.thumbarea:
            # Make copy of image so original isn't changed
            imaddcopy = self.imadd.copy()
            # Set padded regions to global background to avoid abrupt changes during blurring
            imaddcopy[self.padcoadd != 0] = background
            # Create smoothed image and set padded regions to 0
            thumbgauadd = GaussianBlur(imaddcopy, (0,0), sigmaX=sigmas, sigmaY=sigmas, borderType=BORDER_REFLECT )
            thumbgauadd[self.padcoadd != 0] = 0
        else:
            thumbgauadd = GaussianBlur(self.imadd, (0,0), sigmaX=sigmas, sigmaY=sigmas, borderType=BORDER_REFLECT )
        if dotest:
            showArray(thumbgauadd, newframe = 1,  txt = 'coadd' + str(idx)+ str(self.galnum))
        # Convert mask of padded region into format usable by SEP
        coaddmask = self.padcoadd.astype(uint8)
        # Get background parameters
        bkg = sep.Background(array(thumbgauadd), mask = coaddmask)
        if debug:
            showArray(bkg.back(), newframe = 1, txt = self.coadd_band)
        # Background is sky estimate if sky not a free parameter in Sersic fit
        # tract background and create segmentation masks
        thumbgauadd = thumbgauadd - bkg
        stdthumbgau = bkg.globalrms
        if dotest and DOPRINT:
            print('bkg.globalback ', bkg.globalback)
            print('stdthumbgau', stdthumbgau)
        #=======================================================================
        # Create two detection masks using SEP
        #    SEP requires zero-based image, so background is subtracted before processing
        # segmap: Detection mask from smoothed image.  This is used to detect extended objects
        #    such as the UDG candidate
        # segmap2: Detection mask from unsubtracted thumbnail.
        #    Used to detect unresolved objects
        #=======================================================================
        warnings.simplefilter('ignore')
        #if self.coadd_band  == 'all':
        _objects, segmap3 = sep.extract(thumbgauadd, 1.5 * mu0_thresh, segmentation_map = True,
                               mask  = coaddmask, minarea = 10, deblend_cont = .005, filter_kernel = None)
        cntobj = segmap3[self.half_edge, self.half_edge]
        if cntobj != 0:
            segmap3[where(segmap3 == cntobj)] = 0
        segmap3[where(segmap3 !=0)] = 100
        _objects, segmap = sep.extract(thumbgauadd, 2, err=stdthumbgau , segmentation_map = True,
                                   mask  = coaddmask + segmap3, minarea = 10, deblend_cont = .02, deblend_nthresh = 20, filter_kernel = None, clean_param=3 )
        #_objects, segmap = sep.extract(thumbgauadd, 2, err=stdthumbgau , segmentation_map = True,
        #                            mask  = coaddmask + segmap3, minarea = 10, deblend_cont = .02, deblend_nthresh = 20, filter_kernel = None, clean_param=3 )
        imadd_diff = self.imadd - bkg.globalback
        _objects, segmap2 = sep.extract(imadd_diff, 10 * mu0_thresh, segmentation_map = True,
                                   mask  = coaddmask + segmap3, minarea = 10, deblend_cont = .005, filter_kernel = None)


        warnings.simplefilter('always')
        #=======================================================================
        # Mask peripheral objects brighter than 3 * mu0_thresh
        # Region susceptible to masking (outside of radius) has values of 1.0
        # Interior of circle has values of 0.0
        #=======================================================================
        segmap[thumbgauadd * self.peripheral_mask > 3 * mu0_thresh] = 100
        if debug:
            print(self.coadd_band, stdthumbgau, bkg.globalback)
            showArray(segmap, newframe = 1, txt = self.coadd_band +  ' segmap')
            showArray(segmap2, newframe = 1, txt = self.coadd_band +  ' segmap2')
        if SHOWIMAGES:
            showArray(thumbgauadd, newframe = 1,  txt = 'coadd segmap' + str(idx)+ str(self.galnum))
            showArray(segmap, newframe = 1,  txt = 'coadd segmap' + str(idx)+ str(self.galnum))
            showArray(imadd_diff, newframe = 1,  txt = 'coadd segmap' + str(idx)+ str(self.galnum))
            showArray(segmap2, newframe = 1,  txt = 'coaddsegmap2 ' + str(idx)+ str(self.galnum))
        # Flag negative outliers and padded areas
        segmap[thumbgauadd < -2 * stdthumbgau] = 100
        segmap[self.padcoadd != 0] = 100
        # Unmasked regions
        nonmask_locs = segmap == 0
        if len(nonzero(nonmask_locs)[0]) ==0:
            # if no points to analyze, bail out
            fitall = self.blank_im.copy()
            sersic_AR, sersic_mag, sersic_Re = 0.0, 0.0, 0.0
            success = 5
            return fitall, thumbgauadd, segmap, nonmask_locs, success, sersic_AR, sersic_mag, sersic_Re
        # Add name of band coadded to 'all' which will stack coadded thumbnails
        if self.coadd_band  != 'all':
            if not NO_Z:
                self.sersic_filter_dict['all'] = self.sersic_filter_dict['all'] + [self.coaddname]
            else:
                if self.coadd_band != 'z':
                    self.sersic_filter_dict['all'] = self.sersic_filter_dict['all'] + [self.coaddname]
        if SHOWIMAGES:
            showArray(segmap, newframe = 1,  txt = 'coadd' + str(idx)+ str(self.galnum)+ 'test Sersic pre')
            showArray(thumbgauadd, newframe = 1,  txt = 'coadd' + str(idx)+ str(self.galnum))
        obj_at_center = True
        #=======================================================================
        # Attempt to find candidate which should be at center of segmentation image
        # if not at center, see if it is within 2 pixels of center
        # In this case, arbitrarily pick maximum detection number
        # If no candidate is found, set fitting parameters to 0 and return
        #=======================================================================
        theobjsep = segmap[self.half_edge, self.half_edge]
        if theobjsep == 0:
            # Object not at center
            testvals = segmap[self.half_edge - 2:self.half_edge + 3, self.half_edge - 2:self.half_edge + 3]
            theobjsep = npmax(testvals)
            if theobjsep == 0:
                if DOPRINT:
                    print('No object at center')
                obj_at_center = False
        if obj_at_center:
            self.ctrgal_locs = segmap == theobjsep
            #===================================================================
            # Unmask central object in masked from smoothed image
            # Find object numbers of objects in footprint of central object in
            #    unsubtracted image
            # If they are large (>100 pix), unmask them
            #    They may represent the target
            # Results of above will unmask large central object in both detection masks
            #===================================================================
            segmap[self.ctrgal_locs] = 0
            # galvals = set(segmap2[self.ctrgal_locs].ravel())
            # if len(galvals) > 1:
            #     for galval in galvals:
            #         if galval > 0:
            #             galobjlocs =where(segmap2 == galval)
            #             if len(galobjlocs[0]) >100:
            #                 segmap2[galobjlocs] = 0
        else:
            #===================================================================
            # No object at center
            # Do not run Sersic screen and return default values
            #===================================================================
            self.ctrgal_locs = []
            fitall = self.blank_im.copy()
            sersic_AR, sersic_mag, sersic_Re = 0.0, 0.0, 0.0
            success = 5
            return fitall, thumbgauadd, segmap, nonmask_locs, success, sersic_AR, sersic_mag, sersic_Re
        # Mask for Sersic fit will contain both detection images
        segmapSersic = segmap + segmap2 + segmap3
        #exit()
        if SHOWIMAGES:
            showArray(segmap, newframe = 1, txt = "Sersic post")
        if debug:
            showArray(imadd_diff, newframe = 1, txt = self.coadd_band)
            showArray(segmapSersic, newframe = 1, txt = self.coadd_band)
        #===================================================================
        # Ignore underflow warnings caused by large negative exponentials in
        #    Sersic function
        #===================================================================
        if CHECK_WARNINGS:
            seterr(under='ignore')
        #=======================================================================
        # Screening Sersic exponential (n = 1) fit of unsubtracted thumbnail
        #    segmapSersic: Mask of bright objects that might affect fit
        #    return_raw: Return only results of fit (no further calculations)
        # success: Cause of scipy.optimize.leastsq termination
        #    Success < 5 indicates successful fit
        # params: Results of fit
        #    params[0], params[1]: center_x, center_y
        #    params[2], params[3]: semi-axes
        #    params[4]: position angle (theta)
        #    params[5]: height
        #    params[6]: offset
        #=======================================================================
        success, params = mySercicFit(self.brick, self.galnum, imadd_diff, segmapSersic, return_raw=True, sky=0)
        if CHECK_WARNINGS:
            seterr(all='raise')
        if success < 5:
            #===================================================================
            # r1, r2 are ellipse axes from fit
            # Re:  Effective radius
            # AR: Axis ratio
            # if either is < 0, set axis ratio, effective radius to default values
            #===================================================================
            r1 = (params[2])
            r2 = (params[3])
            if (r1 <0) or (r2 < 0):
                sersic_AR = 0.6
                sersic_Re = 15.0
            else:
                # Calculate axis ratio and effective radius
                sersic_AR = min(r1/r2, r2/r1)
                sersic_Re = maj_axis2Re_pix * max(r1,r2)
            #===================================================================
            # Ignore underflow warnings caused by large negative exponentials in
            #    Sersic function
            #===================================================================
            if CHECK_WARNINGS:
                seterr(under='ignore')
            # Create the fit model
            fitall = tofitfuncexplmfitlinls(params, [],(self.Xin, self.Yin))
            if CHECK_WARNINGS:
                seterr(all='raise')
            #===================================================================
            # Check if the Sersic mag will give a negative number
            #     If so, assign arbitrary value of 20
            #     Otherwise, calculate mag
            #===================================================================
            fitall_sum =  sum(fitall)
            bg = params[6]
            diffsum = sum(fitall-bg)
            # Prevent crashes for log of non-positive number
            if diffsum <=0:#fitall_sum <0:
                sersic_mag = 20.0
            else:
                #sersic_mag = self.magzadd - 2.5* log10(sum(fitall))
                try:
                    sersic_mag = self.magzadd - 2.5* log10(diffsum)
                except:
                    print('diffsum', diffsum, self.brick, self.galnum)
                    sersic_mag = self.magzadd - 2.5* log10(diffsum)
            # Reshape model to 2D
            fitall = array(fitall).reshape(self.imshape)
            if dotest:
                print(idx, params, sersic_AR, fitall_sum, sersic_mag, self.magzadd)
                showArray(fitall)
        else:
            # If unsuccessful fit, assign zeros to all parameters
            fitall = self.blank_im.copy()
            sersic_AR, sersic_mag, sersic_Re = 0.0, 0.0, 0.0
        if dotest and SHOWIMAGES:
                showArray(thumbgauadd, newframe = 1, txt = 'thumbgauadd')
                showArray(self.imadd, newframe = 1, txt = 'self.imadd')
                showArray(fitall, newframe = 1, txt = 'fitall thumbgauadd')
                showArray(self.imadd - fitall- bkg.globalback, newframe = 1, txt = 'self.imadd - fitall')
        #return fitall, thumbgauadd, segmap, nonmask_locs, success, sersic_AR, sersic_mag, sersic_Re
        return fitall, thumbgauadd, segmapSersic, nonmask_locs, success, sersic_AR, sersic_mag, sersic_Re

    def create_coadds_corr(self):
        """
        Purpose:
            Creates coadded thumbnails with individual thumbnails weighted to
                account for zero point
        Outputs:
            imadd (float array):  Thumbnail of coadded unsubtracted images
            padcoadd(numpy integer array):  Thumbnail mask showing padded portions of
                coadded images
            brickkeys (string array):  Names of parameters needed for fitting
            paramsout:  Values of parameters needed for fitting
        """
        # Array to show regions that were padded on individual thumbnails
        padcoadd = zeros(self.imshape, dtype = int16)
        # Create arrays to hold output image and sigma squared thumbnails
        imadd = zeros(self.imshape, dtype = float32)
        sigmasqimadd = zeros(self.imshape, dtype = float32)
        # Initialize output parameters for fraction padded and median value
        padfxout = 0
        medout = 0
        # Bands will have 1 entry.  3 image stack will have 3
        thumbnail_data = self.sersic_filter_dict[self.coadd_band]
        #=======================================================================
        # Number of images in stack
        # Should be 1 for the 3 bands and 3 for for the 3-band stack
        #=======================================================================
        data_len = len(thumbnail_data)
        if self.coadd_band == 'all':
            all_bands = True
        else:
            all_bands = False
        if DOPRINT:
            print(thumbnail_data)
        for jdx in range(data_len):
            # Name of thumbnail file to load
            thumbnail_filename = thumbnail_data[jdx]
            if all_bands:
                #===============================================================
                # 3-band stack will be created by adding coadditions of the 3 bands
                # These will have already been created
                #===============================================================
                im, padmap, params = self.loadCoaddcompressed(self.info_folder + 'Thumbnails/' + thumbnail_filename + '.npz')
                brickband, padfx, med = params
                sigmasqim, _padmap, params  = self.loadCoaddcompressed(self.info_folder + 'Thumbnails/' + thumbnail_filename + 'sigmasq.npz')
                # Fraction of thumbnail padded.  Not pertinent for 3-band stack
                padfx = 0
                if DOPRINT:
                    print('allbands', params)
            # Retrieve information from each thumbnail file
            else:
                if DOPRINT:
                    print(thumbnail_filename)
                #===============================================================
                # Load image, mask of padded regions and parameters needed for fitting
                #    edge (integer): length of thumbnail edge in compressed file
                #    padfx (float): Fraction of thumbnail padded because of extension
                #        over brick edge
                #===============================================================
                im, padmap, params  = self.loadSersiccompressed(self.info_folder + 'Thumbnails/' + thumbnail_filename + '.npz')
                brickband, padfx, med = params
                sigmasqim, _padmap, params  = self.loadSersiccompressed(self.info_folder + 'Thumbnails/' + thumbnail_filename + 'sigmasq.npz')
            #===================================================================
            # Continue if < 25% of compressed thumbnail was padded
            #    Should always be the case with bricks
            #===================================================================
            padfxout = float(padfx)
            medout += float(med)
            if padfxout < 0.25:
                if dotest:
                    if SHOWIMAGES:
                        showArray(im, newframe = 1, txt = 'Orig ' + thumbnail_filename)
                        showArray(padmap, newframe = 1, txt = 'pad ' + thumbnail_filename)
                    if DOPRINT:
                        print(params)
                # Sum thumbnails that do not contain any Nan values
                if not isnan(npsum(im)):
                    badshape = False
                    tshape = im.shape
                    if tshape[0] > 201:
                        print(tshape[0])
                        badshape = True
                    if tshape[1] > 201:
                        print('tshape[1]', tshape[1])
                        badshape = True
                    if badshape:
                        print('thumbnail_filename', thumbnail_filename)
                        print('galnum', self.galnum)
                        print('MEFname', str(self.MEFname))
                    imadd[:,:] = imadd[:,:] + im
                    sigmasqimadd[:,:] = sigmasqimadd[:,:] + sigmasqim
                # Keep track of padded regions
                padcoadd[where(padmap > 0)] = 3000
            else:
                print(2167, 'PipelineGalfit: Got a pad fraction > 25%.  Should not occur with bricks')
                exit()
        paramsout = [brickband, padfxout, medout]
        if dotest and SHOWIMAGES:
            showArray(imadd, newframe = 1, txt = 'imadd' + self.coadd_band)
        return imadd, sigmasqimadd, padcoadd, self.brickkeys, paramsout


    def loadSersiccompressed(self, compressed_array):
        """
        Purpose:
            Decompressed and formats previously saved thumbnails and associated data
        Input parameters:
            compressed_array (String): Path to compressed thumbnail information
        Outputs:
            im (float array): Thumbnail of unsubtracted image
            padmap (integer array): Map of any portions of thumbnail that are padded because
                they were to close to a brick edge
            params (float array):  Exposure data associated with thumbnails
            edge (integer): Length of thumbnail side
            padfx (float): Fraction of thumbnail that is padded because it extended
                over brick edge
        """
        # Load array and decompress thumbnails
        data = npload(compressed_array)
        im = data['arr_0']
        # Keys to stored observation information
        #    0: band        Observation filter
        #    1: magz        mag zero point for observation
        #    2: exptime     Exposure time
        #    3: pixscl      Pixel scale
        #    4: gain        Average ampifier gain
        #    5: rdnoise     Average read noise
        #    6: padfx       Fraction of thumbnail padded
        #    7: edge        edge length
        # Values of stored observation information
        vals = data['arr_2']
        if dotest and DOPRINT:
            print(vals)
        # Only return pertinent information
        params = vals#[1:6].astype(float)
        _band, padfx, _med = params
        # Edge length will be used to find center of thumbnail
        # edgeloc = keys.index('edge')
        # edge = int(vals[edgeloc])
        # Extract fraction of thumbnail padded
        padfx = float(padfx)
        # Create a mask of padded regions (padded pixels = 1)
        padmap = zeros_like(im, dtype = int16)
        #=======================================================================
        # Flag padded regions if any exist
        # Don't bother if fraction >= 0.25 since those thumbnails will be rejected anyway
        #=======================================================================
        padmap[where(im ==-1000)] = 1
        return im, padmap, params

    def loadCoaddcompressed(self, compressed_array):
        '''
        Purpose:
            Loads compressed file of stacked images from a single band
            Extracts image, segmentation map and associated parameters
        '''
        try:
            imdata = npload(compressed_array)
        except:
            print('ERROR 2109 PipelineGalfit', compressed_array)
            imdata = npload(compressed_array)
        im = imdata['arr_0']
        segmap = imdata['arr_1']
        padcoadd = zeros_like(im, dtype = int16)
        # First value will contain filter name which is not used
        params = imdata['arr_3'][1:].astype(float)
        padcoadd[where(segmap >= 3000)] = 3000
        return im, padcoadd, params

def run_GF(image_folder, info_folder, RAM_dir, osplatform, core_num, brick_list, gf_info,
                 processedIDs, do_n=True, half_edge=100, UDGs=[]):

    '''
    Purpose:
        Sets up folders that will be used to save data, invokes
            Galfit processing class and calls processing function
    Inputs:
        image_folder (path): Directory where processed bricks are stored
        info_folder (path): Directory where database and other I/O information are stored
        RAM_dir (path):  Directory used for temporary storage including executables
        osplatform (string): Operating system (Mac, HPC, or debian)
        core_num (integer): core number
        gal_list: List containing observations to be processed by core
        brickinfo (array): Brick names and psf's
        processedIDs (integer array): sorted array containing unique IDs for each thumbnail
            created during Sersic screen
        do_n (boolean): Let Sersic index vary during Galfit procedure
        half_edge (integer): Half of thumbnail edge which will also be center
        UDGs (string list): List of UDGs being analyzed
    '''

    # Location of folder for thumbnail storage
    thumbnail_loc = info_folder + 'Thumbnails/'
    if not path.exists(thumbnail_loc):
        makedirs(thumbnail_loc)
    # Initialize dogalfit class that runs Galfit on candidates
    fitUDGs = dogalfit(core_num, brick_list, gf_info,osplatform, info_folder, image_folder, RAM_dir,
        processedIDs, do_n=do_n, half_edge=half_edge, UDGs=UDGs)
    # process candidate galaxies
    fitUDGs.process_galaxies()

def copyinvvar_psf(image_folder, download_folder):
    '''
    Purpose:
        Copies inverse variance images from download folder to node /tmp for faster access if using HPC
    Inputs:
        image_folder (path): Directory where processed bricks are stored on /tmp
        download_folder (path): Directory on main storage containing downloaded bricks

    '''
    # Get list of folders containing processed bricks
    print('Copying files from /xdisk to /tmp')
    print('This may take a few minutes if many files')
    detection_folders = listdir(image_folder)
    for detection_folder in detection_folders:
        # Names of subfolders containing processed bricks end with 'res'
        if detection_folder.endswith('res'):
            # Path to subfolder containing processed brick
            detection_path = image_folder + detection_folder + '/'
            # Downloaded bricks stored in subfolders using first 3 characters of brick as name
            download_num = detection_folder[:3]
            # Create name and path of inverse variance brick
            fsplit = detection_folder.split('_')
            invarname = 'legacysurvey-' + fsplit[0]+ '-invvar-' +  fsplit[1] + '.fits.fz'
            invar_path = download_folder + download_num + '/' + invarname
            if not path.exists(invar_path):
                print (invar_path + ' does not exist')
                print ('This is needed to create Galfit sigma images')
                print ('Exiting')
            else:
                # Source path to inverse variance brick
                src = invar_path
                # Destination path
                dest = (detection_path + fsplit[0] +
                        '_invvar_' +  fsplit[1] + '.fits.fz')
                if not path.exists(dest):
                    copy2(src, dest)
            psfname = 'legacysurvey-' + fsplit[0]+ '-psfsize-' +  fsplit[1] + '.fits.fz'
            psf_path = download_folder + download_num + '/' + psfname
            if not path.exists(psf_path):
                print (psf_path + ' does not exist')
                print ('This is needed to get Galfit psf values')
                print ('Exiting')
            else:
                # Source path to inverse variance brick
                src = psf_path
                # Destination path
                dest = (detection_path + fsplit[0] +
                        '_psfsize_' +  fsplit[1] + '.fits.fz')
                if not path.exists(dest):
                    copy2(src, dest)

def GalfitStart(num_cores, image_folder, info_folder, download_folder, osplatform, _brickinfo, do_n=True, db='',
        half_edge=100, UDGs=[], RAMdiskloc=''):
    '''
    Purpose:
        Entry point for obtaining Galfit estimates of UDG parameters
        Inputs:
            num_cores (integer):  The number of cores that will be used for processing
            image_folder (path): Directory where processed bricks are stored on /tmp
            info_folder (path): Directory where database and other I/O information will be stored
            download_folder (path): Directory containing downloaded bricks
            osplatform (string): Operating system (Mac, HPC, or debian)
            brickinfo (array): Brick names and psf's
            do_n (Boolean): Let Sersic index vary during Galfit procedure
            db (string): Name of database where information will be saved
            half_edge (integer): Half of thumbnail edge which will also be center
            UDGs (string list): List of UDGs being analyzed
            RAMdiskloc (path) Directory used for temporary storage including executables
                Was actual RAM Disk on Mac, but now standard disk storage on all systems
    '''
    # Create temporary folders for saving processing information
    if osplatform == 'Mac':
        RAM_dir = RAMdiskloc
    # Copy files needed for executables
    if osplatform == 'HPC':
        tmp_RAM_loc = RAMdiskloc + 'GF/'
        if not path.exists(tmp_RAM_loc):
            makedirs(tmp_RAM_loc)
        copy2(RAMdiskloc + 'sex.nnw', tmp_RAM_loc + 'sex.nnw')
        copy2(RAMdiskloc + 'gauss_3.0_7x7.conv', tmp_RAM_loc + 'gauss_3.0_7x7.conv')
        copy2(RAMdiskloc +'sex', tmp_RAM_loc + 'sex')
        copy2(RAMdiskloc +'galfit', tmp_RAM_loc + 'galfit')
        copy2(RAMdiskloc +'swarp', tmp_RAM_loc + 'swarp')
        RAM_dir = tmp_RAM_loc
    # Copy inverse variance images from download folder to node /tmp for faster access if HPC
    copyinvvar_psf(image_folder, download_folder)
    # Get sorted array containing unique IDs for each thumbnail created during Sersic screen
    processedIDs = get_processed(info_folder)

    # Obtain list of galaxy numbers to be processed by Galfit
    galnums, bricks = gen_GFtables(info_folder, db, do_n, UDGs=UDGs)
    gf_info = [galnums, bricks]
    brick_list = list(set(bricks))
    brick_list = array(brick_list)

    # If only single candidate, insert in array to prevent crash
    if galnums.size == 1:
        galnums = asscalar(galnums)
        galnums = [galnums]
    if doshuffle:
        # If flag set, randomly shuffle list
        random.seed(12)
        random.shuffle(brick_list)
    #split by number of cores used
    split_list = array_split(brick_list, num_cores)
    # Set up jobs for each core
    jobs = []
    # Keep track of elapsed time
    start = time()
    for idx in range(num_cores):
        if DOPRINT:
            print(idx, 'Length of job list', len(split_list[idx]))
        #=======================================================================
        # run_GF: Sets up folders that will be used to save data, invokes
        #            Galfit processing class and calls processing function
        #    image_folder (path): Directory where processed bricks will be stored
        #    info_folder (path): Directory where database and other I/O information will be stored
        #    RAM_dir (path):  Directory used for temporary storage including executables
        #    osplatform (string): Operating system (Mac, HPC, or debian)
        #    idx (integer): core number
        #    split_list[idx]: List containing observations to be processed by core
        #    processedIDs (integer array): sorted array containing unique IDs for each thumbnail
        #        created during Sersic screen
        #    brickinfo (array): Brick names and psf's
        #    do_n (boolean): Let Sersic index vary during Galfit procedure
        #    half_edge (integer): Half of thumbnail edge which will also be center
        #    UDGs (string list): List of UDGs being analyzed
        #=======================================================================
        process = Process(target=run_GF, args=(image_folder, info_folder, RAM_dir, osplatform, idx,  split_list[idx], gf_info,
                            processedIDs, do_n, half_edge, UDGs))
        jobs.append(process)
        process.start()
        # Introduce small delay so that cores are not synchronous
        sleep(0.1)
    for job in jobs:
        # Join core jobs so that system will wait for all cores to finish before continuing
        job.join()
    if osplatform == 'Mac':
        desktop = path.expanduser('~') + '/Desktop/'
        tmp_RAM_loc = desktop + 'donnerst/'
    print("Elapsed Time for run: %s" % (time() - start,))
