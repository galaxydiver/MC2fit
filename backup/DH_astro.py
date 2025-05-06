import numpy as np
import copy
from scipy.special import gamma, gammainc, gammaincinv ## Sersic post processing
import uncertainties as unc
import uncertainties.unumpy as unumpy

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import AutoMinorLocator
import matplotlib.colors

from scipy import stats
import bces.bces as BCES
from scipy.odr import ODR, Model, Data, RealData

#====================================== Email =========================
def send_email(title='Run Done', message='Job finished!', receiver_email="galaxydiver@arizona.edu"):
    import smtplib
    import os.path
    from email.mime.text import MIMEText

    msg = MIMEText(message)
    msg['Subject'] = title

    s = smtplib.SMTP_SSL("smtp.gmail.com:465")
    s.login("galaxydiver@arizona.edu","tap!subtle!spill")
    s.sendmail("galaxydiver@arizona.edu","galaxydiver@arizona.edu", msg.as_string())
    s.quit()
    print("Email sent!")

# ===================================== ASTRO ====================
def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])

def ang_dist(r1, d1, r2, d2):
    """
    INPUT : RA1, Dec1, RA2, Dec2 in deg unit
    OUTPUT : Angular distance in deg unit
    """
    d2r=np.pi/180
    res=np.sin(d1*d2r)*np.sin(d2*d2r)+np.cos(d1*d2r)*np.cos(d2*d2r)*np.cos((r2-r1)*d2r)
    return np.arccos(res)/d2r

def dms2deg(dmsarray, hms=False, return_hms=False):
    dmsarray=np.array(dmsarray)
    if(np.ndim(dmsarray)==1):  # If input value = 1D array
        is_1darray=True
        dmsarray=np.array([dmsarray])
    else: is_1darray=False

    answer=dmsarray[:,0]+dmsarray[:,1]/60+dmsarray[:,2]/3600
    if(hms==True): answer=answer*360/24
    if(return_hms==True): answer=answer*24/360

    if(is_1darray==True): return answer[0] # If input value = 1D ndarray -> return single value
    else: return answer # return 1D array

def deg2dms(deg, hms=False, return_hms=False):
    deg=np.array(deg)
    if(np.ndim(deg)==0):  # If input value = just value, not ndarray
        is_justvalue=True
        deg=np.array([deg])
    else: is_justvalue=False
    if(hms==True): deg=deg*360/24

    if(return_hms==True): deg=deg/360*24
    answer=np.zeros((len(deg),3))
    answer[:,0]=deg.astype(int)
    answer[:,1]=(deg*60-answer[:,0]*60).astype(int)
    answer[:,2]=(deg*3600-answer[:,0]*3600 - answer[:,1]*60)

    if(is_justvalue==True): return answer[0] # If input value = just value, not ndarray -> return 1D array
    else: return answer # return 2D array


#==================================================================
def mag2flux(mag, m_zero=22.5):
    return 10**((m_zero-mag)/2.5)

def flux2mag(flux, m_zero=22.5, is_unump=False):
    if(is_unump): return m_zero-2.5*unumpy.log10(flux)
    else: return m_zero-2.5*np.log10(flux)

def pix2mu(array, plate_scale=0.262, m_zero=22.5, is_unump=False):
    # array : pixel values --> flux / pixel2
    f=array/(plate_scale**2)  # flux/arcsec2
    return flux2mag(f, m_zero, is_unump=is_unump) # mag / asec2


def mu2pix(array, plate_scale=0.262, m_zero=22.5):
    # array= (array - m_zero)/-2.5
    # return (10**array)*plate_scale**2
    array = mag2flux(array, m_zero)
    return array * (plate_scale**2)

def get_bn(n):
    return gammaincinv(2*n, 1/2)

def get_mu2re_flux(Re, n, ar=1):  ## Flux R<Re
    term1 = Re**2 * 2*np.pi*n * ar
    bn=get_bn(n)
    term2 = np.exp(bn) / (bn)**(2*n)
    R=Re
    x = bn*(R/Re)
    term3 = gammainc(2*n, x)*gamma(2*n)
    return term1*term2*term3

def mu2mag(mu_e, Re, n, ar=1, plate_scale=0.262, m_zero=22.5, is_unump=False):
    amp = mag2flux(mu_e, m_zero) ## Amplitude (Flux / arcsec^2)  ## m_zero will not affect the results
    conversion = get_mu2re_flux(Re*plate_scale, n, ar)
    tot_flux = 2 * amp * conversion ## Total flux = Flux(R<Re)*2
    return flux2mag(tot_flux, m_zero, is_unump=is_unump)

def mag2mu(mag, Re, n, ar=1, plate_scale=0.262, m_zero=22.5, is_unump=False):
    flux = mag2flux(mag, m_zero) ## Amplitude (Flux / arcsec^2)  ## m_zero will not affect the results
    conversion = get_mu2re_flux(Re*plate_scale, n, ar)
    amp = flux / 2 / conversion ## Total flux = Flux(R<Re)*2
    return flux2mag(amp, m_zero, is_unump=is_unump)


def get_ml_ratio(g_r):
    logml=-0.8+ (g_r)*((1.244+0.8)/1.35)
    return 10**logml

def get_mass(m, g_r):
    ## Sun : g : 5.45,  r: 4.76 (Blanton et al.)
    lum_solar= 10**((m-4.76)/-2.5)  # Luminosity / Solar
    ml_ratio = get_ml_ratio(g_r)
    return lum_solar * ml_ratio  # Mass / Solar

## ========================= Drawing =====================


def datacut(cutdata, xdata, ydata, xerr=None, yerr=None, cdata=None):
    newx=np.copy(xdata[cutdata])
    newy=np.copy(ydata[cutdata])
    if(hasattr(xerr, "__len__")):
        if(np.ndim(xerr)==1): new_xerr=np.copy(xerr[cutdata])
        if(np.ndim(xerr)==2):
            print(np.shape(xerr))
            if(len(xerr)==len(xdata)): new_xerr=np.copy(xerr[cutdata])
            else: new_xerr=np.copy(xerr[:,cutdata])
    else: new_xerr=np.copy(xerr)

    if(hasattr(yerr, "__len__")):
        if(np.ndim(yerr)==1): new_yerr=np.copy(yerr[cutdata])
        if(np.ndim(yerr)==2):
            if(len(yerr)==len(ydata)): new_yerr=np.copy(yerr[cutdata])
            else: new_yerr=np.copy(yerr[:,cutdata])
    else: new_yerr=np.copy(yerr)

    if(hasattr(cdata, "__len__")): new_cdata=np.copy(cdata[cutdata])
    else: new_cdata=np.copy(cdata)

    return newx, newy, new_xerr, new_yerr, new_cdata

def get_nonnan(xdata, ydata, cdata=None, ax0=None, inside_box=True):
    if(hasattr(cdata, "__len__")==False): cdata=np.zeros_like(xdata) ## if cdata==None, make zero array
    if(hasattr(ax0, "get_xlim") & (inside_box==True)):
        return np.where(np.isfinite(xdata) & np.isfinite(ydata) &
                       (xdata<=ax0.get_xlim()[1]) & (xdata>=ax0.get_xlim()[0]) &
                       (ydata<=ax0.get_ylim()[1]) & (ydata>=ax0.get_ylim()[0]) &
                       (np.isfinite(cdata)))[0]
    else: return np.where(np.isfinite(xdata) & np.isfinite(ydata) & np.isfinite(cdata))[0]

def fitting_odr(fit_range, usingftn, usingftn_rev, xdata, ydata, xerr=None, yerr=None, is_swap=False, is_ordinary_lsq=False):
    if(is_swap==False):
        data=RealData(xdata, ydata, xerr, yerr)
        model = Model(usingftn)
    else:
        data=RealData(ydata, xdata, yerr, xerr)
        model = Model(usingftn_rev)

    odr = ODR(data, model, [1,1], maxit=200)
    if(is_ordinary_lsq==True): odr.set_job(fit_type=2) ## Simple least square
    else: odr.set_job(fit_type=0) ## Orthogonal Distance Regression
    output = odr.run()

    if(is_swap==False):
        x_model=fit_range
        y_model=usingftn(output.beta, x_model)
    else:
        y_model=fit_range
        x_model=usingftn_rev(output.beta, y_model)

    fitting_sig=np.sqrt(np.diag(output.cov_beta))
    return output, x_model, y_model, fitting_sig

def fitting_odr_res(output, is_swap=False, use_cov=False):
    if(use_cov): fitting_sig=np.sqrt(np.diag(output.cov_beta))
    else: fitting_sig=output.sd_beta
    slope=output.beta[1]
    slope_err=fitting_sig[1]
    intercept=output.beta[0]
    intercept_err=fitting_sig[0]

    if(is_swap==True):
        intercept_err=np.abs(intercept)*((intercept_err/intercept)**2 + (slope_err/slope)**2 + 2*output.cov_beta[0,1]/slope/intercept)
        intercept=-intercept/slope

        slope_err=slope_err / slope**2
        slope=1/slope

    return slope, slope_err, intercept, intercept_err


def func_linear(beta, x):
    y = beta[0]+beta[1]*x
    return y


def fitting_linear_bound(N, x_model, slope, intercept, cov_mat, conf=0.64):

    m = 2                             # number of parameters
    dof = N - m                       # degrees of freedom

    # Let's calculate the mean resposne (i.e. fitted) values again:
    x_matrix = np.column_stack((np.ones(len(x_model)), x_model))
    y_fit = np.dot(x_matrix,  np.array([intercept, slope])) ## Beta
    #  Calculate mean resposne SE:
    ym_res_se = np.dot(np.dot(x_matrix, cov_mat), np.transpose(x_matrix))
    ym_res_se = np.sqrt(np.diag(ym_res_se))

    ym_lower = y_fit - stats.t.ppf(q = 1 - (1-conf)/2, df = dof) * ym_res_se
    ym_upper = y_fit + stats.t.ppf(q = 1 - (1-conf)/2, df = dof) * ym_res_se

    return y_fit, ym_lower, ym_upper

def fitting_odr_bound(xdata, x_model, output, conf=0.64):
    intercept, slope = output.beta
    cov_mat = output.cov_beta * output.res_var
    return fitting_linear_bound(len(xdata), x_model, slope, intercept, cov_mat, conf=conf)

def fitting_bces_bound(xdata, x_model, a, b, aerr, berr, covab, fitting_index=0, conf=0.64):
    intercept, slope = b[fitting_index], a[fitting_index]
    cov_mat=np.zeros((2,2))
    cov_mat[0,0]=berr[fitting_index]**2
    cov_mat[1,1]=aerr[fitting_index]**2
    cov_mat[0,1]=covab[fitting_index]
    cov_mat[1,0]=covab[fitting_index]
    return fitting_linear_bound(len(xdata), x_model, slope, intercept, cov_mat, conf=conf)






def fitting_odr_bound_old(x_model, output, shift=True, sigma=1):
    """
    You may just use this if you are not considering the shift.
    plt.fill_between(x_model,
                 func_linear(output.beta + 1*output.sd_beta, x_model),
                 func_linear(output.beta - 1*output.sd_beta, x_model),
                 color='k', alpha=0.15)
    """
    if(shift):
        xshift=np.nanmean(x_model)
        yshift=np.nanmean(func_linear(output.beta, x_model))
        yshift_xshift=np.nanmean(func_linear(output.beta, x_model-xshift))
    else:
        xshift=0
        yshift=0
        yshift_xshift=0
    lowb=func_linear(output.beta - sigma*output.sd_beta-[yshift_xshift,0], x_model-xshift) + yshift
    uppb=func_linear(output.beta + sigma*output.sd_beta-[yshift_xshift,0], x_model-xshift) + yshift
    return lowb, uppb

def fitting_bces_bound_old(x_model, a, b, aerr, berr, shift=True, sigma=1):

    if(shift):
        xshift=np.nanmean(x_model)
        yshift=np.nanmean(func_linear([b,a], x_model))
        yshift_xshift=np.nanmean(func_linear([b,a], x_model-xshift))
    else:
        xshift=0
        yshift=0
        yshift_xshift=0
    lowb=func_linear([b,a] - sigma*np.array([berr,aerr])-[yshift_xshift,0], x_model-xshift) + yshift
    uppb=func_linear([b,a] + sigma*np.array([berr,aerr])-[yshift_xshift,0], x_model-xshift) + yshift
    return lowb, uppb


def err2D_center_data(err2D, data, change_nan=np.nan):
    """
    Descr - Change the error data for fitting error bar
          e.g.) plt.errorbar(x=[3], xerr=[1,5]) -> errorbar: 2 - 8
          but in the case for errorbar 1 - 5
    """
    ## Data format
    if(len(err2D)==len(data)): err2D=np.copy(err2D-data[:,None])
    else: err2D=np.copy(err2D-data[None,:])
    if(change_nan!=None):
        err2D[np.isnan(err2D)]=change_nan

    err2D[0]=-err2D[0]
    ## check range
    target=np.where(err2D[0]<0)[0]
    if(len(target)!=0):
        print("Error! Data point is out of the error range!")
    return err2D


def cdata_classification(cdata, N=7):
    newcdata=copy.deepcopy(cdata)
    newcdata[newcdata==3]=6
    newcdata[newcdata==4]=3
    newcdata[newcdata==7]=6
    newcdata[newcdata==11]=5
    newcdata[newcdata==12]=4
    newcdata[newcdata==14]=6

    ## cmap
    cmap = plt.get_cmap('Set1')
    cmap_usinglist=np.array([0,1,2,3,4,6,7]).astype(int)
    cmap_usinglist=cmap_usinglist[:N]
    colors = cmap(cmap_usinglist)
    # Create a new colormap from those colors
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_jet', colors, N=N)
    return newcdata, color_map

def cdata_sdss_spec(Thisbox):
    sdssclass=np.full_like(Thisbox.catalog['class'], np.nan)
    sdssclass[Thisbox.catalog['class']=='GALAXY']=0
    sdssclass[Thisbox.catalog['class']=='QSO   ']=1
    sdssclass[Thisbox.catalog['class']=='STAR  ']=2
    sdssclass=sdssclass.astype(int)
    cdata=sdssclass

    ## cmap
    cmap = plt.get_cmap('Set1')
    colors = cmap([2,1,0,3,4])
    # Create a new colormap from those colors
    color_map = matplotlib.colors.LinearSegmentedColormap.from_list('cut_jet', colors, N=3)
    return sdssclass, color_map


def colorbar_classification(mainplot, cbaxes, ticklabels=['Sersic', 'S+S', 'S+PSF', '2comp', 'S+2PSF', 'SS+PSF', 'complex'],
                            N=7, fontsize_label=13):
    cb=plt.colorbar(mainplot, cax=cbaxes, orientation='horizontal')
    cb.set_label(label="Classification",
                 fontsize=fontsize_label, fontname='DejaVu Serif', fontweight='semibold') #weight='bold'
    ticks=np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5])[:N]
    ticklabels=np.array(ticklabels)[:N]
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels, weight='semibold')
    return cb

def colorbar_sdss_spec(mainplot, cbaxes, ticklabels=['Galaxy', 'QSO', 'STAR'],
                            N=3, fontsize_label=13):
    cb=plt.colorbar(mainplot, cax=cbaxes, orientation='horizontal')
    cb.set_label(label="SDSS class",
                 fontsize=fontsize_label, fontname='DejaVu Serif', fontweight='semibold') #weight='bold'
    ticks=np.array([0.5,1.5,2.5])[:N]
    ticklabels=np.array(ticklabels)[:N]
    cb.set_ticks(ticks)
    cb.set_ticklabels(ticklabels, weight='semibold')
    return cb

def side_histogram(gs, ax0, xdata, ydata, histcolor='grey', histtype='bar', linestyle='-', linewidth=2,
                   ax1=None, ax2=None, Nbin=30, normhist=False,
                   is_draw_top=True, is_draw_right=True):
    #gs_geo=gs.get_geometry() ## Size of gs
    if(normhist): weights=np.ones_like(xdata)/len(xdata)
    else: weights=None
    ## ======================= Side histogram ==========================
    if(is_draw_top):
        if(ax1==None):
            if(is_draw_right): ax1=plt.subplot(gs[0,0:-1])  ## gs[0,0:3] for (4,4)
            else: ax1=plt.subplot(gs[0,:])
        ax1.hist(xdata, bins=np.linspace(ax0.get_xlim()[0],ax0.get_xlim()[1], Nbin),
                 color=histcolor, histtype=histtype, weights=weights, linestyle=linestyle, linewidth=linewidth)
        ax1.tick_params(labelsize=13, width=1, length=6, axis='both', direction='in', right=True, top=True)
        ax1.tick_params(width=0.5, length=3, which='minor', axis='both', direction='in', right=True, top=True)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.set_xlim(ax0.get_xlim())

    ## =======================================
    if(is_draw_right):
        if(ax2==None):
            if(is_draw_top): ax2=plt.subplot(gs[1:,-1])
            else: ax2=plt.subplot(gs[:,-1])
        ax2.hist(ydata, orientation='horizontal', bins=np.linspace(ax0.get_ylim()[0],ax0.get_ylim()[1], Nbin),
                 color=histcolor, histtype=histtype, weights=weights, linestyle=linestyle, linewidth=linewidth)
        ax2.tick_params(labelsize=13, width=1, length=6, axis='both', direction='in', right=True, top=True)
        ax2.tick_params(width=0.5, length=3, which='minor', axis='both', direction='in', right=True, top=True)
        ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.set_ylim(ax0.get_ylim())

    return ax1, ax2


def get_frac_err(frac, N_in_bin):
    ## Bernoulli variable, Binomial distribution
    ## https://suchideas.com/articles/maths/applied/histogram-errors/
    ## https://stats.stackexchange.com/questions/484837/error-bars-for-histogram-with-uncertain-data

    ## mu = np = Nk
    ## var = npq, where q=1-p
    ## err = sqrt(npq) = sqrt (np(1-p))
    ## err(p) = 1/n err(mu) = sqrt(np(1-p)/n^2) = sqrt(p(1-p)/n)
    var=(frac*(1-frac))/N_in_bin
    return var**0.5

def get_frac_err_possible(frac, N_in_bin, Ntest=1000):
    testbins=np.arange(Ntest+1)/Ntest   ## if Ntest=100, testbins= [0.01, 0.02, 0.03 ... 0.99, 1]
    err_possible=np.zeros((len(frac), 2))
    for i in range (len(N_in_bin)):
        if(np.isnan(frac[i])): err_possible[i]=np.nan
        else:
            var=testbins*(1-testbins)/N_in_bin[i]
            std=var**0.5
            possible_range1=testbins[frac[i]<=(testbins+std)] ## upper bound
            possible_range2=testbins[frac[i]>=(testbins-std)] ## lower bound
            err_possible[i,0]=np.nanmin(possible_range1)
            err_possible[i,1]=np.nanmax(possible_range2)
    return err_possible


class Frac1D():
    def __init__(self, data, condition, bins, weights=None, minNtot=-1, minNfrac=-1, Ntest=1000):
        """
        INPUT
        """
        self.midbins=(bins[1:]+bins[:-1])/2
        if(hasattr(weights, "__len__")==False):
            weights=np.ones_like(data)

        self.totaldata=np.histogram(data, bins=bins, weights=weights)
        self.fracdata=np.histogram(data[condition], bins=bins, weights=weights[condition])
        self.frac=self.fracdata[0]/self.totaldata[0] ## frac for each bin
        self.frac[np.where((self.totaldata[0]<minNtot) | (self.fracdata[0]<minNfrac))]=np.nan

        self.err=get_frac_err(self.frac, self.totaldata[0])
        if(Ntest!=0):
            self.err_possible = get_frac_err_possible(self.frac, self.totaldata[0], Ntest=Ntest)

        self.err_h=self.frac+self.err
        self.err_l=self.frac-self.err

#         self.err_h=self.frac+(((self.frac)*(1-(self.frac)))/len(self.totaldata[0]))**0.5
#         self.err_l=self.frac-(((self.frac)*(1-(self.frac)))/len(self.totaldata[0]))**0.5
        self.err_l[self.err_l<0]=0

def side_histogram_2dfrac(gs, ax0, xdata, ydata, inclist, xbins, ybins, Ndata_min, color='k',
                          alpha=0.2, ax1=None, ax2=None, Nbin=30, normhist=False,
                          is_err_possible=True, is_draw_top=True, is_draw_right=True):
    #gs_geo=gs.get_geometry() ## Size of gs
    if(normhist): weights=np.ones_like(xdata)/len(xdata)
    else: weights=None
    ## ======================= Side histogram ==========================
    if(is_draw_top):
        if(ax1==None):
            if(is_draw_right): ax1=plt.subplot(gs[0,0:-1])  ## gs[0,0:3] for (4,4)
            else: ax1=plt.subplot(gs[0,:])
        FracGal=Frac1D(xdata, inclist, bins=xbins, minNtot=Ndata_min)

        ax1.plot(FracGal.midbins, FracGal.frac, c=color, marker='.')
        if(is_err_possible):
            ax1.fill_between(FracGal.midbins, FracGal.err_possible[:,0], FracGal.err_possible[:,1],
                             color=color, alpha=alpha)
        else:
            ax1.fill_between(FracGal.midbins, FracGal.err_l, FracGal.err_h, color=color, alpha=alpha)
        ax1.set_ylabel(r"$\rm \mathbf{f}_{\rm \mathbf{NSC,o}}$", fontsize=17, fontname='DejaVu Serif', fontweight='semibold')
        ax1.tick_params(labelsize=13, width=1, length=6, axis='both', direction='in', right=True, top=True)
        ax1.tick_params(width=0.5, length=3, which='minor', axis='both', direction='in', right=True, top=True)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.set_xlim(ax0.get_xlim())

    ## =======================================
    if(is_draw_right):
        if(ax2==None):
            if(is_draw_top): ax2=plt.subplot(gs[1:,-1])
            else: ax2=plt.subplot(gs[:,-1])
        FracGal=Frac1D(ydata, inclist, bins=ybins, minNtot=Ndata_min)

        ax2.plot(FracGal.frac, FracGal.midbins, c=color, label='nsc+2comp frac', marker='.')
        if(is_err_possible):
            ax2.fill_betweenx(FracGal.midbins, FracGal.err_possible[:,0], FracGal.err_possible[:,1],
                             color=color, alpha=alpha)
        else:
            ax2.fill_betweenx(FracGal.midbins, FracGal.err_l, FracGal.err_h, color=color, alpha=alpha)

        ax2.set_xlabel(r"$\rm \mathbf{f}_{\rm \mathbf{NSC,o}}$", fontsize=17, fontname='DejaVu Serif', fontweight='semibold')
        ax2.tick_params(labelsize=13, width=1, length=6, axis='both', direction='in', right=True, top=True)
        ax2.tick_params(width=0.5, length=3, which='minor', axis='both', direction='in', right=True, top=True)
        ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.set_ylim(ax0.get_ylim())

    return ax1, ax2
