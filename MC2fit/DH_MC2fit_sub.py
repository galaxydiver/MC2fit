import DH_array as dharray
import numpy as np
from scipy.special import gamma, gammainc, gammaincinv ## Sersic post processing
from astropy.modeling.functional_models import Sersic1D, Sersic2D


def pix2mu(array, plate_scale=0.262, zeromag=22.5):
    # array : pixel values --> flux / pixel2
    f=array/plate_scale**2  # flux/arcsec2
    return zeromag-2.5*np.log10(f) # mag / asec2

def mu2pix(array, plate_scale=0.262, zeromag=22.5):
    array= (array - zeromag)/-2.5
    return (10**array)*plate_scale**2

def get_bn(n):
    return gammaincinv(2*n, 1/2)

def get_mu_to_re_flux(Re, n, ar=1):  ## Flux R<Re
    term1 = Re**2 * 2*np.pi*n * ar
    bn=get_bn(n)
    term2 = np.exp(bn) / (bn)**(2*n)
    R=Re
    x = bn*(R/Re)
    term3 = gammainc(2*n, x)*gamma(2*n)
    return term1*term2*term3

def mu_to_mag(mu_e, Re, n, ar=1, plate_scale=0.262, zeromag=22.5):
    amp = 10**((zeromag-mu_e)/2.5) ## Amplitude (Flux / arcsec^2)  ## zeromag will not affect the results
    conversion = get_mu_to_re_flux(Re*plate_scale, n, ar)
    tot_flux = 2 * amp * conversion ## Total flux = Flux(R<Re)*2
    return zeromag-2.5*np.log10(tot_flux)

def mag_to_mu(mag, Re, n, ar=1, plate_scale=0.262, zeromag=22.5):
    flux = 10**((zeromag-mag)/2.5) ## Amplitude (Flux / arcsec^2)  ## zeromag will not affect the results
    conversion = get_mu_to_re_flux(Re*plate_scale, n, ar)
    amp = flux / 2 / conversion ## Total flux = Flux(R<Re)*2
    return zeromag-2.5*np.log10(amp)

def sersic_integral2D(mu_e, reff, n, ar, plate_scale, res=500, scale_fac=100):
    """
    Descr - measure total magnitude of a given galaxy
    INPUT
     - mu_e : surface brightness at Reff [mag/arc2]
     - reff : Reff
     - n : Sersic index
     - ar : axis ratio (b/a)
     - plate_scale : Plate scale [arcsec / pix]
     OUTPUT
     - Total magnitude [mag]

    """

    mag_factor=scale_fac/reff

    amp=10**((25-mu_e)/2.5)
    amp=amp*((plate_scale/mag_factor)**2)  # amp : flux / pix2
    e=e=1-ar  # ar : b/a,   e = (a-b)/a

    model=Sersic2D(amplitude=amp, r_eff=reff*mag_factor, n=n, ellip=e)

    x,y = np.meshgrid(np.arange(res)-res/2, np.arange(res)-res/2)
    ## Multi-gals
    if(hasattr(mu_e, "__len__")):
        x=np.repeat(x.reshape(np.shape(x)[0], np.shape(x)[1], 1), len(mu_e), axis=2)
        y=np.repeat(y.reshape(np.shape(y)[0], np.shape(y)[1], 1), len(mu_e), axis=2)
        img=model(x,y)
        flux=np.sum(np.sum(img, axis=0), axis=0)
    else:
        img=model(x,y)
        flux=np.sum(img)
    return 25-2.5*np.log10(flux)
