# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:44:22 2022

@author: Hy-Q @NBI, KU
"""

import numpy as np
import os
import sys
import csv
sys.path.append('D:/Sejrs bachelor project/Data')
from iminuit import Minuit, cost
from iminuit import describe
from iminuit.util import make_func_code
import matplotlib.pyplot as plt
import pickle
from inspect import signature
from matplotlib.widgets import Slider
import scipy
from scipy import stats
import time


def gauss_2d(x, N, mu_x, mu_y, sigma_x, sigma_y, d):

    
    """
    A 2D gaussian.

    Parameters
    ----------
    x : numpy array shape(n,2)
        The x and y values for the gaussians structured so the first coloum is x, and the second y.
    N : float
        The higth of the peak og the gausian.
    mu_x : float
        x coordinate for the center of the gaussian.
    mu_y : float
        y coordinate for the center of the gaussian.
    sigma_x : float
        The width of the gaussian in the x direction.
    sigma_y : float
        The wodth of the gaussian in the y direction .
    d : float
        Constant offset of the gausian.

    Returns 
    -------
    The value of the gaussian at the point x as a float

    """

    return N * np.exp(-1/2 * (((x[:,0]-mu_x)/sigma_x)**2 + 
                              ((x[:,1]-mu_y)/sigma_y)**2))
"""                          
def gauss_2d(pixdata, amp, mux, muy, sigx, sigy, offset):
    X, Y = pixdata
    zdata = amp*np.exp(-(X-mux)**2/(2*sigx**2)-(Y-muy)**2/(2*sigy**2)) + offset
    return zdata.ravel()
"""


def gauss_1d(x, N, mu, sigma, d):
    """
    A 1D gaussian

    Parameters
    ----------
    x : float
        The point where the gaussian is calculated at.
    N : float
        The higth of the peak og the gausian.
    mu : float
        Center of the gaussian.
    sigma : float
        Width of the gaussian.
    d : float
        Constant offset of the gaussian.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return N * np.exp(-1/2 * ((((x-mu)/sigma)**2))) + d

def linear(x, a, b):
    return x * a + b




def find_peak(foto):

    """
    Finds the middle of the laser spot in a foto

    Parameters
    ----------
    foto : "2D numpy array
        A foto with brightness of each pixel.

    Returns
    -------
    peak_index : numpy arra shape(2)
        The x and y coordinates of the middle of the laser spot.

    """
    peak_raw = np.where(foto == np.max(foto))  # finds the indexes of maxima on the image

    
    # Check if peak_raw has at least 2 elements
    if len(peak_raw) < 2:
        # Handle the error here
        # For example, you could return a default value or raise an error
        return np.array([0, 0])
    
    peak_index = np.array([np.round(np.mean(peak_raw[0])), 
                           np.round(np.mean(peak_raw[1]))])   # takes the average
    return peak_index

def zoom(foto, zoom_amount, center):
    """
    Zoomes on a foto given a zoom amout 

    Parameters
    ----------
    foto : 2D numpy array
        Foto that is to be zoomed.
    zoom_amount : float
        Amount of magnification.
    center : numpy array shape(2)
        The position to zoom in on.

    Returns
    -------
    foto_zoomed : 2D numpy array
        The zoomed in foto.
    lower : numpy array shape(2)
        Position of the lower left corner of the zommed image compered to the original.
    higher : numpy array shape(2)
        Position of the upper rigth corner of the zommed image compered to the original.

    """
    
    foto_shape = np.shape(foto)
    _zoom = np.array([int(foto_shape[0] / (2 * zoom_amount)), 
                      int(foto_shape[1] / (2 * zoom_amount))])
    lower = np.array([center[0]-_zoom[0], center[1]-_zoom[1]])
    higher = np.array([center[0]+_zoom[0], center[1]+_zoom[1]])
    
    for i in range(len(lower)): # to make sure the zoomed image does not go out of bound
        if lower[i] < 0:
            higher[i] += -lower[i]
            lower[i] = 0
            
        if higher[i] >= foto_shape[i]:
            lower[i] += foto_shape[i] - higher[i]
            higher[i] = foto_shape[i]
        
        
    
    foto_zoomed = foto[lower[0] : higher[0], lower[1] : higher[1]]
    
    
    
    return foto_zoomed, lower, higher

def zoom_pixel(foto, pixel_nr, center):
    """
    Crops an image given a center and size

    Parameters
    ----------
    foto : 2D numpy array
        Foto that is to be zoomed.
    pixel_nr : int
        The size of the cropped image.
    center : numpy array shape(2)
        The position of the center of the cropped image.

    Returns
    -------
    foto_zoomed : 2D numpy array with size (pixel_nr, pixel_nr)
        The zoomed in foto.
    lower : numpy array shape(2)
        Position of the lower left corner of the zommed image compered to the original.
    higher : numpy array shape(2)
        Position of the upper rigth corner of the zommed image compered to the original.

    """
    foto_shape = np.shape(foto)
    _zoom = int(pixel_nr/2)
    lower = np.array([center[0]-_zoom, center[1]-_zoom], dtype=int)
    higher = np.array([center[0]+_zoom, center[1]+_zoom],dtype=int)
    
    for i in range(len(lower)): # to make sure the zoomed image does not go out of bound
        if lower[i] < 0:
            higher[i] += -lower[i]
            lower[i] = 0
            
        if higher[i] >= foto_shape[i]:
            lower[i] += foto_shape[i] - higher[i]
            higher[i] = foto_shape[i]
        
        
    
    foto_zoomed = foto[lower[0] : higher[0], lower[1] : higher[1]]
    
    return foto_zoomed, lower, higher
    



def save_dictionery_pickle(dictionery ,file_name):
    """
    Saves data in the pickle format

    Parameters
    ----------
    dictionery : dictionery
        The dectionery to be saved.
    file_name : str
        Name of the file where the data is saved.

    Returns
    -------
    None.

    """
    pickle.dump(dictionery, open( rf'N:\SCI-NBI-qplab-data\OldLab\HyQ1000\Data\Piezzo_rf_Sejr_bachlor_Thesis\{file_name}' + ".p", "wb" ) )

def save_calibration_2dscan(step_size, unitvector, filename):
    
    calibration = {}
    calibration["step_size"] = step_size
    calibration["unit_vector"] = unitvector
    
    save_dictionery_pickle(calibration, filename)
    


def read_dictionery_pickle(file_name):
    """
    Loads data in the pickle format

    Parameters
    ----------
    file_name : str
        Name of the file to load data from.

    Returns
    -------
    dictionery
        The loaded dictionery.

    """
    return pickle.load(open(rf'N:\SCI-NBI-qplab-data\OldLab\HyQ1000\Data\Piezzo_rf_Sejr_bachlor_Thesis\{file_name}' + ".p", "rb"))

def read_dictionery_pickle_linux(file_name):
    return pickle.load(open(rf'/home/sk0rt3/auto_excite/Data/{file_name}' + ".p", "rb"))
    """
    Loads data in the pickle format just from an other path

    Parameters
    ----------
    file_name : str
        Name of the file to load data from.

    Returns
    -------
    dictionery
        The loaded dictionery.

    """

def extract_dict(pics):
    fotos = pics['foto']
    number_of_steps = pics['number_of_steps']
    pics_per_step = pics['pics_per_step']
    step_size = pics['step_size']
    lower = pics["lower"]
    size = pics["size"]
    return fotos, number_of_steps, pics_per_step, step_size, lower, size

def extract_dict_2_camera(pics):
    fotos_1 = pics['foto_1']
    fotos_2 = pics['foto_2']
    number_of_steps = pics['number_of_steps']
    pics_per_step = pics['pics_per_step']
    step_size = pics['step_size']
    lower_1 = pics["lower_1"]
    size = pics["size"]
    return fotos_1, fotos_2 , number_of_steps, pics_per_step, step_size, lower_1, size

def extract_dict_2d_scan(pics):
    fotos = pics['foto']
    number_of_steps = pics['number_of_steps']
    pics_per_step = pics['pics_per_step']
    step_size = pics['step_size']
    lower = pics["lower"]
    size = pics["size"]
    #theo_pos = pics['theoretical_pos']
    return fotos, number_of_steps, pics_per_step, step_size, lower, size#, theo_pos

def extract_dict_2d_scan_calibration(calibration):
    step_size = calibration["step_size"]
    unit_vector = calibration["unit_vector"]
    return step_size, unit_vector

    
def show_foto(foto):
    """
    Displays a foto

    Parameters
    ----------
    foto : 2D numpy array
        Displays the given image.

    Returns
    -------
    None.

    """
    plt.imshow(foto, vmin=0 , vmax= 255, cmap="hot")
    plt.colorbar()
    plt.show()
    plt.pause(1) # make the picture be desplayted at once
 
    

      
        
class LeastSquares:
    """
    Generic least-squares cost function with error.
    """

    errordef = Minuit.LEAST_SQUARES # for Minuit to compute errors correctly

    def __init__(self, model, x, y, err, weigths=None):
        self.model = model  # model predicts y for given x
        self.x = x
        self.y = y
        self.err = err
        try:
            if weigths == None:
                self.weigths = self.y * 0 + 1
        except:    
            self.weigths = weigths
        

    def __call__(self, *par):  # we accept a variable number of model parameters
        ym = self.model(self.x, *par)
        #print(np.shape(self.weigths),np.shape(self.y),np.shape(ym),np.shape(self.err))
        #print(self.weigths, self.y, ym, self.err)
        
        return np.sum((self.y - ym) ** 2 / self.err ** 2)








def fit_chi2(func, X, foto, sy, guess, weights=None):
    """
    Preforms an chi2 fit

    Parameters
    ----------
    func : function
        The function to be fitted.
    x : numpy array any shape
        The x data witch the function takes.
    y : numpy array
        The y data the function shal compere with.
    sy : numpy array shape same as y
        the error on y.
    guess : numpy array
        Starting guess for dunction perameters.
    weights : numpy array shape same as y, optional
        Weigths to do a weigthed chi2 fit. The default is None.

    Returns
    -------
    numpy array
        The found parameters.
    numpy array
        The errors on the found parameters.

    """
    
    
    
    chi2fit = LeastSquares(func, X, foto, sy, weigths=weights)
    #sig = signature(func)
    #number = len(sig.parameters)
    minuit_chi2 = Minuit(chi2fit, *guess)
    """
    if len(guess) == number - 1:
        
        minuit_chi2 = Minuit(chi2fit, *guess)
    elif len(guess) == number:
        minuit_chi2 = Minuit(chi2fit, *(guess[:-1]))
    else:
        print(f"Error: guess has length {len(guess)} but the fit funktion takes {number} perameters")
    """
    minuit_chi2.errordef = 1.0
    minuit_chi2.migrad()
    
    return minuit_chi2.values[:], minuit_chi2.errors[:]


def foto_flattening(foto, lower, sy):
    """
    Takes a foto and formates it so it can be fitted to

    Parameters
    ----------
    foto : 2D numpy array
        The foto to be formatted for fitting.
    lower : numpy array shape(2)
        position of the lower left corner of the foto.
    sy : 2D numpy array
        The error on the values in the foto.

    Returns
    -------
    X : 2D numpy array
        Contains all the pairs of x and y positions in the foto.
    Y : 1D numpy array
        The flattened pixel values in foto.
    SY : 1D numpy array
        The flattened error on the pixel values in foto.

    """
    
    X = np.zeros((len(foto)*len(foto[0]),2))
    Y = foto.flatten()
    SY = sy.flatten()
    lower = lower.astype(int) # lower is always an integer value, but is stored as a float
    
    for i in range(len(foto)):
        for j in range(len(foto[0])):
            X[i + j * len(foto)] = np.array([i + lower[1], j + lower[0]])
    
    return X, Y, SY


def foto_2Dguess_gen(foto, lower):
    """
    Generates guesses complatible with 2D gaussian fit.

    Parameters
    ----------
    foto : 2D numpy array
        The foto to be fitted to.
    lower : numpy array shape(2)
        position of the lower left corner of the foto.

    Returns
    -------
    guess : numpy array
        A guess for a 2D gaussian.

    """
    #guess = np.array([100, 10, 10, 3, 3, 0])
    guess = np.array([np.max(foto)*0.9, (lower[1]+np.mean(np.where(foto == 
                                    np.max(foto))[0])),(lower[0]
                                    +np.mean(np.where(foto == 
                                    np.max(foto))[1])) , 2.8, 2.8, 3,])
    return guess



def foto_1Dguess_gen(foto, lower):
    """
    Generates guesses complatible with 2 1D gaussian fits.

    Parameters
    ----------
    foto : 2D numpy array
        The foto to be fitted to.
    lower : numpy array shape(2)
        position of the lower left corner of the foto.

    Returns
    -------
    guess_x : numpy array
        A guess for a 1D gaussian in x direction.
    guess_y : numpy array
        A guess for a 1D gaussian in y direction.

    """
    guess_x = np.array([np.max(foto), lower[0]+np.mean(np.where(foto == 
                                    np.max(foto))[0]), 2.8, 3])
    guess_y = np.array([np.max(foto), lower[1]+np.mean(np.where(foto == 
                                    np.max(foto))[1]), 2.8, 3])
    
    return guess_x, guess_y


def fit_2d_gauss_multi(foto, lower, number_of_steps, sy=None):
    try:
        if sy == None:
            sy = np.full(int(number_of_steps), None)
    except:
        pass
    pa, er, gues = fit_2d_gauss(foto[0], lower[0], number_of_steps, sy=sy[0])
    par = np.zeros((int(number_of_steps), len(pa)))
    err = np.zeros((int(number_of_steps), len(pa)))
    guess = np.zeros((int(number_of_steps), len(pa)))
    par[0] = pa
    err[0] = er
    guess[0] = gues
    for i in range(1, int(number_of_steps)):
        par[i], err[i], guess[i] = fit_2d_gauss(foto[i], lower[i], number_of_steps, sy=sy[i])
        
    return par, err, guess



def fit_2d_gauss(foto, lower, number_of_steps, sy=None):
    try:
        if sy == None:
            sy = np.ones_like(foto)
    except:
        sy[sy < 1] = 1
        pass
        
    guess = foto_2Dguess_gen(foto, lower)
    x, y, sy = foto_flattening(foto, lower, sy)
   
    par, err = fit_chi2(gauss_2d, x, y, sy, guess)
    return np.array(par), np.array(err), guess

"""

def fit_2d_gauss(foto, lower, sy=None):
    x = np.arange(20) + lower[1]
    xstep = x[1]- x[0]
    y = np.arange(20) + lower[0]
    ystep = y[1]- y[0]
    x , y = np.meshgrid(x, y)
    guess = foto_2Dguess_gen(foto, lower)
    foto_raveled = foto.ravel()
    try:
        if sy == None:
            sy = np.ones_like(foto)
    except:
        sy[sy < 1] = 1
        pass
    #sy = np.ones_like(foto)
    sy_ravel = sy.ravel()
    par, err = fit_chi2(gauss_2d, (x, y), foto_raveled, sy_ravel, guess)
    #par, cov = scipy.optimize.curve_fit(gauss_2d, (x, y), foto_raveled, p0=guess)
    #err = np.sqrt(np.diag(cov))
    return np.array(par), np.array(err), guess

"""
       
def foto_slice(foto, lower, point, axis:int):
    size = len(foto)
    point = point.astype(int)
    lower = lower.astype(int)
    
    if axis == 0:
        X = np.arange(size) + lower[1]
        Y = (foto[point[1] - lower[0]])
        XX = np.transpose(np.array([np.linspace(lower[1], lower[1] + size,
                                        1000),np.ones(1000) * point[1]]))
        
    elif axis == 1:
        X = np.arange(size) + lower[0]
        Y = (foto[:,point[0] - lower[1]])
        XX = np.transpose(np.array([np.ones(1000) * point[0], 
                np.linspace(lower[0], lower[0] + size, 1000)]))
        
        
    return X, Y, XX
 
    
def plot_gauss_2d(foto, lower, par):
    fig, ax = plt.subplots(figsize=(12,18), ncols=3)
    
    ax[0].imshow(foto, vmin=0, vmax = 255, cmap='hot')
    ax[0].plot(par[2]-lower[1], par[1]-lower[0], "x")
    
    x0, y0, X0 = foto_slice(foto, lower, par[1:3], 0)
    Y0 = gauss_2d(X0, *par)
    
    
    ax[1].plot(x0, y0, ".")
    ax[1].plot(X0[:,0], Y0)
    
    x1, y1, X1 = foto_slice(foto, lower, par[1:3], 1)
    Y1 = gauss_2d(X1, *par)
    
    ax[2].plot(x1, y1, ".")
    ax[2].plot(X1[:,1], Y1)
    
    plt.show()
    
def foto_slider(fotos, lowers, number_of_fotos, pars, errs):
    fig, ax = plt.subplots(ncols=3)
    fig.subplots_adjust(bottom=0.25)
    
    foto = fotos[0]
    par = pars[0]
    lower = lowers[0]
    err = errs[0]    
    
    foto_ax = ax[0].imshow(foto, vmin=0, vmax=255, cmap='hot')
    foto_fit, = ax[0].plot(*(par[1:3] - np.roll(lower, 1)), "x")
    #foto_fit_err, (foto_fit_err_top, foto_fit_err_bot, xym, yym), (bars,)  = ax[0].errorbar(*(par[1:3] - 
     #               lower), xerr=err[1], yerr=err[2], fmt="none", capsize=0)
        
    x0, y0, X0 = foto_slice(foto, lower, par[1:3], 0)
    Y0 = gauss_2d(X0, *par)

    xcut_ax, = ax[1].plot(x0, y0, ".")
    xcut_fit, = ax[1].plot(X0[:,0], Y0)
        
    ax[1].set_title(f"xcut, x=")
    ax[1].set_ylim(0,255)
        
        
    x1, y1, X1 = foto_slice(foto, lower, par[1:3], 1)
    Y1 = gauss_2d(X1, *par)
        
    ycut_ax, = ax[2].plot(x1, y1, ".")
    ycut_fit, = ax[2].plot(X1[:,1], Y1)
        
    ax[2].set_title(f"ycut, y=")
    ax[2].set_ylim(0,255)
    
        
        
        
    ax_pic = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        
        
    samp = Slider(ax_pic, "pic nr", 0, number_of_fotos - 1, valinit=0, valstep=1, color="green")
        
        
    def update(val):
    
        index = samp.val
        foto_ax.set_data(fotos[index])
        foto_fit.set_data(*(pars[index,1:3] - np.roll(lowers[index],1)))
        #foto_fit_err_bot = pars[index,2] - errs[index,2]
        #foto_fit_err_top = pars[index,2] + errs[index,2]
        
        x0, y0, X0 = foto_slice(fotos[index], lowers[index], (pars[index])[1:3], 0)
        Y0 = gauss_2d(X0, *(pars[index]))
        
        
        xcut_ax.set_data(x0, y0)
        xcut_fit.set_data(X0[:,0], Y0)
        
        x1, y1, X1 = foto_slice(fotos[index], lowers[index], (pars[index])[1:3], 1)
        Y1 = gauss_2d(X1, *(pars[index]))
        
        ycut_ax.set_data(x1, y1)
        ycut_fit.set_data(X1[:,1], Y1)
            
        
        ax[1].set_title(f"xcut, x=")
        ax[2].set_title(f"ycut, y=")
        fig.canvas.draw_idle()
    
    samp.on_changed(update)
    plt.show()
        
        
def position_graf(pos, err, lower):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    x1 = pos[:,0]  
    x2 = pos[:,1]  
    x_err = err[:,0]
    y_err = err[:,1]
    
    
    
    guess = np.array([-0.1, 400])
    par, uss = fit_chi2(linear, np.arange(len(x1)), x1, x_err+1, guess)
    print(par[0] * 1000 * 5)
    
    xx = np.linspace(0, len(x1), 1000)
    yy = linear(xx, *par)
        
    ax[0].plot(np.arange(len(x1)), x1, ".")
    ax[0].errorbar(np.arange(len(x1)), x1, yerr=x_err, fmt="none")
    ax[0].plot(xx, yy)
    #ax[0].plot(np.arange(len(x1)), lower[:,0][:-1])
    #ax[0].set_xlabel('Step_nr')
    ax[0].set_ylabel('Distance [pixels]')
    ax[0].set_title("y axis")
    
    guess = np.array([-0.1, 400])
    par, uss = fit_chi2(linear, np.arange(len(x2)), x2, y_err+1, guess)
    print(par[0] * 1000 * 5)
    
    xx = np.linspace(0, len(x1), 1000)
    yy = linear(xx, *par)
    
    ax[1].plot(np.arange(len(x2)), x2, ".")
    ax[1].plot(xx, yy)
    ax[1].errorbar(np.arange(len(x2)), x2, yerr=y_err, fmt="none")
    ax[1].set_xlabel('Step_nr')
    ax[1].set_ylabel('Distance [pixels]')
    ax[1].set_title('x axis')
    plt.show()
    
def position_graf_with_residuals(pos, err, lower, title_, text_pos):
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True)
    x1 = pos[:,0]  
    x2 = pos[:,1]  
    x_err = err[:,0]
    y_err = err[:,1]
    
    
    
    guess = np.array([-0.1, 400])
    pary, ussy = fit_func(np.arange(len(x1)), x1, y_err, linear, guess, ax[0,0], text_pos[:2])
    #pary, ussy = fit_chi2(linear, np.arange(len(x1)), x1, x_err+1, guess)
    print(pary[0] * 1000 * 5)
    
    xx = np.linspace(0, len(x1), 1000)
    yy = linear(xx, *pary)
        
    ax[0,0].plot(np.arange(len(x1)), x1, ".")
    ax[0,0].errorbar(np.arange(len(x1)), x1, yerr=x_err, fmt="none")
    ax[0,0].plot(xx, yy)
    #ax[0].plot(np.arange(len(x1)), lower[:,0][:-1])
    #ax[0].set_xlabel('Step_nr')
    ax[0,0].set_ylabel('Distance [pixels]')
    ax[0,0].set_title(f"y axis {title_}")
    
    
    guess = np.array([-0.1, 400])
    parx, ussx = fit_func(np.arange(len(x2)), x2, x_err, linear, guess, ax[1,0], text_pos[2:])
    #parx, ussx = fit_chi2(linear, np.arange(len(x2)), x2, y_err+1, guess)
    print(parx[0] * 1000 * 5)
    
    xx = np.linspace(0, len(x1), 1000)
    yy = linear(xx, *parx)
    
    ax[1,0].plot(np.arange(len(x2)), x2, ".")
    ax[1,0].plot(xx, yy)
    ax[1,0].errorbar(np.arange(len(x2)), x2, yerr=y_err, fmt="none")
    ax[1,0].set_xlabel('Step_nr')
    ax[1,0].set_ylabel('Distance [pixels]')
    ax[1,0].set_title(f"x axis {title_}")
    
    
    
    model_data_y = linear(np.arange(len(x1)), *pary)
    
    
    ax[0,1].plot(np.arange(len(x1)), x1 - model_data_y, ".")
    ax[0,1].errorbar(np.arange(len(x1)), x1 - model_data_y, yerr=x_err, fmt="none")
    ax[0,1].plot(np.array([0, len(x1)]), np.zeros(2))
    #ax[0].plot(np.arange(len(x1)), lower[:,0][:-1])
    #ax[0].set_xlabel('Step_nr')
    ax[0,1].set_ylabel('Distance [pixels]')
    ax[0,1].set_title(f"y axis residuals {title_}")
    
    
    
    model_data_x = linear(np.arange(len(x2)), *parx)
    
    ax[1,1].plot(np.arange(len(x2)), x2 - model_data_x, ".")
    ax[1,1].plot(np.array([0, len(x2)]), np.zeros(2))
    ax[1,1].errorbar(np.arange(len(x2)), x2 - model_data_x, yerr=y_err, fmt="none")
    ax[1,1].set_xlabel('Step_nr')
    ax[1,1].set_ylabel('Distance [pixels]')
    ax[1,1].set_title(f'x axis residuals {title_}')
    
    residuals_x = x2 - model_data_x
    residuals_y = x1 - model_data_y
    
    residuals = np.array([residuals_x, residuals_y])
    
    plt.show()
    return residuals



def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res

def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))

def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None

def fit_func(x, y, error, func, guess, ax, text_pos=np.array([0.1, 0.85])):
    chi2_object = LeastSquares(func, x, y, error)
    chi2_object.errordef = 1.0
    minuit = Minuit(chi2_object, *guess)
    minuit.migrad();
    par = minuit.values[:]
    err = minuit.errors[:]
    chi2 = minuit.fval
    Ndof = len(y) - len(par)
    p_val = scipy.stats.chi2.sf(chi2, Ndof)
    d = {
     'fit-type':    "chi2 fit",
     'a':          [par[0], err[0]],
     'b':       [par[1], err[1]],
     'Chi2':     chi2,
     'ndf':      Ndof,
     'Prob':     p_val,
    }
    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(*text_pos, text, ax, fontsize=8)
    ax.legend(loc="best")
    
    return par, err

def fit_func_v2(x, y, error, func, guess):
    chi2_object = LeastSquares(func, x, y, error)
    chi2_object.errordef = 1.0
    minuit = Minuit(chi2_object, *guess)
    minuit.migrad();
    par = minuit.values[:]
    err = minuit.errors[:]
    
    return par, err

def hist_chi2fit(X, xmin, xmax, bins, func, guess, ax):
    global number, bin_width
    counts, bin_edges = np.histogram(X, bins = bins, range=(xmin, xmax))
    ax.hist(X, bins=bins, range=(xmin, xmax), histtype="step")
    

    bin_width = bin_edges[1] - bin_edges[0]
    
    bin_m = bin_edges[1:] - (bin_edges[1] - bin_edges[0]) / 2
    number = len(X)
    
    mask = counts > 0
    x = bin_m[mask]
    y = counts[mask]
    
    ax.errorbar(x, y, yerr=np.sqrt(y), fmt=".", capsize=2, color="black")
    
    chi2_object = LeastSquares(func, x, y, np.sqrt(y))
    chi2_object.errordef = 1.0
    minuit = Minuit(chi2_object, *guess)
    minuit.migrad();
    par = minuit.values[:]
    err = minuit.errors[:]
    chi2 = minuit.fval
    XX = np.linspace(xmin, xmax, 1000)
    YY = func(XX, *par)
    ax.plot(XX, YY, label="chi2 fit")
    Ndof = len(x) - len(par)
    p_val = scipy.stats.chi2.sf(chi2, Ndof)
    ax.set_ylabel(f"Counts / {bin_width}")
    
    d = {
     'fit-type':    "chi2 fit",
     'mu':          [par[0], err[0]],
     'sigma':       [par[1], err[1]],
     'Chi2':     chi2,
     'ndf':      Ndof,
     'Prob':     p_val,
    }

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.1, 0.85, text, ax, fontsize=8)
    ax.legend(loc="best")
    
def unbinned_fit(X, xmin, xmax, func, guess, ax):
    c = cost.UnbinnedNLL(X, func)
    minuit = Minuit(c, *guess)
    minuit.migrad();
    par = minuit.values[:]
    err = minuit.errors[:]
    XX = np.linspace(xmin, xmax, 1000)
    YY = func(XX, *par)
    Ndof = len(X) - len(par)
    ax.plot(XX, YY, label="unbbined likelyhoodfit")
    rank_corr = scipy.stats.spearmanr(X, np.arange(len(X)))
    d = {
     'fit-type':    "Unbinned likelyhood",
     'sigma':       [minuit.values['sigma'], minuit.errors['sigma']],
     'ndf':         Ndof,
     #'rank corr':   rank_corr.statistic,
     'rank p_val':   rank_corr.pvalue,
    }

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.1, 0.65, text, ax, fontsize=8)
    ax.legend(loc="best")

def gauss_hist_fit(x, mu, sigma):
    global number, bin_width
    return 1/np.sqrt(2 * np.pi) / sigma *np.exp(-1/2 * (x - mu) ** 2 / (sigma ** 2)) * number * bin_width
    

def residual_analasis(x, bins, xmin, xmax, title):
    
    fig, ax = plt.subplots(figsize=(8,6))

    
    number = len(x)
    bin_width = (xmax - xmin) / bins
    guess = np.array([np.mean(x), np.std(x)])
    
    #ax.hist(x)
    
    hist_chi2fit(x, xmin, xmax, bins, gauss_hist_fit, guess, ax)
    unbinned_fit(x, xmin, xmax, gauss_hist_fit, guess, ax)
    
    ax.set_xlabel("x value")
    
    ax.set_title("Residual distribution" + title)
    #plt.savefig("monecarlo_with_unbinned.eps")
    
    plt.show()
    

    
def scatterplot(pos, err, title):
    fig, ax = plt.subplots()
    x1 = pos[:,0]  
    x2 = pos[:,1]  
    x_err = err[:,0]
    y_err = err[:,1]
    
        
    ax.plot(x1, x2,".-")
    ax.errorbar(x1, x2,  yerr=x_err, xerr=y_err, fmt="none")
    
   
    ax.set_ylabel('Distance [pixels]')
    ax.set_xlabel('Distance [pixels]')
    ax.set_title("laser position "+ title)
    
   
    plt.show()
    
    
def foto_mean(fotos, lower, N_pics_step):
    number_of_fotos = len(fotos)
    fotos_new = np.zeros((int(number_of_fotos / N_pics_step), len(fotos[0]), len(fotos[0, 0])))
    lower_new = np.zeros((int(number_of_fotos / N_pics_step), len(lower[0])))
    uncertenty = np.zeros((int(number_of_fotos / N_pics_step), len(fotos[0]), len(fotos[0, 0])))
    print(np.shape(fotos_new))
    for i in range(len(fotos_new)):
        fotos_new[i] = np.mean(fotos[i * N_pics_step: (i + 1) * N_pics_step], axis=0)
        uncertenty[i] = np.std(fotos[i * N_pics_step: (i + 1) * N_pics_step], axis=0)
        lower_new[i] = np.mean(lower[i * N_pics_step: (i + 1) * N_pics_step], axis=0)
        mask = uncertenty[i] == 0
        uncertenty[i][mask] = 0.5
        uncertenty[i] /= np.sqrt(N_pics_step)
        
    return fotos_new, lower_new, uncertenty



"""
def find_spot(X, foto):
    
    center = find_peak(foto)
    size_ = 20
    foto_zoomed, lower, higher = zoom_pixel(foto, size_, center)
        
    pars, errs, guesss = fit_2d_gauss_multi(np.array([foto_zoomed]), np.array([lower]), 1)

    laser_pos = pars[:,1:3][0]
    print(laser_pos)
        
    delta_pos = X - laser_pos
    
    return delta_pos
"""

def find_spot(mirror, cam, end_pos, steps_moved, expected_step_size=np.array([None, None]), max_steps=None):
    
    
    if expected_step_size[0] == None:
        if max_steps == None:
            while True:
                foto = cam.take_foto()
                center = find_peak(foto)
                size_ = 20
                foto_zoomed, lower, higher = zoom_pixel(foto, size_, center)
                pars, errs, guesss = fit_2d_gauss_multi(np.array([foto_zoomed]), np.array([lower]), 1)
            
                laser_pos = pars[:,1:3][0]
                delta_pos = end_pos - laser_pos
                mirror.move_step(1, -int(delta_pos[0]*50))
                mirror.move_step(2, -int(delta_pos[1]*50))
                steps_moved[0] += -int(delta_pos[0]*50)
                steps_moved[1] += -int(delta_pos[1]*50)
                if int(delta_pos[0]*50) == 0 and int(delta_pos[1]*50) == 0:
                    break
        else:
            for _ in range(max_steps):
                foto = cam.take_foto()
                center = find_peak(foto)
                size_ = 20
                foto_zoomed, lower, higher = zoom_pixel(foto, size_, center)
                pars, errs, guesss = fit_2d_gauss_multi(np.array([foto_zoomed]), np.array([lower]), 1)
            
                laser_pos = pars[:,1:3][0]
                delta_pos = end_pos - laser_pos
                mirror.move_step(1, -int(delta_pos[0]*50))
                mirror.move_step(2, -int(delta_pos[1]*50))
                steps_moved[0] += -int(delta_pos[0]*50)
                steps_moved[1] += -int(delta_pos[1]*50)
                if int(delta_pos[0]*50) == 0 and int(delta_pos[1]*50) == 0:
                    break
    
    else:
        foto = cam.take_foto()
        center = find_peak(foto)
        size_ = 20
        foto_zoomed, lower, higher = zoom_pixel(foto, size_, center)
        pars, errs, guesss = fit_2d_gauss_multi(np.array([foto_zoomed]), np.array([lower]), 1)
    
        laser_pos = pars[:,1:3][0]
        steps_expected = np.zeros(2)
        if max_steps == None:
        
            while True:
                foto = cam.take_foto()
                center = find_peak(foto)
                size_ = 20
                foto_zoomed, lower, higher = zoom_pixel(foto, size_, center)
                pars, errs, guesss = fit_2d_gauss_multi(np.array([foto_zoomed]), np.array([lower]), 1)
                
                laser_pos = pars[:,1:3][0]
                #print('laser pos, expected stepsize', laser_pos, expected_step_size)
                delta_pos = end_pos - laser_pos
                
                steps_expected += expected_step_size
                step_dif = steps_expected - steps_moved
                mirror.move_step(1, int(np.round(step_dif[0])))
                mirror.move_step(2, int(np.round(step_dif[1])))
                steps_moved[0] += int(np.round(step_dif[0])) 
                steps_moved[1] += int(np.round(step_dif[1]))
                #print("expected, moved, dif", steps_expected, steps_moved, step_dif)
                if int(delta_pos[0]*50) == 0 and int(delta_pos[1]*50) == 0:
                    break
        
        
        else:
            for _ in range(max_steps):
                #foto = cam.take_foto()
                #center = find_peak(foto)
                #size_ = 20
                #foto_zoomed, lower, higher = zoom_pixel(foto, size_, center)
                #pars, errs, guesss = fit_2d_gauss_multi(np.array([foto_zoomed]), np.array([lower]), 1)
                
                #laser_pos = pars[:,1:3][0]
                #print('laser pos, expected stepsize', laser_pos, expected_step_size)
                #delta_pos = end_pos - laser_pos
                
                steps_expected += expected_step_size
                step_dif = steps_expected - steps_moved
                mirror.move_step(1, int(np.round(step_dif[0])))
                mirror.move_step(2, int(np.round(step_dif[1])))
                steps_moved[0] += int(np.round(step_dif[0])) 
                steps_moved[1] += int(np.round(step_dif[1]))
                #print("expected, moved, dif", steps_expected, steps_moved, step_dif)
                #if int(delta_pos[0]*50) == 0 and int(delta_pos[1]*50) == 0:
                    #break
            
            
            
            
            
    return steps_moved
    
def length(vector):
    """
    Calculates the length of a vector

    Parameters
    ----------
    vector : Numpy array
        The numpy array schuld be of shape (n, x).

    Returns
    -------
    float or 1 dimentional numpy array
        Returns a numpy array with shape(n) with the legnths oif the vectors.

    """
    return np.sqrt(np.sum(vector ** 2, axis=0))
    
    
    
    
def move_to_spot(mirror, dX):
    """
    Moves the laser a number of pixels

    Parameters
    ----------
    mirror : object
        The object that controlls the mirrors.
    dX : 2d numpy array
        The distance to move.

    Returns
    -------
    None.

    """
    
    mirror.move_step(1, -int(dX[0]*50))
    time.sleep(1)
    mirror.move_step(2, -int(dX[1]*50))
    time.sleep(1)    
    
    
def find_spot_simple(mirror, cam, X):
    """
    Gradualy moves the laser to a given point X

    Parameters
    ----------
    mirror : object
        The object that controlls the mirrors.
    cam : object
        The object that controlls the camera.
    X : 2d numpy array
        The coordinates of the desiered laser position.

    Returns
    -------
    None.

    """
    
    foto = cam.take_foto()
    center = find_peak(foto)
    size_ = 20
    foto_zoomed, lower, higher = zoom_pixel(foto, size_, center)
    
    pars, errs, guesss = fit_2d_gauss_multi(np.array([foto_zoomed]), np.array([lower]), 1)

    laser_pos = pars[:,1:3][0]
    
    delta_pos = X - laser_pos
   
    move_to_spot(mirror, delta_pos)
    

def correct_mirror_pos(position_expected, position_wanted, unit_vector, mirror):
    """
    Corrects the laser position given a expected position and a wanted position

    Parameters
    ----------
    position_expected : 2d numpy array
        The expected position of the laser.
    position_wanted : 2d numpy array
        The wanted position of the laser.
    unit_vector : (4, 2)d numpy array
        A colection of the 4 vector describing a single step of the mirror in worward x backward x forward y backward y.
    mirror : object
        The object controlling the mirros.

    Returns
    -------
    position_expected : 2d numpy array
        The new expected position.
    position_wanted : 2d numpy array
        The wanted position.

    """
    corrected = False
    diff_vector = position_wanted - position_expected
    while not corrected:
        corrected = True
        if diff_vector[0] >= 1:
            position_expected[0] += 1
            diff_vector = position_wanted - position_expected
            mirror.move_step(1, 1)
            corrected = False
            
            
        elif diff_vector[0] <= -1:
            position_expected[0] += -1 
            diff_vector = position_wanted - position_expected
            mirror.move_step(1, -1)
            corrected = False
            
        
        if diff_vector[1] >= 1:
            position_expected[1] += 1
            diff_vector = position_wanted - position_expected
            mirror.move_step(2, 1)
            corrected = False
            
        
        elif diff_vector[1] <= -1:
            position_expected[1] += -1
            diff_vector = position_wanted - position_expected
            mirror.move_step(2, -1)
            corrected = False
            
        
    return position_expected, position_wanted
    
    
    
def generate_unit_vector(cam, mirror, axis, step_size, start_pos, direction, 
                         steps, foto_nr, foto_size = 20):
    """
    Generates the unitvector that discribe how the laser will move when the mirror steps

    Parameters
    ----------
    cam : object
        A object that controlles the camera.
    mirror : object
        A object that controlls the mirrors.
    axis : int
        Witch mirror axis to find the unitvector.
    step_size : int
        The step size used unde calibration.
    start_pos : 2d numpy array
        The laser position from where the calibration schuld start.
    direction : 1 or -1
        Describing if the mirror shuld step in the positive or negative direction along the axis.
    steps : int
        Number of steps taken durring calibration.
    foto_nr : int
        Number of fotos taken for each step.
    foto_size : int, optional
        The size of the cropped image. that is square. The default is 20.

    Returns
    -------
    unit_vector : 2d numpy array
        A vector that describes how the laser will move with one step by the mirror along the given axis and direction.

    """
    
    
    for _ in range(6):
        find_spot_simple(mirror, cam, start_pos)
    
    calibration_fotos = np.zeros((2, 2 * foto_nr, foto_size, foto_size))
    lower = np.zeros((2, (steps + 1) * foto_nr, 2))
    higher = np.zeros((2, (steps + 1) * foto_nr, 2))
    calibration_fotos = np.zeros((2, (steps + 1) * foto_nr, foto_size, 
                                  foto_size))
    lower = np.zeros((2, (steps + 1) * foto_nr, 2))
    higher = np.zeros((2, (steps + 1) * foto_nr, 2)) 
    calibration_fotos_mean = np.zeros(((steps + 1), foto_size, foto_size))
    lower_mean = np.zeros(((steps + 1), 2))
    pic_nr = 0
    
    for j in range(foto_nr):
        foto = cam.take_foto()
        if j == 0:
            center = find_peak(foto)
        calibration_fotos[0, pic_nr], lower[0, pic_nr], \
        higher[0, pic_nr] = zoom_pixel(foto, foto_size, center)
        pic_nr += 1
    
    for _ in range(steps):
        mirror.move_step(axis, step_size * direction)
        
    
        for j in range(foto_nr):
            foto = cam.take_foto()
            if j == 0:
                center = find_peak(foto)
            calibration_fotos[0, pic_nr], lower[0, pic_nr], \
            higher[0, pic_nr] = zoom_pixel(foto, foto_size, center)
            pic_nr += 1
            
    for i in range(steps + 1):
        calibration_fotos_mean[i] = np.mean(calibration_fotos[0, i * foto_nr:
                                                        (1 + i) * foto_nr], 
                                                                axis=0)
        lower_mean[i] = np.mean(lower[0, i * foto_nr:(1 + i) * foto_nr],
                                axis=0)
    
    
    pars, errs, guesss = fit_2d_gauss_multi(calibration_fotos_mean, 
                                               lower_mean, (steps + 1))
    laser_pos = pars[:,1:3]
    
    par, uss = fit_func_v2(laser_pos[:,0], laser_pos[:,1], errs[:,2], 
                              linear, np.array([-0.1, 400]))
    
    unit_vector = np.array([-direction, par[0]])
    unit_vector *= length(laser_pos[-1] - laser_pos[0]) / length(unit_vector)
    unit_vector /= steps * step_size

    return unit_vector

def decompose_vector(vector, unit_vector_1, unit_vector_2):
    """
    Decomposes a vector in to a linear combination of the 2 unit vectors

    Parameters
    ----------
    vector : 2d numpy array
        The vector to be decomposed.
    unit_vector_1 : 2d numpy array
        The first unitvector oif the basis.
    unit_vector_2 : 2d numpy array
        The second unit vecto of the basis.

    Returns
    -------
    a : float
        The first decomposition constant.
    b : float
        The secound decomposition constant.

    """
    b = (vector[1] - vector[0] * unit_vector_1[1] / unit_vector_1[0]) / (
     unit_vector_2[1] - unit_vector_2[0] * unit_vector_1[1] / unit_vector_1[0])
    a = (vector[0] - b * unit_vector_2[0]) / unit_vector_1[0]

    return a, b


def decomposition_coeficent(vector, unit_vector):
    """
    Decomposes the vector in to a linear combinatio of two other vectors, but allaows for differant unit vector for back and forth

    Parameters
    ----------
    vector : 2d numpy array
        Vector to be decomposed.
    unit_vector : (4, 2)d numpy array
        The unitvectors of the decomposition. There are 4 as the stepsize is different backwards and forwards, so all coeficent are keept positive
    

    Returns
    -------
    2d numpy array
        A vector in the new basis.
    direction : 2d numpy array
        A numpy array to keep track of if the coeficents are forwards or backwards.

    """
    unit_vector_1 = unit_vector[0]
    unit_vector_2 = unit_vector[2]
    direction = np.ones(2)
    a, b = decompose_vector(vector, unit_vector_1, unit_vector_2)
    if a < 0:
        unit_vector_1 = unit_vector[1]
        direction[0] = -1
    if b < 0:
        unit_vector_2 = unit_vector[3]
        direction[1] = -1
    a, b = decompose_vector(vector, unit_vector_1, unit_vector_2)
    if a < 0 or b < 0:
        print("error, cooeficents are negative", a, b, vector)
        
    return np.array([a, b]), direction
    
def calibrate_stepsize(mirror, cam, step_size, steps, corner_pos, 
                       foto_nr, N_step_x, N_step_y):
    """
    Calibrates the mirror stepsize for a square 2d scan

    Parameters
    ----------
    mirror : object
        object for controlling the mirrors.
    cam : object
        object for controlling the cameras.
    step_size : 4d numpy array
        The stepsize to be sused under calibration for the 4 directions.
    steps : int
        How many steps to use for the calibration.
    corner_pos : (4,2)d numpy array
        The corners of the wanted grid, with the startig point first.
    foto_nr : int
        Number of fotos per step.
    N_step_x : int
        number of steps in the first direction.
    N_step_y : int
        number of steps in the secound direction.

    Returns
    -------
    step_size_new : (4,2)d numpy array
        The calibrated stepsizes.
    unit_vector : (4,2)d numpy array
        The unit vectors of the basis that the step space is in.

    """
    
    unit_vector_x_f = generate_unit_vector(cam, mirror, 1, step_size[0],
                                           corner_pos[0], 1, steps, foto_nr)
    unit_vector_x_b = generate_unit_vector(cam, mirror, 1, step_size[1],
                                           corner_pos[1], -1, steps, foto_nr)
    unit_vector_y_f = generate_unit_vector(cam, mirror, 2, step_size[2],
                                           corner_pos[0], 1, steps, foto_nr)
    unit_vector_y_b = generate_unit_vector(cam, mirror, 2, step_size[3],
                                           corner_pos[2], -1, steps, foto_nr)
    
    unit_vector = np.array([unit_vector_x_f, unit_vector_x_b, unit_vector_y_f, 
                            unit_vector_y_b])
    
    
    vector = np.zeros((4, 2))
    vector[0] = corner_pos[1] - corner_pos[0]
    vector[1] = corner_pos[0] - corner_pos[1]
    vector[2] = corner_pos[2] - corner_pos[0]
    vector[3] = corner_pos[0] - corner_pos[2]
    
    decomposed_coeficcents = np.zeros((4, 2))
    direction = np.zeros((4, 2))
    
    for k in range(4):
        decomposed_coeficcents[k], direction[k] = decomposition_coeficent(
                                    vector[k], unit_vector)
    
    
    step_size_new = decomposed_coeficcents * direction

    step_size_new[:2] /= N_step_x
    step_size_new[2:] /= N_step_y
    
    return step_size_new , unit_vector   
    

def step_to_pixel_transformation(step, unit_vector):
    """
    Transforms a number of steps to a pixel location on the camera

    Parameters
    ----------
    step : 2d numpy array
        The steps to be transformed to pixels.
    unit_vector : (4,2)d numpy array
        The unitvectors that describes 1 step in pixels.

    Returns
    -------
    pixel : 2d Numpy array
        The transformed distance in pixelses.

    """
    pixel = np.zeros(2)
    
    for i in range(2):
        if step[i] >= 0:
            pixel[0] += unit_vector[i * 2, 0] * step[i]
            pixel[1] += unit_vector[i * 2, 1] * step[i]
        else:
            pixel[0] += unit_vector[i * 2 + 1, 0] * step[i]
            pixel[1] += unit_vector[i * 2 + 1, 1] * step[i]
    
    return pixel


def pixels_to_steps(vector, unit_vector):
    """
    Takes a vector in pixelses ind converts it to steps

    Parameters
    ----------
    vector : 2d numpy array
        The pixels to be convered to steps.
    unit_vector : (4, 2)d numpy array
        The unitvectors to convert.

    Returns
    -------
    TYPE
        The coresponding steps to the pixels.

    """
    coeficent, direction = decomposition_coeficent(vector, unit_vector)
    
    return coeficent * direction
    
