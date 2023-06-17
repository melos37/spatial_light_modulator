# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:44:22 2022

@author: QP
"""
import instrumental as ins
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

sys.path.append('C:\Anaconda\Lib\site-packages\instrumental\drivers\cameras')
from instrumental.drivers.cameras import uc480

sys.path.append('C:\Program Files\Thorlabs\Scientific Imaging\DCx Camera Support\Develop/lib')
#sys.path.append('C:/Users/QP/Sejr Bachlor project git reposetory/functions')
import Data_handling as dh
import numpy as np


class Camera:
    """
    Class for controlling the cameras
    """
    
    def __init__(self, camera_nr=1):
        """
        Activates the camera

        Parameters
        ----------
        camera_nr : int, optional
            1 or 2, Chosses witch camera to connect to. The default is 1.

        Returns
        -------
        None.

        """
        self.exposure_time = ins.u('0.1 millisecond')
        if camera_nr == 1:
            cam_str = "<ParamSet[UC480_Camera] serial=b'4102636780' model=b'D1024G13M' id=1>"
        if camera_nr == 2:
            cam_str = "<ParamSet[UC480_Camera] serial=b'4102893399' model=b'SC1280G12N' id=2>"
        cam = ins.list_instruments() # returns list of avalible cameras
        
        if len(cam) == 0:
            print("No cameras detected")
            return
        
        print('The following cameras was detected:', cam)
        
        self.camera = ins.drivers.cameras.uc480.UC480_Camera(cam_str, reopen_policy='reuse')

        
        
        
    def take_foto(self, show=False):
        """
        Take a foto and returns the foto as a 2D numpy array

        Parameters
        ----------
        show : bool, optional
            If True the image will be shown with a plt.imshow. The default is False.

        Returns
        -------
        foto : 2D numpy array
            The foto the camera has taken. It is in gray scale, and between 0 and 255 in each pixel.

        """
        foto = self.camera.grab_image(exposure_time=self.exposure_time)
        if show:
            plt.imshow(foto, vmin=0, vmax=255, cmap='hot')
            plt.colorbar()
            plt.show()
            
        if np.max(foto) < 50:
            print('Warning, Laser not found on camera') 
        plt.pause(0.05)
        return foto
    
    def close(self, force=False):
        self.camera1.close(force=force)
    
    @property
    def exposure(self):
        """
        Gets the current exposuretiem of the camera, it is by deafult 1 millisecond.

        Returns
        -------
        str
            The current exposuretime of the camrea.

        """
        return self.exposure_time
    
    @exposure.setter
    def exposure(self, value):
        """
        Sets the exposure time of the camera

        Parameters
        ----------
        value : float
            The desiered exposure time of the camera in units of milliseconds.

        Returns
        -------
        None.

        """
        self.exposure_time = ins.u(f'{value} milliseconds')
        
    
    
    def live_view(self, zoom_amount=1, center=None):
        
        def new_foto():
        
           
            foto = self.take_foto()
            
           # center_flat = np.argmax(foto)
           # center = np.unravel_index(center_flat, np.shape(foto))
           # foto_zoomed, lower, higher = dh.zoom(foto, zoom_amount, center)
            
            
            ln.set_data(foto) #_zoomed)
            ax.set_title(f"max intensity = {np.max(foto)}")
        foto = self.take_foto()
       # center_flat = np.argmax(foto)
       # center = np.unravel_index(center_flat, np.shape(foto))
       # foto_zoomed, lower, higher = dh.zoom(foto, zoom_amount, center)
        fig, ax = plt.subplots()
        ln = ax.imshow(foto, vmin=0, vmax=255, cmap="hot") #foto_zoomed
        
        cbar = fig.colorbar(ln)
        
        fignr = fig.number
        while True:
            if plt.fignum_exists(fignr):
                new_foto()
                plt.pause(0.05)
            else:
                break
            