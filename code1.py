# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:44:22 2022

@author: Hy-Q @NBI, KU
"""
import numpy as np
from scipy.signal import sawtooth
import time
import matplotlib.pyplot as plt
import sys
sys.path.append("D:\Controll_of_Camera.py")
import Controll_of_Camera as cc
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from code2 import SLMscan as ssscan


# Camera-picture size
size_a = 768
size_b = 1024
pathth = f'N:\SCI-NBI-qplab-data\OldLab\HyQ1000\Data\slm_nikos\Images'

def main():
    
    cam = cc.Camera()
    cam.exposure = 0.1
    scan = 1

    if scan == 1:
        
        # Size of the SLM display after zooming
        zoom_amount = 1
        
        # Create an instance of the SLMscan class
        slm_scan = ssscan(cam, zoom_amount)
        
        PeriodList = np.arange(1, 150.1, 1)  
        AmplitudeList = np.arange(125, 125.1, 1.0) 

        img_array = slm_scan.scan(PeriodList, AmplitudeList)
    
        my_slm = ssscan(cam, size_a, size_b)
        my_slm.save_images(img_array, pathth)


    else:    
    # Create an instance of the SLMscan class
        my_slm = ssscan(cam, size_a, size_b)
    
    # Call the constant_pattern method with desired amplitude and period
        amplitude = 125.0  # Set the desired amplitude
        frequency = 50.0  # Set the desired period
        phase_shift = 0.0
        my_slm.constant_pattern(amplitude, frequency, phase_shift)
        

if __name__ == '__main__':
    main()
