# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 14:44:22 2022

@author: Hy-Q @NBI, KU
"""
import numpy as np
from scipy.signal import sawtooth
import time
import matplotlib.pyplot as plt
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
import cv2
import os
import re
import tkinter as tk
from PIL import Image, ImageTk
import threading
import pyslm
import slmpy
import sys
sys.path.append("D:\Controll_of_Camera.py")
import Controll_of_camera as cc



class SLMscan:
    def __init__(self, cam, zoom_amount, size_a=None, size_b=None):
        self.cam = cam
        self.zoom_amount = zoom_amount
        if size_a is None:
            self.size_a = int(768 / self.zoom_amount)
        else:
            self.size_a = size_a
        if size_b is None:
            self.size_b = int(1024 / self.zoom_amount)
        else:
            self.size_b = size_b

    def update_zoom_amount(self, zoom_amount):
        self.zoom_amount = zoom_amount
        self.size_a = int(768 / self.zoom_amount)
        self.size_b = int(1024 / self.zoom_amount)

    def scan(self, PeriodList, AmplitudeList, PhaseList=None, vertical=True):
        M = len(AmplitudeList)
        N = len(PeriodList)
        if PhaseList is not None:
            O = len(PhaseList)
            L = M * N * O
        else:
            L = M * N

        slm = slmpy.SLMdisplay()

        resX, resY = slm.getSize()

        # Make array to save pictures
        foto_zoomed = np.zeros((L, self.size_a, self.size_b), dtype=np.uint8)

        # Create a Tkinter window
        window = tk.Tk()

        # Create a Tkinter label to display the image
        image_label = tk.Label(window)
        image_label.pack()

        # Initializing for-loops to take measurements
        index = 0
        capture_thread_finished = False

        def capture_image(frequency, amplitude, phase_shift, scaling_factor):
            nonlocal index
            nonlocal foto_zoomed

            # Generate a sawtooth pattern
            x = np.linspace(0, 2 * np.pi, resX)
            pattern = amplitude * sawtooth(frequency * x + phase_shift)

            # Scale pattern values to fit within the range 0 to 255
            shifted_pattern = pattern + 127.5
            pattern_clipped = np.clip(shifted_pattern, 0, 255)
            pattern_uint8 = pattern_clipped.astype(np.uint8)

            # Generate an image by repeating the sawtooth pattern
            image = np.tile(pattern_uint8, (resY, 1))

            # Update the Tkinter label with the image
            image_pil = Image.fromarray(image)

            # Convert the PIL image to a numpy array
            array = np.array(image_pil)

            image_tk = ImageTk.PhotoImage(image_pil)
            image_label.config(image=image_tk)
            image_label.image = image_tk

            slm.updateArray(array)

            foto = self.cam.take_foto(False)
            zoom_amount = self.zoom_amount
            center = np.array([393, 540])
            zoomed_foto, _, _ = self.zoom(foto, zoom_amount, center)

            # Update the size of the foto_zoomed array if the dimensions have changed
            if zoomed_foto.shape != foto_zoomed.shape[1:]:
                foto_zoomed = np.zeros((L, zoomed_foto.shape[0], zoomed_foto.shape[1]), dtype=np.uint8)

            foto_zoomed[index] = zoomed_foto

            index += 1
            print(f'Image {index} corresponds to frequency = {frequency}, amplitude = {amplitude}')

            if index == L:
                # All images captured, notify that the capture thread has finished
                nonlocal capture_thread_finished
                capture_thread_finished = True

                # Schedule the destruction of the Tkinter window after a delay (in milliseconds)
                window.after(2000, window.destroy)

        def capture_thread():
            for i in range(len(PeriodList)):
                for j in range(len(AmplitudeList)):
                    frequency = PeriodList[i]
                    amplitude = AmplitudeList[j]
                    if PhaseList is not None:
                        phase_shift = PhaseList[j]
                    else:
                        phase_shift = 0.0

                    # Adjust the scaling factor based on the current amplitude value
                    scaling_factor = 10 / amplitude

                    capture_image(frequency, amplitude, phase_shift, scaling_factor)

        # Start the capture thread
        capture_thread = threading.Thread(target=capture_thread)
        capture_thread.start()

        # Run the Tkinter main loop
        window.mainloop()

        # Wait for the capture thread to finish
        while not capture_thread_finished:
            time.sleep(0.1)

        # Save the foto_zoomed array to a file
        path = f'N:\SCI-NBI-qplab-data\OldLab\HyQ1000\Data\slm_nikos\Images\Periods_=_{PeriodList[0]}_to_{PeriodList[-1]}_Amplitude_=_{AmplitudeList[0]}_to_{AmplitudeList[-1]}.npy'
        np.save(path, foto_zoomed)

        slm.close()

        # Return the foto_zoomed array
        return foto_zoomed
