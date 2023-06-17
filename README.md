# spatial_light_modulator
Here some scripts for control of a spatial light modulator are submitted. It contains script for control and scanning of different parameters as well as tools to analyze the different results.


Key Libraries
Several key libraries were utilized throughout the characterization process, playing essential roles in SLM and camera control, sawtooth pattern generation, and data handling.

slmpy:
The "slmpy" package was specifically designed for SLM control. It provided commands for SLM initialization, closure, and the crucial "update array" command for projecting custom patterns onto the SLM screen. This specialized functionality was necessary since generic libraries treating the SLM as a secondary screen and projecting images onto it were insufficient for our purposes.

scipy.signal:
The "scipy.signal" package's "sawtooth" method was employed to generate the sawtooth pattern required for our experiments. It offered efficient and reliable tools for signal processing and waveform generation.

Control of Camera:
The "Control of Camera" package played a crucial role in our project. It facilitated the calibration of the setup through the "live view" method and provided functionality for capturing and saving images in a suitable format for subsequent analysis. This package ensured accurate and efficient image acquisition.

Threading:
Threading was utilized to enable simultaneous processes such as image creation and projection, parameter scanning, and data acquisition. By implementing threading, we ensured the parallel operation of these processes, enhancing overall efficiency and performance.

Data Handling:
The "Data Handling" script was utilized in multiple steps of the process for data acquisition and analysis. It provided convenient methods for storing, organizing, and processing the acquired data, allowing for effective analysis and interpretation.

Key Steps of the Scanning Process
The scanning process involved several key steps to characterize and optimize the performance of the SLM. Here is an overview of the main steps:

Initialization and Configuration (code1):
The primary script initialized the camera and configured the desired zoom level for acquired images. It also allowed the input of parameter values to be scanned, specifying start and stop values and step sizes for each parameter.

Secondary Script Execution (code2):
The secondary script utilized the specified parameters and saved them in an array for further analysis. It introduced the "SLMScan" class, which encapsulated essential methods for the project.

Establishing Connections and Setting Up:
The secondary script established a connection with the camera, retrieved the zoom amount from the primary script, and initialized the SLM. The size of the SLM was determined to facilitate the creation of the grating pattern image.

Threading Implementation:
To enable concurrent operations, threading was implemented. One thread was responsible for scanning through the parameters and modifying their values incrementally at each step. Simultaneously, another thread read these updated values and generated the corresponding sawtooth pattern image using the amplitude, frequency, and phase shift.

Rescaling and Pattern Projection:
To ensure coherence in the projected image, a rescaling process was implemented. After the pattern creation stage, the pattern values were scaled to fall within the range of 0 to 255 using commands in the script. The rescaled pattern was then projected onto the SLM using the "slm.updateArray" command.

Image Capture and Storage:
At each step, the camera captured an image using the "Control of Camera" package, preserving the parameter values associated with that particular step. The captured images were stored in an array.

Image Conversion:
A separate method was employed to convert the captured images into both PNG and GIF formats. This conversion facilitated further analysis and examination of the acquired images.

By developing this Python script and utilizing the key libraries mentioned above, we achieved comprehensive control and manipulation of the SLM, enabling precise characterization and optimization of its performance. The customized script empowered us to tailor the sawtooth pattern generation process to our specific experimental requirements, enhancing the accuracy and versatility of our SLM-based investigations.
