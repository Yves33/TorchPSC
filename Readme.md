TorchPSC
========
Torchpsc is an experimental tool to perform Post Synaptic Current (PSC) analysis on a GPU (NVIDA only, for now, although it should be possible to use AMD through ROCm cupy interface).  
(c) Yves33  
![screenshot](screenshot.png "")  

1- Requirements:
----------------
Torchpsc uses CUPY to filter, extract and analyse PSC (post synaptic events). Depsite it's name, it does not use PyTorch (for now!)
+ cuda toolkit (GPU side computations)
+ cupy  (GPU side computations)
+ cusignal (for argrelextrema. cusignal should not be required with cupy>=13)
+ cuML (PCA, UMAP on GPU)
+ numpy (CPU side computations)
+ scipy (curve_fit, iirnotch, filtfilt, savgol_filter) (till these are implemented in cupy...)
+ h5py to save the results
+ pypeclip to quickly copy summary
+ python-neo
+ pyOpenGL
+ imgui-bundle (many thanks to the authors!)

2- Features:
------------
+ The software can accept axon *.abf or spike2 *.smr files (and maybe axographX files). Signal units has to be A (or a multiple), and only the first signal in file will be processed.
+ Blazing fast display of current traces with detected PSCs using chunked min/max simplification of large traces
+ Real time rectification/filtering of input traces
+ Real time Wiener deconvolution of input signals using user defined template (biexponential)
+ Real time event extraction and feature caracterization
+ Real time measurements of PSC params:
  - Peak
  - Amplitude
  - Time to Peak
  - Half width
  - Area
  - Sharpness (number of crosses at half amplitude)
+ Real time visualisation of events PCA/UMAP
+ Fast PSC exponential decay fitting and residual error calculation
+ Powerfull event filtering mechanism based on extracted parameters
+ Additional manual event filtering
+ Burst detection
+ Cumulative amplitudes and intervals display
+ Save/load results files to hdf stores (easy to reparse in python)
+ Random crashes (It's not a feature, It's a bug!! hopefully less and less)

3- Usage:
---------
+ Start program using `python torchpsc.py`
+ Drop file on program main window.  
+ Start playing with parameters.  
+ Save results to hdf5 or copy results in clipboard.  
  
Notes:  
Events are extracted each frame. If too many events are extracted (for example, if you threshold is too low), the program will stop dynamic event extraction. You will then need to i) move threshold and/or adapt parameters to more suitable values ii) rearm the dynamic event extraction checkbox (in parametes->Event extraction).  

4- User interaction:
--------------------
+ **Controls:**  
  Most interactive controls are self explanatory (buttons, checkboxes, collapsibles, etc)
+ **Graphs:**
  the usual controls used in imgui::implot library are enabled in the program.   
  specifically:
  + mouse wheel on x-y axis to zoom in-out (mouse wheel on graph to zoom both x and y).
  + click& drag on x-y axis to pan axis.
  + right click on axis to adjust scales.
  + right click and drag on graph to zoom.

4- Detailed usage: Parameters:
------------------------------
+ **GUI options:**
  + Use sliders: choose to use slider as inputs instead of text
  + Show PCA: shows the PCA window (PCA of all extracted events. Only on Linux or WSL2)
  + Enable voltage traces: by default, the program will reject any file is units is not Amperes (or a multiple). This option enables to load signals even if units is not A.
+ **Filters:**  
  + Low pass: low pass butterworth filter with indicated cutoff frequency.  0 disables the filter
  + Band sop: Band stop filter (iirnotch) to remove electrical interference. 
  + Savgol: Savitzky-Golay filter. first parameter indicates number of points, second parameter the degree of polynom (only 2 and 3 are accepted).  
+ **Baseline correction:**
  + Scale factor: multiplies signal by value. This can be use to correct for improper scaling (e.g when using signals with wrong units) or signal rectification (the program will only detect upward events).  
  + Baseline average: duration to take into account for baseline correction.  
  + Smooth pts: number of points used for additional signal smoothing using sliding window mean algorithm.
  + Rectification: methods used to retify signal (keep=no transformation, abs=absolute value of signal, rms=Root Mean Saquare, clip additionnaly clips the signal to 0.
  + rms pts: number of points to consider for RMS filtering.
+ **Convolution:**
  The filter uses Wienner filter convolution followed by threshold extraction on the deconvolved trace. You therefore need to adapt the deconvolution parameters to the kinetics of the PSCs that you expect to extract.
  + Skip convolution: skips the convolution step and use a copy of signal for psc extraction
  + PSC rise time (ms): the rise time constant for template
  + PSC decay time (ms): the decay time constant fro template
  + Convolution llambda: "strength" of deconvolution.
+ **Event extraction:**  
  Event extraction occurs whenever the value of deconvolved signal (lower signal) crosses the threshold (white draggable cursor in lower signal). Crossings correspond to event peaks. 
  Once the peaks are found, the program walks back the signal to find event onset, and extract the entire event signal.
  + Enable/Disable current range: marks the current visible time window to disable enable event extraction during this time range.
  + Dynamic extraction: when set, events are extracted every frame (30-60 times per second). If number of events is too high, the program will stop dynamic extraction and  you will need to rearm the checkbox.
  + PSC duration (s): the length of the signal to consider (from peak) for decay fitting, event averaging and saving
  + PSC max rise time (s): the duration to extract before psc peak (also the duration before peak to search for psc onset)
  + PSC threshold: threshold for event extraction. Can also be set through cursor in bottom signal
  + Max burst gap: duration to sto considering events are grouped in a burst.
  + Min burst count: minimal number of events required to create a burst
  + Fit all: fits all events. fit is *NOT* performed in real time as it is performed on *CPU*. You should click the button to update all time constants.
+ **Grouped analysis:**  
  When opening simultaneously multiple files, the threshold is shared, but the filtering parameters are specific to each file, unless explicitely specified
  + Share filter: use the same Low/High values of filters for all currently opened files. Does not interfere with user filter.  
+ **Export:**  
  + Save: save the file to hdf5 store. the saved files can be reopened with PSCTorch and can be easily parsed using any programming language implementing hdf5 file parsing (h5py for python)
  + Copy csv: will dump individual event amplitudes, onsets, time constants and intervals to clipboad.
  + Copy summary: will dump a summary of current analysis.  
    - SYN_evt_ampl_mean:	average amplitude (peak - onset) of individual events.  
    - SYN_evt_ampl_std:	amplitude  standard deviation for individual events.  
    - SYN_evt_peak:	average peak of individual events.
    - SYN_evt_inter:	average interval between events
    - SYN_evt_wtc_mean:	average of individual events time constant
    - SYN_evt_wtc_std: standard deviation of individual events time constants
    - SYN_evt_overallfreq:	number of events / recording length
    - SYN_avgpsc_wtc: time constant of averaged event
    - SYN_avgpsc_ttp:	time to peak of averaged event
    - SYN_evt_fano: fano factor for event times
    - SYN_burst2psc_ratio:	ratio of number of events in burts / number of isolated events.
    - SYN_burst_content: average number of events in each burst.
    - SYN_burst_dura: average duration of bursts.
    - SYN_burst_overallfreq:  number of bursts / recording length.
+ **Event filtering:**
  It is often required to filter the extracted events based on their properties. The three graphs on the bottom of the main window enable visual fast and reproducible rejection of events using graphical tools.
  + Left panel: shows an histogram and enbales setting high low threshold for event acceptance/ rejection.
    - Top dropdown menu selects the measurement to consider (warning: in order to use Tau, you must first fit all events using Fit all button in parameters). 
    - Number of bins in histogramm is adjusted using the Bins control.
    - Left and right cursors on the histogram set low and high threshold for events  (seeing the cursors may require you to unzoom the x axis).
    - Low and high threshods may also be set using low/high bounds controls
    - Reset current filter and Reset all filters respectivelly reset current or all filterers.
  + Middle panel: shows individual event traces (only events in time range of top signal are displayed).
    - Normalize: normalizes individual events
    - Events: show individual events
    - Average: show event average (if normalize is checked, events are normalized before averaging)
    - Template: diplay the convolution event.
    - Valid/Rejected: show valid events or rejected events
    - Partial filter: when set shows accepted rejected events due to current filter settings.
  + Right panel: enables manual rejection of events. The panel show the currently selected event, and optionnally a fit on this event.  
    - click on red dots in upper signal to select event (may require to zoom top trace).
    - Fit, Autoscale and Reset user filter should be self explanatory.
    - Use mouse wheel on slider to move to next/previous event. click on slider to enable/disable event. Disabled events appear in red (if rejection is due to filter, the parameter not passing the filter is mentionned)

+ **Multiple files and cumulative curves:**  
  When multiple files are loaded (usually to assess the effect of drugs), the program will display cumulative probability curves in a separated window.
  + The top dopdown menu lets you choose between event amplitudes and event intervals.
  + The bins control sets the number of bins used for cumulative curve calculation.

4- Companion Jupyter notebooks:
------------------------------
Not ready yet. Be patient.
 
5-  Limitations and warning:
----------------------------
+ This software has been tested with fedora38 / cupy 12.8 
+ This software has been tested with windows 10 / cupy 13
+ It should theoretically work on windows using wsl2
+ This software is still alpha stage and may crash unexpectedly!

6- Todo:
--------
+ Selecting input channel when opening files
+ Implement machine learning approaches to classify IPSCs/EPSCs (using pytorch, aeon, pyts)
+ Jupyter companion notebook for high quality figure export (cumulative curves, statts per psc / per file)
+ More input file formats through neo-python
