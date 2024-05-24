#conda create -n cupy
#conda activate cupy
#conda install -c conda-forge cupy
#conda install -c conda-forge python-neo pyopengl scipy h5py pyperclip
#pip install imgui-bundle
#pip install nvidia-cuda-nvrtc-cu11

##
## TODO:
# - compare with older versions.
# - import *.itsq files and convert to h5df (not trivial. not that usefull!)
# - change PCA checkbox to combo None|PCA|UMAP
# - create CLassifier checkbox HDBScan|GMM
# - auto threshold button (super-easy)
# - replace most changed,x=imgui by _,x=imgui. when changed is not used
# - random forest / cnn classifier aeon, pyts,tslearn?
# - stumpy template matching. too slow!
#######################################################
#######################################################
## user settings
invalidation_window=0.002       ## width of the area to invalidate when an event is manually invalidated. default is 0.002 (2ms)
pixel_pick_radius=5             ## pick radius for traces and pca window
pixel_pt_size=2                 ## size in pixels of points for PCA
downsample_stride=50            ## downsample factor for wide views
downsample_threshold=50000      ## the threshold for switching between low resolution and high resolution traces, in pts
default_cmap=6                  ## default color map
default_shared_filters=False    ## should we share filters between files
default_share_threshold=True    ## should we share threshold between files.ignored. always True
default_show_cumul=False        ## show cumulative curves if at least two files are opened
default_show_pca=True           ## show PCA window
default_use_sliders=True        ## use sliders for parameters

## signal preprocessing
default_savgol=(9,2)            ## parameters for savgol filter
default_scale_factor=-1.0       ## optional signal scaling, in case your scales are wrong use negative to get upward peaks!
default_baseline_smooth=1.0     ## duration for baseline average
default_instant_smooth=1        ## signal convolution for smoothing. ignored if <2
default_rectify_method=0        ## keep,abs,rectify default is keep
default_rms_pts=5               ## number of points for rms rectification

## signal convolution
default_conv_risetime=0.2       ## rise time in milliseconds
default_conv_decaytime=2.0      ## decay time in milliseconds
default_llambda=2.333           ## ??

## epsc extraction
default_psc_threshold=None      ## automatically determined first time a file is processed
default_psc_duration=0.015      ## number of points to keep for epsc tracks (s).
default_ttp_searchwindow=0.005           ## how long to look for a peak after psc onset (s) must be less than default_psc_duration
default_psc_prepts=0.001        ## unused for now (s)
default_max_psc_number=5000     ## hard cutoff to number of psc. if number of psc is higher than that, continuous extraction is automatically turned off

default_search_units='A'        ## by default, search for traces in Ampères
default_display_units='pA'      ## display in pA
default_search_channel=0        ## take first channel with appropriate units
default_trace_id=0              ## in case there are severall episodes of recordings

benchmark=False                 ## developpers only!
## specify machine learning parameters
#default_decomposition= {"name":"PCA",
#                        "params":dict(n_components=4,svd_solver='full',random_state=42)
#                        }
default_decomposition= {"name":"UMAP",
                        "params":dict(n_neighbors=15,
                                    n_components=5,
                                    metric='cosine',
                                    learning_rate=0.1,
                                    min_dist=0.5,
                                    spread=3,
                                    hash_input=True,         ## mandatory
                                    random_state=0,          ## mandatory
                                    #deterministic=True,     ## mandatory
                                    init="random"            ## mandatory
                                    )           
                        }

default_scaler={"name":"RobustScaler",
                "params":dict()
                }
#default_clusterer={"name":"KMeans",
#                "params":dict(n_clusters=4)
#                }
default_clusterer={"name":"HDBSCAN",
                "params":dict(min_cluster_size=10, metric='euclidean', prediction_data=True)
                }

debug={'enable':False,
       'break':False}

####################
####################
from enum import IntEnum
import json
import time
import argparse
import OpenGL.GL as GL
from imgui_bundle import imgui,implot,ImVec2, ImVec4
# Always import glfw *after* imgui_bundle
# (since imgui_bundle will set the correct path where to look for the correct version of the glfw dynamic library)
import glfw
import numpy as np
import quantities as pq
from neo.core import AnalogSignal
from experiment import Experiment   ## to be removed or oversimplified? neo.autoio.readblock(0).segment()
from neomonkey import installmonkey ## to be removed
installmonkey()
## GPU accelerated
## the next version (13.0) of cupy should incorporate the following funcions
## argrelextrema, find_peaks, butter, lfilter, savgol_filter

try:
    from cusignal import argrelextrema ## should be incorporated to release cupy v13.0
except:
    try:
        from cupyx.scipy.signal import argrelextrema
    except:
        print("hwaccelerated argrelextream not available")
import cupy as cp
try:
    import cuml
    cuml_available=True
except:
    cuml_available=False
#from cuml import PCA ,UMAP
#from cuml.preprocessing import RobustScaler,MaxAbsScaler,StandardScaler,MinMaxScaler
#from cuml.cluster import KMeans,AgglomerativeClustering


## CPU
#from scipy.signal import argrelextrema
#import numpy as cp
#cp.asnumpy=lambda X:X
#from sklearn.decomposition import PCA ##,UMAP not available in sklearn
#from sklearn.preprocessing import RobustScaler,MaxAbsScaler,StandardScaler,MinMaxScaler
#from sklearn.cluster import KMeans,AgglomerativeClustering

from scipy.optimize import curve_fit
from scipy.signal import iirnotch,filtfilt,savgol_filter,butter,lfilter
import h5py
import pyperclip

from glfwapp import GLFWapp

######################
## retifying methods
lowpass_freqs=[0,1000,1500,2000,2500,3000,3500,4000,4500,5000]
rectify_method_names=["keep","abs","rms",'clip+rms']
class Rectify(IntEnum):
    keep=0          ## keep original signal
    abs=1           ## take abs. not suitable if you have both upwards and downward events 
    rms=2           ## take rms. not suitable if you haved both upwards and downward events
    cliprms=3       ## clipped rms

histogram_target_names=['Time to peak','Half-width','Peak','Amplitude','Area','Sharpness',"Tau","Log(Residual error)",'Score']
pca_histogram_target_names=['Time to peak','Half-width','Peak','Amplitude','Area','Sharpness',"Tau","Log(Residual error)",'Score','Clusters','Valid']
class Target(IntEnum):
    ttp=0           ## display time to peak
    halwidth=1      ## display hlafwidth
    peak=2          ## display peak value
    amp=3           ## display amplitude
    area=4          ## display area
    sharpness=5     ## display sharpness
    tau=6           ## psc time constant
    fiterr=7        ## fit residual error
    score=8         ## value of concolved signal
    clusters=9      ## clusters from hdbscan
    valid=10        ## rejected accepted pscs

cumul_target_names=["Amplitudes,Intervals","Tau"]
class Cumul(IntEnum):
    amplitude=0     ## display cumulated amplitudes
    interval=1      ## display cumulated intervals
    tau=2           ## display cumulated tau. not implemented

## colormaps.
cmap_names=["Deep","Dark","Pastel","Paired","Viridis","Plasma","Hot","Cool","Pink","Jet","Twilight","RdBu","BrBG","PiYG","SpectralnGreys"]
cmap_Deep     = 0   ## seaborn deep 
cmap_Dark     = 1   ## matplotlib "Set1" 
cmap_Pastel   = 2   ## matplotlib "Pastel1"
cmap_Paired   = 3   ## matplotlib "Paired" 
cmap_Viridis  = 4   ## matplotlib "viridis"
cmap_Plasma   = 5   ## matplotlib "plasma"
cmap_Hot      = 6   ## matplotlib/MATLAB "hot" 
cmap_Cool     = 7   ## matplotlib/MATLAB "cool"
cmap_Pink     = 8   ## matplotlib/MATLAB "pink"
cmap_Jet      = 9   ## MATLAB "jet"
cmap_Twilight = 10  ## matplotlib "twilight"
cmap_RdBu     = 11  ## red/blue, Color Brewer
cmap_BrBG     = 12  ## brown/blue-green, Color Brewer
cmap_PiYG     = 13  ## pink/yellow-green, Color Brewer 
cmap_Spectral = 14  ## color spectrum, Color Brewer 
cmap_Greys    = 15  ## white/black 

## cpu array of pscs
psc_fields=['onset_s','onset_t','onset_v','length_s','length_t','ttp_s','ttp_t','peak_s','peak_t','peak_v',\
            'amp_v','area_v','halfwidth_t','sharpness','tau','fiterr','score']
c_onset_s=0         ## onset in samples
c_onset_t=1         ## onset in s
c_onset_v=2         ## onset value
c_length_s=3        ## length in samples
c_length_t=4        ## length in seconds
c_ttp_s=5           ## time to peak in samples
c_ttp_t=6           ## time to peak in seconds
c_peak_s=7          ## peak in sample
c_peak_t=8          ## peak in seconds
c_peak_v=9          ## peak value
c_amp_v=10          ## amplitudes
c_area_v=11         ## areas
c_halfwidth_t=12    ## half width. only in seconds
c_sharpness=13      ## sharpness
c_tau=14            ## time constant of fitted exp
c_fiterr=15         ## log(residual fit err)
c_score=16          ## value of convolved signal
c_interval_s=17     ## interval in seconds

## flags to avoid reprocessing
flag_preprocess=1   ## notch and savgol filters
flag_rectify=2      ## we should re rerun rectification
flag_convolve=4     ## re run convolve
flag_extract=8      ## extract events again
flag_filter=16      ## filter events again. not used!

## commands issued from main panel
## for now commands are unused!
cmd_none=0
cmd_enable=1
cmd_disable=2
cmd_fit_events=3
cmd_copy=4
cmd_save_h5df=5
cmd_save_csv=6
cmd_classify_rf=7
cmd_add_to_training_set=8 ## takes all events and adds them to training set

wdtype='f4'       ## cp.float32 ## working dtype

def chunkedminmax(data,chunk_size=100):
    ## fast visually conformat resampling of np.array
    ## todo provide output array will avoid repetitive reallocation
    skip=len(data)%(chunk_size*2)
    if skip>0:
        arr=data[:-skip].reshape(-1,chunk_size*2) #arr=data[skip:].reshape(-1,chunk_size*2)
    else:
        arr=data[:].reshape(-1,chunk_size*2) #arr=data[skip:].reshape(-1,chunk_size*2)
    out=cp.zeros((arr.shape[0],2),dtype=cp.float32)
    out[:,0]=arr.min(axis=1)
    out[:,1]=arr.max(axis=1)
    return out.reshape(-1) ## flatten makes a copy, which is slow!

def chunkedmax(data,chunk_size=100):
    skip=len(data)%(chunk_size*2)
    return data[skip:].reshape(-1,chunk_size).max(axis=1)

def chunkedmin(data,chunk_size=100):
    skip=len(data)%(chunk_size*2)
    return data[skip:].reshape(-1,chunk_size).min(axis=1)

def lttb(data, n_bins=None,chunk_size=None):
    ## gives good results in numpy, but weird and slow using cupy!
    def area(a, bs, c):
        return cp.abs((a-c) - (a-bs)*2) ## 2*bs-c-a
    ## oversimplified visual resampling!
    if chunk_size:
        n_bins=len(data)//chunk_size+int(len(data)%chunk_size!=0)
    data_bins = cp.array_split(data, n_bins)
    out=cp.zeros(len(data_bins))
    out[0]=cp.mean(data_bins[0])
    out[-1]=cp.mean(data_bins[-1])
    for i in range(1,len(data_bins)-1):
        areas = area(out[i-1], data_bins[i], cp.mean(data_bins[i+1]))
        out[i]= data_bins[i][cp.argmax(areas)]
    return cp.array(out)

def _ge_(i,v=0): ## greater or equal and not None
    return not (i is None) and i>=v

def pygui_timeline(label,pos,lo,hi,active,height_px=4,rightbtn=2,margin=5):
    style=imgui.get_style()
    barcolor=imgui.color_convert_float4_to_u32(style.color_(imgui.Col_.frame_bg))
    if active:
        handlecolor=imgui.color_convert_float4_to_u32(style.color_(imgui.Col_.slider_grab))
    else:
        handlecolor=imgui.color_convert_float4_to_u32(ImVec4(0.7,0.0,0.0,1.0))
    wori=imgui.get_window_pos()                             ## screen position of window top left
    wsize = imgui.get_window_content_region_max()           ## actual size of window content (after padding substraction)
    wcursor=imgui.get_cursor_pos()                          ## window coordinates of cursor (top left of next widget,including padding)
    height=imgui.get_font_size()+2*style.frame_padding.y    ## height of a standard slider
    width=wsize.x-wcursor.x -2*margin                       ## width of slider to fill window
    if width<0:
        return False,None
    top=wori.y+wcursor.y                                    ## top of the bar (screen)
    left=wori.x+wcursor.x + margin                          ## left of bar (screen)
    ## rendering scrollbar and handle
    imgui.invisible_button("timeline",ImVec2(width,height))   ## invisible button is required to get drag.
    draw_list = imgui.get_window_draw_list()
    draw_list.add_rect_filled(ImVec2(left,top+height//2-height_px//2),ImVec2(left+width,top+height//2+height_px//2), barcolor)
    if not(pos is None) and pos>=0:  ## __ge__(pos)
        xc=left+width*(pos-lo)/(hi-lo)
        yc=top+height//2
        draw_list.add_circle_filled(ImVec2(xc,yc),height_px*2,handlecolor)
        if imgui.is_item_hovered():
            for btn in range(3):
                if imgui.is_mouse_clicked(btn): return True,int(10+btn)
            wheel=imgui.get_io().mouse_wheel
            if wheel:return True,int(wheel)
    return False,0

def detect_peaks(signal, threshold=0.5):
    root_mean_square = cp.sqrt(cp.sum(cp.square(signal) / len(signal)))
    ratios = cp.power(signal/root_mean_square,2)
    peaks = (ratios > cp.roll(ratios, 1)) & (ratios > cp.roll(ratios, -1)) & (ratios > threshold)
    return cp.where(peaks)[0]

def sharpness(data):
    #return cp.sum(cp.diff(cp.sign((data)))!=0)    ## (1) slowest
    #return cp.count_nonzero(cp.signbit(data[1:]*data[:-1])) ## (2) slightly faster
    return cp.count_nonzero(cp.logical_xor(cp.signbit(data[1:]),cp.signbit(data[:-1]))) ## (3) equivalent to (2)

class PSCFrame:
    def __init__(self,*args,**kwargs):
        ## fake method
        pass

    @classmethod
    def from_experiment(cls,parent,filename):
        cls=cls()
        cls.p=parent
        if filename:
            exp=Experiment(filename)
            if not parent._V2A:
                current=exp.signal(default_search_channel,default_search_units)[default_trace_id]
            else:
                current=exp.signal(default_search_channel,"V")[default_trace_id]
                current=AnalogSignal(current.magnitude, units='pA', sampling_rate=current.sr)
            cls.name=exp.name
            cls.filename=filename
        
        ## end awfull hack
        cls.sr=float(current.sr) ## make it float, don't bother with units
        cls.si=float(1/current.sr) ## make it float, don't bother with units
        cls.t_min=current.s[0]
        cls.t_max=current.s[-1]
        cls.length=len(current)
        cls.a_min=np.min(current.rescale(default_display_units))
        cls.a_max=np.max(current.rescale(default_display_units))
        cls.units=default_display_units
        cls.x_data={}
        cls.x_data[1]=np.array(current.s-current.s[0], dtype=wdtype)
        cls.x_data[downsample_stride]=np.array(current.s-current.s[0], dtype=wdtype)[::downsample_stride].copy()
        
        cls.current=current                                   ## original signal. on cpu
        cls.gpu_signal=cp.array(current.pA, dtype=wdtype)     ## signal on gpu, after optionnal notch and savgol
        cls.gpu_rectified=cp.array(current.pA, dtype=wdtype)  ## transformed signal on gpu
        cls.gpu_convolved=cp.array(current.pA, dtype=wdtype)  ## convolved signal on gpu
        try:
            cls.gpu_enabled=cp.ones(cls.length,dtype=cp.uint8)    ## boolean array indicating per point enabled/diabled
            cls.gpu_corrected=cp.ones(cls.length,dtype=cp.uint8)  ## boolean array indicating per point manually enabled events
        except:
            cls.gpu_enabled=cp.array(np.ones(cls.length,dtype=cp.uint8))    ## boolean array indicating per point enabled/diabled
            cls.gpu_corrected=cp.array(np.ones(cls.length,dtype=cp.uint8)) ## boolean array indicating per point manually enabled events

        cls.psc_onsets=cp.array([])            ## onset of pscs, in samples
        cls.psc_bases=cp.array([])             ## current at onset
        cls.psc_scores=cp.array([])            ## current sore, i.e the value of convolved data at peak
        cls.psc_ttp=cp.array([])               ## time to peak, in samples
        cls.psc_peaks=cp.array([])             ## current at peak
        cls.psc_length=cp.array([])            ## duration of each psc, in samples. shoerter than psc_duration when pscs are overlapping
        cls.psc_amps=cp.array([])
        cls.psc_areas=cp.array([])
        cls.psc_halfwidth=cp.array([])
        cls.psc_sharpness=cp.array([],dtype=cp.uint8)
        cls.psc_tau=cp.array([])
        cls.psc_fiterr=cp.array([])
        cls.psc_mask=cp.array([],dtype=cp.uint8)## psc mask
        cls.lohifilters={tgt:None for tgt in Target}
        cls.threshold_hard=None
        cls.selected_idx=None
        cls.burst_onoffs=cp.array([])
        cls.burst_counts=cp.array([])
        cls.flags= flag_rectify|flag_convolve|flag_extract ## on first run, signals must undergo full reprocessing
        return cls
    
    def preprocess(self):
        ## todo: should be implemented in hardware!
        ## neither cusignal nore cupy handle iir filters
        ## next release of cupy (13.0) should handle butterworth and savgol filters
        ## better be prepared!
        if self.flags & flag_preprocess:
            tmpsignal=self.current.pA
            if self.p.notch_filter:
                b_notch, a_notch = iirnotch(50.0, 30, self.sr)
                tmpsignal=filtfilt(b_notch,a_notch,tmpsignal)
                '''
                ##butterworth version
                bw=0.05
                nyq = 0.5 * self.sr
                fr=50 if self.p.notch_filter==1 else 60
                low = fr*(1-bw) / nyq
                high = fr*(1+bw) / nyq
                i, u = butter(3, [low, high], btype='bandstop')
                tmpsignal=lfilter(i, u, tmpsignal)'''
            if self.p.lopass_filter:
                cutoff=lowpass_freqs[self.p.lopass_filter]
                from scipy.signal import butter, lfilter, freqz
                b,a=butter(N=6, Wn=cutoff, fs=self.sr, btype='low', analog=False)
                tmpsignal = lfilter(b, a, tmpsignal)
            if all([s>=2 for s in self.p.savgol_filter]) and self.p.savgol_filter[0]>self.p.savgol_filter[1]:
                tmpsignal=savgol_filter(tmpsignal, *self.p.savgol_filter)
            self.gpu_signal=cp.array(tmpsignal).astype(wdtype)
        self.flags=self.flags & ~flag_preprocess

    def rectify(self):
        if self.flags & flag_rectify:
            npts=int(self.p.baseline_smooth*self.sr)
            self.gpu_rectified=self.gpu_signal*self.p.scale_factor
            if npts>10:
                self.gpu_rectified=self.gpu_rectified-cp.convolve(self.gpu_rectified,cp.ones(npts,dtype=cp.float32)/(npts), 
                                                            mode='same')#,method='fft')
            self.gpu_enabled[:npts//2]=0
            self.gpu_enabled[-npts//2:]=0
            if self.p.instant_smooth>1:
                self.gpu_rectified=cp.convolve(self.gpu_rectified,cp.ones(self.p.instant_smooth,dtype=cp.float32)/(self.p.instant_smooth), 
                                                mode='same')#,method='fft')
            match self.p.rectify_method:
                case Rectify.keep:
                    pass
                case Rectify.abs:
                    self.gpu_rectified=cp.fabs(self.gpu_rectified)
                case Rectify.rms:
                    sqsignal=cp.power(self.gpu_rectified,2)
                    window=cp.ones(self.p.rms_pts,dtype=cp.float32)/(self.p.rms_pts)
                    self.gpu_rectified=cp.sqrt(cp.convolve(sqsignal, window,mode='same'))#,method='fft'))
                case Rectify.cliprms:
                    self.gpu_rectified=cp.clip(self.gpu_rectified,cp.quantile(self.gpu_rectified,0.05),cp.inf)
                    sqsignal=cp.power(self.gpu_rectified,2)
                    window=cp.ones(self.p.rms_pts,dtype=cp.float32)/(self.p.rms_pts)
                    self.gpu_rectified=cp.sqrt(cp.convolve(sqsignal, window,mode='same')).astype(cp.float32)#,method='fft'))
            
            ## rms and abs may introduce a shift in baseline. we need to recenter!, by substracting the median or convolving (but slower)
            #self.gpu_rectified-=cp.median(self.gpu_rectified)
            self.gpu_rectified-=cp.mean(self.gpu_rectified)
            self.a_min=float(cp.min(self.gpu_rectified))
            self.a_max=float(cp.max(self.gpu_rectified))
        self.flags=self.flags & ~flag_rectify ## clear rectify flag and return

    def convolve(self):
        ## wiener filter deconvolution
        ## https://gist.github.com/dansuh17/20e7d50dbdf214f8d26c6f41a1d76dc9
        ## https://groups.google.com/g/scipy-user/c/4zQMRAb1rsk/m/RzUrrhNY6ukJ
        #https://github.com/AMikroulis/xPSC-detection/blob/master/xPSC%20detection/xPSC_detection.py
        if self.flags & flag_convolve:
            if self.p.skip_convolution:
                self.gpu_convolved=self.gpu_rectified.copy()
                self.template=cp.linspace(0,float(self.t_max),len(self.gpu_signal),dtype=cp.float32)*0
                ## adjust extraction
                #self.p.psc_threshold=cp.min(self.gpu_convolved)+(cp.max(self.gpu_convolved)-cp.min(self.gpu_convolved))*0.5#2*float(cp.quantile(self.gpu_convolved,0.99))
                self.p.psc_threshold=cp.max(self.gpu_convolved)
            else:
                t_psc=cp.linspace(0,float(self.t_max),len(self.gpu_signal),dtype=cp.float32)
                tau_1=self.p.conv_risetime/1000 ##*pq.ms,
                tau_2=self.p.conv_decaytime/1000 ##*pq.ms,
                Aprime = (tau_2/tau_1)**(tau_1/(tau_1-tau_2))
                #Aprime = (tau_2**(tau_1/(tau_1-tau_2)))/tau_1 ## https://molecularbrain.biomedcentral.com/articles/10.1186/s13041-021-00847-x
                self.template = 1./Aprime * (-cp.exp(-t_psc/tau_1) + cp.exp((-t_psc/tau_2))) #*int(2*int(p.upward)-1)
                self.template/=cp.max(self.template)
                H = cp.fft.fft(self.template)
                self.gpu_convolved = cp.real(cp.fft.ifft(cp.fft.fft(self.gpu_rectified)*cp.conj(H)/(H*cp.conj(H) + self.p.llambda**2)))
            self.threshold_hard=float(cp.quantile(self.gpu_convolved,0.999))
        self.flags=self.flags & ~flag_convolve ## clear rectify flag and return

    def extract(self):
        if not self.p._online_extraction:
            return
        if self.flags & flag_extract:
            if not self.p.psc_threshold:
                #self.p.psc_threshold=float(cp.mean(self.gpu_convolved)+cp.std(self.gpu_convolved)*4)
                self.p.psc_threshold=2*float(cp.quantile(self.gpu_convolved,0.99))
            ## hard limit on psc_threshold
            #self.p.psc_threshold=max(self.threshold_hard,self.p.psc_threshold)
            ## extract pscs in visible region.
            ## should move everything to one big array.
            ## self.pscprops=cp.zeros((10,len(self.psc_onsets)))
            _cnt=cp.array([int(self.sr*self.p.psc_duration)])
            _cnt2=cp.array([int(self.sr*self.p.psc_ttp_searchwindow)])
            self.psc_onsets   = argrelextrema(cp.clip(self.gpu_convolved,self.p.psc_threshold,None), cp.greater, order=7)[0]
            #self.psc_onsets = detect_peaks(cp.clip(self.gpu_convolved,self.p.psc_threshold,None),1.0) ## slower
            _valids           = self.gpu_enabled[self.psc_onsets]
            #_valids           = cp.take(self.gpu_enabled,self.psc_onsets)
            self.psc_onsets   = self.psc_onsets[_valids==1.0]
            self.psc_length   = cp.clip( cp.ediff1d(self.psc_onsets,to_end=_cnt) , 0, _cnt )
            #self.psc_complete = self.psc_length<_cnt
            self.psc_bases    = cp.take(self.gpu_rectified, self.psc_onsets)
            self.psc_data     = cp.split(self.gpu_rectified,cp.asnumpy(self.psc_onsets))[1:] ## cupy.split(): not that sequence on device memory is not allowed!
            self.psc_mask     = cp.ones_like(self.psc_onsets,dtype=cp.uint8)
            self.psc_scores   = cp.take(self.gpu_convolved, self.psc_onsets)
            if len(self.psc_data):
                padarr=cp.array([cp.nan]*int(_cnt),dtype=cp.float32)
                d=[data[:self.psc_length[i]] for i,data in enumerate(self.psc_data)]
                d=cp.array([cp.resize(cp.concatenate((data,padarr)),int(_cnt)) for data in d ])
                self.psc_ttp=cp.nanargmax(d[:,:_cnt2],axis=1) ## TODO MOVE TO self.p.max_ttp
                self.psc_peaks=cp.take(self.gpu_rectified, self.psc_onsets+self.psc_ttp)
                self.psc_amps=self.psc_peaks - self.psc_bases
                self.psc_areas=cp.nansum(d,axis=1)
                d2=d-self.psc_amps.reshape(len(self.psc_onsets),1)/2
                self.psc_halfwidth=cp.nansum(d2>0,axis=1)/self.sr
                self.psc_sharpness=cp.nansum(cp.diff(cp.sign(d2),
                                                    n=1,
                                                    axis=1)!=0,
                                            axis=1) ## behavior with nan is unknow
                self.psc_tau=cp.zeros_like(self.psc_onsets,dtype=cp.float32)    ## keep memory for tau. fitting is done on cpu, and only when requested
                self.psc_fiterr=cp.zeros_like(self.psc_onsets,dtype=cp.float32) ## keep memory for fit err (goodness of fit)!
                self.psc_data=d
                '''d=[data[:self.psc_length[i]] for i,data in enumerate(self.psc_data)]
                self.psc_ttp=cp.array([cp.argmax(data) for data in d])
                self.psc_peaks=cp.take(self.gpu_rectified, self.psc_onsets+self.psc_ttp)
                self.psc_amps=self.psc_peaks - self.psc_bases
                self.psc_areas=cp.array([cp.sum(data) for data in d])
                if fastcompute:
                    self.psc_areas=cp.array([cp.sum(data) for data in d])
                    self.psc_halfwidth=cp.zeros_like(self.psc_onsets)
                    self.psc_sharpness=cp.zeros_like(self.psc_onsets)
                else:
                    self.psc_areas=cp.array([cp.sum(data) for data in d])/self.sr
                    ## using intermediate array slightly fastens computaions on my machine!
                    d2=[data-self.psc_amps[i]/2 for i,data in enumerate(d)]
                    # approximate half width by substracting amplitude/2 summing all positive points
                    self.psc_halfwidth=cp.array([cp.sum(data>0) for data in d2])/self.sr
                    ## compute sharpness by substracting amplitude/2 and counting number of sign changes
                    self.psc_sharpness=cp.array([sharpness(data) for data in d2])
                self.psc_tau=cp.zeros_like(self.psc_onsets,dtype=cp.float32)    ## keep memory for tau. fitting is done on cpu, and only when requested
                self.psc_fiterr=cp.zeros_like(self.psc_onsets,dtype=cp.float32) ## keep memory for fit err (goodness of fit)!
                #self.psc_noise=cp.array([cp.quantile(data[self.psc_length[i]-50:self.psc_length[i]],0.95)-
                #                         cp.quantile(data[self.psc_length[i]-50:self.psc_length[i]],0.05) for i,data in enumerate(self.psc_data)])
                ## now that we have ttp and length for each psc, we can fit easily an exp that goes to 0
                ## we have i=Aoexp(t/tau), hence ln(i)=ln(Ao)+t/tau, but :
                ## - baseline is not strictly zero...
                ## - baseline can be negative (problem with log())
                ## - the wieight of the points should not be uniform
                ## - not much faster than scipy (0.5 s for 200-300 fits), and less accurate results
                #pr.disable()'''
            ## safeguard: if number of psc is too high, disable online event extraction and stop drawing
            if len(self.psc_onsets)>default_max_psc_number:
                self.p._online_extraction=False 
            self.selected_idx=None
        self.flags=self.flags & ~flag_extract ## clear rectify flag and return

    def fit(self):
        psc=self.psc_to_cpu()
        x=np.linspace(0,self.p.psc_duration,num=int(self.p.psc_duration*self.sr))
        s=cp.asnumpy(self.gpu_rectified)
        singleexp=lambda x, a, b, c: a*np.exp(-x/b)+c
        for i,evt in enumerate(psc):
            o=int(evt[c_onset_s])
            l=int(evt[c_length_s])
            ttp=int(evt[c_ttp_s])
            try:
                popt,pcov=curve_fit(singleexp, x[:l-ttp], 
                                                s[o+ttp:o+l],
                                                bounds=([0., 0.0002, -20],   [2*max(s[o+ttp:o+l]), 0.040, 20])
                                                )
                self.psc_tau[i]=1000*popt[1]
                self.psc_fiterr[i]=float(np.log10(1+np.sqrt(np.diag(pcov))[1]))
            except:
                popt=[np.nan,np.nan]
                pcov=[np.nan,np.nan]
                self.psc_tau[i]=np.nan
                self.psc_fiterr[i]=np.nan
        self.psc_fiterr=cp.clip(self.psc_fiterr,cp.quantile(self.psc_fiterr,0.05),cp.quantile(self.psc_fiterr,0.95))
        self.lohifilters[Target.fiterr]=None
        self.lohifilters[Target.tau]=None

    def make_bursts(self):
        ## the algorith is entirely done on gpu. compared to old algo, we do not enable for longer interval on start of burst
        ## a possibility would be to check if the event immediately before a burst has shorter than required interval, and change onset and count accordingly
        inters=cp.concatenate( (cp.array([-1]),
                               (self.psc_inter<self.p.burst_gap_min)*2-1,
                               cp.array([-1])
                               ))
        transitions=cp.where(cp.diff(cp.sign(inters)))[0]-1 ## index immediately before a transition occurs
        ##https://stackoverflow.com/questions/24342047/count-consecutive-occurences-of-values-varying-in-length-in-a-numpy-array
        burstcounts=1+cp.diff(cp.where(cp.concatenate( (cp.array([False]),
                                                 inters[:-1] != inters[1:],
                                                cp.array([True]))
                                            ))[0])[::2]
        burstonoffs=self.psc_onsets[self.psc_mask.astype(cp.bool_)][transitions+1].reshape(-1,2)
        self.burst_onoffs=burstonoffs[burstcounts>=self.p.burst_count_min]
        self.burst_counts=burstcounts[burstcounts>=self.p.burst_count_min]

    def update_mask(self):
        if len(self.psc_onsets):
            self.psc_mask=cp.ones_like(self.psc_onsets)
            for tgt,tgtdata in enumerate([self.psc_ttp/self.sr,
                                self.psc_halfwidth,
                                self.psc_peaks,
                                self.psc_amps,
                                self.psc_areas,
                                self.psc_sharpness,
                                self.psc_tau,
                                self.psc_fiterr,
                                self.psc_scores]):
                if self.lohifilters[tgt]:
                    self.psc_mask &= (self.lohifilters[tgt][0]<=tgtdata)
                    self.psc_mask &= (tgtdata<=self.lohifilters[tgt][1])
            self.psc_mask&=self.gpu_corrected[self.psc_onsets]
            ## intervals needs to be recomputed each time the mask has changed.
            ## as a consequence, the size of psc_intervals is always less than the size of mask
            ## assert( len(self.psc_mask)==cp.sum(self.psc_mask))
            self.psc_inter=cp.ediff1d(self.psc_onsets[self.psc_mask.astype(cp.bool_)],
                                 to_end=cp.array([len(self.gpu_rectified)-self.psc_onsets[-1]])) / self.sr
            self.make_bursts()

    def psc_to_cpu(self):
        if len(self.psc_onsets)<1:
            return np.array([])#np.array((0,16))
        #psc_fields=['onset_s','onset_t','onset_v','length_s','length_t','ttp_s','ttp_t','peak_s','peak_t','peak_v',\
        #'amp_v','area_v','halfwidth_t','sharpness','tau','fiterr','score']
        return cp.asnumpy(cp.array([
                      self.psc_onsets,                  ## 0 onset in samples
                      self.psc_onsets/self.sr,          ## 1 onset in s
                      self.psc_bases,                   ## 2 onset value
                      self.psc_length,                  ## 3 length in samples
                      self.psc_length/self.sr,          ## 4 length in seconds
                      self.psc_ttp,                     ## 5 time to peak in samples
                      self.psc_ttp/self.sr,             ## 6 time to peak in seconds
                      self.psc_onsets+self.psc_ttp,     ## 7 peak in sample
                      (self.psc_onsets+self.psc_ttp)/self.sr,## 8 peak in seconds
                      self.psc_peaks,                   ## 9 peak value
                      self.psc_amps,                    ## 10 amplitudes
                      self.psc_areas,                   ## 11 areas
                      self.psc_halfwidth,               ## 12 half width. only in seconds
                      self.psc_sharpness,               ## 13 sharpness
                      self.psc_tau,                     ## 14 time constant or fitterd curve
                      self.psc_fiterr,                  ## 15 log(residual error) of fit
                      self.psc_scores,                  ## 16 scores
                      ]).T).astype(np.float32)
    @classmethod
    def from_hdf(cls,parent,filename):
        cls=cls()
        if not filename:
            return None
        with h5py.File(filename, 'r') as f:
            groupname=list(f.keys())[0]
            g=f[groupname]
            info=json.loads(g.attrs['info'])
            cls.p=parent
            cls.name=groupname
            cls.sr=info['sr']
            cls.t_min=info['t_min']
            cls.units=info['units']
            cls.lohifilters={int(k):(*v,) if v else None for k,v in info['lohifilters'].items()}
            current=AnalogSignal(g['raw'][:],
                                units=cls.units,
                                t_start=cls.t_min*pq.s,
                                sampling_rate=cls.sr*pq.Hz)
            #cls.sr=float(current.sr) ## make it float, don't bother with units
            cls.si=float(1/current.sr) ## make it float, don't bother with units
            cls.t_min=current.s[0]
            cls.t_max=current.s[-1]
            cls.length=len(current)
            cls.a_min=np.min(current.pA)
            cls.a_max=np.max(current.pA)
            #cls.units='pA'
            cls.x_data={}
            cls.x_data[1]=np.array(current.s-current.s[0], dtype=wdtype)
            cls.x_data[downsample_stride]=np.array(current.s-current.s[0], dtype=wdtype)[::downsample_stride].copy()
            
            cls.current=current                                   ## original signal. on cpu
            cls.gpu_signal=cp.array(current.pA, dtype=wdtype)     ## signal on gpu, after optionnal notch and savgol
            cls.gpu_rectified=cp.array(current.pA, dtype=wdtype)  ## transformed signal on gpu
            cls.gpu_convolved=cp.array(current.pA, dtype=wdtype)  ## convolved signal on gpu
            cls.gpu_enabled=cp.array(np.unpackbits(g['enabled'][:].astype(np.uint8)),dtype=cp.uint8)      ## boolean array indicating per point enabled/diabled
            cls.gpu_corrected=cp.array(np.unpackbits(g['corrected'][:].astype(np.uint8)),dtype=cp.uint8)  ## boolean array indicating per point manually enabled events
            ## most psc_props are computed in real time and will be overriden at first frame.
            ## the only exception is tau and fiterr which are computer on demand!
            if 'psc_prop' in g.keys():
                pscs=g['psc_prop'][:]
                cls.psc_onsets=cp.array(pscs[:,0])            ## onset of pscs, in samples
                cls.psc_bases=cp.array(pscs[:,2])             ## current at onset
                cls.psc_scores=cp.array(pscs[:,16])           ## current sore, i.e the value of convolved data at peak
                cls.psc_ttp=cp.array(pscs[:,5])               ## time to peak, in samples
                cls.psc_peaks=cp.array(pscs[:,9])             ## current at peak
                cls.psc_length=cp.array(pscs[:,3])            ## duration of each psc, in samples. shorter than psc_duration when pscs are overlapping
                cls.psc_amps=cp.array(pscs[:,10])
                cls.psc_areas=cp.array(pscs[:,11])
                cls.psc_halfwidth=cp.array(pscs[:,12])
                cls.psc_sharpness=cp.array(pscs[:,13],dtype=cp.uint8)
                cls.psc_tau=cp.array(pscs[:,14])
                cls.psc_fiterr=cp.array(pscs[:,15])
                cls.psc_mask=cp.array(g['psc_mask'],dtype=cp.uint8)## psc mask
            else:
                cls.psc_onsets=cp.array([])            ## onset of pscs, in samples
                cls.psc_bases=cp.array([])             ## current at onset
                cls.psc_scores=cp.array([])            ## current sore, i.e the value of convolved data at peak
                cls.psc_ttp=cp.array([])               ## time to peak, in samples
                cls.psc_peaks=cp.array([])             ## current at peak
                cls.psc_length=cp.array([])            ## duration of each psc, in samples. shoerter than psc_duration when pscs are overlapping
                cls.psc_amps=cp.array([])
                cls.psc_areas=cp.array([])
                cls.psc_halfwidth=cp.array([])
                cls.psc_sharpness=cp.array([],dtype=cp.uint8)
                cls.psc_tau=cp.array([])
                cls.psc_fiterr=cp.array([])
                cls.psc_mask=cp.array([],dtype=cp.uint8)## psc mask
            #cls.lohifilters={tgt:None for tgt in Target}
            cls.threshold_hard=None
            cls.selected_idx=None
            for k,v in info['params'].items():
                setattr(cls.p,k,v)
            cls.burst_onoffs=cp.array([])
            cls.burst_counts=cp.array([])
            cls.flags= flag_rectify|flag_convolve|flag_extract ## on first run, signals must undergo full reprocessing
            return cls
        
    def to_hdf(self):
        prepoints=int(0.001*self.sr)
        postpoints=int(0.015*self.sr)
        evts=np.array([cp.asnumpy(self.gpu_rectified[o-prepoints:o+postpoints]) 
                       for o in self.psc_onsets ],dtype=np.float32)
        with h5py.File(pathlib.Path(self.filename).resolve().with_suffix('.hdf5'), 'w') as f:
            g=f.create_group(self.name)
            ## save arrays
            ## we save original signal (current), the two maskes (enabled and corrected).
            ## we also save the rectified signal (i.e notch+savgol+scale+convolutions+rectifications),
            ## but this will be reloaded upon reload
            ## other programs, however, may need this signal
            dataset=g.create_dataset('raw',data=np.array(self.current.pA),dtype=np.float32,
                                     compression="gzip", compression_opts=9) ## should move to self.signal!
            dataset.attrs['info']='raw untouched signal'
            dataset=g.create_dataset('enabled',data=np.packbits(cp.asnumpy(self.gpu_enabled)),dtype=np.uint8,
                                     compression="gzip", compression_opts=9) ## should move to self.signal!
            dataset.attrs['info']='boolean mask indicating enabled/disabled parts for psc extraction'
            dataset=g.create_dataset('corrected',data=np.packbits(cp.asnumpy(self.gpu_corrected)),dtype=np.uint8,
                                     compression="gzip", compression_opts=9) ## should move to self.signal!
            dataset.attrs['info']='boolean mask for discarding events'
            dataset=g.create_dataset('rectified',data=np.array(cp.asnumpy(self.gpu_rectified)),dtype=np.float32,
                                     compression="gzip", compression_opts=9) ## we may save it, but will not be used when reloading
            dataset.attrs['info']='filtered and rectified signal'
            dataset=g.create_dataset('psc_data',data=evts,compression="gzip", compression_opts=9)
            dataset.attrs['info']='the trace for each psc'
            dataset=g.create_dataset('psc_mask',data=cp.asnumpy(self.psc_mask))
            dataset.attrs['info']='the current mask used to enable/disable pscs'
            dataset=g.create_dataset('psc_prop',data=self.psc_to_cpu())
            dataset.attrs['info']='individual psc properties'
            dataset.attrs['columns']='|'.join(psc_fields)
            dataset=g.create_dataset('burst_prop',data=cp.asnumpy(cp.array([
                                                            self.burst_onoffs[:,0],
                                                            self.burst_onoffs[:,1],
                                                            self.burst_counts,
                                                            ]).T).astype(np.float32))
            dataset.attrs['info']='onset/offset and count of each burst'
            dataset.attrs['columns']='onset|offset|count'
            ## attributes should be related to groups!
            info={}
            info['sr']=self.sr
            info['t_min']=float(self.t_min)
            info['units']=self.units
            info['lohifilters']={int(k):v for k,v in self.lohifilters.items()}
            info['params']={attr:getattr(self.p,attr) for attr in ['notch_filter','savgol_filter','scale_factor','baseline_smooth','instant_smooth',
                                                                   'rectify_method','rms_pts','conv_risetime','conv_decaytime',
                                                                   'llambda','psc_threshold','psc_threshold_max','psc_duration',
                                                                   'burst_gap_min','burst_count_min']}
            g.attrs['info']=json.dumps(info)

    def _fit_avg_(self,normalize=True):
        ## todo check if there are some events. return np.nan,np.nan
        m=self.psc_mask.astype(cp.bool_)
        if np.sum(m)<1:
            return np.nan, np.nan
        if not normalize:
            avg_data=cp.nanmean(self.psc_data[m],
                                     axis=0)
        else:
            avg_data=cp.nanmean( (self.psc_data[m]-cp.amin(self.psc_data[m],axis=1,keepdims=True))/cp.ptp(self.psc_data[m],axis=1,keepdims=True),
                                    axis=0)
        ttp=int(cp.argmax(avg_data))
        ## todo: find point where signa is 80% of its maximal value!
        ## i.e search last positive point in psc downshifted of 80% of its value
        ttp=int(cp.where( (avg_data-cp.max(avg_data)*0.8) >0)[0][-1])
        
        l=len(avg_data)
        s=cp.asnumpy(avg_data)
        x=np.linspace(0,self.p.psc_duration,num=l)
        singleexp=lambda x, a, b, c: a*np.exp(-x/b)+c
        popt,pcov=curve_fit(singleexp, x[:l-ttp], 
                                        s[ttp:].astype(float),
                                        bounds=([0., 0.0002, -20],   [2*max(s[ttp:]), 0.040, 20]),
                                        p0=[max(s[ttp:]), 0.001, 0.0]
                                        )
        return ttp/self.sr,popt[1]
    
    def dump(self):
        mask=cp.asnumpy(self.psc_mask).astype(np.bool_)
        pscs=cp.asnumpy(self.psc_to_cpu()).astype(np.float32)[mask]
        ##assert(len(self.psc_inter)==np.sum(mask))
        buffer='Amplitude\tOnset\tTau\tInterval\n'
        for i in range(len(pscs)):
            buffer+=f"{pscs[i][c_amp_v]}\t{pscs[i][c_onset_t]}\t{pscs[i][c_tau]}\t{self.psc_inter[i]}\n"
        pyperclip.copy(buffer)

    def summarize(self):
        pscs=cp.asnumpy(self.psc_to_cpu()).astype(np.float32)
        mask=cp.asnumpy(self.psc_mask.astype(cp.bool_)).astype(np.bool_)
        if len(pscs[mask]):
            r={}
            r["SYN_evt_ampl_mean"]=np.nanmean(pscs[mask,c_amp_v])*pq.pA
            r["SYN_evt_ampl_std"]=np.nanstd(pscs[mask,c_amp_v])*pq.pA
            r["SYN_evt_peak"]=np.nanmean(pscs[mask,c_peak_v])*pq.pA
            if len(self.psc_inter)>1:
                r["SYN_evt_inter"]=float(cp.nanmean(self.psc_inter[:-1]))*pq.s
            else:
                r["SYN_evt_inter"]=np.nan*pq.s
            ## todo. replace by nan when fit/psc is not complete
            taumask=cp.asnumpy(self.psc_length==cp.max(self.psc_length)).astype(np.bool_)
            r["SYN_evt_wtc_mean"]=np.nanmean(pscs[mask|taumask,c_tau])/1000*pq.s  ## this one may contain nan
            r["SYN_evt_wtc_std"]=np.nanstd(pscs[mask|taumask,c_tau])/1000*pq.s    ## this one may contain nan
            r["SYN_evt_overallfreq"]=np.sum(mask)/(self.t_max-self.t_min)
            ttp,wtc=self._fit_avg_()
            r["SYN_avgpsc_wtc"]=wtc * pq.s
            r["SYN_avgpsc_ttp"]=ttp * pq.s
            r["SYN_evt_fano"]=float(cp.std(self.psc_inter[:-1])/cp.mean(self.psc_inter[:-1]))
            r["SYN_burst2psc_ratio"]=float(cp.sum(self.burst_counts)/cp.sum(self.psc_mask))
            r["SYN_burst_content"]=float(cp.mean(self.burst_counts))
            r["SYN_burst_dura"]=float(cp.mean(self.burst_onoffs[:,1]-self.burst_onoffs[:,0])/self.sr)*pq.s
            r["SYN_burst_inter"]:float(cp.mean(cp.ediff1d(self.burst_onoffs[:,1])))*pq.s
            r["SYN_burst_overallfreq"]=len(self.burst_onoffs[:,1])/(self.t_max-self.t_min)
        else:
            r={
                'SYN_evt_ampl_mean': np.nan*pq.pA,
                "SYN_evt_ampl_std":np.nan*pq.pA,
                "syn_evt_peak":np.nan*pq.pA,
                "SYN_evt_inter":np.nan*pq.s,
                "SYN_evt_wtc_mean":np.nan*pq.s,
                "SYN_evt_wtc_std":np.nan*pq.s,
                "SYN_evt_overallfreq":0/pq.s,
                "SYN_avgpsc_wtc":np.nan * pq.s,
                "SYN_avgpsc_ttp":np.nan * pq.s,
                "SYN_evt_fano":np.nan,
                "SYN_burst2psc_ratio":np.nan,
                "SYN_burst_content":0,
                "SYN_burst_dura":0,
                "SYN_burst_inter":np.nan*pq.s,
                "SYN_burst_overallfreq":0/pq.s
            }
        buffer=''
        for k,v in r.items():
            buffer+=f"{k}\tdimension\t{float(v)}\n"
        pyperclip.copy(buffer)
        return r
    
##https://pyts.readthedocs.io/en/stable/generated/pyts.classification.TimeSeriesForest.html
class PSCapp(GLFWapp):
    def __init__(self,*args):
        super().__init__()

        implot.create_context()
        implot.get_input_map().zoom_rate=0.5

        self._signals=[]
        self._current_signal=-1

        ## signal processing parameters
        self.notch_filter=0
        self.lopass_filter=0                          ## choice for lopass (0,1500,2000,2500,4000Hz)                    
        self.savgol_filter=default_savgol             ## parameters for savgol
        self.scale_factor=default_scale_factor        ## optional signal scaling, in case your scales are wrong!
        self.baseline_smooth=default_baseline_smooth  ## duration for baseline average
        self.instant_smooth=default_instant_smooth    ## signal convolution for smoothing
        self.rectify_method=default_rectify_method    ## kee,negate,rectify
        self.rms_pts=default_rms_pts                  ## number of points for rms rectification

        ## signal convolution
        self.skip_convolution=False                   ## by default, make a convolution
        self.conv_risetime=default_conv_risetime      ## rise time in msec
        self.conv_decaytime=default_conv_decaytime    ## decay time in msec
        self.llambda=default_llambda                  ## ??

        ## epsc extraction
        self.psc_threshold=default_psc_threshold      ## automatically determined first time a file is processed
        self.psc_threshold_max=None                   ## initial value of upper bounds for threshold
        self.psc_duration=default_psc_duration        ## number of points to keep for epsc tracks. also maximal duration to search for peak
        self.psc_ttp_searchwindow=default_ttp_searchwindow              ## how long to look for a peak after onsets have been identified
        self.psc_prepts=default_psc_prepts            ## unused for now

        ## burst grouping
        self.burst_gap_min=0.1                        ## minimum gap between two events for considering as a burst
        self.burst_count_min=4                        ## minimum number of events to make a burst
        
        ## 
        self._enable_selection=0     ## shouldn't be here. rather move to command
        self._disable_selection=0    ## shouldn't be here. rather move to command
        self._online_extraction=True ## automatically turns to False if number of events is too high
        self._thrstep=0.0

        ## epsc filtering
        self._filt_tgt=0         ## the value that is filtered
        self._filt_bins=20       ## number of bins in histogram
        self._normalize=False    ## display normalized pscs
        self._average=False      ## dispaly average of pscs
        self._template=False      ## dispaly average of pscs
        self._each_psc=True      ## display all pscs (eats half of the frames)
        self._partial_filter=False
        self._valid_psc=True     ## display valid/invalid pscs (eats half of the frames)
        self._fit=True           ## show individual psc fits
        self._fit_autoscale=False

        ## gui otions
        self._want_sliders=True      ## use sliders instead of input box
        self._want_shared_filters=False ## by default,filters low and high thresholds are not shared across files
        self._pixel_tolearnce=5      ## psc picking tolerance (pixels)
        self._V2A=False              ## enable fake V to A convertion
        self._inited=True              ## wether we are at the very first frame

        ## internal use only
        self._slo=None
        self._shi=None
        self._flags=0x00
        self._pca_tgt=0
        self._pca_all=0
        self._cumul_tgt=0
        self._cumul_bins=300
        self._pca_cmap=default_cmap
        self._show_pca=default_show_pca

        ## dimensionality reduction and clusterering
        if cuml_available:
            self._scaler_cuml=getattr(cuml.preprocessing,default_scaler["name"])(**default_scaler["params"])
            self._reducer_cuml=getattr(cuml,default_decomposition["name"])(**default_decomposition["params"])
            self._clusterer_cuml=getattr(cuml,default_clusterer["name"])(**default_clusterer["params"])
        #self._reducer_cuml = PCA(**pca_params)
        #self._reducer_cuml=UMAP(**umap_params)
        #self._clusterer_cuml=KMeans(n_clusters=8)

        ## argument parsing
        parser=argparse.ArgumentParser()
        parser.add_argument("-s", "--save", help='Auto save results on output', action="store_true")
        parser.add_argument('-f','--file', nargs='+', help='Input files', required=False)

        self.autosave=parser.parse_args().save
        if parser.parse_args().file:
            self.on_file_drop(None,parser.parse_args().file)

    def on_file_drop(self,window,files):
        for f in files:
            try:
                if pathlib.Path(f).suffix in ['.smr','.abf']:
                    self._signals.append(PSCFrame.from_experiment(parent=self,filename=f))
                    self._current_signal=len(self._signals)-1
                elif pathlib.Path(f).suffix in ['.hdf5','.h5py']:
                    self._signals.append(PSCFrame.from_hdf(parent=self,filename=f))
                    self._current_signal=len(self._signals)-1
            except:
                print("Unknown file type or corrupted file")    

    def slider_float(self,*args,**kwargs):
        if self._want_sliders:
            return imgui.slider_float(*args)
        else:
            return imgui.input_float(*args[0:2],**kwargs)

    def slider_int(self,*args,**kwargs):
        if self._want_sliders:
            return imgui.slider_int(*args)
        else:
            return imgui.input_int(*(args[0:2]),**kwargs)

    def paramwindow(self):
        global fastcompute
        imgui.begin("Parameters")
        flags=0x00 ## we have not modified anythin
        cmds=0x00  ## we have no command
        if imgui.collapsing_header("GUI options"):
            _,self._want_sliders=imgui.checkbox("Use sliders",self._want_sliders)
            _,self._show_pca=imgui.checkbox("Show PCA",self._show_pca)
            _,self._V2A=imgui.checkbox("Enable voltage traces",self._V2A)
            if debug["enable"]:
                _,debug['break']=imgui.checkbox("Break",debug['break'])
            imgui.text(f"FPS : {imgui.get_io().framerate:.1f}")

        if imgui.collapsing_header("Filters",imgui.TreeNodeFlags_.default_open):
            changed,self.lopass_filter=imgui.combo('Low pass',self.lopass_filter,[str(i) for i in lowpass_freqs])
            flags|=changed*(flag_preprocess|flag_rectify|flag_convolve|flag_extract|flag_filter)
            changed,self.notch_filter=imgui.combo('Band stop',self.notch_filter,['Disabled','50Hz','60Hz'])
            flags|=changed*(flag_preprocess|flag_rectify|flag_convolve|flag_extract|flag_filter)
            changed,self.savgol_filter=imgui.input_int2('Savgol (npts,deg)',self.savgol_filter)
            flags|=changed*(flag_preprocess|flag_rectify|flag_convolve|flag_extract|flag_filter)
            self.savgol_filter=(max(0,min(self.savgol_filter[0],20)),
                                max(0,min(self.savgol_filter[1],4)))

        if imgui.collapsing_header("Baseline correction",imgui.TreeNodeFlags_.default_open):
            ## baseline correction
            changed,self.scale_factor=imgui.input_float("Scale factor",self.scale_factor)
            flags|=changed*(flag_rectify|flag_convolve|flag_extract|flag_filter)
            changed,self.baseline_smooth=self.slider_float("Baseline average",self.baseline_smooth,0.5,10)
            flags|=changed*(flag_rectify|flag_convolve|flag_extract|flag_filter)
            changed,self.instant_smooth=self.slider_int("Smooth pts",self.instant_smooth,1,10)
            flags|=changed*(flag_rectify|flag_convolve|flag_extract|flag_filter)
            changed,self.rectify_method=imgui.combo("Rectification",self.rectify_method,rectify_method_names)
            flags|=changed*(flag_rectify|flag_convolve|flag_extract|flag_filter)
            changed,self.rms_pts=self.slider_int("RMS pts",self.rms_pts,4,50)
            flags|=changed*(flag_rectify|flag_convolve|flag_extract|flag_filter)

        if imgui.collapsing_header("Convolution",imgui.TreeNodeFlags_.default_open):
            ## convolution
            changed, self.skip_convolution=imgui.checkbox("Skip convolution",self.skip_convolution)
            flags|=changed*(flag_convolve|flag_extract|flag_filter)
            changed,self.conv_risetime=self.slider_float("PSC rise time (ms)",self.conv_risetime,0.01,10)
            flags|=changed*(flag_convolve|flag_extract|flag_filter)
            changed,self.conv_decaytime=self.slider_float("PSC decay time (ms)",self.conv_decaytime,self.conv_risetime*1.1,40)
            flags|=changed*(flag_convolve|flag_extract|flag_filter)
            changed,self.llambda=self.slider_float("Convolution llambda",self.llambda,0.5,10)
            flags|=changed*(flag_convolve|flag_extract|flag_filter)

        if imgui.collapsing_header("Event extraction",imgui.TreeNodeFlags_.default_open):
            ## event extraction. should add max_ttp and max_evt_length
            imgui.text("Current range");imgui.same_line()
            self._enable_selection=imgui.button("Enable");imgui.same_line()
            self._disable_selection=imgui.button("Disable")
            _,self._online_extraction=imgui.checkbox("Dynamic extraction",self._online_extraction)
            changed,self.psc_duration=self.slider_float("PSC duration (s)",self.psc_duration,0.002,0.040)
            flags|=changed*flag_extract
            changed,self.psc_ttp_searchwindow=self.slider_float("PSC max rise time (s)",self.psc_ttp_searchwindow,0.001,0.020)
            flags|=changed*flag_extract
            if self.psc_threshold:
                changed,self.psc_threshold=self.slider_float("PSC threshold",self.psc_threshold,
                                                             0,min(self.psc_threshold_max,3.4e38),step=self._thrstep)
                flags|=changed*(flag_extract|flag_filter)
            changed,self.burst_gap_min=imgui.input_float("Max burst gap (s)",self.burst_gap_min)
            self.burst_gap_min=max(0.01,self.burst_gap_min)
            #flags|=changed*flag_extract ## not required for bursts
            changed,self.burst_count_min=imgui.input_int("Min burst count",self.burst_count_min)
            self.burst_count_min=max(2,self.burst_count_min)
            #flags|=changed*flag_extract ## not required for bursts
            ## Todo: check if all time constants are zero (i.e events have just been extracted) and enable/diable button accordingly
            ## regroup all non real time controls (for now "fit") in a separate collaplible header 
            if imgui.button("Fit all"):
                self._signals[self._current_signal].fit()
            ## if we invalidated/revalidated a section, force event extraction
            flags|=(flag_extract|flag_filter)*(self._enable_selection or self._disable_selection)
        if imgui.collapsing_header("Grouped analysis",imgui.TreeNodeFlags_.default_open):
            _,self._want_shared_filters=imgui.checkbox("Share filters",self._want_shared_filters)

        if len(self._signals):
            if imgui.collapsing_header("Export",imgui.TreeNodeFlags_.default_open):
                if imgui.button("Save"):
                    self._signals[self._current_signal].fit()
                    self._signals[self._current_signal].to_hdf()
                    center = imgui.get_main_viewport().get_center()
                    imgui.set_next_window_pos(center, imgui.Cond_.appearing.value, ImVec2(0.5, 0.5))
                    imgui.open_popup("Save file")
                if imgui.begin_popup_modal("Save file", None, imgui.WindowFlags_.always_auto_resize.value)[0]:
                    imgui.text('Analyses was successfully saved')
                    if imgui.button("OK", ImVec2(120, 0)):
                        imgui.close_current_popup()
                    imgui.end_popup()

                imgui.same_line()
                if imgui.button("Copy csv"):
                    self._signals[self._current_signal].fit()
                    self._signals[self._current_signal].dump()
                    center = imgui.get_main_viewport().get_center()
                    imgui.set_next_window_pos(center, imgui.Cond_.appearing.value, ImVec2(0.5, 0.5))
                    imgui.open_popup("Copy results")
                if imgui.begin_popup_modal("Copy results", None, imgui.WindowFlags_.always_auto_resize.value)[0]:
                    imgui.text('xPSC amplitudes and intervals were copied into the clipboard')
                    if imgui.button("OK", ImVec2(120, 0)):
                        imgui.close_current_popup()
                    imgui.end_popup()

                imgui.same_line()
                if imgui.button("Copy Summary"):
                    self._signals[self._current_signal].fit()
                    self._signals[self._current_signal].summarize()
                    center = imgui.get_main_viewport().get_center()
                    imgui.set_next_window_pos(center, imgui.Cond_.appearing.value, ImVec2(0.5, 0.5))
                    imgui.open_popup("Copy summary")
                if imgui.begin_popup_modal("Copy summary", None, imgui.WindowFlags_.always_auto_resize.value)[0]:
                    imgui.text('The summarized analyses are now in the clipboard')
                    if imgui.button("OK", ImVec2(120, 0)):
                        imgui.close_current_popup()
                    imgui.end_popup()
        imgui.end()
        return flags,0x00

    def plotsignals(self):
        ## fetch epsc data to cpu, to avoid too many transfers
        signal=self._signals[self._current_signal] if self._current_signal>=0 else None
        cpu_pscs=signal.psc_to_cpu()
        cpu_mask=cp.asnumpy(signal.psc_mask).astype('bool')
        if implot.begin_subplots("##linked",2,1, ImVec2(-1, 2*imgui.get_font_size() * 15),implot.SubplotFlags_.link_cols):
            if implot.begin_plot("##current"):
                if self._inited:
                    implot.setup_axes_limits(signal.t_min, signal.t_max, signal.a_min, signal.a_max)
                bounds=implot.get_plot_limits()
                shi=min(int(bounds.x.max*signal.sr),len(signal.gpu_signal))
                slo=max(0,int(bounds.x.min*signal.sr))
                pltsize=implot.get_plot_size()
                xtol5=pixel_pick_radius*(shi-slo)/pltsize[0]                    ## converts 5 pixels to offset in array
                ytol5=pixel_pick_radius*(bounds.y.max-bounds.y.min)/pltsize[1]  ## converts 5 pixels to y units
                #hi0=min(int(bounds.x.max*signal.sr/10),len(signal.current))
                #lo0=max(0,int(bounds.x.min*signal.sr/10))
                # at that stage, it is not worth transferring only the visible part or cupy buffer.
                # the huge bottleneck here is drawing more than 300000 points.
                # visually conservative downsampling (LTTB) is too slow and unreliable on cupy!
                # minmax downsampling gives correct results
                ## huge stride gives epsc which are disconnected from traces
                if self._enable_selection:
                    signal.gpu_enabled[slo:shi]=1
                if self._disable_selection:
                    signal.gpu_enabled[slo:shi]=0
                if (bounds.x.max-bounds.x.min)*signal.sr > downsample_threshold:
                    #implot.plot_line("", signal.x_data[stride],cp.asnumpy(signal.gpu_rectified[::stride]),1)
                    #implot.plot_line("",cp.asnumpy(chunkedminmax(signal.gpu_rectified,chunk_size=downsample_stride)),xscale=downsample_stride/signal.sr)
                    implot.plot_line("", signal.x_data[downsample_stride][:-1],
                                            cp.asnumpy(chunkedminmax(signal.gpu_rectified,chunk_size=downsample_stride))
                                            )
                    implot.plot_digital("##enabled", signal.x_data[downsample_stride], 
                                                    cp.asnumpy(signal.gpu_enabled[::downsample_stride]).astype(np.float32))
                    #implot.plot_digital("##user", signal.x_data[downsample_stride], 
                    #                                cp.asnumpy(signal.gpu_corrected[::downsample_stride]).astype(np.float32),1)
                else:
                    if debug['break']:
                        debug['break']=False

                    implot.plot_line("", signal.x_data[1][slo:shi], 
                                        cp.asnumpy(signal.gpu_rectified[slo:shi]).astype(np.float32))
                    implot.plot_digital("##enabled", signal.x_data[1][slo:shi], 
                                                    cp.asnumpy(signal.gpu_enabled[slo:shi]).astype(np.float32)
                                                    )
                    #implot.plot_digital("##user", signal.x_data[1][slo:shi], 
                    #                                cp.asnumpy(signal.gpu_corrected[slo:shi]).astype(np.float32))
                if len(signal.psc_onsets):
                    ## fetches several times the cupy array. theoretically slower, but does not change a lot
                    #implot.plot_scatter("##onsets",
                    #                    cp.asnumpy(signal.psc_onsets/signal.sr).astype(cp.float32)[cpu_mask],
                    #                    cp.asnumpy(signal.psc_bases)[cpu_mask])
                    #implot.plot_scatter("##ttp",
                    #                    cp.asnumpy((signal.psc_onsets+signal.psc_ttp)/signal.sr).astype(cp.float32)[cpu_mask],
                    #                    cp.asnumpy(signal.psc_peaks)[cpu_mask])
                    #implot.set_next_marker_style(implot.Marker_.cross, pixel_pt_size)
                    implot.plot_scatter("##onsets",
                                        cpu_pscs[:,c_onset_t][cpu_mask],
                                        cpu_pscs[:,c_onset_v][cpu_mask])
                    #implot.set_next_marker_style(implot.Marker_.cross, pixel_pt_size)
                    implot.plot_scatter("##ttp",
                                        cpu_pscs[:,c_peak_t][cpu_mask],
                                        cpu_pscs[:,c_peak_v][cpu_mask])
                    ## check for mouse click on epsc selection:
                    if implot.is_plot_hovered and imgui.is_mouse_clicked(0) and not imgui.is_mouse_dragging(0):
                        pt = implot.get_plot_mouse_pos()
                        if slo<pt[0]*signal.sr<shi and bounds.y.min<pt[1]<bounds.y.max:
                            delta=cp.abs(signal.psc_onsets+signal.psc_ttp-(pt[0]*signal.sr))
                            selected_idx=cp.argmin(delta)  ## get closest psc
                            if delta[selected_idx]>xtol5 or np.abs(signal.psc_peaks[selected_idx]-pt[1])>=ytol5:
                                selected_idx=None
                                signal.selected_idx=None ## should we deselect points when clicking outside?
                            else:
                                signal.selected_idx=selected_idx
                            if selected_idx and imgui.get_io().key_ctrl:
                                o=signal.psc_onsets[selected_idx]
                                start,stop=o-int(invalidation_window*signal.sr),o+int(invalidation_window*signal.sr)
                                signal.gpu_corrected[start:stop]=1-signal.gpu_corrected[start:stop]
                                #signal.flags|=flag_extract
                    if _ge_(signal.selected_idx):
                        implot.tag_x(cpu_pscs[int(signal.selected_idx)][c_onset_t],ImVec4(1.0,1.0,0.0,1.0),f"{signal.selected_idx}")
                if len(signal.burst_onoffs):
                    implot.push_style_var(implot.StyleVar_.line_weight,6)
                    #implot.push_style_var(implot.StyleVar_.marker_size,6)
                    #implot.push_style_color(implot.Col_.marker_fill,ImVec4(1.0,0,0,1.0))
                    #implot.push_style_color(implot.Col_.marker_outline,ImVec4(1.0,0,0,1.0))
                    implot.push_style_color(implot.Col_.line,ImVec4(1.0,0,0,1.0))
                    #implot.set_next_marker_style(implot.Marker_.diamond)
                    #ImPlot::PlotLine("g(x)", xs2, ys2, 20,ImPlotLineFlags_Segments)
                    implot.plot_line("bursts", cp.asnumpy(signal.burst_onoffs.flatten()).astype(np.float32)/signal.sr,
                                         (bounds.y.min+3*ytol5)*np.ones(len(signal.burst_onoffs.flatten())).astype(np.float32),
                                         implot.LineFlags_.segments)
                    #implot.pop_style_color()
                    #implot.pop_style_color()
                    implot.pop_style_color()
                    #implot.pop_style_var()
                    implot.pop_style_var()
                ## let other graphs knwo the current bounds of graph
                self._slo,self._shi=slo,shi
                implot.end_plot()
            if implot.begin_plot("##deconvolved"):#, ImVec2(-1, plot_height)):
                bounds=implot.get_plot_limits()
                shi=min(int(bounds.x.max*signal.sr),len(signal.gpu_signal))
                slo=max(0,int(bounds.x.min*signal.sr))
                pltsize=implot.get_plot_size()
                self._thrstep=(bounds.y.max-bounds.y.min)/pltsize[1]
                if (bounds.x.max-bounds.x.min)*signal.sr>downsample_threshold:
                    #implot.plot_line("##convolved", signal.x_data[stride], cp.asnumpy(signal.gpu_convolved[::stride]))
                    implot.plot_line("##convolved", signal.x_data[downsample_stride][:-1],
                                     cp.asnumpy(chunkedminmax(signal.gpu_convolved,chunk_size=downsample_stride))
                                     )
                else:
                    implot.plot_line("##convolved", signal.x_data[1][slo:shi], 
                                     cp.asnumpy(signal.gpu_convolved[slo:shi]).astype(np.float32))
                #changed,self.vcursorpos,a,b,c=implot.drag_line_x(1,self.vcursorpos,ImVec4(1.0,1.0,1.0,1.0))
                changed,self.psc_threshold,a,b,c=implot.drag_line_y(1,self.psc_threshold,ImVec4(1.0,1.0,1.0,1.0))
                self.psc_threshold_max=float(cp.max(signal.gpu_convolved))*1.1
                self._flags|=changed*flag_extract
                implot.end_plot()
        implot.end_subplots()

    def plothistograms(self):
        signal=self._signals[self._current_signal] if self._current_signal>=0 else None
        changed,self._filt_tgt=imgui.combo("Filter",self._filt_tgt,histogram_target_names)
        _,self._filt_bins=imgui.slider_int("Bins",self._filt_bins,10,50)
        if implot.begin_plot("Histograms", ImVec2(-1, imgui.get_font_size() *12)):
            match self._filt_tgt:
                case Target.ttp:tgtdata=signal.psc_ttp/signal.sr
                case Target.halwidth:tgtdata=signal.psc_halfwidth
                case Target.peak:tgtdata=signal.psc_peaks
                case Target.amp:tgtdata=signal.psc_amps
                case Target.area:tgtdata=signal.psc_areas
                case Target.sharpness:tgtdata=signal.psc_sharpness
                case Target.tau:tgtdata=signal.psc_tau
                case Target.fiterr:tgtdata=cp.clip(signal.psc_fiterr,0,cp.max(signal.psc_fiterr[signal.psc_fiterr!=cp.inf]))
                case Target.score:tgtdata=signal.psc_scores
            if len(tgtdata)==0:
                implot.end_plot()
                return
            ## plot histogram
            h=cp.histogram(tgtdata,bins=self._filt_bins) ## for so little data, would it be faster to work on cpu?
            bar=cp.asnumpy(h[0]).astype(np.float32)
            x=cp.asnumpy(h[1]).astype(np.float32)
            implot.setup_axes(histogram_target_names[self._filt_tgt], "Occurence")
            scalecond=implot.Cond_.always if changed or self._flags else implot.Cond_.once ## need to reset scales when we change tgtdata
            implot.setup_axes_limits(min(x), max(x), 0, max(bar)*1.5,scalecond)
            barwidth=(x[1]-x[0])
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1,0, max(x)*1.1)
            implot.setup_axis_limits_constraints(implot.ImAxis_.y1, 0,max(bar)*1.5)
            implot.plot_bars("", x+barwidth/2,bar, (x[1]-x[0])*0.66)
            ##
            ## plot cursors for hi/lo filtering events
            ##
            if signal.lohifilters[self._filt_tgt] is None:
                #m=cp.mean(tgtdata)
                #hw=cp.ptp(tgtdata)
                #signal.lohifilters[self._filt_tgt]=(m-hw/2,m+hw/2)
                signal.lohifilters[self._filt_tgt]=(cp.min(tgtdata)*0.9,cp.max(tgtdata)*1.1)
            lof,hif=signal.lohifilters[self._filt_tgt]
            changed,hif,a,b,c=implot.drag_line_x(self._filt_tgt+10,hif,ImVec4(1.0,1.0,1.0,1.0))
            changed,lof,a,b,c=implot.drag_line_x(self._filt_tgt+20,lof,ImVec4(1.0,1.0,1.0,1.0))
            signal.lohifilters[self._filt_tgt]=(min(hif,lof),max(hif,lof)) ## don't need to re_extract. only update mask
            ##signal.update_mask() ##NOTAGOODIDEA
            ## kee a copy of partial filter
            self.current_filter_mask=(tgtdata<=hif)&(lof<=tgtdata)
            implot.end_plot()
        lof,hif=signal.lohifilters[self._filt_tgt]
        changed,lohi=imgui.input_float2("low / high bounds",ImVec2(lof,hif),format="%g")
        signal.lohifilters[self._filt_tgt]=(min(lohi[0],lohi[1]),max(lohi[0],lohi[1]))
        if imgui.button("Reset current filter"):
            signal.lohifilters[self._filt_tgt]=None
        imgui.same_line()
        if imgui.button("Reset all filters"):
            for tgt in range(Target.score+1):
                signal.lohifilters[tgt]=None
        if self._want_shared_filters:
            for tgtsignal in self._signals:
                tgtsignal.lohifilters=signal.lohifilters.copy()

    def plotpscs(self):
        signal=self._signals[self._current_signal] if self._current_signal>=0 else None
        _, self._normalize=imgui.checkbox("Normalize",self._normalize);imgui.same_line()
        _, self._each_psc = imgui.checkbox("Evtents",self._each_psc);imgui.same_line()
        _, self._average=imgui.checkbox("Average",self._average);imgui.same_line()
        _, self._template = imgui.checkbox("Template",self._template)
        changed=imgui.radio_button("Valid",self._valid_psc);imgui.same_line()
        changed|=imgui.radio_button("Rejected",not self._valid_psc)
        if changed:
            self._valid_psc=not self._valid_psc
        imgui.same_line()
        _,self._partial_filter=imgui.checkbox("Partial filter",self._partial_filter)
        x=np.linspace(0,self.psc_duration,num=int(self.psc_duration*signal.sr))
        s=cp.asnumpy(signal.gpu_rectified)
        avg=np.zeros_like(x)
        avg_cnt=0
        norm=lambda x: (x-np.min(x)) /(np.max(x)-np.min(x)) if self._normalize else x
        if implot.begin_plot("Traces",ImVec2(-1, imgui.get_font_size() *12)):
            implot.setup_axes_limits(0, self.psc_duration, 0, 100)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1,0, self.psc_duration)
            if len(signal.psc_onsets):
                cpu_pscs=cp.asnumpy(cp.array([signal.psc_onsets,signal.psc_length,signal.psc_mask])).T ## one transfer only
                if self._partial_filter: ## when using parial filter, replace global filter by patial one
                    cpu_pscs[:,2]=cp.asnumpy(self.current_filter_mask)
                for o,l,m in cpu_pscs:
                    if bool(m)==self._valid_psc and self._slo<o<self._shi:
                        if self._each_psc:
                            implot.plot_line("",x[:l],norm(s[o:o+l].astype(float)))
                        if self._average:
                            avg+=norm(s[o:o+len(x)].astype(float))
                            avg_cnt+=1
                if self._average:
                    implot.plot_line("##avg",x,avg/avg_cnt)
                if self._template:
                    if self._average:
                        implot.plot_line("##tmpl",x,cp.max(avg)/avg_cnt*cp.asnumpy(signal.template[0:len(x)]).astype(np.float64))
                    else:
                        bounds=implot.get_plot_limits()
                        implot.plot_line("##tmpl",x,bounds.y.min+(bounds.y.max-bounds.y.min)*0.9*cp.asnumpy(signal.template[0:len(x)]).astype(np.float64))
            implot.end_plot()

    def plotselected(self):
        signal=self._signals[self._current_signal] if self._current_signal>=0 else None
        changed, self._fit=imgui.checkbox("Fit",self._fit);imgui.same_line()
        autoscale, self._fit_autoscale=imgui.checkbox("Autoscale",self._fit_autoscale);imgui.same_line()
        if imgui.button("Reset user filter"):
            signal.gpu_corrected=cp.ones_like(signal.gpu_corrected)
        valid=int(signal.psc_mask[signal.selected_idx]) if _ge_(signal.selected_idx) else None
        changed,m=pygui_timeline('',signal.selected_idx,0,len(signal.psc_onsets),valid)
        #rescale=(m<10) & changed  ## mark that we should adjust scale
        if changed and m<10:
            signal.selected_idx=max(0,min(signal.selected_idx+m,len(signal.psc_mask)-1))
        elif changed and m>=10:
            o=signal.psc_onsets[signal.selected_idx]
            start,stop=o-int(invalidation_window*signal.sr),o+int(invalidation_window*signal.sr)
            signal.gpu_corrected[start:stop]=1-signal.gpu_corrected[start:stop]
            
        text="Fit failed"
        if implot.begin_plot(f"Selection",ImVec2(-1, imgui.get_font_size() *12)):
            implot.setup_axes_limits(0, self.psc_duration, 0,100,implot.Cond_.once)
            implot.setup_axis_limits_constraints(implot.ImAxis_.x1,0, self.psc_duration)
            implot.setup_legend(implot.Location_.north_east)
            if _ge_(signal.selected_idx):
                o=int(signal.psc_onsets[signal.selected_idx])                   
                l=int(signal.psc_length[signal.selected_idx])
                ttp=int(signal.psc_ttp[signal.selected_idx])
                x=np.linspace(0,self.psc_duration,num=int(self.psc_duration*signal.sr))
                #s=cp.asnumpy(signal.gpu_rectified[o:o+l])
                s=cp.asnumpy(signal.gpu_rectified)
                if self._fit_autoscale:
                    implot.setup_axes_limits(0, self.psc_duration, 0,cp.max(s[o+ttp:o+l]),implot.Cond_.always)
                implot.plot_line("##nsel",x[:l],
                                            s[o:o+l].astype(float))
                if self._fit:
                    try:
                        singleexp=lambda x, a, b, c: a*np.exp(-x/b)+c
                        popt,pcov=curve_fit(singleexp, x[:l-ttp], 
                                                            s[o+ttp:o+l].astype(float),
                                                            #s.astype(float),
                                                            bounds=([0., 0.0002, -20],   [2*max(s[o+ttp:o+l]), 0.040, 20]),
                                                            p0=[max(s[o+ttp:o+l]), 0.001, 0.0]
                                                            )
                        implot.plot_line(f"fit",x[ttp:l],singleexp(x[:l-ttp],*popt))
                        text=f"Time constant {1000*popt[1]:.4f} ms; Error : {np.sqrt(np.diag(pcov))[1]}"
                        ## pure numpy: y=a*exp(-x/b)+c => ln(y-c)= ln(a) - (1/b)*x is easy to fit!. should move to cupy!
                        ## the fits are not as nice, however: requires to introduce a small offset to cope with negative values (log(x<0))
                        ## cupy solves the problem by putting nan when asked for negative log
                        ## scipy.curve_fit does not have this problem as  it does not require log transformation!
                        #popt,pcov=np.polyfit(x[:l-ttp],
                        #                     np.log(np.clip(s[o+ttp:o+l].astype(float),0.001,500)+2),
                        #                     1,
                        #                     #w=np.power(np.linspace(100,0,l-ttp),3),
                        #                     w=np.power(s[o+ttp:o+l].astype(float),2),
                        #                     cov=True)
                        #popt,pcov=cp.polyfit(cp.linspace(0,self.psc_duration,l-ttp),
                        #                      cp.nan_to_num(cp.log(signal.gpu_rectified[o+ttp:o+l])),
                        #                      1,
                        #                      w=cp.power(signal.gpu_rectified[o+ttp:o+l],2),
                        #                      cov=True)
                        #tau,a=float(popt[0]),float(popt[1])
                        #implot.plot_line(f"fit2",x[ttp:l],np.exp(a)*np.exp(tau*x[:l-ttp]))
                        #text=f"Time constant {-1000/tau:.4f} ms; Error : {np.sqrt(np.diag(pcov))[1]}"

                        ## fitting the entire epsc (rise and decay.
                        #doubleexp=lambda x, a, b, c, d: a*(-np.exp(-x*1000/b)+np.exp(-x*1000/c))+d
                        #popt,pcov=curve_fit(doubleexp, x[:l], 
                        #                                s[o:o+l].astype(float),
                        #                                p0=[max(s),self.conv_risetime,self.conv_decaytime,0],
                        #                                #bounds=([-np.inf, 0.00005,0.001 -20],   [np.inf, 0.001,0.005, 20]),
                        #                                maxfev=500
                        #                                )
                        #implot.plot_line(f"fit2",x[:l],doubleexp(x[:l],*popt))
                        #text=f"Time constant {-tau:.4f} ms; Error : {np.sqrt(np.diag(pcov))[1]}"
                    except:
                        pass
            implot.end_plot()
            if _ge_(signal.selected_idx):
                imgui.text(text)
                cpu_pscs=cp.asnumpy(cp.array([signal.psc_ttp/signal.sr,
                                                signal.psc_halfwidth,
                                                signal.psc_peaks,
                                                signal.psc_amps,
                                                signal.psc_areas,
                                                signal.psc_sharpness,
                                                signal.psc_tau,
                                                signal.psc_fiterr,
                                                signal.psc_scores
                                                ])).T
                for i,text in enumerate(histogram_target_names[:]):
                    if signal.lohifilters[i]:
                        lo,hi=signal.lohifilters[i]
                        if not lo<cpu_pscs[int(signal.selected_idx)][i]<hi:
                            imgui.push_style_color(imgui.Col_.text,ImVec4(1.0,0,0,1.0))
                            imgui.text(text)
                            imgui.same_line()
                            imgui.pop_style_color()

    def plotpca(self):
        ## could also run an RF classifier on the main features
        signal=self._signals[self._current_signal] if self._current_signal>=0 else None
        #cp.cuda.stream.get_current_stream().synchronize()
        imgui.push_item_width(100)
        changed,self._pca_tgt=imgui.combo("Filter",self._pca_tgt,pca_histogram_target_names)
        imgui.same_line()
        changed,self._pca_cmap=imgui.combo("Colormap",self._pca_cmap,cmap_names)
        imgui.same_line()
        changed,self._pca_all=imgui.checkbox("All events",self._pca_all)
        imgui.pop_item_width()
        if not self._pca_all:
            gpumask=signal.psc_mask.astype(cp.bool_)
        else:
            gpumask=cp.array([True]*len(signal.psc_onsets))
        match self._pca_tgt:
            case Target.ttp:tgtdata=signal.psc_ttp[gpumask]/signal.sr
            case Target.halwidth:tgtdata=signal.psc_halfwidth[gpumask]
            case Target.peak:tgtdata=signal.psc_peaks[gpumask]
            case Target.amp:tgtdata=signal.psc_amps[gpumask]
            case Target.area:tgtdata=signal.psc_areas[gpumask]
            case Target.sharpness:tgtdata=signal.psc_sharpness[gpumask]
            case Target.tau:tgtdata=signal.psc_tau[gpumask]
            case Target.fiterr:tgtdata=signal.psc_fiterr[gpumask]
            case Target.score:tgtdata=signal.psc_scores[gpumask]
            case _:tgtdata=None
        implot.push_colormap(self._pca_cmap)
        imgui.begin_group()
        width=imgui.get_content_region_avail().x - imgui.get_font_size() * 5
        if implot.begin_plot("##pca",ImVec2(width,width)):
            if cp.sum(gpumask)>5:#len(signal.psc_onsets)>5 and cp.sum(signal.psc_mask)>5:
                X_cupy=cp.transpose(cp.array([signal.psc_ttp,
                                signal.psc_peaks,
                                signal.psc_amps,
                                signal.psc_areas,
                                signal.psc_halfwidth,
                                signal.psc_sharpness,
                                signal.psc_tau,
                                signal.psc_fiterr,
                                signal.psc_scores
                                ]))
                X_cupy=X_cupy[gpumask,:]
                indices=np.arange(len(signal.psc_onsets))[cp.asnumpy(gpumask)]
                ## start dataset transformation
                X_cupy=self._scaler_cuml.fit_transform(X_cupy)
                X_reduced=self._reducer_cuml.fit_transform(X_cupy)
                if self._pca_tgt==Target.clusters: ## or if self._pca_tgt>Target.score
                    y_cluster=self._clusterer_cuml.fit_predict(X_reduced)
                    tgtdata=y_cluster
                elif self._pca_tgt==Target.valid:
                    tgtdata=signal.psc_mask[gpumask]
                x=cp.asnumpy(X_reduced[:,0])
                y=cp.asnumpy(X_reduced[:,1])
                ptp=cp.ptp(tgtdata)
                if ptp: ## we are not sure if we will have tau and fiterr at that stage!
                    scaled=cp.asnumpy((tgtdata-cp.min(tgtdata))/(cp.ptp(tgtdata)))
                else:
                    scaled=cp.asnumpy(tgtdata)
                implot.push_plot_clip_rect()
                dl=implot.get_plot_draw_list()
                for i in range(len(x)):
                    xy=ImVec2(implot.plot_to_pixels(x[i],y[i]))
                    if implot.is_plot_hovered and imgui.is_mouse_clicked(0) and not imgui.is_mouse_dragging(0):
                        pt = imgui.get_mouse_pos()
                        if abs(pt[0]-xy[0])<pixel_pick_radius and abs(pt[1]-xy[1])<pixel_pick_radius:
                            signal.selected_idx=indices[i]
                    try:
                        dl.add_circle_filled(xy,pixel_pt_size,imgui.get_color_u32(implot.sample_colormap(scaled[i])))
                    except:
                        pass
                implot.pop_plot_clip_rect()
            implot.end_plot()
        imgui.end_group()
        imgui.same_line()
        implot.colormap_scale("##heatmap_scale", 0.0, 1.0, ImVec2(-1,width))
        implot.pop_colormap()
        ## plot histogram of filtered data
        '''if implot.begin_plot("PCA Histogram", ImVec2(-1, imgui.get_font_size() *12)):
            if cp.sum(gpumask)>5:
                h=cp.histogram(tgtdata,bins=self._filt_bins) ## for so little data, would it be faster to work on cpu?
                bar=cp.asnumpy(h[0]).astype(np.float32)
                x=cp.asnumpy(h[1]).astype(np.float32)
                implot.setup_axes(pca_histogram_target_names[self._pca_tgt], "Occurence")
                scalecond=implot.Cond_.always if changed or self._flags else implot.Cond_.once ## need to reset scales when we change tgtdata
                implot.setup_axes_limits(min(x), max(x), 0, max(bar)*1.5,scalecond)
                barwidth=(x[1]-x[0])
                implot.setup_axis_limits_constraints(implot.ImAxis_.x1,0, max(x)*1.1)
                implot.setup_axis_limits_constraints(implot.ImAxis_.y1, 0,max(bar)*1.5)
                implot.plot_bars("##pca_his", x+barwidth/2,bar, (x[1]-x[0])*0.66)
            implot.end_plot()'''

    def plotcumul(self):
        _,self._cumul_tgt=imgui.combo("Cumulate",self._cumul_tgt,["Amplitudes","Intervals"])
        _,self._cumul_bins=imgui.input_int("Bins",self._cumul_bins)
        if implot.begin_plot(f"Selection",ImVec2(-1, -1)):
            implot.setup_axis_limits_constraints(implot.ImAxis_.y1,-0.2, 1.2)
            implot.setup_legend(implot.Location_.south_east)
            try:
                match(self._cumul_tgt):
                    case Cumul.amplitude:
                        hi=int(max([cp.max(sig.psc_amps[sig.psc_mask.astype(cp.bool_)]) for sig in self._signals]))
                        implot.setup_axis_limits_constraints(implot.ImAxis_.x1,0.0, hi)
                        x=np.linspace(0,hi,self._cumul_bins,dtype=np.float32)
                        for s in self._signals:
                            y=cp.asnumpy(cp.cumsum(cp.histogram(s.psc_amps[s.psc_mask.astype(cp.bool_)],self._cumul_bins,(0,hi))[0]))
                            implot.plot_line(s.name, x,cp.asnumpy(y/y[-1]).astype(np.float32))
                    case Cumul.interval:
                        hi=int(max([cp.max(sig.psc_inter) for sig in self._signals]))
                        implot.setup_axis_limits_constraints(implot.ImAxis_.x1,0.0, hi)
                        x=np.linspace(0,hi,self._cumul_bins,dtype=np.float32)
                        for s in self._signals:
                            y=cp.asnumpy(cp.cumsum(cp.histogram(s.psc_inter,self._cumul_bins,(0,hi))[0]))
                            implot.plot_line(s.name, x,cp.asnumpy(y/y[-1]).astype(np.float32))
            except:
                pass
            implot.end_plot()

    def run(self):
        while not glfw.window_should_close(self.window):
            ## todo return results on exit!
            glfw.poll_events()
            display_w, display_h = glfw.get_framebuffer_size(self.window)
            GL.glViewport(0, 0, display_w, display_h)
            GL.glClearColor(0,0,0,0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            imgui.backends.opengl3_new_frame()
            imgui.backends.glfw_new_frame()
            imgui.new_frame()
            imgui.push_font(self.roboto )
            
            signal=self._signals[self._current_signal] if len(self._signals) and self._current_signal>=0 else None  ## in case we changed the signal!
            ## each iteration, we process the signal if something has changed
            ## then we draw the gui and collect the flags that have changed
            ## update flags for all signals
            #plot_height = imgui.get_font_size() * 15
            self._flags,cmds=self.paramwindow()

            if len(self._signals)==0:
                imgui.begin("info")
                imgui.text("Dop files on this window to start analysis")
                imgui.end()
                imgui.pop_font()
                imgui.render()
                imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
                glfw.swap_buffers(self.window)
                continue
            ## process signal. these functions will do nothing if parameters have not changed    
            #signal.rectify()
            #signal.convolve()
            #signal.extract()
            #signal.update_mask()
            for s in self._signals:
                s.preprocess()
                s.rectify()
                s.convolve()
                s.extract()
                s.update_mask() ## all other occurences of update_mask should be removed! but check before!


            imgui.begin("Signals")
            closingtabs=[]
            if imgui.begin_tab_bar("##Currently opened files"):
                for cnt,s in enumerate(self._signals):
                    tab=imgui.begin_tab_item(f"{s.name}",True)
                    closingtabs.append(not tab[1])
                    if tab[0]:
                        self._current_signal=cnt
                        imgui.end_tab_item()
                imgui.end_tab_bar()

            self.plotsignals()
            imgui.columns(3)
            self.plothistograms()
            imgui.next_column()
            self.plotpscs()
            imgui.next_column()
            self.plotselected()
            imgui.end()
            if self._show_pca:
                imgui.begin("PCA")
                self.plotpca()
                imgui.end()
            if len(self._signals)>1:
                imgui.begin("Cumul")
                self.plotcumul()
                imgui.end()
            for s in self._signals:
                s.flags=s.flags|self._flags ## update flags on all signal. hence if we switch to next signal, the signals will be modified according to what has changed
                s.flags|=flag_extract*benchmark ## benchmark!
            
            if any(closingtabs):
                del self._signals[closingtabs.index(True)]
                if len(self._signals):
                    if closingtabs.index(True)<=self._current_signal:
                        self._current_signal-=1
                else:
                    self._current_signal=None

            imgui.pop_font()
            imgui.render()
            ##############################
            ##############################
            ## Begin opengl rendering
            ##
            #display_w, display_h = glfw.get_framebuffer_size(self.window)
            #GL.glViewport(0, 0, display_w, display_h)
            #GL.glClearColor(0,0,0,0)
            #GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            ##
            ## End opengl rendering
            ##############################
            ##############################
            imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
            glfw.swap_buffers(self.window)
        ## end while loop
        if self.autosave:
            for s in self._signals:
                s.to_hdf()

        if len(self._signals)==1:
            self.results=self._signals[self._current_signal].summarize() ## will copy in clipboard
        else:
            self.results={}




if __name__ == "__main__":
    import os,pathlib
    os.chdir(pathlib.Path(__file__).parent)
    app=PSCapp()
    app.run()
    app.cleanup()
    print(app.results) ##should be parsed by subprocess.popen()