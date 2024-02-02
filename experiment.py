#!/usr/bin/env python3
# Copyright (c)2020-2022, Yves Le Feuvre <yves.le-feuvre@u-bordeaux.fr>
#
# All rights reserved.
#
# This file is prt of the intrinsic program
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted under the terms of the BSD License. See
# LICENSE file in the root of the Project.

#
# using neo, neither axograph nore axon (v1) protocols provide any scaling information.
# for axon v1 (pclamp9.x) protocols, there's a hack that retrieves the information
# for axograph, you must provide a scaling:
# > exp.protocol.scaleoutput (0,[-500*pq.pA,500*pq.pA]) ## ensures that all amplitues are in -500,500pA range (1000 steps)
#

import numpy as np
import neo
import quantities as pq
import pathlib,logging

def Experiment(filename,**kwargs):
    if pathlib.Path(filename).is_file():
        suffix=pathlib.Path(filename).suffix
        if suffix in[".abf"]:
            return ABFexperiment(filename,**kwargs)
        if suffix in[".axgd",".axgx"]:
            return AXGexperiment(filename,**kwargs)
        if suffix in[".smr"]:
            return SMRexperiment(filename,**kwargs)
        if suffix in[".maty"]:
            return MATYexperiment(filename,**kwargs)
        #if suffix in[".txt",".ascii"]:
        #    return ASCIIexperiment(filename)

class GENexperiment:
    def __init__(self,filename,**kwargs):
        self.path=filename
        self.name=pathlib.Path(filename).resolve().name
        self.suffix=pathlib.Path(filename).resolve().suffix
        self.signalcount=len(self.blk.segments[0].analogsignals)
        self.sweepcount=len(self.blk.segments)
    
    def signal(self,channel,filter="V"):
        #return [seg.analogsignals[channel].shift() for seg in self.blk.segments]
        signals=[]
        for seg in self.blk.segments:
            sas=[s.shift() for s in seg.analogsignals if str(s.units).endswith(filter)]
            if len(sas):
                signals.append(sas[channel])
        return signals

class GENprotocol:
    def __init__(self,exp,**kwargs):
        self.samplecount=len(exp.blk.segments[0].analogsignals[0])
        self.sampling_rate=np.floor(exp.blk.segments[0].analogsignals[0].sampling_rate)
        self.sweepcount=len(exp.blk.segments)
    
    def assinglestep(self,channel=0):
        return [e[1] for e in self.asepochs(channel)]
    
    def assignal(self, channel=0):
        signals=[]
        for epochs in self.asepochs(channel):
            baseline=np.zeros(self.samplecount) ## no units
            for i,epoch in enumerate(epochs):
                lo,hi=epoch['s_start'] ,epoch['s_stop']
                if epoch['type']==0:
                    baseline[lo:hi]= np.zeros(hi-lo)
                elif epoch['type']==1:
                    baseline[lo:hi]= np.ones(hi-lo)*float(epoch['lvl'])
                elif epoch['type']==2: ## to be checked for ramp
                    ## ramp start from prefious level!
                    ramplevelstart=epochs[i-1]['lvl'] if i>0 else 0
                    ramplevelstop=epoch['lvl']
                    baseline[lo:hi]=np.linspace(ramplevelstart,ramplevelstop,hi-lo)
                elif epoch['type']==3: ## to be checked for train
                    for c in range(epoch['cnt']):
                        up=epoch['s_up']
                        down=epoch['s_down']
                        baseline[lo+c*(up+down):lo+c*(up+down)+up]=epoch['lvl']
            signals.append(neo.AnalogSignal(baseline*self.dacunits[channel],sampling_rate=self.sampling_rate))
        return signals

'''
epochs are stored as dict (but may change to addict)
's_start':  start of epoch in samples,
's_stop':   end of epoch in samples,
's_dur':    duration of epoch in samples,
'lvl':      final level of epoch with appropriate units (pA,mV). for ramp, we may introduce ilvl,
#'ilvl':     initial level of epoch (usually the level at the end of previous epoch) [not implemented]
'start':    start of epoch in time units (s),
'stop':     end of epoch in time units (s),
'dur':      duration of epoch in samples,
'cnt':      for trains, number of pulses in epoch,
's_up':     for trains, duration of up phase, in samples,
's_down':   for trains, duration of down phase, in samples
'up':       for trains, duration of up phase, in time units,
'down':     for trains, duration of down phase, in time units
#'freq':     for trains, the frequency of pulses [not implemented]
#'per':      for trains: the peiod of trains [not implemented]
'type':     0=off,1=step,2=ramp,3=train
'''
class SMRprotocol(GENprotocol):
    def __init__(self,exp,**kwargs):
        super(SMRprotocol,self).__init__(exp,**kwargs)
        self.dacnames=['']
        self.dacunits=[pq.V]
        self.name="smr"

    def asepochs(self,channel):
        return [[{  's_start':0,
                    's_stop':self.samplecount,
                    's_dur':self.samplecount,
                    'lvl':0,
                    'start':0.0*pq.s,
                    'stop':self.samplecount/(float(self.sampling_rate))*pq.s,
                    'dur':self.samplecount/(float(self.sampling_rate))*pq.s,
                    'cnt':0*pq.s,
                    'up':0*pq.s,
                    'down':0*pq.s,
                    's_up':0,
                    's_down':0,
                    'type':0,
        }]]

    def scaleoutput(self,channel,absrange):
        pass

class SMRexperiment(GENexperiment):
    def __init__(self,filename,**kwargs):
        self.file=neo.io.Spike2IO(str(filename))
        self.blk=self.file.read_block(signal_group_mode='split-all')
        super(SMRexperiment,self).__init__(filename,**kwargs)
        self.protocol=SMRprotocol(self,**kwargs)

class ABFprotocol(GENprotocol):
    def __init__(self,exp,**kwargs):
        super(ABFprotocol,self).__init__(exp,**kwargs)
        if exp.file._axon_info["fFileSignature"]==b'ABF ' and \
           exp.file._axon_info["fFileVersionNumber"]>=1.8:
            self.init_v1(exp,**kwargs)
        else:
            self.init_v2(exp,**kwargs)
    
    def init_v2(self,exp,**kwargs):
        #logging.getLogger().warning("Protocol parser has not been heavily tested on ABF2 files")
        #logging.getLogger().warning("Please contact the author to check for field consistancy")
        self.name=str(exp.file._axon_info["sProtocolPath"]).split('/')[-1]
        self.samplecount=len(exp.blk.segments[0].analogsignals[0])
        self.sampling_rate=np.floor(exp.blk.segments[0].analogsignals[0].sampling_rate)
        self.offset_p=int(np.floor(self.samplecount/64))                    ## number of samples
        self.offset_t=(self.offset_p/self.sampling_rate).simplified         ## should be pq.s
        self.episode_cnt=exp.file._axon_info['lActualEpisodes']
        self.episode_repeat=exp.file._axon_info['lActualEpisodes']//exp.file._axon_info['protocol']['lEpisodesPerRun']  ## number of repeats per episode. no repeats
        ## parsing epochs neo implementation (incomplete)
        self.dacnames= [exp.file._axon_info['listDACInfo'][i]['DACChNames'] for i in range(4)] ## todo should be range(len(exp.file._axon_info['listDACInfo']))
        self.dacunits= [exp.file._axon_info['listDACInfo'][i]['DACChUnits'] for i in range(4)] ## todo should be range(len(exp.file._axon_info['listDACInfo']))
        self.dacunits=[pq.quantity.Quantity(1.0,u.decode()) for u in self.dacunits]
        ## DAC0 only
        '''
        self.epochproperties=exp.file._axon_info['dictEpochInfoPerDAC'][0]
        nepochs=len(self.epochproperties.keys())
        self.epochs=[]
        for i in range(nepochs):
            self.epochs.append(self.epochproperties[i])
            self.epochs[-1]['lEpochInitTrainPeriod']=self.epochs[-1]['lEpochPulsePeriod']
            self.epochs[-1]['lEpochInitPulseWidth']=self.epochs[-1]['lEpochPulseWidth']
            self.epochs[-1]['lEpochTrainPeriodInc']=0         ## not handled by pClamp / clampex v11
            self.epochs[-1]['lEpochPulseWidthInc']=0          ## not handled by pClamp / clampex v11
        ## stored in exp.file._axon_info['EpochInfo'][0]{nDigitalValue} ??
        self.dac=[self.epochs,[],[],[]] ## todo should be range(len(exp.file._axon_info['listDACInfo']))
        '''
        ## all DAC (hopefully!)
        self.dac=[]
        for dacnum in exp.file._axon_info['dictEpochInfoPerDAC'].keys():
            epochproperties=exp.file._axon_info['dictEpochInfoPerDAC'][dacnum]
            nepochs=len(epochproperties.keys())
            epochs=[]
            for i in range(nepochs):
                epochs.append(epochproperties[i])
                epochs[-1]['lEpochInitTrainPeriod']=epochs[-1]['lEpochPulsePeriod']
                epochs[-1]['lEpochInitPulseWidth']=epochs[-1]['lEpochPulseWidth']
                epochs[-1]['lEpochTrainPeriodInc']=0         ## not handled by pClamp / clampex v11
                epochs[-1]['lEpochPulseWidthInc']=0          ## not handled by pClamp / clampex v11
            self.dac.append(epochs)

    def init_v1(self,exp,**kwargs):
        self.name=str(exp.file._axon_info["sProtocolPath"]).split('/')[-1]
        ## epochtype:enum(step,ramp,pulse,train,biphasic_train,triangle_train,cosine_train). Old versions do not have trains
        ## units are pA and samples!
        self.samplecount=len(exp.blk.segments[0].analogsignals[0])
        self.sampling_rate=np.floor(exp.blk.segments[0].analogsignals[0].sampling_rate)
        self.offset_p=int(np.floor(self.samplecount/64))                    ## number of samples
        self.offset_t=(self.offset_p/self.sampling_rate).simplified         ## should be pq.s
        self.episode_cnt=exp.file._axon_info['lActualEpisodes']             ## number of episodes. no units
        self.episode_repeat=exp.file._axon_info['lActualEpisodes']//exp.file._axon_info['lEpisodesPerRun']  ## number of repeats per episode. no repeats
        ## parsing epochs neo implementation (incomplete)
        self.dacnames= ['dac0','dac1','dac2','dac3']    ## neo does not specify the dac names.
        self.dacunits= [np.nan,np.nan,np.nan,np.nan]    ## neo does not specify the dac units.
        self.epochproperties=[k for k in exp.file._axon_info.keys() if "Epoch" in k ]
        nepochs=len(exp.file._axon_info[self.epochproperties[0]])
        self.epochs=[]
        for i in range(nepochs):
            self.epochs.append({})
            for k in self.epochproperties:
                self.epochs[-1][k]=exp.file._axon_info[k][i]
        ## reverse engeneered values of dac names, units and train properties
        ## not documented by pyABF
        import struct
        f = open(exp.path, "rb")
        f.seek(1306); self.dacnames=struct.unpack('40s',f.read(40))[0].decode()
        f.seek(1346); self.dacunits=struct.unpack('32s',f.read(32))[0].decode()
        f.seek(2136); lEpochInitTrainPeriod=struct.unpack('20i',f.read(80))
        f.seek(2216); lEpochInitPulseWidth=struct.unpack('20i',f.read(80))
        f.close()
        ## todo find digital outputs
        self.dacnames= [self.dacnames[i:i+10].rstrip(' ') for i in range(0, len(self.dacnames), 10)]
        self.dacunits= [self.dacunits[i:i+8].rstrip(' ') for i in range(0, len(self.dacunits), 8)]
        self.dacunits= [pq.quantity.Quantity(1.0,u) for u in self.dacunits]
        for i,e in enumerate(self.epochs):
            e['lEpochInitTrainPeriod']=lEpochInitTrainPeriod[i]
            e['lEpochTrainPeriodInc']=0         ## not handled by pClamp / clampex v9.2
            e['lEpochInitPulseWidth']=lEpochInitPulseWidth[i]
            e['lEpochPulseWidthInc']=0          ## not handled by pClamp / clampex v9.2
        ## end of hack
        self.dac=[  [e for i,e in enumerate(self.epochs) if 0<=i<=9],
                    [e for i,e in enumerate(self.epochs) if 10<=i<=19],
                    [e for i,e in enumerate(self.epochs) if 20<=i<=29], ## should be empty with current neo implementation
                    [e for i,e in enumerate(self.epochs) if 30<=i<=39], ##should be empty with current neo implementation
                ]
  
    def scaleoutput(self,channel,absrange):
        self.dacunits[channel]=1.0                                                  ## erase old unit for channel and use dimensionless signal
        lvls=np.array( [abs(e['lvl']) for f in self.asepochs(channel) for e in f] ) ## compute list of alllvls throughout protocol
        scalefactor=1e12                                                            ## multiply by high number (10**(3,6,9,12,15)))
        lvls*=scalefactor*absrange[0].units ## should be big enough!
        while not all([absrange[0]<l<absrange[1] for l in lvls]):                   ## downscale while we are no in range
            lvls/=1000
            scalefactor/=1000
        ## once this is done, print the scale factor
        for e in self.dac[channel]:                                                 ## correct amplitudes
            e['fEpochInitLevel']*=scalefactor
            e['fEpochLevelInc']*=scalefactor
        self.dacunits[channel]=absrange[0].units

        
    def asepochs(self,channel):
        sampling_period=(1/self.sampling_rate).simplified
        frames=[]
        if self.dacunits[channel]==np.nan:
            logging.getLogger().warning('You have specified auto units, but the protocol parser could not determine DAC units! ')
            logging.getLogger().warning('Please run protocol.fixoutputscale(channel,range,ignore)! ')
            raise(NotImplementedError)
        units= self.dacunits[channel] ## may be dimentionless, if units=1.0
        for i in range(self.episode_cnt):
            steps=[]
            offset=self.offset_p
            for j,e in enumerate(self.dac[channel]):
                ## compute start ,stop and duration in samples and time units
                stepproperties={'s_start':offset,
                                's_dur':e['lEpochInitDuration']+i*e['lEpochDurationInc'],
                                's_stop':offset+e['lEpochInitDuration']+i*e['lEpochDurationInc'],
                                'lvl':(e['fEpochInitLevel']+i*e['fEpochLevelInc'])*units,
                                'type':e['nEpochType'],
                                's_down':e['lEpochInitTrainPeriod']-e['lEpochInitPulseWidth'] if e['nEpochType']==3 else np.nan,
                                's_up':e['lEpochInitPulseWidth'] if e['nEpochType']==3 else np.nan,
                                'cnt':(e['lEpochInitDuration']+i*e['lEpochDurationInc']) // e['lEpochInitTrainPeriod'] if e['nEpochType']==3 else np.nan
                              }
                #compute value in seconds
                stepproperties['start']=stepproperties['s_start']*sampling_period
                stepproperties['dur']=stepproperties['s_dur']*sampling_period
                stepproperties['stop']=stepproperties['s_stop']*sampling_period
                stepproperties['down']=stepproperties['s_down']*sampling_period
                stepproperties['up']=stepproperties['s_up']*sampling_period
                #stepproperties.update(e)
                ## for first pulse, correct pulse start and duration to incorporate offset
                ## in fact this should not be done , as protocol preview in clampex does not perfom this correction
                #if j==0 : 
                #    stepproperties['s_start']=0
                #    stepproperties['s_dur']=int(e['lEpochInitDuration']+i*e['lEpochDurationInc']+offset)
                #    stepproperties['start']=0.0*pq.s
                #    stepproperties['dur']= stepproperties['s_dur']*sampling_period 
                steps.append(stepproperties)                                 ## add step to list of steps
                offset=offset+e['lEpochInitDuration']+i*e['lEpochDurationInc']  ## compute current offset
            frames.append(steps)
        return frames

class ABFexperiment(GENexperiment):
    def __init__(self,filename,**kwargs):
        self.file=neo.io.AxonIO(str(filename))
        self.blk=self.file.read_block(signal_group_mode='split-all')
        super(ABFexperiment,self).__init__(filename,**kwargs)
        self.protocol=ABFprotocol(self,**kwargs)

def is_num(s):
    if (s.find('-') <= 0) and s.replace('-', '', 1).isdigit():
        if (s.count('-') == 0):
            s_type = 'Positive Integer'
        else:
            s_type = 'Negative Integer'
    elif (s.find('-') <= 0) and (s.count('.') < 2) and \
         (s.replace('-', '', 1).replace('.', '', 1).isdigit()):
        if (s.count('-') == 0):
            s_type = 'Positive Float'
        else:
            s_type = 'Negative Float'
    else:
        s_type = "Not alphanumeric!"
        return False
    return True

def num(x):
    try:
        return float(x) if '.' in x else int(x)
    except:
        return x

class AXGprotocol(GENprotocol):
    def __init__(self, exp,**kwargs):
        super(AXGprotocol,self).__init__(exp,**kwargs)
        ## protocol pulse fields. some remain unknown...
        ## for now, all pulses are attached to DAC channel 0.
        ## not that sure for gap_inc and average, but as they're not used by current protocols...
        fields=['average','wavrepeat','cnt','start','start_inc','dur','dur_inc', 7 ,'gap','gap_inc',10,'lvl','lvl_inc',13,14]
        ## if a pulse has cnt>1, the interval between two pulses is given by gap+episode_cnt*gap_inc
        notes=exp.blk.annotations['notes'].split('\n')
        self.name=exp.file.info["comment"].split(':')[1].rstrip().lstrip() ##not valid on files converted from axon
        self.pulses=[] ## create new list of pulses
        '''
        for n in notes:
            ## protocols description may vary...
            if n.startswith("Start an episode every"): self.episode_interval=[float(x) for x in n.split() if  is_num(x)][0]
            elif n.startswith("Pause after waveform series"): self.episode_gap=[float(x) for x in n.split() if is_num(x)][0]
            elif n.startswith("Repeat protocol"): self.episode_repeat=[int(x) for x in n.split() if is_num(x)][0]
            ##['Repeat each waveform, then step to next waveform']
            elif n.startswith("DAC Holding Levels"):self.dac_hold=[int(x) for x in n.split('\t') if is_num(x)]
            elif n.startswith("Episodes"):self.episode_cnt=[int(x) for x in n.split(' ') if is_num(x)][0]
            elif n.startswith("Pulses") and '#' not in n:self.pulse_cnt=[int(x) for x in n.split(' ') if is_num(x)][0]
            elif n.startswith("Pulse #") : self.pulses.append({'type':'pulse'}) ## create new list of pulses
            elif n.startswith("Train #") : self.pulses.append({'type':'train'}) ## create new list of pulses
            elif len(n.split('\t'))==16:
                if len(self.pulses)==0: #sometimes the first pusle is not created as there is no 'Pulse #' label!
                    self.pulses.append({'type':'train'}) ## create new list of pulses)
                self.pulses[-1].update({k:num(v) for k,v in zip(fields,n.split('\t'))})
        '''
        ## new parser. not sure if it is more reliable!
        currentpulse=0
        for n in notes:
            ## protocols description may vary...
            if n.startswith("Start an episode every"): self.episode_interval=[float(x) for x in n.split() if  is_num(x)][0]
            elif n.startswith("Pause after waveform series"): self.episode_gap=[float(x) for x in n.split() if is_num(x)][0]
            elif n.startswith("Repeat protocol"): self.episode_repeat=[int(x) for x in n.split() if is_num(x)][0]
            ##['Repeat each waveform, then step to next waveform']
            elif n.startswith("DAC Holding Levels"):self.dac_hold=[int(x) for x in n.split('\t') if is_num(x)]
            elif n.startswith("Episodes"):self.episode_cnt=[int(x) for x in n.split(' ') if is_num(x)][0]
            elif n.startswith("Pulses") and '#' not in n:
                self.pulse_cnt=[int(x) for x in n.split(' ') if is_num(x)][0]
                for p in range(self.pulse_cnt):
                    self.pulses.append({'type':'pulse'})
            elif n.startswith("Pulse #") : self.pulses[currentpulse].update({'type':'pulse'}) ## create new list of pulses. Increasing currentpulse should be performed here (starting from -1)
            elif n.startswith("Train #") : self.pulses[currentpulse].update({'type':'train'}) ## create new list of pulses. Increasing currentpulse should be performed here (starting from -1)
            elif len(n.split('\t'))==16:
                self.pulses[currentpulse].update({k:num(v) for k,v in zip(fields,n.split('\t'))})
                currentpulse+=1

        self.dacnames=['dac0']
        self.dacunits=[np.nan]
        ## we start parsepulses() here, but (although I do not have any file to test! nore any specs) 
        ## is is very likely that each pulse can be associated with its **own** pulse table.
        ## hence,rather than creating self.pulsetables={...}, one should create self.pulses[currentpulse]['pulsetable']={}
        ## and correct accordingly in asepochs()
        self.parsepulsetables(notes)

    def parsepulsetables(self, notes):
        self.pulsetables={'Amplitude':[],'Onset':[],'Width':[],'Inter-Pulse':[]}
        #currentpulse=0
        target=''
        idx=0
        while idx<len(notes):
            n=notes[idx]
            #if "Pulse #" in n or "Train #" in n:
            #    #"Train #1 : Current Stimulus"
            #    self.currentpulse=int(n.split('#')[-1])
            if 'Table Entries' in n:
                l=int(n.split(' ')[-1])
                target=n.split(' ')[0]
            elif target!='':
                try:
                    self.pulsetables[target].append(float(n))
                except:
                    target=''
            idx+=1

    def scaleoutput(self,channel,absrange):
        self.dacunits[channel]=1.0                                                  ## erase old unit for channel and use dimensionless signal
        lvls=np.array( [abs(e['lvl']) for f in self.asepochs(channel) for e in f] ) ## compute list of alllvls throughout protocol
        scalefactor=1e12                                                            ## multiply by high number (10**(3,6,9,12,15)))
        lvls*=scalefactor*absrange[0].units ## should be big enough!
        while not all([absrange[0]<l<absrange[1] for l in lvls]):                   ## downscale while we are no in range
            lvls/=1000
            scalefactor/=1000
        ## once this is done, print the scale factor
        for p in self.pulses:                                                 ## correct amplitudes
            p['lvl']*=scalefactor
            p['lvl_inc']*=scalefactor
        self.dacunits[channel]=absrange[0].units

    def asepochs(self,channel):
        def _separate(st,unit=pq.pA):
            ## creates epochs for empty regions non empty and ovelapping regions. 
            ## overlapping single pulses should be correctly handled
            ## results may be weird for ramp or train/repeated pulses
            ## not thoroughfully tested!
            bounds=[s['start'] for s in st]+[s['stop'] for s in st]+[0.0]  ## compute list of steps onsets and offsets
            bounds=list(set(bounds))                                             ## remove duplicates
            bounds.sort()                                                        ## sort bounds
            newsteps=[]
            for i in range(len(bounds)-1):
                ## iterate through steps and find if one ovelaps with current epoch
                lvl=0
                etype=0
                up=np.nan
                down=np.nan
                cnt=0
                for step in st:
                    if step['start']<=bounds[i] and step['stop']>=bounds[i+1]:
                        lvl+=step['lvl']            ## if pulses are additive, else lvl=step['lvl']
                        etype=step["type"]
                        up=step['up']
                        down=step['down']
                        cnt=step['cnt']
                ## create epoch
                epoch={ 'start':bounds[i]*pq.s,
                        'stop': bounds[i+1] * pq.s,
                        'dur':  (bounds[i+1]-bounds[i])* pq.s,
                        'lvl':lvl,
                        'type':etype,
                        'up':up*pq.s,
                        'down':down*pq.s,
                        'cnt':cnt
                        }
                ## as axograph stores pulse times in seconds / ms, calcultae corresponding samples
                epoch['s_start']=int(epoch['start']*self.sampling_rate)
                epoch['s_stop']=int(epoch['stop']*self.sampling_rate)
                epoch['s_dur']=int(epoch['dur']*self.sampling_rate)
                epoch['s_up']= np.nan if np.isnan(epoch['up']) else int(epoch['up']*self.sampling_rate)
                epoch['s_down']=np.nan if np.isnan(epoch['up']) else int(epoch['down']*self.sampling_rate)
                newsteps.append(epoch)
            return newsteps
        ## highly experimental. for now only tested for simple current steps and / or trains, optionnally overlapping
        ## 
        frames=[]
        #for i in range(self.episode_cnt):
        #    steps=[]
        for cnt in range(self.episode_cnt):
            for rep in range(self.episode_repeat):
                steps=[]
                for pulse in self.pulses:
                    ## compute start ,stop and duration in samples and time units
                    nb=pulse['cnt']
                    if nb==1:
                        stepproperties={'start':(pulse['start']+cnt*pulse['start_inc'])/1000 ,
                                        'stop': (pulse['start']+pulse['dur']+cnt*pulse['start_inc']+cnt*pulse['dur_inc']) /1000,
                                        'dur':  (pulse['dur']+cnt*pulse['dur_inc'])/1000,
                                        'lvl':  pulse['lvl']+cnt*pulse['lvl_inc'],
                                        'type': 3 if pulse['cnt']>1 else 1, ##to be adjusted for ramps,...
                                        ## todo provide gap and gap_incr, recompute dur and stop
                                        'up':  (pulse['dur']+cnt*pulse['dur_inc']) / 1000,
                                        'down':(pulse['gap']+cnt*pulse['gap_inc']) / 1000,
                                        'cnt':pulse['cnt']
                                        }
                    else:
                        ## check whether we have a pulse table
                        ## if pulse tables are not empty, use provided values instead of those provided by pulse AND ignore ANY increment
                        ## HIGHLY EXPERIMENTAL. NO SPECS!
                        if len(self.pulsetables['Amplitude'])==self.episode_cnt:
                            pulse['lvl']=self.pulsetables['Amplitude'][cnt]
                            pulse['lvl_inc']=0 ## theoretically, but not entirely sure!
                        if len(self.pulsetables['Onset'])==self.episode_cnt:
                            pulse['start']=self.pulsetables['Onset'][cnt]
                            pulse['start_inc']=0 ## theoretically, but not entirely sure!
                        if len(self.pulsetables['Width'])==self.episode_cnt:
                            pulse['dur']=self.pulsetables['Width'][cnt]
                            pulse['dur_inc']=0 ## theoretically, but not entirely sure!
                        if len(self.pulsetables['Inter-Pulse'])==self.episode_cnt:
                            pulse['gap']=self.pulsetables['Inter-Pulse'][cnt]
                            pulse['gap_inc']=0 ## theoretically, but not entirely sure!
                        ## end of HIGHLY EXPERIMENTAL section
                        stepproperties={'start':(pulse['start']+cnt*pulse['start_inc'])/1000 ,
                                        'stop': (pulse['start']+cnt*pulse['start_inc']+(pulse['dur']+pulse['gap']+cnt*pulse['dur_inc']+cnt*pulse['gap_inc'])*nb) /1000,
                                        'dur':  (pulse['dur']+pulse['gap']+cnt*pulse['dur_inc']+cnt*pulse['gap_inc'])*nb/1000,
                                        'lvl':  pulse['lvl']+cnt*pulse['lvl_inc'],
                                        'type':3 if pulse['cnt']>1 else 1, ##to be adjusted for ramps,...
                                        ## todo provide gap and gap_incr, recompute dur and stop
                                        'up':  (pulse['dur']+cnt*pulse['dur_inc']) / 1000,
                                        'down':(pulse['gap']+cnt*pulse['gap_inc']) / 1000,
                                        'cnt':pulse['cnt']
                                        }
                    steps.append(stepproperties)
                frames.append(_separate(steps))
        ## fix units
        for f in frames:
            for s in f:
                s['lvl']=s['lvl']*self.dacunits[channel]
        return frames
    
class AXGexperiment(GENexperiment):
    def __init__(self,filename,**kwargs):
        self.file=neo.io.AxographIO(str(filename))
        self.blk=self.file.read_block(signal_group_mode='split-all')
        self.protocol=AXGprotocol(self)
        super(AXGexperiment,self).__init__(filename,**kwargs)

class MATYprotocol(GENprotocol):
    def __init__(self,exp,**kwargs):
        super(MATYprotocol,self).__init__(exp,**kwargs)
        for k,v in exp.blk.annotations.items():
            setattr(self,k,v)
            self.name=self.protocolname
        self.dacnames=['dac0']
        self.dacunits=[np.nan]

    def scaleoutput(self,channel,absrange):
        self.dacunits[channel]=1.0                                                  ## erase old unit for channel and use dimensionless signal
        lvls=np.array( [abs(e['lvl']) for f in self.asepochs(channel) for e in f] ) ## compute list of alllvls throughout protocol
        scalefactor=1e12                                                            ## multiply by high number (10**(3,6,9,12,15)))
        lvls*=scalefactor*absrange[0].units ## should be big enough!
        while not all([absrange[0]<l<absrange[1] for l in lvls]):                   ## downscale while we are no in range
            lvls/=1000
            scalefactor/=1000
        ## once this is done, print the scale factor
        self.steps=[s*scalefactor for s in self.steps]
        self.dacunits[channel]=absrange[0].units

    def asepochs(self,units=pq.pA):
        ## very simplified: assumes a single pulse per sweep
        frames=[]
        for e in range(self.episode_cnt):
            steps=[]
            steps.append({'s_start':0,
                          's_stop':int(self.start*self.sampling_rate),
                          's_dur':int(self.start*self.sampling_rate),
                          'start':0*pq.s,
                          'stop':self.start*pq.s,
                          'dur':self.start*pq.s,
                          'lvl':0.0*units,
                          'type':0
                        })
            steps.append({'s_start':int(self.start*self.sampling_rate),
                          's_stop':int(self.stop*self.sampling_rate),
                          's_dur':int(self.dur*self.sampling_rate),
                          'start':self.start*pq.s,
                          'stop':self.stop*pq.s,
                          'dur':self.dur*pq.s,
                          'lvl':self.steps[e]*pq.pA,
                          'type':1
                          })
            frames.append(steps)
        return frames

class MATYexperiment(GENexperiment):
    """only used for merged ccsteps files"""
    def __init__(self,filename,**kwargs):
        self.file=neo.io.NeoMatlabIO(str(filename))
        self.blk=self.file.read_block()
        self.protocol=MATYprotocol(self)
        super(MATYexperiment,self).__init__(filename,**kwargs)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    import pprint
    import neomonkey
    
    neomonkey.installmonkey()
    ROOT="../../samples/"
    ############################
    ## axographx protocols
    ############################
    ## time constant
    #PATH="yukti&clara/WTC1-V2.2-12w-A1-DIV24-2022.03.11-Cell4/WTC1-V2.2-12w-A1-DIV24-2022.03.11 cell4 004.axgd"
    ##resistance
    #PATH="yukti&clara/WTC1-V2.2-12w-A1-DIV24-2022.03.11-Cell4/WTC1-V2.2-12w-A1-DIV24-2022.03.11 cell4 005.axgd"
    ## AHP_MED
    #PATH="yukti&clara/WTC1-V2.2-12w-A1-DIV24-2022.03.11-Cell2/WTC1-V2.2-12w-A1-DIV24-2022.03.11 cell2 009.axgd"
    ## AHP_SLOW
    PATH="yukti&clara/WTC1-V2.2-12w-A1-DIV24-2022.03.11-Cell2/WTC1-V2.2-12w-A1-DIV24-2022.03.11 cell2 011.axgd"
    # AHP_3AP
    #PATH="yukti&clara/3APs/2022.06.02_1818_Cell3 009.axgd"
    #PATH="yukti&clara/3APs/Shank3ex4-22_104_Slice1_Cell1 013.axgd"
    ## IV
    #PATH="yukti&clara/WTC1-V2.2-12w-A1-DIV24-2022.03.11-Cell2/WTC1-V2.2-12w-A1-DIV24-2022.03.11 cell2 001.axgd"
    #PATH="yukti&clara/WTC1-V2.2-12w-A1-DIV24-2022.03.11-Cell1/WTC1-V2.2-12w-A1-DIV24-2022.03.11 cell1 002.axgd"
    ## SPON
    #PATH="yukti&clara/WTC1-V2.2-12w-A1-DIV24-2022.03.11-Cell1/WTC1-V2.2-12w-A1-DIV24-2022.03.11 cell1 005.axgd"
    ## SAG by Yukti!
    #PATH="yukti&clara/sag/0220509_4473 cell 4 006.axgd"
    ## AHP with varying pulse tables
    #PATH="yukti&clara/AHPVarying/16.02.18 004.axgd"

    ############################
    ## pclamp protocols
    ############################
    ## IV
    #PATH="yves/iv/iv.abf"
    ## RAMP
    #PATH="yves/ramps/ramp1.abf"
    ## TRAINS
    #PATH="yves/train/test_pulses_0000.abf"
    #PATH="Anna/20230201_w006/20230201_w006_0015.abf"

    #############################
    ## reconstructed matlab files (maty)
    ############################
    PATH="yves/maty/20220314_113s_0002.maty"

    #############################
    ## spike2 protocols (always empty!)
    ############################
    #PATH="yves/smr/test.smr"

    
    ROOT="/run/media/rapids/VERBATIM HD YVES/Data/data-yves/labo/devel/patchan/samples/"
    ROOT="../samples/"
    myexp=Experiment(pathlib.Path(ROOT+PATH).resolve())
    ## scaling output is mandatory for axograph files, as the units for protocol outputs are unknown!
    ## that step is optionnal for abf files, as output command units are directly parsed from abf file
    if np.isnan(myexp.protocol.dacunits[0]):
        myexp.protocol.scaleoutput(0,absrange=[-1510*pq.pA,1510*pq.pA])
    pprint.pprint(myexp.protocol.asepochs(0),sort_dicts=False)
    #pprint.pprint(myexp.protocol.assinglestep(0),sort_dicts=False)
    #pprint.pprint([ (f[1]['start'],f[1]['stop'],f[1]['lvl']) for f in myexp.protocol.asepochs(channel=0)])
    sigs=myexp.signal(0)
    cmds=myexp.protocol.assignal(channel=0)
    for i in range(myexp.sweepcount):
        #plt.plot(cmds[i].ms,cmds[i]+i*1500*pq.pA)
        plt.plot(sigs[i].ms,sigs[i])
    for e in myexp.protocol.asepochs(0)[-1]:
        pass
        #plt.axvline(e['start'].rescale(pq.ms),c='r')
        #plt.axvline(e['stop'].rescale(pq.ms),c='g')
    plt.show()
