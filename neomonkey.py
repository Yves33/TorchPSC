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

import neo
import numpy as np
import quantities as pq
from quantities import uV,mV,V
from quantities import pA,nA,uA,mA,A
from quantities import ns,us,ms,s

def neo_shift(self,newstart=None):
    '''
    Shifts the times of the signal, so that newstart corresponds to first point
    in order to be consistent, the first time point by default is self.sampling_interval
    '''
    if newstart==None:
        newstart=self.sp
    return neo.AnalogSignal(self.magnitude*self.units,sampling_rate=self.sampling_rate,t_start=newstart)

def neo_time_slice(self,start,stop,qty=pq.s):
    '''
    Returns a time slice
    if start and stop are quantities, they are used. else qty is used.
    Compared to neo.AnalogSignal.time_slice(), accepts slices starting at 0.0
    clamps start and stop values to return valid slices
    see also __call__()
    '''
    if not isinstance(start,pq.quantity.Quantity):
        start=start*qty
    if not isinstance(stop,pq.quantity.Quantity):
        stop=stop*qty
    start=max(self.times[0], min(start,self.times[-1]))
    stop=max(self.times[0], min(stop,self.times[-1]))
    return self.time_slice(start,stop)

def neo_at(self,start,qty=pq.s,avg=0*pq.s):
    '''
    Returns value of signal at specific time
    '''
    clamp=lambda x, l, u: l if x < l else u if x > u else x
    
    if not isinstance(start,pq.quantity.Quantity):
        start=start*qty
    sstart=int(clamp(start-avg, self.times[0],self.times[-1])*self.sr)
    sstop=int(clamp(start+avg, self.times[0],self.times[-1])*self.sr)
    sstart=clamp(sstart, 0,len(self)-1)
    sstop=clamp(sstop, sstart,len(self)-1)
    if sstart==sstop:
        return self[sstart][0]
    else:
        return self[sstart:sstop].mean()

def get_item(self,k):
    try:
        return self.__old_getitem__(k)
    except:
        if isinstance (k,slice) :
            start=k.start*0.0 if k.start is None else k.start
            stop=k.stop*np.inf if k.stop is None else k.stop
            return self(start,stop)[::k.step]
        else:
            start=0.0 if k[0].start is None else k[0].start
            stop=np.inf if k[0].stop is None else k[0].stop
            return self(start,stop,k[1])[::k[0].step]
    
def installmonkey(experimental_features=False):
    baseunits={u:V for u in ['uV','mV','V']}
    baseunits.update({u:A for u in ['pA','nA','uA','mA','A']})
    baseunits.update({u:s for u in ['us','ms','s']})
    neo.AnalogSignal.t=neo_time_slice
    neo.AnalogSignal.shift=neo_shift
    neo.AnalogSignal.at=neo_at
    neo.AnalogSignal.uV=property(lambda self:self.rescale(uV).magnitude.flatten()*uV)
    neo.AnalogSignal.mV=property(lambda self:self.rescale(mV).magnitude.flatten()*mV)
    neo.AnalogSignal.V=property(lambda self:self.rescale(V).magnitude.flatten()*V)
    neo.AnalogSignal.pA=property(lambda self:self.rescale(pA).magnitude.flatten()*pA)
    neo.AnalogSignal.nA=property(lambda self:self.rescale(nA).magnitude.flatten()*nA)
    neo.AnalogSignal.uA=property(lambda self:self.rescale(uA).magnitude.flatten()*uA)
    neo.AnalogSignal.mA=property(lambda self:self.rescale(mA).magnitude.flatten()*mA)
    neo.AnalogSignal.A=property(lambda self:self.rescale(A).magnitude.flatten()*A)
    neo.AnalogSignal.us=property(lambda self:self.times.rescale(us))
    neo.AnalogSignal.ms=property(lambda self:self.times.rescale(ms))
    neo.AnalogSignal.s=property(lambda self:self.times.rescale(s))
    neo.AnalogSignal.sr=property(lambda self:self.sampling_rate)
    neo.AnalogSignal.sp=property(lambda self:self.sampling_period.simplified)
    neo.AnalogSignal.baseunit=property(lambda self:baseunits[self.units.dimensionality.string])
    pq.quantity.Quantity.str=property(lambda self:self.dimensionality.string)
    ## not tested yet
    neo.AnalogSignal.__call__=lambda self,a,b,c=pq.s:self.t(a,b,c) ## avoids .t accessor!)
    neo.AnalogSignal._from=lambda self,t:self.t(t,self.times[-1].magnitude)
    neo.AnalogSignal._to=lambda self,t:self.t(self.times[0].magnitude,t)
    ## hackish!
    if experimental_features:
        neo.AnalogSignal.__old_getitem__=neo.AnalogSignal.__getitem__
        neo.AnalogSignal.__getitem__=get_item
    
def average(sigs):
    '''
    computes the average of signals. the final unit is the unit of first signal in list
    '''
    return neo.AnalogSignal( np.mean( [s.rescale(s[0].units) for s in sigs],axis=0)*sigs[0].units,
                             sampling_rate=sigs[0].sampling_rate, t_start=sigs[0].times[0])


if __name__=='__main__':
    import matplotlib.pyplot as plt
    installmonkey()
    inpath="D:\\data-yves\\labo\\devel\\patchan\\samples\\yukti&clara\\WTC1-V2.1-12w-B1-DIV28-2022.03.08-Cell2\\wtc1-v2.1-12w-b1-div28-2022.03.08-cell2 002.axgd"
    inpath="../samples/yukti&clara/WTC1-V2.1-12w-B1-DIV28-2022.03.08-Cell2/wtc1-v2.1-12w-b1-div28-2022.03.08-cell2 002.axgd"
    #inpath="C:\\Users\\ylefeuvre\\Desktop\\example files\\sag\\0220509_4473 cell 1 008.axgd"
    #inpath="../samples/yves/maty/20220314_113s_0002.maty"

    f = neo.io.AxographIO(str(inpath),force_single_segment=False)
    blk = f.read_block(signal_group_mode='split-all')
    sigs=[sig for seg in blk.segments for sig in seg.analogsignals if sig.baseunit==pq.V ]
    s1=sigs[9]
    print("testing functions...")
    print("signal.at(t,average) : ",s1.at(0.21,avg=0.01*pq.s))
    print("signal.at(t) : ",s1.at(0.21))
    print("__call__(t0,t1) : ",np.mean(s1(0.2,np.inf).V))
    print("__getitem__(f0:f1,unit) : ",np.mean(s1[0.1:0.2,pq.s]))
    print("__getitem__(f0:,unit) : ",np.mean(s1[0.1:,pq.s]))
    print("__getitem__(:hi,unit) : ",np.mean(s1[:0.2,pq.s]))
    print("__getitem__(:hi,unit) : ",np.mean(s1[0.1*pq.s:0.2*pq.s]))

    plt.plot(s1.s,s1.mV)
    plt.plot(s1[10000:20000].s,s1[10000:20000].mV)
    plt.plot(s1[0.5*pq.s:].s,s1[.5*pq.s:].mV)
    #s2=average([s for s in sigs if s.baseunit==V])
    #plt.plot(sigs[0].ms,sigs[0].mV,color='b')
    #plt.axhline(np.mean(sigs[0].t(0.0,0.1).mV))
    plt.show()