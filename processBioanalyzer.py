import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig
import warnings

from . import myUtils as mypy

###############################################################################
# Import Data
###############################################################################

def importTraces(data_pth, prefix, suffix=None, ext = '.csv', header = 13):
    
    ladder_file = prefix + '_Ladder_' + suffix + ext

    len_header = header

    header = pd.read_csv(data_pth / ladder_file, header = None).iloc[:len_header].set_index(0).to_dict()[1]

    n_samples = int(header['Number of Samples Run'])
    n_pts = int(header['Number of Events'])

    ladder = pd.read_csv(data_pth / ladder_file, header = len_header).iloc[:-1].astype(float)
    ladder = pd.Series(ladder.to_dict('list'), name='Ladder')
    ladder.Time = np.array(ladder.Time)
    ladder.Value = np.array(ladder.Value)
    ladder['Color'] = 'k'
    lanes = [ladder]
    assert len(ladder.Value) == n_pts
    
    for i in range(n_samples):
        sample_file = prefix + f'_Sample{i+1}_' + suffix + ext

        header = pd.read_csv(data_pth / sample_file, header = None).iloc[:len_header].set_index(0).to_dict()[1]
        sample = pd.read_csv(data_pth / sample_file, header = len_header).iloc[:-1].astype(float)
        
        lane = pd.Series(sample.to_dict('list'), name=header['Sample Name'])
        
        lane.Time = np.array(lane.Time)
        lane.Value = np.array(lane.Value)
        lane['Color'] = f'C{i}'
        lanes.append(lane)
        assert len(sample.Value) == n_pts

    traces = pd.DataFrame(lanes).reset_index().rename(columns={'index':'Sample'})
    return traces


def importPeaks(data_pth, prefix, suffix=None, ext = '.csv', header = 13, skip_inc = 9):
    
    len_header = header
    
    results_file = prefix + '_Results_' + suffix + ext
    results = pd.read_csv(data_pth / results_file, names=range(10), encoding = "ISO-8859-1")
    columns = list(results.iloc[len_header].values)
    num_columns = [col for col in columns if col != 'Observations']

    n_peaks = [int(i) for i in results[results[0] == 'Number of peaks found:'][1].values]
    sample_names = results[results[0] == 'Sample Name'][1].values
    markers = ['Lower Marker','Upper Marker']
    skip = len_header+1

    peaks = []
    for i,(n,name) in enumerate(zip(n_peaks,sample_names)):
        lane = results.iloc[skip:skip+n+2].rename(columns={c:col for c,col in enumerate(columns)})
        lane.replace(to_replace='1,500',value='1500',inplace=True)
        lane.insert(0,'Sample',name)
        for marker in markers:
            lane.insert(1,marker,lane['Observations']==marker)
        lane = lane[['Sample']+num_columns+markers]
        lane.reset_index(drop=True,inplace=True)

        peaks.append(lane)

        skip += n+skip_inc

    peaks = pd.concat(peaks,ignore_index=True)
    peaks = peaks.astype({col:float for col in num_columns})
    peaks = peaks.astype({'Size [bp]':int})
    
    return peaks

###############################################################################
# Conversions
###############################################################################

def calibrate(traces,kit='DNA1000'):
    ladder = traces[traces['Sample']=='Ladder'].squeeze()
    kit = kit.casefold()
    
    pks_bp = {
        'dna1000' : [15,25,50,100,150,200,300,400,500,700,850,1000,1500],
        'dna7500' : [50,100,300,500,700,1000,1500,2000,3000,5000,7000,10380]
    }

    valid_bp_rng = {
        'dna1000' : [25,1000],
        'dna7500' : [100,7500],
    }
    
    pks_bp = pks_bp[kit]
    valid_bp_rng = valid_bp_rng[kit]
    pks,_ = sig.find_peaks(ladder.Value,height=10)
    t = np.array(ladder.Time)
    pks_s = t[pks]

    #plt.plot(ladder.Time,ladder.Value)
    #plt.plot(np.array(ladder.Time)[pks],np.array(ladder.Value)[pks],'o')
    #plt.plot(pks_s,pks_bp,'o')

    res = 0.01
    t_interp = np.arange(t.min(),t.max()+res,res)
    bp_interp = np.interp(t_interp,pks_s,pks_bp)
    
    def bp_to_s(bp, validate=True):
        if validate:
            assert (bp>=min(pks_bp))&(bp<=max(pks_bp)), f'Cannot extrapolate below {min(pks_bp)} or above {max(pks_bp)} bp'
            if (bp<min(valid_bp_rng))|(bp>max(valid_bp_rng)):
                warnings.warn(f'Input outside valid sizing range: {valid_bp_rng[0]}-{valid_bp_rng[1]} bp')
        return t_interp[closest(bp_interp,bp)]
        
    valid_t_rng = [bp_to_s(bp, validate=False) for bp in valid_bp_rng]

    def s_to_bp(s, validate=True):
        if validate:
            assert (s>=min(pks_s))&(s<=max(pks_s)), f'Cannot extrapolate below {min(pks_s)} or above {max(pks_s)} s'
            if (s<min(valid_t_rng))|(s>max(valid_t_rng)):
                warnings.warn(f'Input outside valid sizing range: {valid_t_rng[0]:.2f}-{valid_t_rng[1]:.2f} s')
        return bp_interp[closest(t_interp,s)]

    return bp_to_s, s_to_bp


def getHeights(samples, peaks, peak_list, traces=None, tol=10):
    heights = np.zeros([len(samples),len(peak_list)])
    ref_q = np.zeros(len(samples))

    for i,row in samples.reset_index().iterrows():
        
        ref_q[i] = row['Tar_Q']

        these_peaks = peaks[peaks['Sample']==row['Sample']]
        if traces is not None:
            trace = traces[traces['Sample']==row['Sample']].iloc[0]
            bp_to_s,_ = calibrate(traces)
        for j,pk in enumerate(peak_list):
            closest_peak = closest(these_peaks['Size [bp]'],pk)
            bkg = trace.Value[closest(trace.Time,bp_to_s(pk))] if traces is not None else 0
            heights[i,j] = these_peaks.loc[closest_peak,'Peak Height'] if np.abs(these_peaks.loc[closest_peak,'Size [bp]']-pk)<tol else bkg
        
    return ref_q,heights


def closest(lst,val,check_all=True):
    if type(lst) is pd.Series:
        idx = (lst-val).abs().idxmin()
    else:
        idx = np.argmin(np.abs(lst-val))
    if check_all:
        matches = [i for i,v in enumerate(lst) if v==lst[idx]]
        n_matches = len(matches)
        if n_matches>1:
            warnings.warn(f'{n_matches} elements ({matches}) are equidistant to input value')
    return idx

###############################################################################
# Plotting
###############################################################################

def plotTraces(traces, peaks, skip_traces=[], label_peaks=None, skip_peaks=[], bp_min = -np.inf, bp_max = np.inf, skip_ladder=True, stagger_labels=False):
    
    t = traces.iloc[0].Time
    
    t_rng = [np.min(t),np.max(t)]
    
    bp_to_s,_ = calibrate(traces)
    
    if bp_min>-np.inf:
        t_rng[0] = bp_to_s(bp_min)
    if bp_max<np.inf:
        t_rng[1] = bp_to_s(bp_max)
    
    for i,trace in traces.iterrows():
        
        if i in skip_traces: continue
        if skip_ladder & (trace.Sample=='Ladder'): continue

        these_peaks = (peaks['Sample'] == trace['Sample'])
        UM = these_peaks & peaks['Upper Marker']
        LM = these_peaks & peaks['Lower Marker']
        norm = np.mean([peaks[M]['Peak Height'].values[0] for M in [UM,LM]])

        plt.plot(trace.Time,trace.Value/norm+i,color=trace.Color)
        plt.annotate(trace.Sample,
                     xy = (t_rng[1]-1, trace.Value[closest(t,t_rng[1])]/norm+i+0.1),
                     horizontalalignment='right', fontsize=14)    
            
    plt.setp(plt.gca(),
             xlim = t_rng,
             yticks = [],
             xticks = []
            )

    if label_peaks is None: return
    
    trans = plt.gca().get_xaxis_transform()
    
    gray=[0.75, 0.75, 0.75]
    for i,bp in enumerate(label_peaks):
        if bp in skip_peaks: continue
        if (bp<bp_min)|(bp>bp_max): continue
            
        pk = peaks[peaks['Size [bp]']==bp]
        t = pk['Aligned Migration Time [s]'].mean()
        plt.axvline(t, color=gray, linestyle='--', zorder=-1)
        labely = (i % 2 -1)/20-0.025 if stagger_labels else -0.025
        
        plt.annotate(bp, xy=(t,labely), 
                     xycoords=trans, ha="center", va="top", fontsize=14)
        if all(pk['Upper Marker']):
            plt.annotate('UM', xy=(t,labely-1/20), 
                     xycoords=trans, ha="center", va="top", fontsize=14)
        if all(pk['Lower Marker']):
            plt.annotate('LM', xy=(t,labely-1/20), 
                     xycoords=trans, ha="center", va="top", fontsize=14)
        
    plt.annotate('bp', xy=(max(t_rng)-1,-0.025-stagger_labels/20/2),
                 xycoords=trans, ha="right", va="top", fontsize=14)
        
    
    return