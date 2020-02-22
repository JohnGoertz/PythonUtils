import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    lanes = [pd.Series(ladder.to_dict('list'), name='Ladder')]
    assert len(ladder.Value) == n_pts
    for i in range(n_samples):
        sample_file = prefix + f'_Sample{i+1}_' + suffix + ext

        header = pd.read_csv(data_pth / sample_file, header = None).iloc[:len_header].set_index(0).to_dict()[1]
        sample = pd.read_csv(data_pth / sample_file, header = len_header).iloc[:-1].astype(float)
        
        lane = pd.Series(sample.to_dict('list'), name=header['Sample Name'])
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
# Plotting
###############################################################################

def plotTraces(traces, peaks, skip_traces=[], label_peaks=None, skip_peaks=[], skip_ladder=True, stagger_labels=False):
    for i,trace in traces.iterrows():
        
        if i in skip_traces: continue
        if skip_ladder & (trace.Sample=='Ladder'): continue

        these_peaks = (peaks['Sample'] == trace['Sample'])
        UM = these_peaks & peaks['Upper Marker']
        LM = these_peaks & peaks['Lower Marker']
        norm = np.mean([peaks[M]['Peak Height'].values[0] for M in [UM,LM]])

        plt.plot(trace.Time,trace.Value/norm+i,color=trace.Color)
        plt.annotate(trace.Sample,
                     xy = (trace.Time[-1]-1, trace.Value[-1]/norm+i+0.1),
                     horizontalalignment='right', fontsize=14)    

    plt.setp(plt.gca(),
             xlim = [np.min(trace.Time),np.max(trace.Time)],
             yticks = [],
             xticks = []
            )

    if label_peaks is None: return
    
    trans = plt.gca().get_xaxis_transform()
    
    gray=[0.75, 0.75, 0.75]
    for i,bp in enumerate(label_peaks):
        if bp in skip_peaks: continue
            
        t = peaks[peaks['Size [bp]']==bp]['Aligned Migration Time [s]'].mean()
        plt.axvline(t, color=gray, linestyle='--', zorder=-1)
        labely = (i % 2 -1)/20-0.025 if stagger_labels else -0.025
        plt.annotate(bp, xy=(t,labely), 
                     xycoords=trans, ha="center", va="top", fontsize=14)
        
    plt.annotate('bp', xy=(trace.Time[-1]-1,-0.025),
                 xycoords=trans, ha="right", va="top", fontsize=14)
    
    return