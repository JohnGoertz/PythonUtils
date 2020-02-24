import matplotlib as mpl
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


def getLadderPeaks(traces, kit='DNA1000'):
    ladder = traces[traces['Sample']=='Ladder'].squeeze()
    
    columns = ['Sample', 'Size [bp]', 'Conc. [ng/µl]', 'Molarity [nmol/l]', 'Area',
               'Aligned Migration Time [s]', 'Peak Height', 'Peak Width', '% of Total',
               'Time corrected area', 'Lower Marker', 'Upper Marker']
    
    pks,_ = sig.find_peaks(ladder.Value,height=10)
    t = np.array(ladder.Time)
    pks_s = t[pks]
    pks_bp = kit_pks(kit)
    rows = [{'Sample' : 'Ladder',
             'Size [bp]' : bp,
             'Aligned Migration Time [s]' : s,
            } for bp,s in zip(pks_bp,pks_s)]
    
    return pd.DataFrame(rows,columns=columns)

###############################################################################
# Conversions
###############################################################################

def kit_pks(kit='DNA1000'):
    kit = kit.casefold()
    return {
        'dna1000' : [15,25,50,100,150,200,300,400,500,700,850,1000,1500],
        'dna7500' : [50,100,300,500,700,1000,1500,2000,3000,5000,7000,10380]
    }[kit]

def kit_rng(kit='DNA1000'):
    kit = kit.casefold()
    return {
        'dna1000' : [25,1000],
        'dna7500' : [100,7500],
    }[kit]

def calibrate(ladder_peaks, kit='DNA1000'):
    
    valid_bp_rng = kit_rng(kit)
    pks_bp = ladder_peaks['Size [bp]'].values
    pks_s = ladder_peaks['Aligned Migration Time [s]'].values
    
    res = 0.01
    t_interp = np.arange(pks_s.min(),pks_s.max()+res,res)
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


def getHeights(samples, peaks, peak_list, traces=None, tol=10, kit='DNA1000'):
    heights = np.zeros([len(samples),len(peak_list)])
    ref_q = np.zeros(len(samples))

    if traces is not None:
        ladder_peaks = getLadderPeaks(traces, kit)
        bp_to_s, _ = calibrate(ladder_peaks, kit)
    
    for i,row in samples.reset_index().iterrows():
        
        ref_q[i] = row['Tar_Q']

        these_peaks = peaks[peaks['Sample']==row['Sample']]
        if traces is not None:
            trace = traces[traces['Sample']==row['Sample']].iloc[0]
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

def plotTraces(traces, peaks, skip_traces=[], label_peaks=None, skip_peaks=[], bp_min = -np.inf, bp_max = np.inf, skip_ladder=True, stagger_labels=False, kit='DNA1000'):
    
    t = traces.iloc[0].Time
    
    t_rng = [np.min(t),np.max(t)]
    
    ladder_peaks = getLadderPeaks(traces, kit)
    bp_to_s, _ = calibrate(ladder_peaks, kit)
    
    if bp_min>=kit_pks(kit)[0]:
        t_rng[0] = bp_to_s(bp_min)
    if bp_max<=kit_pks(kit)[-1]:
        t_rng[1] = bp_to_s(bp_max)
        
    _,ax = plt.subplots()
    ax.set_xlim(t_rng)
    
    # coordinate transform for annotations
    xtrans = plt.gca().get_xaxis_transform() # x in data units, y in axes fraction
    ytrans = plt.gca().get_yaxis_transform() # y in data units, x in axes fraction
    
    for i,trace in traces.iterrows():
        
        if i in skip_traces: continue
        if skip_ladder & (trace.Sample=='Ladder'): continue

        these_peaks = (peaks['Sample'] == trace['Sample'])
        UM = these_peaks & peaks['Upper Marker']
        LM = these_peaks & peaks['Lower Marker']
        norm = np.mean([peaks[M]['Peak Height'].values[0] for M in [UM,LM]])

        plt.plot(trace.Time,trace.Value/norm+i,color=trace.Color)
        labely = trace.Value[closest(t,t_rng[1])]/norm+i+0.1
        plt.annotate(trace.Sample,
                     xy = (0.99, labely), xycoords=ytrans,
                     horizontalalignment='right', fontsize=14)    
            
    plt.setp(plt.gca(),
             xlim = t_rng,
             yticks = [],
             xticks = []
            )

    if label_peaks is None: return
    
    gray=[0.75, 0.75, 0.75]
    for i,bp in enumerate(label_peaks):
        if bp in skip_peaks: continue
        if (bp<bp_min)|(bp>bp_max): continue
            
        pk = peaks[peaks['Size [bp]']==bp]
        t = pk['Aligned Migration Time [s]'].mean()
        plt.axvline(t, color=gray, linestyle='--', zorder=-1)
        labely = (i % 2 -1)/20-0.025 if stagger_labels else -0.025
        
        plt.annotate(bp, xy=(t,labely), 
                     xycoords=xtrans, ha="center", va="top", fontsize=14)
        if all(pk['Upper Marker']):
            plt.annotate('UM', xy=(t,labely-1/20), 
                     xycoords=xtrans, ha="center", va="top", fontsize=14)
        if all(pk['Lower Marker']):
            plt.annotate('LM', xy=(t,labely-1/20), 
                     xycoords=xtrans, ha="center", va="top", fontsize=14)
        
    plt.annotate('bp', xy=(0.99,-0.025-stagger_labels/20/2),
                 xycoords='axes fraction', ha="right", va="top", fontsize=14)
        
    return plt.gca()


def GelLikeImage(traces, peaks, skip_traces=[], label_peaks=[], skip_peaks=[],
                 bp_min = -np.inf, bp_max = np.inf, hlines = True, 
                 label_lanes = True, ladder_pos = 'left', kit='DNA1000', label_kwargs={}, im_kwargs={}):
    
    ladder_peaks = getLadderPeaks(traces, kit)
    bp_to_s, s_to_bp = calibrate(ladder_peaks, kit)
    
    laneprops = dict(
        aspect='auto', 
        cmap='Greys',
        interpolation='nearest',
        vmax = traces.apply(lambda row: np.max(row.Value) if row.Sample!='Ladder' else -np.inf, axis=1).max(),
        vmin = traces.apply(lambda row: np.min(row.Value) if row.Sample!='Ladder' else np.inf, axis=1).min(),
        origin = 'Lower'
    )
    
    labelprops = dict(
        fontsize=14
    )
            
    im_kwargs.update({k:v for k,v in laneprops.items() if k not in im_kwargs.keys()})
    label_kwargs.update({k:v for k,v in labelprops.items() if k not in label_kwargs.keys()})
    
    t = traces.iloc[0].Time
    
    t_rng = [np.min(t),np.max(t)]
    
    if bp_min>=kit_pks(kit)[0]:
        t_rng[0] = bp_to_s(bp_min)
    if bp_max<=kit_pks(kit)[-1]:
        t_rng[1] = bp_to_s(bp_max)
    
    min_idx = closest(t,t_rng[0])
    max_idx = closest(t,t_rng[1])
    
    def plotLane(trace, offset):
        ax = fig.add_axes([0.1*offset, 0.1, 0.08, 0.8])
        ax.set_axis_off()
        ax.imshow(trace.Value[min_idx:max_idx].reshape(-1,1), label=trace.Sample, **im_kwargs)
        if label_lanes:
            ax.set_title(trace.Sample, **label_kwargs)

    ladder = traces[traces['Sample']=='Ladder'].iloc[0]

    fig = plt.figure()

    offset = 0
    
    if ladder_pos is not None:
        if ladder_pos.casefold() in ['left','both']:
            plotLane(ladder, offset)
            offset += 1

    for i,trace in traces.iterrows():
        if trace.Sample == 'Ladder': continue
        if i in skip_traces: continue
        plotLane(trace, offset)
        offset += 1

    if ladder_pos is not None:
        if ladder_pos.casefold() in ['right','both']:
            plotLane(ladder, offset)
            offset += 1

    grey=[0.75, 0.75, 0.75]

    AX = fig.add_axes([0, 0.1, 0.1*offset-0.02, 0.8])
    AX.set_axis_off()
    AX.set_ylim(t_rng)
    AX.set_xlim([0,1])

    # coordinate transform for annotations
    trans = AX.get_yaxis_transform() # y in data untis, x in axes fraction

    for pk in label_peaks:
        if pk in skip_peaks: continue
        if (pk<bp_min)|(pk>bp_max): continue
        s = bp_to_s(pk)
        if hlines:
            AX.axhline(s, linestyle='--', color=grey)
        if ladder_pos is not None:
            if ladder_pos.casefold() in ['left','both']:
                plt.annotate(int(pk), xy=(1.025,s),
                             xycoords=trans, ha="left", va="center", **label_kwargs)
        else:
            plt.annotate(int(pk), xy=(-0.025,s),
                         xycoords=trans, ha="right", va="center", **label_kwargs)
        
    if ladder_pos is not None:
        for pk,s in ladder_peaks[['Size [bp]','Aligned Migration Time [s]']].values:
            if pk in skip_peaks: continue
            if (pk<bp_min)|(pk>bp_max): continue
            if ladder_pos.casefold() in ['left','both']:
                plt.annotate(int(pk), xy=(-0.025,s),
                             xycoords=trans, ha="right", va="center", **label_kwargs)
            else:
                plt.annotate(int(pk), xy=(1.025,s),
                             xycoords=trans, ha="left", va="center", **label_kwargs)
                
        
    return plt.gcf()


def peak_box(bp, bp_to_s, GLI_fig, height=30, bias=0.6, lr_pad=0.0125,
             label=None, label_pos = 'right', label_pad = 0,
             rect_kwargs={}, label_kwargs={}):
    
    ladders = [ax.get_children()[-2].get_label()=='Ladder' for ax in GLI_fig.axes][:-1]
    ladder_pos = {(1,1) : 'both',
                  (1,0) : 'left',
                  (0,1) : 'right',
                 }[(ladders[0],ladders[-1])]
    
    ax = GLI_fig.gca()
    
    n_lanes = len(GLI_fig.axes)-1
    
    lane_w = 1/(0.8+n_lanes-1)
        
    if ladder_pos is None:
        rect_x = 0
    elif ladder_pos=='right':
        rect_x = 0
    elif ladder_pos in ['left','both']:
        rect_x = lane_w
        
    rect_x -= lr_pad
    rect_y = bp_to_s(bp-(1-bias)*height)
    
    if ladder_pos is None:
        rect_w = 1
    elif ladder_pos in ['left','right']:
        rect_w = 1-lane_w
    elif ladder_pos=='both':
        rect_w = 1-2*lane_w
        
    rect_w += 2*lr_pad
    rect_h = bp_to_s(bp+bias*height)-rect_y
    
    if not any([k in ['ec','edgecolor'] for k in rect_kwargs.keys()]):
        rect_kwargs.update({'ec': 'k'})
        
    if not any([k in ['fc','facecolor'] for k in rect_kwargs.keys()]):
        rect_kwargs.update({'fc': 'none'})
    
    box = mpl.patches.Rectangle((rect_x, rect_y), rect_w, rect_h, clip_on=False, **rect_kwargs);
    ax.add_patch(box)
    
    if label is not None:
        
        labelprops = dict(
            fontsize=14
        )

        label_kwargs.update({k:v for k,v in labelprops.items() if k not in label_kwargs.keys()})
        
        label_kwargs.update({
            'left'   : {'xy': (rect_x+label_pad,rect_y+rect_h/2), 'va':'center', 'ha':'right'},
            'right'  : {'xy': (rect_x+rect_w+label_pad,rect_y+rect_h/2), 'va':'center', 'ha':'left'},
            'top'    : {'xy': (rect_x+rect_w/2,rect_y+rect_h+label_pad), 'va':'bottom', 'ha':'center'},
            'bottom' : {'xy': (rect_x+rect_w/2,rect_y-label_pad), 'va':'top', 'ha':'center'},
            'center' : {'xy': (rect_x+rect_w/2,rect_y+rect_h/2), 'va':'center', 'ha':'center'},
        }[label_pos])
        
        ax.annotate(label, clip_on=False, **label_kwargs)