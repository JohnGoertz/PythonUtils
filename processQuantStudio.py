import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stat
import scipy.optimize as opt
import warnings

from . import myUtils as mypy

###############################################################################
#%% Import Data

def importQuantStudio(data_pth, data_file, *,
                      header = None, debug = False):
    test = pd.read_excel(data_pth / data_file,
                             sheet_name = 'Sample Setup')
    if header is None: 
        header = pd.Index(test.iloc[:,0]).get_loc('Well')+1
    
    try:
        setup = pd.read_excel(data_pth / data_file,
                             sheet_name = 'Sample Setup', header = header)
    except PermissionError as e:
        print('Permission Denied. The file may be open; close the file and try again.')
        return print(e)
    
    if 'Target Name' not in setup.columns:
        display(setup)
    
    wells = setup[~setup['Target Name'].isna()].Well
    
    try:
        raw = pd.read_excel(data_pth / data_file,
                             sheet_name = 'Raw Data', header = header)
    except:
        raw = None
        
    data = pd.read_excel(data_pth / data_file,
                         sheet_name = 'Amplification Data', header = header)
    
    instr_results = pd.read_excel(data_pth / data_file,
                         sheet_name = 'Results', header = header)
    
    instr_results.CT.mask(instr_results.CT == 'Undetermined',np.nan,inplace=True)
    
    imps = {'setup'         : setup,
            'raw'           : raw,
            'data'          : data,
            'instr_results' : instr_results}
    
    for df in imps.values():
        if df is None: continue
        df.columns = df.columns.str.replace(' ', '')
        df = df[df.Well.isin(wells)]
    
    return imps
    
def reshapeQS_SNX(targets, n_r, *, quantities = None,
                  setup = None, raw = None, data = None, instr_results = None):
    
    #Number of cycles (n_c) and array of cycle numbers (c)
    c = mypy.uniq(data.Cycle)
    n_c = c.size
    
    #Number of targets
    n_t = len(targets)
    
    #Number of quantities (n_q) and logged quantities (lg_q)
    if quantities is None:
        quantities = mypy.uniq(setup.Quantity[setup.TargetName.isin(targets)]).sort[::-1]
        
    n_q = len(quantities)
    
    n_c = len(c)
    well_map, ct_instr,thresh_instr = [np.zeros([n_t,n_q,n_r]) for _ in range(3)]
    Rn, dRn = [np.zeros([n_t,n_q,n_r,n_c]) for _ in range(2)]
    
    # Rearrange data into four-dimensional matrix
    #   Dim1 -> targets
    #   Dim2 -> concentrations
    #   Dim3 -> replicates
    #   Dim4 -> cycle data
    for (t,q,r) in np.ndindex(n_t,n_q,n_r):
        setup_mask = (setup.TargetName == targets[t]) & (setup.Quantity == quantities[q])
        
        if sum(setup_mask) == n_r:
            well_map[t,q,r] = setup.Well[setup_mask].iloc[r]
        else:
            warnings.warn(f'{sum(setup_mask)} wells match target {targets[t]} and quantity {quantities[q]}')
            continue
        
        
        data_mask = (data.Well == well_map[t,q,r]) & (data.TargetName == targets[t])
        assert sum(data_mask) == n_c
        
        Rn[t,q,r,:] = data.Rn[data_mask]
        dRn[t,q,r,:] = data.DeltaRn[data_mask]
        
        result = instr_results[(instr_results.Well == well_map[t,q,r]) &
                               (instr_results.TargetName == targets[t])]
        ct_instr[t,q,r] = result.CT
        thresh_instr[t,q,r] = result.CtThreshold
        
    return {'c'             : c,
            'n_t'           : n_t,
            'n_q'           : n_q,
            'lg_q'          : np.log10(quantities),
            'well_map'      : well_map,
            'dRn'           : dRn,
            'ct_instr'      : ct_instr,
            'thresh_instr'  : thresh_instr}


def reshapeQS_MUX(targets, fluors, n_r, *, delin = ' - ',
                  setup = None, raw = None, data = None, instr_results = None):
    
    #Number of cycles (n_c) and array of cycle numbers (c)
    c = mypy.uniq(data.Cycle)
    n_c = c.size
    
    #Number of targets
    n_t = len(targets)
    
    #Number of fluorophores
    n_f = len(fluors)
    
    n_c = len(c)
    
    # Rearrange data into multi-dimensional matrix
    #   Dim1 -> targets
    #   Dim2 -> concentrations
    #   Dim3 -> replicates
    #   Dim4 -> fluorophore
    #   Dim5 -> cycle data
    
    # Use the quantities in the first fluorophore listed to map the wells to the array
    #Number of quantities (n_q)
    q_mask = (setup.TargetName.str.contains(targets[0], regex = False)) & \
             (setup.TargetName.str.contains(fluors[0], regex = False))
    quantities = np.sort(mypy.uniq(setup.Quantity[q_mask]))[::-1]
    n_q = len(quantities)
    
    well_map = np.zeros([n_t,n_q,n_r])
    
    for (t,q,r) in np.ndindex(n_t,n_q,n_r):
        tar = targets[t] + delin + fluors[0]
        setup_mask = (setup.TargetName.str.match(tar)) & \
                     (setup.Quantity == quantities[q])
        
        if sum(setup_mask) == n_r:
            well_map[t,q,r] = setup.Well[setup_mask].iloc[r]
        else:
            warnings.warn(f'{sum(setup_mask)} wells match target {targets[t]} and quantity {quantities[q]}')
            well_map[t,q,r] = np.nan
    
    q_map, ct_instr, thresh_instr = [np.zeros([n_t,n_q,n_r,n_f]) for _ in range(3)]
    Rn, dRn = [np.zeros([n_t,n_q,n_r,n_f,n_c]) for _ in range(2)]
    
    for (t,q,r,f) in np.ndindex(n_t,n_q,n_r,n_f):
        quant = mypy.uniq(setup.Quantity[
                (setup.Well == well_map[t,q,r]) & 
                (setup.TargetName.str.contains(fluors[f]))
                ])
        tar = targets[t] + delin + fluors[f]
        assert len(quant) <= 1, \
            f'Quantity is not unique for target {tar}, fluor {fluors[f]}, quantity {quantities[q]}, rep {r}'
        if len(quant) == 0:
            warnings.warn(f'Quantity {quantities[q]} not found for target {tar}, fluor {fluors[f]}, rep {r}')
            continue
        q_map[t,q,r,f] = quant[0]
        data_mask = (data.Well == well_map[t,q,r]) & \
                (data.TargetName.str.contains(fluors[f]))
        
        if sum(data_mask) != n_c:
            tar = targets[t] + delin + fluors[0]
            print(tar)
            print('{:}|{:}|{:}'.format(t,q,r))
            print(quant)
            continue
        assert sum(data_mask) == n_c, \
            f'{sum(data_mask)} rows match well {well_map[t,q,r]} and fluorophore {fluors[f]}'
        
        Rn[t,q,r,f,:] = data.Rn[data_mask]
        dRn[t,q,r,f,:] = data.DeltaRn[data_mask]
        
        result = instr_results[(instr_results.Well == well_map[t,q,r]) &
                               (instr_results.TargetName.str.contains(fluors[f]))]
        ct_instr[t,q,r,f] = result.CT
        thresh_instr[t,q,r,f] = result.CtThreshold
        
    return {'c'             : c,
            'n_t'           : n_t,
            'n_q'           : n_q,
            'n_f'           : n_f,
            'lg_q'          : np.log10(quantities),
            'well_map'      : well_map,
            'q_map'         : q_map,
            'dRn'           : dRn,
            'ct_instr'      : ct_instr,
            'thresh_instr'  : thresh_instr}

###############################################################################
#%% Calculate CTs from thresholds

def calcCTs(dRn, thresholds, n_t, n_q, n_r, n_c, interp_step = 0.01):
    c = np.arange(n_c)+1
    ct_manual = np.zeros([n_t,n_q,n_r])
    dRn_interp = np.zeros([n_t,n_q,n_r,n_c])
    for (i,j,k) in np.ndindex(n_t,n_q,n_r):
        # Interpolate the amplification curves to get fractional CTs
        cp = np.arange(0,n_c+interp_step,interp_step)+1
        dRn_interp = np.interp(cp,c,dRn[i,j,k])
        # Find the first fractional cycle over the threshold
        ct = np.argmax(dRn_interp>thresholds[i,j,k])*interp_step+1
        # If nothing found, np.argmax will return 0, so enter nan as CT
        ct_manual[i,j,k] = np.where(ct-1,ct,np.nan)
    return ct_manual



###############################################################################
#%% Plot Rn and dRn, overlaying replicates


def setupOverlay(targets, labels, plt_kwargs = {},
                 t_map = None, q_colors = None, show_legend = True,
                 fig = None,):
    n_t = len(targets)
              
    if fig is None:
        fig = plt.figure(constrained_layout = True,
                     figsize = [20,8])
    if t_map is not None:
        #Use the coordinates specified in t_map
        nrows, ncols = max(t_map.values())
        gs = fig.add_gridspec(nrows+1,ncols+1)
        gs_map = {t:gs[coord] for (t,coord) in t_map.items()}
    else:
        #Make the GridSpec layout as square as possible
        n_t_ = n_t+1
        if n_t_ > 25:
            sq = (n_t_)**0.5
            nrows = int(sq)
            ncols = n_t_//nrows + int(n_t_ % nrows != 0)
        elif n_t_ <= 4:
            ncols = n_t_
            nrows = 1
        else:
            ncols = 5
            nrows = n_t_//ncols + int(n_t_ % ncols != 0)
        gs = fig.add_gridspec(nrows,ncols)
        gs_map = {t:gs[i] for i,t in enumerate(targets)}
        gs_map['legend'] = gs[n_t]
    
        # Plot with smaller x- and y-tick labels and axis labels
    with plt.rc_context({'axes.labelsize'  : 'small',
                         'axes.labelweight': 'normal',
                         'xtick.labelsize' : 'small',
                         'ytick.labelsize' : 'small'}):
        # Create axes for each target
        axs = {t:fig.add_subplot(gs_map[t]) for t in targets}
                    
    with plt.rc_context({'axes.titlesize':'large'}):
        for t in targets: axs[t].set_title(t)
    
    if show_legend:
        ## Add legend in blank gridspec location
        # Add a subplot in the location
        axs['legend'] = fig.add_subplot(gs_map['legend'])
        # plot dummy data with appropriate labels, add legend, then remove axes
        
        if q_colors is not None:
            for j,q in enumerate(labels):
                axs['legend'].plot(0,0, **plt_kwargs, **q_colors[j], label=f'{q}')
            axs['legend'].legend(loc = 'center', title = 'log10(Copies)', fontsize = 16)
        else:
            for k,v in plt_kwargs.items():
                axs['legend'].plot(0,0, **v, label=k)
            axs['legend'].legend(loc = 'center', fontsize = 16)
        plt.setp(axs['legend'],**{
                'frame_on' : False,
                'xlim'     : [1,2],
                'ylim'     : [1,2],
                })
        axs['legend'].axes.get_xaxis().set_visible(False)
        axs['legend'].axes.get_yaxis().set_visible(False)
    
    return fig, axs

        
def plotGrid(targets, quantities, y, x = None, CTs = None, t_colors = None,
             fig = None, axs = None):
    n_t, n_q, n_r, n_c = np.shape(y)
    if not x: x = np.arange(n_c)+1

    if None in (fig,axs):
        # Subplot grid for targets and quantities
        fig, axs = plt.subplots(nrows=n_t, ncols=n_q,
                               sharey=True, sharex=True,
                               gridspec_kw={'hspace': 0., 'wspace': 0.})
    
    # Define plot styles for each target
    if t_colors is None:
        t_colors = {t:{'color':'C{:}'.format(i)} for i,t in enumerate(targets)}
    
    for (i,j,k) in np.ndindex(n_t,n_q,n_r):
        axs[i,j].plot(x,y[i,j,k,:],**t_colors[targets[i]])
        plt.setp(axs[i,j],**{
                'xticklabels'   : [],
                'yticklabels'   : []
                })
    
    if CTs is not None: # Denote ct's with vertical dashed lines
        for (i,j) in np.ndindex(n_t,n_q): axs[i,j].axvline(np.nanmean(CTs[i,j,:]), linestyle = ':', color = 'k')
    
    # Adjust y-tick spacing
    yl = axs[0,0].get_ylim()
    axs[0,0].set_yticks(np.linspace(yl[0],yl[1],5))
    
    # Add x and y labels for subplot grid
    for i,t in enumerate(targets):
        axs[i,0].set_ylabel(t, rotation=0, labelpad = 0, ha='right', va='center')
    for j,q in enumerate(quantities): axs[n_t-1,j].set_xlabel(np.log10(q))
    
    fig.text(0.5, 0.02, 'log$_{10}$ Copies', ha='center', weight = 'bold')
    fig.text(0.03, 0.5, 'Target', ha='center', va='center', rotation='vertical', weight = 'bold')
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.9])
        
    return {'fig' : fig, 'axs' : axs}
    

def setupGrid(n_t,n_q):
    return plt.subplots(nrows=n_t, ncols=n_q,
                        sharey=True, sharex=True,
                        gridspec_kw={'hspace': 0., 'wspace': 0.})


###############################################################################
#%% Plot efficiencies from CT fits
# Function for determining efficiency from CT fits with uncertainty
def calc_E (ct_mat,quant_list,n_reps):
    ct_unroll = np.reshape(ct_mat,len(quant_list)*n_reps)
    q_unroll = np.repeat(quant_list,n_reps)
    mask = ~np.isnan(ct_unroll)
    fit_stats = stat.linregress(q_unroll[mask],ct_unroll[mask])
    slope = fit_stats.slope
    E = 10**(-1/slope)-1 # calculation of the efficiency based on the slope of the fit
    dE = (fit_stats.stderr/slope**2)*E
    return {'E'         : 2*E,
            'dE'        : 2*dE,
            'slope'     : fit_stats.slope,
            'intercept' : fit_stats.intercept}

# Plot efficiencies with uncertainty
def plotBox(ax, x, E, dE, style = {}, label = None):
    if not style: style = {'color':'k'}    
    le = E-dE
    ue = E+dE
    ax.fill([x-0.25,x+0.25,x+0.25,x-0.25],[le,le,ue,ue],
            alpha = 0.5,
            **style)
    ax.plot([x-0.25,x+0.25],[E, E],lw=2,**style)
    ax.plot(x,E,'.',ms=20,**style, label = label)
    return

def plotEfit(targets, lg_q, CTs, t_colors = None, style = 'absolute'):
    n_t, n_q, n_r = np.shape(CTs)
    target_stats = {t: calc_E(CTs[i,::],lg_q,n_r) for i,t in enumerate(targets)}
    
    # Overlay CT efficiency fits
    fig_CT_eff, ax_CT_eff = plt.subplots(1,2, figsize = [12,4])
    
    if t_colors is None:
        t_colors = {t:{'color':'C{:}'.format(i)} for i,t in enumerate(targets)}
    
    for i,t in enumerate(targets):
        eff = target_stats[t]
        ax_CT_eff[0].plot(lg_q,CTs[i,::],
                        linestyle = 'none',
                        marker = '.',
                        **t_colors[t])
        ax_CT_eff[0].plot(lg_q,eff['intercept']+eff['slope']*lg_q,
                        **t_colors[t],
                        label = t)
    ax_CT_eff[0].legend()
    plt.setp(ax_CT_eff[0],**{
            'xticks': lg_q,
            'title' : 'Efficiency fits to CTs',
            'xlabel': 'log$_{10}$ Copies',
            'ylabel': 'C$_{T}$'
            })
    
    if style.casefold() in ('%','percent'):
        for v in target_stats.values():
            v['E'] /= 0.02
            v['dE'] /= 0.02
        
    for i,t in enumerate(targets):
        plotBox(ax_CT_eff[1],i+1,target_stats[t]['E'],target_stats[t]['dE'],style = t_colors[t]) 
        
    if style.casefold() == 'absolute':
        ax_CT_eff[1].set_ylim(0.95,3.05)
        ax_CT_eff[1].set_ylabel('Efficiency (abs)')
    elif style.casefold() in ('%','percent'):
        ax_CT_eff[1].set_ylim(0.5,1.5)
        ax_CT_eff[1].set_ylabel('Efficiency (%)')
        
    plt.setp(ax_CT_eff[1],**{
            'title'        : 'Efficiencies from CT Fits',
            'xlabel'       : 'Target',
            'xlim'         : [0.5,n_t+0.5],
            'xticks'       : np.arange(n_t)+1,
            'xticklabels'  : targets
            })
    return {'target_stats' : target_stats,
            'fig_CT_eff'   : fig_CT_eff,
            'ax_CT_eff'    : ax_CT_eff,}
    

###############################################################################
#%% Model definitions and fitting functions

# 4- and 5- parameter logistic models (4PLM, 5PLM)
def logistic_4P(c,F0,Fmax,cflex,b):
    return F0 + ( Fmax - F0 )/( 1 + (c/cflex)**b )

def logistic_4P_inv(Ffrac,cflex,b):
    return np.floor( cflex*( 1/Ffrac - 1 )**(1/b) ).astype(int)

def logistic_5P(c,F0,Fmax,cflex,b,g):
    return F0 + ( Fmax - F0 )/( 1 + ( c/cflex )**b )**g

def logistic_5P_inv(Ffrac,cflex,b,g):
    return np.floor( cflex*( (1/Ffrac)**(1/g) - 1 )**(1/b) ).astype(int)

## Fit full-process kinetics PCR model (FPK-PCR) from Lievens et al. 2011 (DOI:10.1093/nar/gkr775)
# Bilinear model for ln(ln(E)) from fluorescence
def E_BLM(F,a1,a2,a3,eta,chi,Fc):
    return chi + eta*np.log( np.exp( ( a1*(F-Fc)**2 + a2*(F-Fc) )/eta ) + np.exp( ( a3*(F-Fc) )/eta ) )

def fitSigmoid(x,y): 
    y = y - np.min(y) + np.min(np.diff(y)) # Zero data
    
    F0_guess = y[0]
    Fmax_guess = y[-1]
    cflex_guess = x[ np.argmax( y > np.mean([F0_guess,Fmax_guess]) ) ]
    b_guess = -10
    p_opt_4PLM, p_cov_4PLM = opt.curve_fit(logistic_4P, x, y, p0 = (F0_guess,Fmax_guess,cflex_guess,b_guess))

    '''    
    p_std_4PLM = np.sqrt(np.diag(p_cov_4PLM))
    residuals_4PLM = y - logistic_4P(x, *p_opt_4PLM)
    ss_res_4PLM = np.sum(residuals_4PLM**2)
    ss_tot_4PLM = np.sum((y-np.mean(y))**2)
    r2_4PLM = 1 - (ss_res_4PLM / ss_tot_4PLM)
    ''' 
   
    # Truncate the data to improve the 5PLM fit
    stop_plateau = logistic_4P_inv(0.99,*p_opt_4PLM[2:])
    
    try:
        # Fit 5PLM
        g_guess = 1
        p_opt_5PLM, p_cov_5PLM = opt.curve_fit(logistic_5P, x[:stop_plateau], y[:stop_plateau],
                                   p0 = (*p_opt_4PLM,g_guess))
    except RuntimeError:
        print('5PLM fit error')
        start_ground = 4
        start_decline = logistic_4P_inv(0.05,*p_opt_4PLM[2:])+1
        stop_ground = start_decline-7
        start_transition = logistic_4P_inv(0.85,*p_opt_4PLM[2:])-2
        stop_transition = logistic_4P_inv(0.95,*p_opt_4PLM[2:])-2
        
        p_opt = [*p_opt_4PLM,0]
        p_cov = p_cov_4PLM
        is_4PLM = True
    else:
        # Extract reference points from 5PLM
        #skip first 3 cycles for stability
        start_ground = 4
        #ground phase ends at ~5% maximum signal change
        start_decline = logistic_5P_inv(0.05,*p_opt_5PLM[2:])+1
        stop_ground = start_decline-7
        #phase transition btw first and second declines occurs btw 85% and 95%
        start_transition = logistic_5P_inv(0.85,*p_opt_5PLM[2:])
        stop_transition = logistic_5P_inv(0.95,*p_opt_5PLM[2:])
        
        p_opt = p_opt_5PLM
        p_cov = p_cov_5PLM
        is_4PLM = False
        
        
    ctrl_pts = {'start_ground'      : start_ground,
                'stop_ground'       : stop_ground,
                'start_decline'     : start_decline,
                'start_transition'  : start_transition,
                'stop_transition'   : stop_transition}
    
    return {'ctrl_pts'  : ctrl_pts,
            'p_opt'     : p_opt,
            'p_cov'     : p_cov,
            'is_4PLM'   : is_4PLM}
    
def fitLMs(F, ln2E, ctrl_pts):
    (start_ground, stop_ground, start_decline, start_transition, stop_transition) = ctrl_pts.values()
    

    # Fit first decline phase to guess a1, a2, and chi
    try:
        a1,a2,chi = np.polyfit(F[start_decline:start_transition], ln2E[start_decline:start_transition], 2)
    except:
        return
        
    # Fit second decline phase to guess a3
    dec2_y = F[start_transition:-1]
    dec2_x = ln2E[start_transition:]
    mask = ~np.isnan(dec2_x)
    try:
        a3_inv,interc_inv = np.polyfit(dec2_x[mask],dec2_y[mask],1)
    except:
        return
    
    a3 = 1/a3_inv
    interc = -interc_inv*a3
    
    return {'Lin1_params'  : [a1,a2,chi],
            'Lin2_params'  : [a3,interc]}

def fitBLM(F, ln2E, ctrl_pts, Lin1_params, Lin2_params):
    start_decline = ctrl_pts['start_decline']
    
    ln2E_ROI = ln2E[start_decline:]
    mask = ~np.isnan(ln2E_ROI)
    F_ROI = F[start_decline:-1][mask]
    ln2E_ROI = ln2E_ROI[mask]
    
    a1,a2,chi = Lin1_params
    a3,interc = Lin2_params
        
    # Graphically find intersection of two polynomials to guess Fc
    p1 = np.poly1d([a1,a2,chi])
    p2 = np.poly1d([a3,interc])
    F_range = np.linspace(0,np.max(F),1e5)
    Fc_ind = np.argmax(p2(F_range) <= p1(F_range))
    if ~Fc_ind:
        Fc = F[ctrl_pts['start_transition']]
    else:
        Fc = F_range[Fc_ind]
    
    # Standard guess for eta
    eta = -0.5
    
    guess_params = [a1,a2,a3,eta,chi,Fc]
    
    try:
        p_opt_E_BLM, p_cov_E_BLM = opt.curve_fit(E_BLM, F_ROI, ln2E_ROI,
                                                 p0 = guess_params)
        
    except RuntimeError: # Try again with modified guesses
        guess_params = [a1,a2*10,a3*10,eta*10,chi,Fc]
        try:
            p_opt_E_BLM, p_cov_E_BLM = opt.curve_fit(E_BLM, F_ROI, ln2E_ROI,
                                                     p0 = guess_params)
            
        except RuntimeError: # If it still doesn't work just use the guesses
            print('BLM Fit Error')
            return
        
    return p_opt_E_BLM


###############################################################################
#%% Fit each curve
def fit_all_curves(F):
    n_t, n_q, n_r, n_c = F.shape
    c = np.arange(n_c)+1
    # For storing fit parameters
    sig_params = np.zeros([n_t,n_q,n_r,5])
    Lin1_params = np.zeros([n_t,n_q,n_r,3])
    Lin2_params = np.zeros([n_t,n_q,n_r,2])
    BLM_params = np.zeros([n_t,n_q,n_r,6])
    ctrl_pts = np.zeros([n_t,n_q,n_r,5],dtype='int')
    
    E0 = np.zeros([n_t,n_q,n_r])
    skip_Fit, is_4PLM, badLinFit, badBLMFit = [np.full([n_t,n_q,n_r], False, dtype = bool) for _ in range(4)]
    
    #skip_Fit[5,3:] = True
    
    E = F[...,1:]/F[...,:-1]
    ln2E = np.log( np.log(E) )
    
    for (i,j,k) in np.ndindex(n_t,n_q,n_r):
        this_F = F[i,j,k]
        if skip_Fit[i,j,k]: continue
        print('{:}|{:}|{:}'.format(i,j,k))
        
        sig_fit = fitSigmoid(c,this_F)
        pts, sig_params[i,j,k], _, is_4PLM[i,j,k] = sig_fit.values()
        ctrl_pts[i,j,k] = list(pts.values())
        
        LM_fit = fitLMs(this_F, ln2E[i,j,k], pts)
        if LM_fit is None:
            badLinFit[i,j,k] = True
            badBLMFit[i,j,k] = True
            continue
        
        Lin1_params[i,j,k], Lin2_params[i,j,k] = LM_fit.values() 
        
        BLM_fit = fitBLM(this_F, ln2E[i,j,k], pts, Lin1_params[i,j,k], Lin2_params[i,j,k])
        if BLM_fit is None:
            badBLMFit[i,j,k] = True
            continue
        
        BLM_params[i,j,k] = BLM_fit
        E0[i,j,k] = np.exp( np.exp( E_BLM(0,*BLM_params[i,j,k]) ) )
               
    
    print('\n{:} bad fits'.format(np.sum(badBLMFit)))
    bad_Fit = skip_Fit | is_4PLM | badLinFit | badBLMFit
    
    return [sig_params, Lin1_params, Lin2_params, BLM_params, ctrl_pts, E, ln2E, E0,
            skip_Fit, is_4PLM, badLinFit, badBLMFit, bad_Fit]
  
    
###############################################################################
#%% Plot each curve 
def plot_all_curves(F, targets, quantities,
                    sig_params, Lin1_params, Lin2_params, BLM_params, ctrl_pts,
                    E, ln2E, E0, skip_Fit, is_4PLM, badLinFit, badBLMFit, bad_Fit,
                    q_colors = None, t_map = None):
    ## Set up plots for transformed efficiency vs fluorescence with different layers of information
    
    n_t, n_q, n_r, n_c = F.shape
    c = np.arange(n_c)+1
    
    if q_colors is None:
        cmap_q = sns.color_palette('cubehelix',n_q+1)
        q_colors = {j:{'color' : cmap_q[j]} for j in range(n_q)}

    
    dot_style = {'marker'   : '.',
                 'linestyle': 'None',
                 'alpha'    : 0.25}
    
    fit_style = {'linestyle' :'-',
                 'alpha'     : 1}
    
    extrap_style = {'linestyle' : '--'}
        
    fig_ln2EvF_dots, axs_ln2EvF_dots = setupOverlay(targets, quantities, t_map = t_map)
    fig_ln2EvF_LMs, axs_ln2EvF_LMs = setupOverlay(targets, quantities, t_map = t_map)
    fig_ln2EvF_BLM, axs_ln2EvF_BLM = setupOverlay(targets, quantities, t_map = t_map)
    ln2EvF_axs = [axs_ln2EvF_dots, axs_ln2EvF_LMs, axs_ln2EvF_BLM]
    ln2EvF_figs = [fig_ln2EvF_dots, fig_ln2EvF_LMs, fig_ln2EvF_BLM]
    fig_EvC_dots, axs_EvC_dots = setupOverlay(targets, quantities, t_map = t_map)
    fig_EvC_LMs, axs_EvC_LMs = setupOverlay(targets, quantities, t_map = t_map)
    fig_EvC_BLM, axs_EvC_BLM = setupOverlay(targets, quantities, t_map = t_map)
    EvC_axs = [axs_EvC_dots, axs_EvC_LMs, axs_EvC_BLM]
    EvC_figs = [fig_EvC_dots, fig_EvC_LMs, fig_EvC_BLM]
    
    for t in targets:
        for axs in EvC_axs:
            plt.setp(axs[t],**{
                    'title': t,
                    'ylim': [0.5,3],
                    'yticks': np.arange(0.5,3.1,0.5),
                    'yticklabels': ['','1','','2','','3'],
                    'xlabel': 'Cycle',
                    'ylabel': 'Efficiency'})
        
        for axs in ln2EvF_axs:
            plt.setp(axs[t],**{
                    'title': t,
                    'ylim': [-8,1],
                    'xlabel': 'Fluorescence',
                    'ylabel': 'ln$^2$ Efficiency'})
        
    #for i in range(1,len(targets)):
    #    # Share x axes between all targets
    #    for axs in EvC_axs: axs[targets[0]].get_shared_x_axes().join(axs[targets[0]],axs[targets[i]])
    #    # Share x axes between all HEX targets
    #    for axs in ln2EvF_axs: axs[targets[1]].get_shared_x_axes().join(axs[targets[1]],axs[targets[i]])
    
    ## Plot each curve
    for (i,j,k) in np.ndindex(n_t,n_q,n_r):
        if skip_Fit[i,j,k]: continue
        t = targets[i]
        this_F = F[i,j,k]
        this_ln2E = ln2E[i,j,k]
        this_E = E[i,j,k]
    
        start_ground, stop_ground, start_decline, start_transition, stop_transition = ctrl_pts[i,j,k]
        
        # Plot individual data points
        for axs in ln2EvF_axs:
            axs[t].plot(this_F[:-1], this_ln2E, **q_colors[j], **dot_style)
        for ax in EvC_axs:
            ax[t].plot(c[stop_ground:-1], this_E[stop_ground:], **q_colors[j], **dot_style)
    
        # Overlay the linear model fits
        if badLinFit[i,j,k]: continue
        Lin1_F = this_F[stop_ground:start_transition]
        Lin1_c = c[stop_ground:start_transition]
        Lin2_F = this_F[start_transition+2:]
        Lin2_c = c[start_transition+2:]
        p1 = np.poly1d(Lin1_params[i,j,k])
        p2 = np.poly1d(Lin2_params[i,j,k])
        
        axs_ln2EvF_LMs[t].plot(Lin1_F,p1(Lin1_F), **q_colors[j], **fit_style)
        axs_ln2EvF_LMs[t].plot(Lin2_F,p2(Lin2_F), **q_colors[j], **fit_style)
        
        axs_EvC_LMs[t].plot(Lin1_c,np.exp( np.exp( p1(Lin1_F) ) ), **q_colors[j], **fit_style)
        axs_EvC_LMs[t].plot(Lin2_c,np.exp( np.exp( p2(Lin2_F) ) ), **q_colors[j], **fit_style)
        #Extrapolate the first fit back to the beginning
        extrap_F = this_F[:stop_ground]
        extrap_c = c[:stop_ground]
        axs_EvC_LMs[t].plot(extrap_c,np.exp( np.exp( p1(extrap_F) ) ), **q_colors[j], **extrap_style)
    
        #Overlay the bilinear model fits
        if badBLMFit[i,j,k]: continue
        BLM_F = this_F[start_decline:-1]
        x = np.linspace(this_F[start_decline],np.max(this_F),100)
        BLM_c = c[start_decline:-1]
        axs_ln2EvF_BLM[t].plot(x,E_BLM(x,*BLM_params[i,j,k]), **q_colors[j], **fit_style)
        axs_EvC_BLM[t].plot(BLM_c,np.exp( np.exp( E_BLM(BLM_F, *BLM_params[i,j,k]) ) ), **q_colors[j], **fit_style)
        #Extrapolate back to the beginning
        extrap_F = this_F[:start_decline]
        extrap_c = c[:start_decline]
        axs_EvC_BLM[t].plot(extrap_c,np.exp( np.exp( E_BLM(extrap_F, *BLM_params[i,j,k]) ) ), **q_colors[j], **extrap_style)
    
#    for f in EvC_figs: mypy.bigntight(f)
#    for f in ln2EvF_figs: mypy.bigntight(f)
    
    return EvC_figs, EvC_axs, ln2EvF_figs, ln2EvF_axs


###############################################################################
#%% Make detailed side-by-side comparison plots for one target        

def focusTarget(focus_t, F, targets, quantities, 
                    sig_params, Lin1_params, Lin2_params, BLM_params, ctrl_pts,
                    E, ln2E, E0, skip_Fit, is_4PLM, badLinFit, badBLMFit, bad_Fit,
                    q_colors = None, show_fit = ('BLM','LM','none')):
    
    blm = 'BLM' in show_fit
    lm = 'LM' in show_fit
    nf = 'none' in show_fit
    
    i = targets.index(focus_t)
    n_t, n_q, n_r, n_c = F.shape
    c = np.arange(n_c)+1
    
    if q_colors is None:
        cmap_q = sns.color_palette('cubehelix',n_q+1)
        q_colors = {j:{'color' : cmap_q[j]} for j in range(n_q)}
    
    # Plot with smaller x- and y-tick labels and axis labels
    focus_axs = list()
    focus_figs = list()
    with plt.rc_context({'axes.labelsize'  : 'small',
                         'axes.labelweight': 'normal',
                         'xtick.labelsize' : 'small',
                         'ytick.labelsize' : 'small'}):
        
        if blm:
            fig_focus_BLM, axs_focus_BLM = plt.subplots(1,3, figsize=[26,9])
            focus_axs.append(axs_focus_BLM)
            focus_figs.append(fig_focus_BLM)
        if lm:
            fig_focus_LMs, axs_focus_LMs = plt.subplots(1,3, figsize=[26,9])
            focus_axs.append(axs_focus_LMs)
            focus_figs.append(fig_focus_LMs)
        if nf:
            fig_focus_dots, axs_focus_dots = plt.subplots(1,3, figsize=[26,9])
            focus_axs.append(axs_focus_dots)
            focus_figs.append(fig_focus_dots)
    if blm: fig_focus_BLM.suptitle('{:s} focus\nBLM Fits'.format(focus_t))
    if lm: fig_focus_LMs.suptitle('{:s} focus\nLM Fits'.format(focus_t))
    if nf: fig_focus_dots.suptitle('{:s} focus\nPoints Only'.format(focus_t))   
        

    dot_style = {'marker'   : '.',
                 'linestyle': 'None',
                 'alpha'    : 0.25}
    
    fit_style = {'linestyle' :'-',
                 'alpha'     : 1}
    
    extrap_style = {'linestyle' : '--'}
    
    for (j,k) in np.ndindex(n_q,n_r):
        if skip_Fit[i,j,k]: continue
        this_F = F[i,j,k]
        this_ln2E = ln2E[i,j,k]
        this_E = E[i,j,k]
    
        start_ground, stop_ground, start_decline, start_transition, stop_transition = ctrl_pts[i,j,k]
        
        # Plot individual data points
        for axs in focus_axs:
            axs[0].plot(c,this_F,**q_colors[j], **dot_style)
            axs[1].plot(this_F[:-1], this_ln2E, **q_colors[j], **dot_style)
            axs[2].plot(c[stop_ground:-1], this_E[stop_ground:], **q_colors[j], **dot_style)
            
        # Overlay the sigmoidal fits
        if blm:
            if is_4PLM[i,j,k]:
                p = axs_focus_BLM[0].plot(c,logistic_4P(c,*sig_params[i,j,k,:-1]), **q_colors[j], **fit_style)
            else:
                p = axs_focus_BLM[0].plot(c,logistic_5P(c,*sig_params[i,j,k,:]), **q_colors[j], **fit_style)        
            if k == 1: plt.setp(p,'label','{:.0E}'.format(quantities[j]))
            
        if lm:
            if is_4PLM[i,j,k]:
                p = axs_focus_LMs[0].plot(c,logistic_4P(c,*sig_params[i,j,k,:-1]), **q_colors[j], **fit_style)
            else:
                p = axs_focus_LMs[0].plot(c,logistic_5P(c,*sig_params[i,j,k,:]), **q_colors[j], **fit_style)        
            if k == 1: plt.setp(p,'label','{:.0E}'.format(quantities[j]))
            
    
        # Overlay the linear model fits
        if badLinFit[i,j,k]: continue
        Lin1_F = this_F[stop_ground:start_transition]
        Lin1_c = c[stop_ground:start_transition]
        Lin2_F = this_F[start_transition+2:]
        Lin2_c = c[start_transition+2:]
        p1 = np.poly1d(Lin1_params[i,j,k])
        p2 = np.poly1d(Lin2_params[i,j,k])
        
        if lm:
            axs_focus_LMs[1].plot(Lin1_F,p1(Lin1_F), **q_colors[j], **fit_style)
            axs_focus_LMs[1].plot(Lin2_F,p2(Lin2_F), **q_colors[j], **fit_style)
            
            axs_focus_LMs[2].plot(Lin1_c,np.exp( np.exp( p1(Lin1_F) ) ), **q_colors[j], **fit_style)
            axs_focus_LMs[2].plot(Lin2_c,np.exp( np.exp( p2(Lin2_F) ) ), **q_colors[j], **fit_style)
            #Extrapolate the first fit back to the beginning
            extrap_F = this_F[:stop_ground]
            extrap_c = c[:stop_ground]
            axs_focus_LMs[2].plot(extrap_c,np.exp( np.exp( p1(extrap_F) ) ), **q_colors[j], **extrap_style)
        
        if blm:
            #Overlay the bilinear model fits
            if badBLMFit[i,j,k]: continue
            BLM_F = this_F[start_decline:-1]
            x = np.linspace(this_F[start_decline],np.max(this_F),100)
            BLM_c = c[start_decline:-1]
            axs_focus_BLM[1].plot(x,E_BLM(x,*BLM_params[i,j,k]), **q_colors[j], **fit_style)
            axs_focus_BLM[2].plot(BLM_c,np.exp( np.exp( E_BLM(BLM_F, *BLM_params[i,j,k]) ) ), **q_colors[j], **fit_style)
            #Extrapolate back to the beginning
            extrap_F = this_F[:start_decline]
            extrap_c = c[:start_decline]
            axs_focus_BLM[2].plot(extrap_c,np.exp( np.exp( E_BLM(extrap_F, *BLM_params[i,j,k]) ) ), **q_colors[j], **extrap_style)
    
    if lm: axs_focus_LMs[0].legend(title = 'Copies')
    if blm: axs_focus_BLM[0].legend(title = 'Copies')
    
    for ax in focus_axs:
        plt.setp(ax[0],**{
                'xlabel': 'Cycle',
                'ylabel': 'Fluorescence',
                'title' : 'Signal over time'})
        plt.setp(ax[1],**{
                'xlabel': 'Fluorescence',
                'ylabel': 'ln$^2$ Efficiency',
                'title' : 'Transformed Efficiency'})
        plt.setp(ax[2],**{
                'ylim'  : [0.89, 2.5],
                'xlim'  : ax[0].get_xlim(),
                'xlabel': 'Cycle',
                'ylabel': 'Efficiency',
                'title' : 'Efficiency over time'})
    
    for fig in focus_figs:
        plt.pause(1e-1)
        fig.tight_layout(rect=[0, 0, 1, 0.9])
    return focus_figs, focus_axs


 
###############################################################################  
#%% Plot parameters by target and quantity

def DF_params(params, F, targets, quantities):
    
    n_t,n_q,n_r,_ = F.shape

    base = {'Target'    : [targets[i] for (i,_,_) in np.ndindex(n_t,n_q,n_r)],
            'TargetN'   : [i for (i,_,_) in np.ndindex(n_t,n_q,n_r)],
            'Quantity'  : [np.log10(quantities[j]) for (_,j,_) in np.ndindex(n_t,n_q,n_r)],
            'QuantityN' : [j for (_,j,_) in np.ndindex(n_t,n_q,n_r)]}
    
    return pd.DataFrame({**base,**params})

def plotParams(FitParams,thresholds,x = 'Target', hue = 'Quantity', order = None):
    
    params = list(thresholds.keys())
    n_p = len(thresholds)
    if n_p < 4:
        ncols = n_p
        nrows = 1
    else:
        sq = (n_p)**0.5
        nrows = int(sq)
        ncols = n_p//nrows + int(n_p % nrows != 0)

    fig, axs = plt.subplots(nrows, ncols, squeeze = False)
    
    bad = pd.Series(index=FitParams.index,dtype=bool)
    
    for i in range(n_p):
        p = params[i]
        
        thresh = thresholds[p]
        if thresh is np.ndarray:
            assert len(thresh) == 2, 'Threshold must have at most two bounds'
            out = (FitParams[p] < thresh[0]) | (FitParams[p] > thresh[1])
        else:
            out = (np.abs(FitParams[p]) > np.abs(thresholds[p]))
        na = FitParams[p].isin([np.nan,np.inf,-np.inf])
        bad = bad | out | na | FitParams.bad
        
    good = ~bad
    
       
    for i in range(n_p):
        p = params[i]
        ax = axs.flat[i]
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        sns.violinplot(x = x, y = p, data = FitParams[good],
                      order=order, 
                       color ='white', inner = None, cut = 0, ax = ax,zorder=0)
        
        
        for i,tar in enumerate(order):
            this = FitParams[good]
            this = this[this[x]==tar]
            zorder = this.sort_values(by=[hue]).index
            ax.scatter(x=[i for _ in range(len(this))],y=this[p][zorder],c=this.color[zorder],s=10**2,zorder=5)
        '''
        sns.stripplot(x = x, y = p, data = FitParams[good],
                      order=order, 
                      size = 10, alpha = 1, jitter = False, hue = hue, ax = ax)
        '''
        ax.set_title(p)
        
        #ax.legend().remove()
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
    
    for i in range(n_p,axs.size):
        axs.flat[i].axis('off')
        
        
    return fig, axs


    
    
    
    
    
    
    