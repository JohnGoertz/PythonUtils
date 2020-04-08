import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stat
import lmfit as lf
import copy as cp
from scipy.integrate import odeint
import scipy.optimize as opt
import warnings

from . import myUtils as mypy

#%% Import Data

def importQuantStudio(data_pth, data_file, *,
                      header = None, debug = False):
  
    try:
        xlsx = pd.ExcelFile(data_pth / data_file)
    except PermissionError as e:
        print('Permission Denied. The file may be open; close the file and try again.')
        return print(e)
    
    if header is None: 
        test = pd.read_excel(xlsx, sheet_name = 'Sample Setup')
        header = pd.Index(test.iloc[:,0]).get_loc('Well')+1
        
    sheets = pd.read_excel(xlsx, header = header,
                         sheet_name = ['Sample Setup',
                                       #'Raw Data',
                                       'Amplification Data',
                                       'Results',
                                      ])
                           
    
    imps = {'setup'         : sheets['Sample Setup'],
            #'raw'           : sheets['Raw Data'],
            'data'          : sheets['Amplification Data'],
            'instr_results' : sheets['Results'],
           }
    '''
    try:
        raw = pd.read_excel(xlsx, sheet_name = 'Raw Data', header = header)
    except:
        raw = None
    '''    
    
    
    if 'Target Name' not in imps['setup'].columns:
        display(imps['setup'])
    
    undetermined_CT = imps['instr_results'].CT.astype(str).str.contains('Undetermined')
    imps['instr_results'].CT.mask(undetermined_CT,np.nan,inplace=True)
    imps['instr_results'].CT = imps['instr_results'].CT.astype(float).values
    
    wells = imps['setup'][~imps['setup']['Target Name'].isna()].Well
    for df in imps.values():
        if df is None: continue
        df.columns = df.columns.str.replace(' ', '')
        df = df[df.Well.isin(wells)]
    
    return imps

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



#%% Plot each target in own axis

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


#%% Equation Definitions

## Define the (differential) equations to be fit to the data

# Convert the growth rate from base-e to base-2
lg2_e = np.log2(np.exp(1))


# Logistic growth equation with growth rate r and carrying capacity K
def growth(t, params):
    pop0 = params['pop0'].value
    r = params['r'].value
    K = params['K'].value
    return K / (1 + (K - pop0) / pop0 * np.exp(-r / lg2_e * t))


# Logistic growth ODE
def growth_deq(pop, t, params):
    r = params['r'].value
    K = params['K'].value
    x = pop
    return r / lg2_e * x * (1 - x / K)


# Solve the logistic growth ODE
def growth_deq_sol(t, pop0, params):
    return odeint(growth_deq, pop0, t, args=(params,))


# Error in the growth model
def growth_residual(params, t, data):
    pop0 = params['pop0'].value
    model = growth_deq_sol(t, pop0, params)
    return (model - data).ravel()  # *np.gradient(data.ravel())


# Linear drift equation
def drift(t, params):
    intercept = params['intercept'].value
    slope = params['slope'].value
    return intercept + slope * t


# A mixture model for the drift and growth equations (solutions)
def mix_model(t, params):
    return drift(t, params) * growth(t, params)


# Error in the growth model
def mix_residual(params, t, data):
    model = np.reshape(mix_model(t, params), [-1, 1])
    return (model - data).ravel()  # *np.gradient(data.ravel())


# Logistic growth ODE where K (carrying capacity) is a linear function of time
def mix_deq(vals, t, params):
    pop, K = vals
    r = params['r'].value
    m = params['slope'].value

    dx = r / lg2_e * pop * (1 - pop / K)
    dK = m

    return [dx, dK]


# Solve the ODE with tight absolute tolerance
def mix_deq_sol(t, vals0, params):
    abserr = 1.0e-12
    relerr = 1.0e-6
    return odeint(mix_deq, vals0, t, args=(params,),
                  atol=abserr, rtol=relerr)


# Error in the growth/drift ODE model.
# Assess only the full model, not the drift component alone (model[:,1])
def mix_deq_res(params, t, data):
    pop0 = params['pop0'].value
    K0 = params['intercept'].value
    model = mix_deq_sol(t, [pop0, K0], params)
    return (model[:, 0] - data[:, 0]).ravel()

#%% Fitting Strategy
# Fit differential equations directly (True) or use the "empirical" mixture model (False).
# The direct diffeq approach doesn't improve the fit greatly, and takes ~twice as long, but is more explainable.

def driftgrowth(c, data, diffeq=True, row=None):
    # Fit a simple logistic growth ODE model with no drift to provide estimates for the drift model
    init_growth = lf.Parameters()
    init_growth.add('r', value=1, min=0.1, max=2, vary=True)  # Could maybe be a wider range
    # Offset which relates known initial copy number to estimated initial fluorescence intensity
    # There appears to have been a mistake where BP200 and BP240 were not diluted properly, off
    # by an order of magnitude. This would probably be better to address in the initial import, but...
    # that might break things
    pop0_offset = 11.5 if row.Tar not in ['BP200', 'BP240'] else 10.5
    pop0_guess = row.Tar_Q - pop0_offset
    init_growth.add('pop0', value=10 ** (pop0_guess), min=10 ** (pop0_guess - 1), max=10 ** (pop0_guess + 1), vary=True)
    init_growth.add('K', value=float(data[-1]) * 0.8, min=1e-1, max=1.5)

    init_result = lf.minimize(growth_residual, init_growth, args=(c, data), method='leastsq')
    final = data + init_result.residual.reshape(data.shape)

    # Define the "beginning" of the drift region as where the numerical derivative is less than 1% above its final value
    '''
    cutoff = 0.01
    dy = data[1:-1]/data[0:-2]
    dy /= max(dy)
    start = np.argmax((dy>dy[-1]+cutoff)[::-1])
    '''
    if row is not None:
        drift0 = len(c) - 5 if row.Tar_Q == 2 else len(c) - 10
    else:
        drift0 = 45  # c[-start] if start<5 else 45

    init_drift = np.polyfit(c[drift0:], data[drift0:], 1)[:, 0]

    # Use outputs from the preliminary growth fit as initial guesses for the mixture model
    mixture = lf.Parameters()
    # Should probably limit variation to a narrower relative range for both pop0 and r
    mixture['r'] = cp.copy(init_result.params['r'])
    mixture['pop0'] = cp.copy(init_result.params['pop0'])
    # Allow the drift parameters to vary only slightly (20%) from their values found by the linear fit
    # Otherwise the intercept and r in particular become far too correlated
    slope = float(init_drift[0])
    intercept = float(init_drift[1])
    if np.abs(slope) < 1e-15:
        mixture.add('slope', value=slope, min=-1e-10, max=1e-10)
    else:
        mixture.add('slope', value=slope, min=slope * 0.8, max=slope * 1.2)
    mixture.add('intercept', value=intercept, min=intercept * 0.8, max=intercept * 1.2)

    if diffeq:
        mix_result = lf.minimize(mix_deq_res, mixture, args=(c, data), method='leastsq')
        pop0 = mix_result.params['pop0'].value
        K0 = mix_result.params['intercept'].value
        model = mix_deq_sol(c, [pop0, K0], mix_result.params)
        final = model[:, 0]
    else:
        mixture.add('K', value=1, vary=False)
        mix_result = lf.minimize(mix_residual, mixture, args=(c, data), method='leastsq')
        final = data[:, 0] + mix_result.residual.reshape(data[:, 0].shape)  # /np.gradient(data.ravel())
    return {'params': mix_result,
            'fit': final}

#%% Plot Fit parameters by target and quantity

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
        
        
        for i,val in enumerate(order):
            this = FitParams[good]
            this = this[this[x]==val]
            dot_zorder = this.sort_values(by=[hue]).index
            ax.scatter(x=[i for _ in range(len(this))],y=this[p][dot_zorder],c=this.color[dot_zorder],s=10**2,zorder=5)
            median_rxn_val = np.median(this[this.Quantity==np.median(this.Quantity)][p])
            ax.plot([i-0.25, i+0.25],[median_rxn_val,median_rxn_val],'k',zorder=1)
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

