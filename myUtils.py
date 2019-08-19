import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime #For timestamping files
import pathlib as pl
import shelve #For saving/loading variables
import scipy.stats as stats
import scipy.optimize as opt
from tqdm import tqdm


def plotSettings():
    plt.style.use('seaborn-poster')
    
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['patch.linewidth'] = 2
    
    mpl.rcParams['font.size'] = '20'
    mpl.rcParams['figure.titlesize'] = 'large'
    mpl.rcParams['figure.titleweight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 'medium'
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.titlesize'] = 'large'
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['xtick.labelsize'] = 'medium'
    mpl.rcParams['ytick.labelsize'] = 'medium'
    return

    
#For datestamping files
def timeStamped(fname, fmt='{fname} %y-%m-%d'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def setupPath(file,processed = None):
    code_pth = pl.Path(file)
    base_pth = code_pth.parent.parent
    if processed is None:
        data_pth = base_pth / 'Data'
    else:
        data_pth = base_pth / processed   
    rslt_pth_parent = base_pth / 'Results'
    rslt_pth = rslt_pth_parent / pl.Path(datetime.datetime.now().strftime('%y-%m-%d'))
    rslt_pth.mkdir(parents=True, exist_ok=True) 
    return code_pth, base_pth, data_pth, rslt_pth, rslt_pth_parent


def setupShelves(newpath, newname, oldpath = None, oldname = None):
    #Store results into a datestamped shelf file
    new_shelf = str(newpath / (newname + '.shelf'))
    if None not in (oldpath,oldname):
        old_shelf = str(oldpath / (oldname + '.shelf'))
        with shelve.open(old_shelf) as shelf:
            for key in shelf:
                print(key)
        return new_shelf, old_shelf
    return new_shelf


#Maximize figure, pause, call tightlayout
def bigntight(fig_obj):
    mng = fig_obj.canvas.manager.window
    mng.showMaximized()
    mng.activateWindow()
    mng.raise_()
    plt.pause(1e-1)
    fig_obj.tight_layout(rect=[0, 0, 1, 0.95])
    return


#For saving figures
def savemyfig(fig_obj, title, path = pl.Path.cwd()):
    fig_obj.show()
    mng = fig_obj.canvas.manager.window
    mng.activateWindow()
    mng.raise_()
    print('Saving.', end = '')
    fig_obj.savefig(path / (title+'.png'),dpi=300, bbox_inches='tight')
    print('.', end = '')
    fig_obj.savefig(path / (title+'.svg'), bbox_inches='tight')
    print('Done')
    return


############################################################################################################
# From 01: Plotting data, uncertainty, curve fits
def classical_fit_intervals(func,p_opt,x,y,xpts):
    tile_x = np.tile(x,[y.size//x.size,1]).T
    n = y.size
    m = p_opt.size
    dof = n-m                                                # Degrees of freedom
    res = y - func(tile_x,*p_opt)                            # Residuals
    t = stats.t.ppf((1. + 0.95)/2., n - m)                   # Student's t distribution
    #chi2 = np.sum((res / func(tile_x,*p_opt))**2)            # chi-squared; estimates error in data
    #chi2_red = chi2 / dof                                    # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(res**2) / dof)                    # standard error of the fit at each point

    ci = t * s_err * np.sqrt(1/n + (xpts - np.mean(x))**2 / np.sum((x - np.mean(x))**2))

    pi = t * s_err * np.sqrt(1 + 1/n + (xpts - np.mean(x))**2 / np.sum((x - np.mean(x))**2))

    return ci, pi

def classical_fit_param_summary(p_opt,p_cov, names = None):
    nstd = stats.norm.ppf((1. - 0.95)/2.)
    p_std = np.sqrt(np.diag(p_cov))
    p_ci_lower = p_opt - nstd * p_std
    p_ci_upper = p_opt + nstd * p_std
    summary = pd.DataFrame(data = [p_ci_lower,p_opt,p_ci_upper,p_std],
                           index = ('95% CI Lower Limit','Optimal Value','95% CI Upper Limit','Standard Error'),
                           columns = names)
    return summary    


############################################################################################################
# From 02: Bootstrapping confidence intervals

# Performing the bootstrap algorithm
def bootstrap_fits(func, x, y, p_opt, n_straps = 1000, res = 100, xpts = None, guess_gen = None,
                 fit_kws = {}, conservative = True, piecewise = True):
    # If y is a vector of length 'm', x must also be a vector of length 'm'
    # If y is a matrix of shape 'm x n', with replicates in different columns, x must either be a vector of length 'm' or a matrix of shape 'm x n'
    
    # Number of unique x values
    n = len(set(x))
    
    # Number of replicates
    m = y.size//n
    
    # Piecewise bootstrapping is nonsensical if there's only one y per x
    if y.ndim == 1: piecewise = False
    
    # Generate points at which to evaluate the curves
    if xpts is None: xpts = np.linspace(x.min(),x.max(),res)
    elif xpts.size == 2: xpts = np.linspace(xpts[0],xpts[1],res)

    # Predicted y values
    y_fit = func(x,*p_opt)
    
    # Tile the predicted y values if necessary so that they're the same shape as the original data
    if y_fit.shape != y.shape: y_fit = np.tile(y_fit,[y.size//x.size,1]).T
        
    # Get the residuals (and they're )
    resid = y - y_fit
    
    p_strapped = np.zeros([n_straps,p_opt.size])    # Create a matrix of zeros to store the parameters from each bootstrap iteration
    curve_strapped = np.zeros([n_straps,xpts.size]) # Another matrix to store the predicted curve for each iteration
    
    for i in tqdm(range(n_straps)):
        
        # Choose new residuals based on the specified method
        if piecewise and conservative:
            invalid_sample = True
            while invalid_sample:
                resid_resamples = np.array([np.random.choice(resid[row],size = m) for row in range(n)])
                if all(len(set(resid_resamples[row])) > 1 for row in range(n)): invalid_sample = False
                    
        elif piecewise and not conservative:
            sigma_resid = [resid[row].std() for row in range(n)]
            resid_resamples = np.array([np.random.normal(0, size = m) for row in range(n)])
        elif not piecewise and not conservative:
            sigma_resid = resid.std()
            resid_resamples = np.random.normal(0, sigma_resid, size = resid.shape)
            
        elif not piecewise and conservative:
            resid_resamples = np.random.choice(resid.flat, size = resid.shape)
                
        # Generate a synthetic dataset from the sampled residuals
        new_y = y_fit+resid_resamples
        
        if guess_gen is not None:
            # Generate guesses for this dataset
            guesses = guess_gen(x,new_y)
        else:
            # Default guesses
            guesses = np.ones(len(p_opt))
        
        # Additional keyword arguments to curve_fit can be passed as a dictionary via fit_kws
        if y.ndim == 1:
            p_strapped[i], _ = opt.curve_fit(func, x, new_y,
                                             p0 = guesses,
                                             **fit_kws)
        else:
            p_strapped[i], _ = opt.curve_fit(func, x, new_y.mean(1),
                                             sigma = new_y.std(1), absolute_sigma = True,
                                             p0 = guesses,
                                             **fit_kws)
        
        curve_strapped[i] = func(xpts,*p_strapped[i])
    
    return p_strapped, curve_strapped

# Plot the bootstrapped curve and its confidence intervals
def bootstrap_plot(xpts,bootstrap_curves, CI = 95, line_kws ={},fill_kws={}):
    c_lower = np.percentile(bootstrap_curves,(100-CI)/2,axis = 0)
    c_median = np.percentile(bootstrap_curves,50,axis = 0)
    c_upper = np.percentile(bootstrap_curves,(100+CI)/2,axis = 0)
    
    # Additional keyword arguments to plot or fill_between can be passed as a dictionary via line_kws and fill_kws, respectively
    med = plt.plot(xpts, c_median, **line_kws)
    ci = plt.fill_between(xpts, c_upper, c_lower, color = plt.getp(med[0],'color'), alpha = 0.25, **fill_kws)
    return med, ci

# Summarize parameters and confidence intervals resulting from the bootstrap algorithm
def bootstrap_summary(bootstrap_params, CI = 95, names = None):
    p_lower = np.percentile(bootstrap_params,(100-CI)/2,axis = 0)
    p_median = np.percentile(bootstrap_params,50,axis = 0)
    p_upper = np.percentile(bootstrap_params,(100+CI)/2,axis = 0)
    
    summary = pd.DataFrame(data = [p_lower,p_median,p_upper],
                       index = ('{:}% CI Lower Limit'.format(CI),'Median Value','{:}% CI Upper Limit'.format(CI)),
                       columns = names)
    return summary  

# Plot the bootstrapped distributions for each parameter and label with the modal value derived from its KDE
def bootstrap_dists(bootstrap_params, CI = 95, names = None, rug_kws = {}, kde_kws = {}):
    _,n_p = bootstrap_params.shape
    mode = np.zeros([n_p,])
    
    fig, axs = plt.subplots(1, n_p, figsize = (4*n_p,3))
    for p in range(n_p):
        sns.distplot(bootstrap_params[:,p], ax = axs[p], **rug_kws, **kde_kws)
        axs[p].axvline(np.percentile(bootstrap_params[:,p], (100-CI)/2, axis = 0), ls = '--', color = 'k')
        axs[p].axvline(np.percentile(bootstrap_params[:,p], 50, axis = 0), ls = ':', color = 'k')
        axs[p].axvline(np.percentile(bootstrap_params[:,p], (100+CI)/2, axis = 0), ls = '--', color = 'k')

        KDE = axs[p].get_children()[-14]
        mode[p] = KDE.get_xdata()[np.argmax(KDE.get_ydata())]
        
        axs[p].set_title(names[p] + '\n' + 'mode = {:.3f}'.format(mode[p]))
        
    return fig, axs, mode
            

    
    
    
    




