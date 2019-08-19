import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime #For timestamping files
import pathlib as pl
import shelve #For saving/loading variables
import scipy.stats as stats
import scipy.optimize as opt
import numpy as np
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


#Fit specified curve and return 95% confidence and prediction intervals
def curve_fit_ci(func, x, y, method = 'bootstrap', n_steps = 500, res = 100, xpts = None, fit_kwargs = {}):
    
    if y.ndim == 1:
        p_opt, p_cov = opt.curve_fit(func, x, y, **fit_kwargs)
    else:
        p_opt, p_cov = opt.curve_fit(func, x, y.mean(1), sigma = y.std(1), absolute_sigma = True, **fit_kwargs)
        
    gof = {'classical': classical_ci(func, x, y, p_opt, p_cov, res, xpts),
           'bootstrap': None}
    return gof.get(method,None)


def classical_ci(func, x, y, p_opt, p_cov, res = 100, xpts = None):
    if x.size != y.size:
        x = np.tile(x,[y.size//x.size,1]).T

    n = y.size
    m = p_opt.size
    dof = n-m                                                # Degrees of freedom
    resid = y - func(x,*p_opt)                                 # Residuals
    t = stats.t.ppf(0.975, n - m)                            # Student's t distribution
    chi2 = np.sum((resid / func(x,*p_opt))**2)                 # chi-squared; estimates error in data
    chi2_red = chi2 / dof                                    # reduced chi-squared; measures goodness of fit
    s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard error of the fit at each point

    ss_resid = np.sum(resid**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1 - (ss_resid / ss_tot)

    if xpts is None: xpts = np.linspace(x.min,x.max,res)
    elif xpts.size == 2: xpts = np.linspace(xpts[0],xpts[1],res)
        
    ci = t * s_err * np.sqrt(1/n + (xpts - np.mean(x))**2 / np.sum((x - np.mean(x))**2))

    pi = t * s_err * np.sqrt(1 + 1/n + (xpts - np.mean(x))**2 / np.sum((x - np.mean(x))**2))

    return {'95% CI' : ci, '95% PI' : pi, 'Reduced Chi^2' : chi2_red, 'R^2' : r2}



def bootstrap_ci(func, x, y, p_opt, n_straps = 500, res = 100, xpts = None,
                 alpha = 0.05, fit_kwargs = {}, conservative = True, piecewise = True):
    
    if y.ndim == 1: piecewise = False
    
    if xpts is None: xpts = np.linspace(x.min(),x.max(),res)
    elif xpts.ndim == 2: xpts = np.linspace(xpts[0],xpts[1],res)

    y_fit = func(x,*p_opt)
    
    if y.shape != x.shape:
        y_fit = np.tile(y_fit,[y.size//x.size,1]).T
        
    resid = y - y_fit                                 # Residuals
    sigma_resid = resid.std()
    total_sigma = sigma_resid #np.sqrt(sigma_resid**2 + syst_err**2)        
    
    p_strapped = np.tile(np.zeros(p_opt.size),(n_straps,1))
    
    curve_strapped = np.tile(np.zeros(xpts.size),(n_straps,1))
    
    for i in tqdm(range(n_straps)):
        
        if piecewise and conservative:
             resid_resamples = np.array([np.random.choice(resid[row],size = resid[row].shape) for row in range(len(x))])
        elif piecewise and not conservative:
            resid_resamples = np.array([np.random.normal(0,size = resid[row].shape) for row in range(len(x))])
        elif not piecewise and not conservative:
            resid_resamples = np.random.normal(0, total_sigma, size = resid.shape)
        elif not piecewise and conservative:
            resid_resamples = np.random.choice(resid.flat, size = resid.shape)
              
        new_y = y+resid_resamples
        
        if y.ndim == 1:
            p_strapped[i], _ = opt.curve_fit(func, x, new_y, **fit_kwargs)
        else:
            p_strapped[i], _ = opt.curve_fit(func, x, new_y.mean(1), sigma = y.std(1), absolute_sigma = True, **fit_kwargs)
        
        curve_strapped[i] = func(xpts,*p_strapped[i])
    
    lower_bound = (alpha/2)*100
    upper_bound = (1-alpha/2)*100
    
    return {'Param mean'  : p_strapped.mean(0),
            'Param lower' : np.percentile(p_strapped,lower_bound,axis = 0),
            'Param median': np.percentile(p_strapped,50,axis = 0),
            'Param upper' : np.percentile(p_strapped,upper_bound,axis = 0),
            'Curve mean'  : curve_strapped.mean(0),
            'Curve lower' : np.percentile(curve_strapped,lower_bound,axis = 0),
            'Curve median': np.percentile(curve_strapped,50,axis = 0),
            'Curve upper' : np.percentile(curve_strapped,upper_bound,axis = 0)}
            

    
    
    
    




