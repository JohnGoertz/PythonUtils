import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime #For timestamping files
import pathlib as pl
import shelve #For saving/loading variables


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


def setupPath(processed = None):
    code_pth = pl.Path(__file__)
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