import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipywidgets as ipw

from tqdm.notebook import tqdm

import jitcode as jc
#https://aip-scitation-org.iclibezp1.cc.ic.ac.uk/doi/10.1063/1.5019320
import symengine as se

import shelve
import gc
import warnings


class Argument():
    def __init__(self, name):
        self.name = name
        self.sym = se.Symbol(name)
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.__str__()

class Parameter(Argument):    
    def __init__(self, name, value=1.0):
        self.value = value
        super(Parameter, self).__init__(name)
        
class Variable(Argument):
    def __init__(self, name, var_list, init=1.0):
        self.y = jc.y(len(var_list))
        self.init = init
        super(Variable, self).__init__(name)
        var_list.append(self)
        
def copies2molar(copies):
    #Calculates concentration in a 10 μL volume
    return copies / 6.022e23 / (10*10**-6)

def molar2copies(moles):
    #Calculates number of copies in a 10 μL volume
    return moles * 6.022e23 * (10*10**-6)

class CompetitiveReaction:
    
    defaults = {
        'norm' : molar2copies(120*10**-9),
        'cycles' : 60,
        'INT_res' : 1.,
        'INT_rng' : [1,9],
        'diffs' : None,
        'solution' : None,
        'frozen' : False,
        'sweep_INT_idx' : 0,
        'sweep_ax' : None,
        'oligo_inits' : 10**5,
        'rates' : 1,
        'primer_inits' : 120,
    }
    
    def __init__(self, INT_inputs, EXT_inputs, labeled_strands):
        self.INT_inputs = INT_inputs
        self.INTs = list(INT_inputs.keys())
        self.EXT_inputs = EXT_inputs
        self.EXTs = list(EXT_inputs.keys())
        self.labeled_strands = labeled_strands
        self.labels = list(set(labeled_strands.values()))
        assert len(self.list('labels'))<=2, 'No more than two labels may be used (for now)'
        self.input_oligos = {**INT_inputs,**EXT_inputs}
        assert all([len(pair)==2 for pair in self.input_oligos.values()]), 'All strands must have exactly two primers'
        self._primers_list = list(sorted(set(primer for pair in self.input_oligos.values() for primer in pair)))
        self.oligos = list(self.input_oligos.keys())
        self.buildConnections()
        self.buildComponents()
        self.buildEquations()
        self.setDefaults()
        self.buildLabels()
        self.compileODE(verbose=True)
            
    ################################################################################
    ## Convenience Functions
    ################################################################################
    
    def list(self,str_attr):
        return [str(item) for item in getattr(self,str_attr)]
    
    def from_list(self,attribute,item):
        return getattr(self,attribute)[self.list(attribute).index(item)]
        
    ################################################################################
    ## Setters and Getters
    ################################################################################
    
    def set_rate(self, oligo, rate):
        oligo = str(oligo)
        rate_name = 'r_'+oligo
        assert rate>0, 'Rate must be greater than 0'
        assert rate_name in self.list('rates'), f'Oligo {oligo} not found'
        self.from_list('rates',rate_name).value = rate
        
    def get_rate(self, oligo):
        oligo = str(oligo)
        rate_name = 'r_'+oligo
        assert rate>0, 'Rate must be greater than 0'
        assert rate_name in self.list('rates'), f'Oligo {oligo} not found'
        return self.from_list('rates',rate_name).value
        
    def set_primer_init(self, primer, nM):
        primer = str(primer)
        assert primer in self.list('primers'), f'Primer {primer} not found'
        # Set the init in copies
        self.from_list('primers',primer).init = molar2copies(nM*10**-9)
        
    def get_primer_init(self, primer, copies=False):
        primer = str(primer)
        assert primer in self.list('primers'), f'Primer {primer} not found'
        if copies:
            # Init should already be in copies
            return self.from_list('primers',primer).init
        else:
            return copies2molar(self.from_list('primers',primer).init)*10**9
        
    def set_oligo_init(self, oligo, copies):
        oligo = str(oligo)
        assert oligo in self.list('oligos'), f'Oligo {oligo} not found'
        self.from_list('strands',oligo+'_L').init = copies
        self.from_list('strands',oligo+'_R').init = copies
                
    def get_oligo_init(self, oligo):
        oligo = str(oligo)
        assert oligo in self.list('oligos'), f'Oligo {oligo} not found'
        inits = [strand.init for strand in self.strands if str(strand)[:-2]==oligo]
        # Ensure there are exactly two strands for the oligo
        assert len(inits)==2
        # Ensure both inits are the same
        assert all(init==inits[0] for init in inits)
        return inits[0]
    
    ################################################################################
    ## Configure the necessary elements for the simulations
    ################################################################################
        
    def buildConnections(self):
        """Builds a pandas dataframe showing all primer-strand pairs as well as labeled strands"""
        strands = [
            [{
                'Oligo' : oligo,
                'Strand' : strand,
                'Label' : self.labeled_strands[(oligo,strand)] if (oligo,strand) in self.labeled_strands else '',
                primer[0 if strand is 'L' else 1] : True,
            } for strand in ('L','R')]
        for oligo,primer in self.input_oligos.items()]

        strands = [strand for oligo in strands for strand in oligo]
        connections = pd.DataFrame(data=strands).set_index(['Oligo','Strand']).fillna(False)
        for idx,row in connections.iterrows():
            assert sum(row[self._primers_list])==1, f'Oligo {"_".join(idx)} has {sum(row[self._primers_list])} primers (should be 1)'
            
        self.connections = connections[['Label'] + self._primers_list]
        
        return connections
    
    def buildComponents(self):
        var_list=[]
        
        INT_strands_list = ['_'.join(strand) for strand in self.connections.index.to_list() if strand[0] in self.list('INTs')]
        EXT_strands_list = ['_'.join(strand) for strand in self.connections.index.to_list() if strand[0] in self.list('EXTs')]

        self.INT_strands = [Variable(strand, var_list) for strand in INT_strands_list]
        self.EXT_strands = [Variable(strand, var_list) for strand in EXT_strands_list]
        self.strands = self.INT_strands+self.EXT_strands

        self.primers = [Variable(primer, var_list) for primer in self._primers_list]
        rate_list = ['r_'+oligo for oligo in self.list('oligos')]
        self.rates = [Parameter(rate) for rate in rate_list]

        self.INT_inits = [Parameter(INT+'_0') for INT in INT_strands_list]
        self.EXT_inits = [Parameter(EXT+'_0') for EXT in EXT_strands_list]
        self.strand_inits = self.INT_inits+self.EXT_inits
        self.primer_inits = [Parameter(str(primer)+'_0') for primer in self.primers]

    def buildEquations(self):

        lg2_e = np.log2(np.exp(1))
        def get_strand(strand): return self.from_list('strands',strand)
        
        # Lookup table for which strands use a given primer
        strands_per_primer = {
            primer: ['_'.join(idx) for idx in self.connections.index[self.connections[str(primer)]]]
            for primer in self.primers
        }

        # Lookup table for which primers are used by a given primer
        primer_per_strand = {
            get_strand('_'.join(idx)): self.primers[i] for idx,row in self.connections.iterrows() 
            for i,v in enumerate(row[self.list('primers')]) if v
        }

        # Rate-limiting coefficients
        # The 'half-rate' concentration is the 
        mu = {
            primer : primer.y/(sum(self.strands[i].y for i,strand in enumerate(self.list('strands')) if strand in strands_per_primer[primer]) + primer.y) 
            for primer in self.primers
        }

        eqns = {
            oligo : {
                get_strand(oligo+'_R').y: self.rates[i].sym/lg2_e*get_strand(oligo+'_L').y*mu[primer_per_strand[get_strand(oligo+'_R')]],
                get_strand(oligo+'_L').y: self.rates[i].sym/lg2_e*get_strand(oligo+'_R').y*mu[primer_per_strand[get_strand(oligo+'_L')]]
            } for i,oligo in enumerate(self.list('oligos'))
        }

        # Flatten the nested dictionaries to a single level
        eqns = {deriv: expr for oligo in self.list('oligos') for deriv,expr in eqns[oligo].items()}

        eqns.update({
            primer.y: -sum(eqns[get_strand(strand).y] for strand in strands_per_primer[primer])
        for primer in self.primers})

        self.eqns = eqns

        return eqns
    
    def setDefaults(self):
        dflt = self.defaults
        for k,v in dflt.items():
            if k not in ['oligo_inits','rates','primer_inits']:
                setattr(self,k,v)
        self.sweep_INT = self.INTs[self.sweep_INT_idx]
        for oligo in self.oligos:
            self.set_oligo_init(oligo,dflt['oligo_inits'])
            self.set_rate(oligo,dflt['rates'])
        for primer in self.primers:
            self.set_primer_init(primer,dflt['primer_inits'])
    
    def buildLabels(self):
        label1_strands_list = ['_'.join(oligo) for oligo,row in self.connections.iterrows() if row.Label == self.list('labels')[0]]
        self.label1_strands = [self.from_list('strands',strand) for strand in label1_strands_list]
        label2_strands_list = ['_'.join(oligo) for oligo,row in self.connections.iterrows() if row.Label == self.list('labels')[1]]
        self.label2_strands = [self.from_list('strands',strand) for strand in label2_strands_list]
        
    ################################################################################
    ## Running a simulation
    ################################################################################
    
    def freezeEquations(self):
        '''Replace symbolic parameters in equations with explicit values'''
        self.eqns = {k:v.subs({rate.sym:rate.value for rate in self.rates}) for k,v in self.eqns.items()}
        self.frozen = True
        
    def unfreezeEquations(self):
        '''Reset equations to use symbolic parameters'''
        self.buildEquations()
        self.frozen = False
        
    def updateParameters(self):
        """Wrap jitcode's set_paramters to set symbolic parameter values post-compilation"""
        if self.frozen: return
        self.ODE.set_parameters(*[rate.value for rate in self.rates])
    
    def compileODE(self, integrator='dopri5', verbose=False, **kwargs):
        """
        Compile equations with selected integrator
        
        Options for integrator include at least:
            'dopri5','RK45','dop853','RK23','BDF','lsoda','LSODA','Radau','vode'
        """
        if self.frozen:
            self.ODE = jc.jitcode(self.eqns, verbose=verbose, **kwargs)
        else:
            self.ODE = jc.jitcode(self.eqns, control_pars=[rate.sym for rate in self.rates], verbose=verbose, **kwargs)
        
        self.ODE.set_integrator(integrator)
    
    def solveODE(self):
        """Initialize and run compiled solver"""
        inits = {v.y:v.init for v in self.strands+self.primers}
        self.ODE.set_initial_value(inits,0.0)
        cycles = np.arange(0.0,self.cycles,1)
        values = {str(v):np.zeros(len(cycles)) for v in self.strands+self.primers}
        for i,c in enumerate(cycles):
            self.ODE.integrate(c)
            for v in self.strands+self.primers:
                values[str(v)][i] = self.ODE.y_dict[v.y]
        self.solution = values
        return values
    
    def timeIntegrators(self,integrators=['dopri5','RK45','dop853','RK23','BDF','lsoda','LSODA','Radau','vode'],
                        compilation_time=False, with_update=False):
        '''
        Times the execution (and optionally compilation) of various ODE integrators
        Results for a three-competitor, four-primer system with no parameter updating:
        dopri5:
        compile: 951 ms ± 48.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        solve: 34.3 ms ± 5.48 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        RK45:
        compile: 945 ms ± 13.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        solve: 40 ms ± 1.56 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        dop853:
        compile: 981 ms ± 67.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        solve: 31.9 ms ± 950 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

        RK23:
        compile: 956 ms ± 14.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        solve: 41.9 ms ± 1.14 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        BDF:
        compile: 3.12 s ± 114 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        solve: 80.9 ms ± 2.4 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        lsoda:
        compile: 3.41 s ± 373 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        solve: 30.7 ms ± 1.14 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        LSODA:
        compile: 3.39 s ± 218 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        solve: 44.7 ms ± 7.32 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        Radau:
        compile: 3.51 s ± 219 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        solve: 75.2 ms ± 5.95 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

        vode:
        compile: 3.23 s ± 147 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        solve: 48.3 ms ± 21.7 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        for integrator in integrators:
            print(integrator+':')
            if compilation_time:
                print('compile: ',end='')
                %timeit self.compileODE(integrator=integrator,verbose=False)
            else:
                self.compileODE(integrator=integrator,verbose=False)
            if with_update:
                self.updateParameters()
            print('solve: ',end='')
            %timeit self.solveODE()
            print()
        '''
        pass
    
    ################################################################################
    ## Solution plotting functions
    ################################################################################
    
    # TODO: Add INT_grid
    
    def INT_sweep(self, INT=None, rng=None, res=None, progress_bar=False):
        if INT is None: INT=self.sweep_INT
        if rng is None: rng=self.INT_rng
        if res is None: res=self.INT_res 
        
        rng = np.arange(rng[0],rng[1]+res,res)
        N = len(rng)
        diffs = np.zeros(N)
        iterator = tqdm(enumerate(rng),total=N) if progress_bar else enumerate(rng)
            
        solutions = {}
        for i,INT_0 in iterator:
            self.set_oligo_init(INT,10**INT_0)
            self.updateParameters()
            solutions[INT_0] = self.solveODE()
            diffs[i] = self.get_diff()
        self.diffs=diffs
        return diffs, solutions
    
    def INT_grid(self, INT1=None, INT2=None, progress_bar=True):
        if INT1 is None: INT1=self.INTs[0]
        if INT2 is None: INT2=self.INTs[1]
        rng = self.INT_rng
        res = self.INT_res
        rng = np.arange(rng[0],rng[1]+res,res)
        N = len(rng)
        diffs = np.zeros([N,N])
        iterator = tqdm(enumerate(rng),total=N) if progress_bar else enumerate(rng)
        for i,INT_0 in iterator:
            self.set_oligo_init(INT1,10**INT_0)
            for j,INT_0 in enumerate(rng):
                self.set_oligo_init(INT2,10**INT_0)
                self.updateParameters()
                self.solveODE()
                diffs[i,j] = self.get_diff()
        return diffs
    
    def plot_INT_grid(self, ax=None, INT1=None, INT2=None, progress_bar=True):
        if INT1 is None: INT1=self.INTs[0]
        if INT2 is None: INT2=self.INTs[1]
        diffs = self.INT_grid(INT1=INT1, INT2=INT2, progress_bar=progress_bar)
        rng = self.INT_rng
        res = self.INT_res
        rng = np.arange(rng[0],rng[1]+res,res)
        ext = np.ceil(np.max([np.max(diffs),np.abs(np.min(diffs))])*2)/2
        fig,ax = plt.subplots(1,1) if ax is None else (ax.figure,ax)
        pcm = ax.pcolormesh(rng,rng,diffs.T, cmap = 'RdBu_r',
                                  vmin=-ext,vmax=ext,
                                  shading = 'gouraud'
                            )
        ax.axvline(5,color='w',linestyle=':')
        ax.axhline(5,color='w',linestyle=':')

        cbar = plt.colorbar(pcm,ticks=np.arange(-ext,ext+0.5,0.5))#, ax = axs[1],extend=extend,ticks=np.arange(-10,10.1,2.5))
        #cbar_x0s.ax.set_ylabel('$log_{10}$ Tar/Ref Ratio\nProviding Signal Parity',fontsize=16)
        #cbar_x0s.ax.tick_params(labelsize=16) 
        cntr = ax.contour(rng,rng,diffs.T,colors = 'k', 
                           #levels = np.arange(np.around(np.min(diffs)*2)/2,np.around(np.max(diffs)*2)/2+0.5,0.5)
                          )
        plt.clabel(cntr, inline=True, fontsize=16, fmt = '%.1f');
        ax.set_aspect('equal', 'box')
        return diffs
    
    def plot_INT_sweep(self, INT=None, rng=None, res=None, progress_bar=False, annotate='Outer', ax=None, indiv=True, update=False):
        # TODO: Allow target axis to be specified for individual plots
        # TODO: Plot total signal for each label
        
        if INT is None: INT=self.sweep_INT
        if rng is None: rng=self.INT_rng
        if res is None: res=self.INT_res 
        
        diffs, sweep_solutions = self.INT_sweep(INT=INT, rng=rng, res=res, progress_bar=progress_bar)
        
        if ax is not None: indiv=False
        
        if indiv:
            fig,gs,ind_axs = self.plotTracesGrid(sweep_solutions,annotate=annotate)
            ax = fig.add_subplot(gs[:,3:])
            self.sweep_ax = ax
        
        if update:
            if self.sweep_ax is None:
                _,self.sweep_ax = self.plotDiffs(diffs)
            else:
                self.updateDiffs(self.sweep_ax)
        else:
            if ax is None:
                fig,ax = plt.subplots(1,1)
            _,self.sweep_ax = self.plotDiffs(diffs,ax=ax)
            
        if annotate in [True,'Inner','Outer']:
            self.annotate_diff_plot(self.sweep_ax,pos=annotate)
            
        return diffs
    
    def plotDiffs(self,diffs=None,ax=None,rng=None,res=None):
        if diffs is None: diffs = self.diffs
        if rng is None: rng = self.INT_rng
        if res is None: res = self.INT_res
        fig,ax = plt.subplots(1,1) if ax is None else (ax.figure,ax)
        INT = self.sweep_INT
        rng = np.arange(rng[0],rng[1]+res,res)
            
        ax.plot(rng,diffs,'o-')
        plt.setp(ax,**{
            'ylim' : [-1.05,1.05],
            'title' : '{:s}-{:s} after {:d} cycles'.format(*self.list('labels')[::-1],self.cycles),
            'ylabel' : 'Signal Difference',
            'xlabel' : f'log10 {INT} copies',
        })
        return fig, ax
    
    def updateDiffs(self,ax, diffs=None, rng=None, res=None):
        if diffs is None: diffs = self.diffs
        if rng is None: rng = self.INT_rng
        if res is None: res = self.INT_res
        rng = np.arange(rng[0],rng[1]+res,res)
        ax.lines[0].set_xdata(rng)
        ax.lines[0].set_ydata(diffs)
        for l in ax.lines[1:]: l.remove()
        txts = [child for child in self.sweep_ax.get_children() if type(child)==mpl.text.Annotation]
        for txt in txts: txt.remove()
        ax.figure.canvas.draw()
        print(self.get_diff_stats())
        
    def get_diff(self):
        return (sum(self.solution[L2][-1] for L2 in self.list('label2_strands'))-
                sum(self.solution[L1][-1] for L1 in self.list('label1_strands')))/self.norm
    
    def get_diff_stats(self, diffs=None,rng=None):
        if diffs is None: diffs=self.diffs
        if rng is None: rng = self.INT_rng
        res = self.INT_res
        rng = np.arange(rng[0],rng[1]+res,res)
        
        interp_res=0.01
        wt_interp = np.arange(rng[0],rng[-1]+interp_res,interp_res)
        diff_interp = np.interp(wt_interp,rng,diffs)
        #diff_half = (np.max(diff_interp)-np.min(diff_interp))/2+np.min(diff_interp)
        diff0 = np.argmin(abs(diff_interp))
        diff90 = (np.max(diff_interp)-np.min(diff_interp))*0.9+np.min(diff_interp)
        diff10 = (np.max(diff_interp)-np.min(diff_interp))*0.1+np.min(diff_interp)
        
        stats = {
            'Zero' : wt_interp[diff0],
            'DR' : np.abs(wt_interp[np.argmin(abs(diff_interp-diff10))] - wt_interp[np.argmin(abs(diff_interp-diff90))]),
            'Max' : np.max(diffs),
            'Min' : np.min(diffs),
        }
        return stats
        
    def annotate_diff_plot(self,ax,diffs=None,rng=None,pos='Outer'):
        if diffs is None: diffs=self.diffs
        if rng is None: rng=self.INT_rng
        stats = self.get_diff_stats(diffs=diffs,rng=rng)
        ax.axvline(stats["Zero"], ls='--', color='k')

        if pos in [True,'Outer']:
            x_pos = 1.05
        elif pos is 'Inner':
            x_pos = 0.025
        
        ax.annotate(f'Zero: {stats["Zero"]:.2f}',
                     xy=(x_pos, .925), xycoords='axes fraction',
                     horizontalalignment='left')

        ax.annotate(f'DR: {stats["DR"]:.2f}',
                     xy=(x_pos, .825), xycoords='axes fraction',
                     horizontalalignment='left')

        ax.annotate(f'Max: {stats["Max"]:.2f}',
                     xy=(x_pos, .725), xycoords='axes fraction',
                     horizontalalignment='left')

        ax.annotate(f'Min: {stats["Min"]:.2f}',
                     xy=(x_pos, .625), xycoords='axes fraction',
                     horizontalalignment='left')
        
    def plotTraces(self,ax=None,solution=None):
        fig,ax = plt.subplots(1,1) if ax is None else (ax.figure,ax)
        if solution is None: solution = self.solution
        for L1 in self.list('label1_strands'):
            ax.plot(np.arange(self.cycles), solution[L1]/self.norm, ls='-')
        for L2 in self.list('label2_strands'):
            ax.plot(np.arange(self.cycles), solution[L2]/self.norm, ls='--')
        return fig, ax
            
    def plotTracesGrid(self,solution_dict,annotate=True):
        fig = plt.figure(constrained_layout=True,figsize=[16,5])
        N = len(solution_dict)
        gs = fig.add_gridspec(N//3+1,6)
        ind_axs = []
        INT = self.sweep_INT
        for i,(INT_0, solution) in enumerate(solution_dict.items()):
            with plt.rc_context({'axes.labelweight':'normal','font.size':14}):
                ind_axs.append(fig.add_subplot(gs[i//3,i%3], sharey=ind_axs[0] if i>0 else None))
                self.plotTraces(ax=ind_axs[i],solution=solution)
                plt.setp(ind_axs[i].get_yticklabels(), visible=True if i%3==0 else False)
                plt.setp(ind_axs[i].get_xticklabels(), visible=True if i//3+1==(N-1)//3+1 else False)
                if annotate in [True,'Inner','Outer']:
                    plt.annotate(f'{INT_0:.1f} logs {INT}', xy=(.025, .825), xycoords='axes fraction',fontsize=12)
                if (i%3==0)&(i//3+1==(N-1)//3+1):
                    plt.setp(ind_axs[i],**{
                        'ylabel' : 'Norm Signal',
                        'xlabel' : 'Cycles',
                    })
        return fig, gs, ind_axs
    
    ################################################################################
    ## Interactive configurations with simulating and plotting
    ################################################################################

    def interactive_solve(self,**kwargs):
        """Set the necessary attributes from the interactive configuration, then solve with a range of initial INT values"""
        for idx, row in self.connections.iterrows():
            assert set(self.select_label1.value).intersection(set(self.select_label2.value)) == set(), 'No strands may be labeled twice'
            if '_'.join(idx) in self.select_label1.value:
                label = self.list('labels')[0]
            elif '_'.join(idx) in self.select_label2.value:
                label = self.list('labels')[1]
            else: 
                label = ''
            self.connections.at[idx,'Label'] = label
        self.buildLabels()
        for oligo in self.oligos:
            self.set_rate(oligo,kwargs['r_'+str(oligo)])
        for EXT in self.EXTs:
            self.set_oligo_init(EXT,10**kwargs[str(EXT)])
        for p in self.primers:
            self.set_primer_init(p,kwargs[str(p)])
        self.norm = molar2copies(kwargs['norm']*10**-9)
        self.cycles = kwargs['cycles']
        self.INT_rng = self.INT_rng_widget.value
        self.INT_res = np.diff(self.INT_rng)[0]/8# kwargs['INT_res']
        if len(self.list('INTs'))>1:
            self.sweep_INT = self.INT_selector.value
            held_INTs = [INT for INT in self.list('INTs') if INT is not self.INT_selector.value]
            for INT,widg in zip(held_INTs,self.INT_conc_widgets):
                self.set_oligo_init(INT,10**widg.value)
        if kwargs['plt_rslt']:
            self.plot_INT_sweep(indiv=kwargs['indiv'], update=False)
        return

    def interactive(self):
        ui_concentrations = ipw.interactive(self.interactive_solve, #{'manual': True}, 
                 **{
                     str(rate):ipw.FloatSlider(min=0.1, max=1.1, step=0.05, value=rate.value,
                                               description=f'{str(rate)[2:]} rate', continuous_update=False)
                     for rate in self.rates
                 },**{
                     str(p):ipw.FloatSlider(min=1, max=500, step=25, value=self.get_primer_init(p),
                                                description=f'nM {str(p)}', continuous_update=False)
                     for p in self.primers
                 },**{
                     str(EXT):ipw.FloatSlider(min=0, max=10, step=0.25, value=np.log10(self.get_oligo_init(EXT)),
                                                description=f'logs {str(EXT)}', continuous_update=False)
                     for EXT in self.EXTs
                 },**{
                 },
                 norm=ipw.IntSlider(min=1, max=500, step=25, value=copies2molar(self.norm)*10**9, description='Norm (nM)', continuous_update=False),
                 #INT_res=ipw.FloatSlider(min=0.1, max=2, step=0.05, value=self.INT_res, description='resolution', continuous_update=False),
                 cycles=ipw.IntSlider(min=10, max=100, value=self.cycles, description='cycles', continuous_update=False),
                 plt_rslt=ipw.Checkbox(value=False, description='Plot Result'),
                 indiv=ipw.Checkbox(value=False, description='Individual Traces'),
            )
        
        self.INT_rng_widget=ipw.FloatRangeSlider(min=0, max=10, step=0.1, value=self.INT_rng,
                                          description=f"{self.list('INTs')[0]} range", continuous_update=False)
        
        self.select_label1 = ipw.SelectMultiple(options = self.list('strands')+['None',], value = self.list('label1_strands'),
                                                description = f'Label {self.list("labels")[0]:}')
        self.select_label2 = ipw.SelectMultiple(options = self.list('strands')+['None',], value = self.list('label2_strands'),
                                                description = f'Label {self.list("labels")[1]:}')        
                                            
        n_oligos = len(self.oligos)
        n_primers = len(self.primers)
        n_EXTs = len(self.EXTs)
        
        col1 = n_oligos
        col2 = col1+n_primers
        col3 = col2+n_EXTs
        col4 = col3+2
        
        rate_widgets = ui_concentrations.children[:col1]
        primer_widgets = ui_concentrations.children[col1:col2]
        EXT_widgets = list(ui_concentrations.children[col2:col3])
        INT_widgets = [self.INT_rng_widget]
        
        if len(self.list('INTs'))>1:
            self.INT_selector = ipw.RadioButtons(options = self.list('INTs'), description = 'Sweep:')
            self.INT_conc_widgets = [ipw.FloatSlider(min=0, max=10, step=0.25, value=np.log10(self.get_oligo_init(INT)),
                                                     description=f'logs {str(INT)}', continuous_update=False)
                                     for INT in self.INTs if str(INT) is not self.INT_selector.value]
            INT_widgets.extend(self.INT_conc_widgets)
            def update_INT_widgets(*args):
                self.INT_rng_widget.description = f'{self.INT_selector.value} range'
                held_INTs = [INT for INT in self.list('INTs') if INT is not self.INT_selector.value]
                for INT,widg in zip(held_INTs,self.INT_conc_widgets):
                    widg.description = f'logs {INT}'
                    widg.value=np.log10(self.get_oligo_init(INT))
            self.INT_selector.observe(update_INT_widgets,'value')
            INT_widgets.append(self.INT_selector)
        
        oligo_widgets = EXT_widgets + INT_widgets
        
        display(ipw.HBox([
            ipw.VBox(rate_widgets),
            ipw.VBox(primer_widgets),
            ipw.VBox(EXT_widgets),
            ipw.VBox(INT_widgets)
        ]))
            
        display(ipw.HBox([
            self.select_label1,
            self.select_label2, 
            ipw.VBox(ui_concentrations.children[col3:col4]),
            ipw.VBox(ui_concentrations.children[col4:-1]),
        ]))
        
        display(ui_concentrations.children[-1])#Show the output