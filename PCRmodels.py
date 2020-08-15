from autograd import numpy as np
from autograd import grad
from autograd.builtins import tuple
from autograd.scipy.integrate import odeint
import scipy.integrate as integrate
import ipdb
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import matplotlib as mpl
# import ipywidgets as ipw
# from tqdm.notebook import tqdm

# import shelve
# import gc
# import warnings


def copies2molar(copies):
    '''Calculates concentration in a 10 μL volume'''
    return copies / 6.022e23 / (10*10**-6)

def molar2copies(moles):
    '''Calculates number of copies in a 10 μL volume'''
    return moles * 6.022e23 * (10*10**-6)

def calc_tau(rate,init,norm):
    return -np.log(init/norm)/(rate*np.log(2))
def calc_rho(rate,init,norm):
    return -np.log(rate*np.log(2))/np.log(calc_tau(rate,init,norm))
def calc_rate(τ,ρ):
    return τ**(-ρ)/np.log(2)
def calc_init(τ,ρ,norm):
    return norm*np.exp(-τ**(1-ρ))

def FAM_HEX_cmap(N = 64, sat = 'mid'):
    rosest = {
        'light' : [199/256, 57/256, 101/256],
        'mid'  : [179/256, 0/256, 47/256],
        'dark' : [162/256, 0/256, 70/256],
    }
    roses = np.ones((N,4))
    for i in range(3):
        roses[:,i] = np.linspace(rosest[sat][i], 1, N)

    tealest = {
        'light' : [35/256, 135/256, 127/256],
        'mid'  : [10/256, 111/256, 103/256],
        'dark' : [0/256, 102/256, 94/256],
    }
    teals = np.ones((N,4))
    for i in range(3):
        teals[:,i] = np.linspace(1, tealest[sat][i], N)

    concat_colors = np.vstack((roses,teals))
    cmap = mpl.colors.ListedColormap(concat_colors, name='RsTl')

    teals = mpl.cm.get_cmap('BrBG',N*2)(np.linspace(0.5,0.9,N))
    oranges = mpl.cm.get_cmap('PuOr',N*2)(np.linspace(0.1,0.5,N))
    concat_colors = np.vstack((oranges,teals))[::-1]
    cmap = mpl.colors.ListedColormap(concat_colors, name='OrBG')

    return cmap

def grey(): return [132/256,151/256,176/256]


class Primer:
    """
    Primers have a name and a concentration
    
    The class allows concentration to be specified in any units among 'copies',
    'nM', and 'M'. All non-specified units are converted and updated appropriately
    """
    def __init__(self, name, copies=None, nM=None, M=None):
        assert [copies,nM,M].count(None) == 2, 'Exactly one of copies, nM, or M must be supplied'
        self.name = name
        if copies is not None:
            self.copies = copies
        if nM is not None:
            self.nM = nM
        if M is not None:
            self.M = M

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()
    
    # When any of copies, nM, or M are set, convert to the appropriate units and set the others as well
    
    @property
    def copies(self):
        return self._copies
    
    @copies.setter
    def copies(self, copies):
        assert copies>0
        self._copies = copies
        self._nM = copies2molar(copies)*10**-9
        self._M = copies2molar(copies)
    
    @property
    def nM(self):
        return self._nM
    
    @nM.setter
    def nM(self, nM):
        assert nM>0
        self._nM = nM
        self._M = nM * 10**-9
        self._copies = molar2copies(nM * 10**-9)
    
    @property
    def M(self):
        return self._M
    
    @M.setter
    def M(self, M):
        assert M>0
        self._M = M
        self._nM = M * 10**9
        self._copies = molar2copies(M)
        
        
class Strand(Primer):
    """
    Strands have an associated rate, but otherwise behave just like Primers
    """
    def __init__(self, name, rate, copies=None, nM=None, M=None):
        self.rate = rate  
        super(Strand, self).__init__(name, copies, nM, M)      

class Oligo(Strand):
    """
    Oligos consist of two Strands, '+' and '-'.
    
    Updating a property of an Oligo updates the corresponding property in both
    of its Strands. The reverse is not true. Directly setting Strand properties 
    might break things, procede with caution.
    """
    def __init__(self, name, rate, copies=None, nM=None, M=None):
        self.strands = []
        for strand in ['-','+']:
            self.strands.append(Strand(name+strand, rate, copies, nM, M))
        super(Oligo, self).__init__(name, rate, copies, nM, M)
    
    # When setting a property of the oligo, set that same property for its strands as well
    
    @property
    def rate(self):
        return self._rate
    
    @rate.setter
    def rate(self, rate):
        self._rate = rate
        for strand in self.strands:
            strand.rate = rate
    
    @property
    def copies(self):
        return Strand.copies.fget(self)
    
    @copies.setter
    def copies(self, copies):
        Strand.copies.fset(self, copies)
        for strand in self.strands:
            strand.copies = copies
    
    @property
    def nM(self):
        return Strand.nM.fget(self)
    
    @nM.setter
    def nM(self, nM):
        Strand.nM.fset(self, nM)
        for strand in self.strands:
            strand.nM = nM
    
    @property
    def M(self):
        return Strand.M.fget(self)
    
    @M.setter
    def M(self, M):
        Strand.M.fset(self, M)
        for strand in self.strands:
            strand.M = M
            

class PCR:
    """
    Monod simulation of a PCR reaction system.
    
    A reaction consists of a set of Primers and Oligos whose interactions are 
    specified via a connectivity matrix. Lists containing these objects are 
    stored in a PCR instance `rxn` via the `rxn.primers` and `rxn.oligos` 
    properties, respectively; `rxn.strands` behaves similarly. 
    """

    defaults = {
        'oligo_copies' : 10**5,
        'oligo_rate' : 0.9,
        'primer_nM' : 100,
        'norm_nM' : 100,
        'cycles' : 60,
        'sol' : None,
        'solution' : None,
        'integrator' : 'LSODA',
    }

    def __init__(self, connections, labels=None, oligo_names=None, label_names=None, primer_names=None):
        """
        Constructs a Monod simulation of a PCR reaction system.
        
        Example:
            ```python
            OR_gate = np.array([[-1, 0, +1, 0, 0], [0, -1, +1, 0, 0], [0, 0, -1, +1, 0], [0, 0, 0, -1, +1]])
            OR_labels = [[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]]
            rxn = pcr.PCR(OR_gate,OR_labels)
            ```

        Args:
            connections (array): A connectivity matrix of shape (n,p) 
                n is the number of oligos in the system and p is the number of primers.
                A value of +/-1 indicates that the oligo uses the given primer, with -1 indicating that the 
                primer targets the "left" or "negative" strand of an oligo and +1 indicating the "right" or 
                "positive" strand. If possible, -1 entries should appear to before +1 entries. Note that 
                "targeting" means "complementary" to, so a primer that targets the positive strand gets
                extended to generate the negative strand.

            labels (list): A list of arrays that specify which strands are labeled
                A list of vectors, each of length (2*n), indicating which strands are targeted with the given 
                probe color. Strands alternate -/+ in the same order as in `connections`

        """

        assert all(np.sum(connections==-1, axis=1) == 1), 'All rows of connections must have exactly one +1 and one -1 value'
        assert all(np.sum(connections==+1, axis=1) == 1), 'All rows of connections must have exactly one +1 and one -1 value'
        assert all(np.sum(connections!=0, axis=1) == 2), 'All rows of connections must have exactly two non-zero values'
        
        self.connections = connections
        self.n_oligos, self.n_primers = connections.shape
        self.n_strands = self.n_oligos*2
        
        # If no labels supplied, "label" the "-" strand of every oligo
        if labels is None:
            self.labels = [[]]
            self.labels[0].extend([1,0] for _ in range(self.n_oligos))
        else:
            assert all(len(label)==self.n_strands for label in labels)
            self.labels = labels
        self.n_labels = len(self.labels)
        
        # If no oligo names are supplied, name each oligo with successive latin letters
        if oligo_names is None:
            oligo_names = [chr(ord('a')+i) for i in range(self.n_oligos)]
        else:
            assert len(oligo_names)==self.n_oligos
        self.oligo_names = oligo_names
        
        # If no label names are supplied, name each label L0, L1, etc
        if label_names is None:
            label_names = [f'L{i}' for i in range(self.n_labels)]
        else:
            assert len(label_names)==self.n_labels
        self.label_names = label_names
        
        # If no primer names are supplied, name each primer p0, p1, etc
        if primer_names is None:
            primer_names = [f'p{i}' for i in range(self.n_primers)]
        else:
            assert len(primer_names)==self.n_primers
        self.primer_names = primer_names
        
        # A list storing each oligo as an Oligo object
        self.oligos = [Oligo(name, self.defaults['oligo_rate'], self.defaults['oligo_copies']) for name in oligo_names]
        
        # A list storing each strand as an Strand object
        self.strands = []
        for oligo in self.oligos:
            self.strands.extend(oligo.strands)
        # A list storing each primer as a Primer object
        self.primers = [Primer(name, nM=self.defaults['primer_nM']) for name in primer_names]
        self.setDefaults()
        self.buildEquations()

    ################################################################################
    ## Getters and Setters
    ################################################################################
    
    def oligo(self, name):
        '''Get an oligo by name or number'''
        if type(name) is int:
            return self.oligos[name]
        if type(name) is str:
            return self.oligos[self.oligo_names.index(name)]
    
    def strand(self, name):
        '''Get a strand by name'''
        idx = [strand.name for strand in self.strands].index(name)
        return self.strands[idx]
    
    def primer(self, name):
        '''Get a primer by name or number'''
        if type(name) is int:
            name = f'p{name}'
        return self.primers[self.primer_names.index(name)]
    
    @property
    def oligo_copies(self):
        '''Returns the concentration, in copies, of each oligo in the reaction'''
        return [oligo.copies for oligo in self.oligos]
    
    @oligo_copies.setter
    def oligo_copies(self,copies_list):
        for oligo,copies in zip(self.oligos,copies_list):
            oligo.copies = copies

    @property
    def strand_copies(self):
        '''Returns the concentration, in copies, of each strand in the reaction'''
        return [strand.copies for strand in self.strands]
    
    @strand_copies.setter
    def strand_copies(self,copies_list):
        for strand,copies in zip(self.strands,copies_list):
            strand.copies = copies
    
    @property
    def primer_copies(self):
        '''Returns the concentration, in copies, of each primer in the reaction'''
        return [primer.copies for primer in self.primers]
    
    @primer_copies.setter
    def primer_copies(self,copies_list):
        for primer,copies in zip(self.primers,copies_list):
            primer.copies = copies
            
    @property
    def primer_nMs(self):
        '''Returns the concentration, in nanomolar, of each primer in the reaction'''
        return np.array([primer.nM for primer in self.primers])
    
    @primer_nMs.setter
    def primer_nMs(self,nMs_list):
        for primer,nM in zip(self.primers,nMs_list):
            primer.nM = nM
    
    @property
    def copies(self):
        '''Returns an array of the concentration in copies for each strand, then each primer, in the reaction'''
        return np.array(self.strand_copies + self.primer_copies)
    
    @copies.setter
    def copies(self,copies_list):
        '''
        Sets the concentration, in copies, of all reaction components
        
        The supplied list can either specify the copies of each oligo, then each 
        primer, or the copies of every strand, then each primer
        '''
        if len(copies_list) == self.n_strands+self.n_primers:
            self.strand_copies = copies_list[:self.n_strands]
            self.primer_copies = copies_list[self.n_strands:]
        elif len(copies_list) == self.n_oligos+self.n_primers:
            self.oligo_copies = copies_list[:self.n_oligos]
            self.primer_copies = copies_list[self.n_oligos:]
    
    @property
    def rates(self):
        '''Returns an array of the rates for each oligo in the reaction'''
        return np.array([oligo.rate for oligo in self.oligos])
    
    @rates.setter
    def rates(self,rates_list):
        rates_list=np.array(rates_list)
        for oligo,rate in zip(self.oligos, rates_list):
            oligo.rate = rate
            
    @property
    def copies_rates_nMs(self):
        '''
        Returns a list of key reaction parameters in natural units:
        
        Initial copies of each oligo in log10 scale, initial concentration of
        primers in nanomolar, and the rates of each oligo
        '''
        log_oligo_copies = [np.log10(oligo.copies) for oligo in self.oligos]
        primer_nMs = self.primer_nMs
        oligo_rates = self.rates
        return np.hstack([log_oligo_copies,oligo_rates,primer_nMs])
    
    @copies_rates_nMs.setter
    def copies_rates_nMs(self,params):
        params=np.array(params)
        n_o = self.n_oligos
        n_p = self.n_primers
        assert len(params) == n_o*2 + n_p
        #parameter_list = np.array(parameter_list)
        self.oligo_copies = 10**params[:n_o]
        self.rates = params[n_o:-n_p]
        self.primer_nMs = params[-n_p:]
            
    @property
    def taus_rhos_nMs(self):
        '''
        Returns the τ and ρ values for each oligo and primer concentrations in nanomolar
        
        This alternative parameterization may provide a parameter space more amenable
        to optimization. When optimizing in the "standard" parameter space, the rate and
        initial copy number of a given oligo are tightly correlated: a long, narrow band 
        of value pairs give rise to nearly-optimal fits. The τ-ρ space is significantly
        less correlated, improving the likelihood of discovering the global optimum.
        '''
        return np.hstack([self.oligo_taus,self.oligo_rhos,self.primer_nMs])
    
    @taus_rhos_nMs.setter
    def taus_rhos_nMs(self,params):
        n_o = self.n_oligos
        n_p = self.n_primers
        norm = self.norm.copies
        taus = params[:n_o]
        rhos = params[n_o:-n_p]
        nMs = params[-n_p:]
        assert len(params) == n_o*2 + n_p
        self.oligo_copies = [calc_init(τ,ρ,norm) for τ,ρ in zip(taus,rhos)]
        self.rates = [calc_rate(τ,ρ) for τ,ρ in zip(taus,rhos)]
        self.primer_nMs = nMs
                
    @property
    def oligo_taus(self):
        '''
        Returns the τ parameter for each oligo.
        
        Cannot be set directly. Set via `taus_rhos_nMs` property
        '''
        return [calc_tau(oligo.rate,oligo.copies,self.norm.copies)
                for oligo in self.oligos]
                
    @property
    def oligo_rhos(self):
        '''
        Returns the ρ parameter for each oligo.
        
        Cannot be set directly. Set via `taus_rhos_nMs` property
        '''
        return [calc_rho(oligo.rate,oligo.copies,self.norm.copies)
                for oligo in self.oligos]

    ################################################################################
    ## Configure the necessary elements for the simulations
    ################################################################################

    def setDefaults(self):
        for primer in self.primers:
            primer.nM = self.defaults['primer_nM']
        for oligo in self.oligos:
            oligo.copies = self.defaults['oligo_copies']
            oligo.rate = self.defaults['oligo_rate']
            
        self.norm = Primer('norm',nM=self.defaults['norm_nM'])
        
        for k,v in self.defaults.items():
            if k not in ['oligo_copies','oligo_rate','norm_nM','primer_nM']:
                setattr(self,k,v)
        # self.sweep_INT = self.INTs[self.sweep_INT_idx]
        # for oligo in self.oligos:
        #     self.set_oligo_init(oligo,dflt['oligo_inits'])
        #     self.set_rate(oligo,dflt['rates'])
        # for primer in self.primers:
        #     self.set_primer_init(primer,dflt['primer_inits'])

    def buildEquations(self):
        n_p = self.n_primers
        n_o = self.n_oligos
        n_s = self.n_strands
        cm = self.connections
        def equations(copies,c,rates):
            # rate-limiting contribution from each primer
            mus = []
            for p in range(n_p):
                # all oligos that utilize the primer
                targets = np.argwhere(cm[:, p] != 0).flatten()
                # strands that bind to the primer
                strands = [copies[2*oligo+(cm[oligo,p]+1)//2] for oligo in targets]
                # total concentration of all strands that bind to the primer
                strand_concs = np.sum(strands)
                # position of the primer in the copies vector
                idx = p+n_s
                mus.append(copies[idx]/(copies[idx] + strand_concs))
            mus = np.array(mus)
            # concentration of each oligo strand at the next time point
            oligo_eqns = []
            for i in range(n_o):
                for j,strand in enumerate([-1,+1]):
                    # rate for the oligo
                    rate = rates[i]*np.log(2)
                    # concentration of the complementary strand
                    complement = copies[i*2+np.abs(j-1)]
                    # rate-limiting contribution from the generating primer
                    # the generating primer binds to the complementary stran+d
                    mu = mus[cm[i,:]==-strand][0]
                    oligo_eqns.append(complement*mu*rate)
            # concentration of each primer at the next time point
            primer_eqns = []
            for p in range(n_p):
                # all oligos that utilize the primer
                targets = np.argwhere(cm[:, p] != 0).flatten()
                # derivatives of strands that bind to the primer
                strands = [oligo_eqns[2*oligo+np.abs(cm[oligo,p]-1)//2] for oligo in targets]
                primer_eqns.append(-np.sum(strands))
            return np.array(oligo_eqns + primer_eqns)

        self.eqns = equations
        return self.eqns


    ################################################################################
    ## Running a simulation
    ################################################################################


    def solve_ivp(self, **kwargs):
        '''
        Method profiling:
            
        '''
        kwargs.setdefault('dense_output',True)
        def eq(c,copies,rates):
            return self.eqns(copies,c,rates)
        sol = integrate.solve_ivp(eq, t_span=(0,self.cycles), y0=self.copies, args=tuple((np.array(self.rates),)),
                                  method=self.integrator, **kwargs)
        self.sol = sol
        return sol
    
    def solution_at(self, c):
        if self.sol is None: self.solve_ivp()
        solution = self.sol.sol(c)
        # Ensure solution is an array of shape (n_s+n_p, len(c)) 
        if solution.ndim==1:
            solution = solution[:,None]
        self.solution = solution
        return self.solution
    
    def odeint(self):
        solution = odeint(self.eqns,self.copies,np.arange(self.cycles),tuple((self.rates,)))
        self.sol = solution
        self.solution = solution[-1,:][:,None]
        return self.solution
        
    
    def solution_sweep(self, oligos=[0], rng=[1,10], res=1, method='solve_ivp', mode='loop'):
        '''
        Calculates solutions for a range of oligo concentrations
        
        Args:
            oligos (list): Indices of oligos to be swept.
                Can be oligo names or numeric indices corresponding to the list
                self.oligos.
                
            rng (list): Concentration range (in log scale) for each oligo.
                Can either be a two-element list or a list of two-element lists.
                If a 1D list, the range is applied to all oligos indicated by
                `idx`. Otherwise, each element corresponds to the range of each
                oligo in `idx`.
                
            res (float or list): Concentration resolution (in decades) of sweep.
                Can either be a single value or a 1D list. If a single value, 
                resolution is applied to all oligos. Otherwise, each element 
                corresponds to the resolution of each oligo in `idx`.
                
        Returns:
            sweep_solution (array): An array containing the `diffs` at each test point
                Note that if the oligos swept are indices 0,1,2,3,3,... then the 
                axes of the output array will correspond to oligos 1,0,2,3,4...
                
        '''
        assert self.n_labels == 2, 'solution_sweep is only supported for exactly two labels'
        assert type(oligos) is list
        n_oligos = len(oligos)
        
        if type(res) is not list:
            # Isotropic grid, same range and resolution for every oligo
            assert all(type(el) is not list for el in rng)
            rng = [rng for _ in range(n_oligos)]
            res = [res for _ in range(n_oligos)]
        else:
            # Anisotropic grid, different range and resolution for each oligo
            assert type(rng) is list
            assert all(type(rng_) is list for rng_ in rng)
            assert all(len(rng_) == 2 for rng_ in rng)
            assert type(res) is list
            assert len(rng) == n_oligos
            assert len(res) == n_oligos
        
        # Build 1D arrays and multidimensional meshgrids of all evaluation points
        arrays = [np.arange(rng_[0], rng_[1]+res_,res_) for rng_,res_ in zip(rng,res)]
        grids = np.meshgrid(*arrays)
        pts = np.vstack([grid.ravel() for grid in grids]).T
        
        if mode == 'vectorized':
            n_copies = len(self.copies)
            n_pts = len(pts)
            
            tiled_copies = np.tile(self.copies,n_pts)
            mask = np.array([1 if i//2 in oligos else 0 for i in range(n_copies)])
            tiled_mask = np.tile(mask, n_pts)
            value_mask = np.hstack([10**i*mask for i in pts.flatten()])
            tiled_copies = tiled_copies - tiled_mask*tiled_copies + tiled_mask*value_mask
                
            def vectorized_eqns(copies_list, c, rates):
                return np.hstack([self.eqns(copies_list[i*n_copies:(i+1)*n_copies], c, rates)
                                  for i in range(n_pts)])
        
            sweep_solution = odeint(vectorized_eqns, tiled_copies, np.arange(self.cycles), tuple((self.rates,)))[-1,:]
            sweep_strands = np.reshape(sweep_solution,[n_pts,-1])[:,:self.n_strands]/self.norm.copies
            sweep_signals = [np.sum(sweep_strands[:,np.array(label, dtype=bool)], axis=1) for label in self.labels]
            sweep_diffs = np.subtract(*sweep_signals)
        else:
            sweep_diffs = []
            for i,pt in enumerate(pts):
                # Clear any previous solution
                self.sol, self.solution = [None,None]
                # Set the oligo concentrations accordingly
                for oligo,lg_copies in zip(oligos,pt):
                    if type(oligo) is str:
                        oligo = [str(oligo_) for oligo_ in self.oligos].index(oligo)
                    self.oligos[oligo].copies = 10**lg_copies
                if method == 'odeint':
                    self.odeint()
                elif method == 'solve_ivp':
                    self.solution_at(self.cycles)
                sweep_diffs.append(self.diffs)
        # Reshape sweep_solution into a grid
        self.sweep_diffs = np.array(sweep_diffs).T.reshape(*grids[0].shape)
        return self.sweep_diffs
        
    
    @property
    def signals(self):
        if self.solution is None: self.solution_at(self.cycles)
        
        strand_solution = self.solution[:self.n_strands,:]/self.norm.copies
        return [strand_solution[np.array(strands,dtype=bool),:].sum(axis=0) for strands in self.labels]
    
    @property
    def diffs(self):
        return np.subtract(*self.signals)
    
    def diffs_at(self, c):
        self.solution_at(c)
        return np.subtract(*self.signals)


class CAN(PCR):
    
    defaults = {**PCR.defaults,**{
        'INT_res' : 1.,
        'INT_rng' : [1,9],
        'sweep_INT_idx' : 0,
        'sweep_ax' : None,
        }}
    
    def __init__(self, INT_connections, EXT_connections, labels=None, INT_names=None, EXT_names=None, label_names=None, primer_names=None):
        """
        Constructs a competitive reaction system consisting of multiple oligos, some of which are labeled.

        Args:
            INTs (array): "Natural" oligos internal to the system that have design constraints.
                A connectivity matrix of shape (n,p), where p is the number of primers in the system and n is the
                number of INT oligos. A value of +/-1 indicates that the oligo uses the given primer, with -1
                indicating that the primer targets the "left" or "negative" strand of an oligo and +1 indicating 
                the "right" or "positive" strand. If possible, -1 entries should appear to before +1 entries. 
                Note that "targeting" means "complementary" to, so a primer that targets the positive strand gets
                extended to generate the negative strand.

            EXTs (array): Synthetic oligos added to the system that have no/fewer design constraints.
                A connectivity matrix of shape (m,p), where p is the number of primers in the system and m is the
                number of EXT oligos.

            labels (list): Specify which strands are labeled.
                A list of vectors, each of length 2*(n+m), indicating which strands are targeted with the given 
                probe color

        """
        
        assert INT_connections.shape[1] == EXT_connections.shape[1], 'INT and EXT matrices must have same number of columns'

        self.INT_matrix = INT_connections
        self.EXT_matrix = EXT_connections
        self.n_INTs, self.n_primers = self.INT_matrix.shape
        self.n_EXTs, _ = self.EXT_matrix.shape
        
        if INT_names  is None:
            INT_names = [chr(ord('α')+i) for i in range(self.n_INTs)]
        if EXT_names is None:
            EXT_names = [chr(ord('a')+i) for i in range(self.n_EXTs)]
        oligo_names = INT_names + EXT_names
        self.connections = np.vstack([self.INT_matrix, self.EXT_matrix])
        
        super(CAN, self).__init__(self.connections, labels=labels, oligo_names=oligo_names, label_names=label_names, primer_names=primer_names)
        
        self.INTs = self.oligos[:self.n_INTs]
        self.EXTs = self.oligos[self.n_INTs:]
        
    
    ################################################################################
    ## Solution plotting functions
    ################################################################################


    def INT_sweep(self, INT=None, rng=None, res=None, progress_bar=False, pts=None):
        if INT is None: INT=self.sweep_INT
        if rng is None: rng=self.INT_rng
        if res is None: res=self.INT_res
        if pts is None: pts = np.arange(rng[0],rng[1]+res,res)

        N = len(pts)
        diffs = np.zeros(N)
        iterator = tqdm(enumerate(pts),total=N) if progress_bar else enumerate(pts)

        solutions = []
        for i,INT_0 in iterator:
            self.set_oligo_init(INT,10**INT_0)
            self.updateParameters()
            solutions.append(self.solveODE())
            diffs[i] = self.get_diff()
        self.diffs=diffs
        return diffs, solutions

    # def INT_grid(self, INT1=None, INT2=None, progress_bar=True):
    #     if INT1 is None: INT1=self.INTs[0]
    #     if INT2 is None: INT2=self.INTs[1]
    #     rng = self.INT_rng
    #     res = self.INT_res
    #     rng = np.arange(rng[0],rng[1]+res,res)
    #     N = len(rng)
    #     diffs = np.zeros([N,N])
    #     iterator = tqdm(enumerate(rng),total=N) if progress_bar else enumerate(rng)
    #     for i,INT_0 in iterator:
    #         self.set_oligo_init(INT1,10**INT_0)
    #         for j,INT_0 in enumerate(rng):
    #             self.set_oligo_init(INT2,10**INT_0)
    #             self.updateParameters()
    #             self.solveODE()
    #             diffs[i,j] = self.get_diff()
    #     return diffs

    # def plot_INT_grid(self, ax=None, INT1=None, INT2=None, progress_bar=True, cmap = FAM_HEX_cmap()):
    #     if INT1 is None: INT1=self.INTs[0]
    #     if INT2 is None: INT2=self.INTs[1]
    #     diffs = self.INT_grid(INT1=INT1, INT2=INT2, progress_bar=progress_bar)
    #     rng = self.INT_rng
    #     res = self.INT_res
    #     rng = np.arange(rng[0],rng[1]+res,res)
    #     ext = np.ceil(np.max([np.max(diffs),np.abs(np.min(diffs))])*2)/2
    #     fig,ax = plt.subplots(1,1) if ax is None else (ax.figure,ax)
    #     pcm = ax.pcolormesh(rng,rng,diffs.T, cmap = cmap,
    #                               vmin=-ext,vmax=ext,
    #                               shading = 'gouraud'
    #                         )
    #     ax.axvline(5,color='w',linestyle=':')
    #     ax.axhline(5,color='w',linestyle=':')

    #     cbar = plt.colorbar(pcm,ticks=np.arange(-ext,ext+0.5,0.5))#, ax = axs[1],extend=extend,ticks=np.arange(-10,10.1,2.5))
    #     #cbar_x0s.ax.set_ylabel('$log_{10}$ Tar/Ref Ratio\nProviding Signal Parity',fontsize=16)
    #     #cbar_x0s.ax.tick_params(labelsize=16)
    #     cntr = ax.contour(rng,rng,diffs.T,colors = 'k',
    #                        #levels = np.arange(np.around(np.min(diffs)*2)/2,np.around(np.max(diffs)*2)/2+0.5,0.5)
    #                       )
    #     plt.clabel(cntr, inline=True, fontsize=16, fmt = '%.1f');
    #     ax.set_aspect('equal', 'box')
    #     ax.set_xlabel('log10 '+str(INT1))
    #     ax.set_ylabel('log10 '+str(INT2))
    #     return diffs

    # def plot_INT_sweep(self, INT=None, rng=None, res=None, progress_bar=False, annotate='Outer', ax=None, indiv=True, update=False):
    #     # TODO: Allow target axis to be specified for individual plots
    #     # TODO: Plot total signal for each label

    #     if INT is None: INT=self.sweep_INT
    #     if rng is None: rng=self.INT_rng
    #     if res is None: res=self.INT_res

    #     diffs, sweep_solutions = self.INT_sweep(INT=INT, rng=rng, res=res, progress_bar=progress_bar)

    #     if ax is not None: indiv=False

    #     pts = np.arange(rng[0],rng[1]+res,res)

    #     if indiv:
    #         fig,gs,ind_axs = self.plotTracesGrid(sweep_solutions,pts,annotate=annotate)
    #         ax = fig.add_subplot(gs[:,3:])
    #         self.sweep_ax = ax

    #     if update:
    #         if self.sweep_ax is None:
    #             _,self.sweep_ax = self.plotDiffs(diffs)
    #         else:
    #             self.updateDiffs(self.sweep_ax)
    #     else:
    #         if ax is None:
    #             fig,ax = plt.subplots(1,1)
    #         _,self.sweep_ax = self.plotDiffs(diffs,ax=ax)

    #     if annotate in [True,'Inner','Outer']:
    #         self.annotate_diff_plot(self.sweep_ax,pos=annotate)

    #     return diffs

    # def plotDiffs(self,diffs=None,ax=None,rng=None,res=None):
    #     if diffs is None: diffs = self.diffs
    #     if rng is None: rng = self.INT_rng
    #     if res is None: res = self.INT_res
    #     fig,ax = plt.subplots(1,1) if ax is None else (ax.figure,ax)
    #     INT = self.sweep_INT
    #     rng = np.arange(rng[0],rng[1]+res,res)

    #     ax.plot(rng,diffs, color=grey(), zorder=0)
    #     ax.scatter(rng,diffs, c=diffs, cmap=FAM_HEX_cmap(),
    #                s=10**2, edgecolor=grey(), zorder=1,
    #                vmin=-1, vmax=1,
    #               )
    #     plt.setp(ax,**{
    #         'ylim' : [-1.05,1.05],
    #         #'title' : '{:s}-{:s} after {:d} cycles'.format(self.labels[0],self.labels[1],self.cycles),
    #         'ylabel' : 'Signal Difference',
    #         'xlabel' : f'log10 {INT} copies',
    #     })
    #     return fig, ax

    # def updateDiffs(self,ax, diffs=None, rng=None, res=None):
    #     if diffs is None: diffs = self.diffs
    #     if rng is None: rng = self.INT_rng
    #     if res is None: res = self.INT_res
    #     rng = np.arange(rng[0],rng[1]+res,res)
    #     ax.lines[0].set_xdata(rng)
    #     ax.lines[0].set_ydata(diffs)
    #     for l in ax.lines[1:]: l.remove()
    #     txts = [child for child in self.sweep_ax.get_children() if type(child)==mpl.text.Annotation]
    #     for txt in txts: txt.remove()
    #     ax.figure.canvas.draw()
    #     print(self.get_diff_stats())

    # def get_diff(self):
    #     '''
    #     Gets the endpoint signal difference from the current solution

    #     Currently, this is defined by "convention" as the intensity of the alphabetically
    #     first label (typically FAM) subtracted from the intensity of the second label
    #     (typically HEX). Obviously, this is very fragile and inflexible and should be fixed ASAP.
    #     '''
    #     return (sum(self.solution[L2][-1] for L2 in self.list('label2_strands'))-
    #             sum(self.solution[L1][-1] for L1 in self.list('label1_strands')))/self.norm

    # def get_diff_stats(self, diffs=None,rng=None):
    #     if diffs is None: diffs=self.diffs
    #     if rng is None: rng = self.INT_rng
    #     res = self.INT_res
    #     rng = np.arange(rng[0],rng[1]+res,res)

    #     interp_res=0.01
    #     wt_interp = np.arange(rng[0],rng[-1]+interp_res,interp_res)
    #     diff_interp = np.interp(wt_interp,rng,diffs)
    #     #diff_half = (np.max(diff_interp)-np.min(diff_interp))/2+np.min(diff_interp)
    #     diff0 = np.argmin(abs(diff_interp))
    #     diff90 = (np.max(diff_interp)-np.min(diff_interp))*0.9+np.min(diff_interp)
    #     diff10 = (np.max(diff_interp)-np.min(diff_interp))*0.1+np.min(diff_interp)

    #     stats = {
    #         'Zero' : wt_interp[diff0],
    #         'DR' : np.abs(wt_interp[np.argmin(abs(diff_interp-diff10))] - wt_interp[np.argmin(abs(diff_interp-diff90))]),
    #         'Max' : np.max(diffs),
    #         'Min' : np.min(diffs),
    #     }
    #     return stats

    # def annotate_diff_plot(self,ax,diffs=None,rng=None,pos='Outer'):
    #     if diffs is None: diffs=self.diffs
    #     if rng is None: rng=self.INT_rng
    #     stats = self.get_diff_stats(diffs=diffs,rng=rng)
    #     ax.axvline(stats["Zero"], ls='--', color='k', zorder=-1)

    #     if pos in [True,'Outer']:
    #         x_pos = 1.05
    #     elif pos is 'Inner':
    #         x_pos = 0.025

    #     ax.annotate(f'Zero: {stats["Zero"]:.2f}',
    #                  xy=(x_pos, .925), xycoords='axes fraction',
    #                  horizontalalignment='left')

    #     ax.annotate(f'DR: {stats["DR"]:.2f}',
    #                  xy=(x_pos, .825), xycoords='axes fraction',
    #                  horizontalalignment='left')

    #     ax.annotate(f'Max: {stats["Max"]:.2f}',
    #                  xy=(x_pos, .725), xycoords='axes fraction',
    #                  horizontalalignment='left')

    #     ax.annotate(f'Min: {stats["Min"]:.2f}',
    #                  xy=(x_pos, .625), xycoords='axes fraction',
    #                  horizontalalignment='left')

    # def plotTraces(self,ax=None,solution=None):
    #     fig,ax = plt.subplots(1,1) if ax is None else (ax.figure,ax)
    #     if solution is None: solution = self.solution
    #     L1s = self.list('label1_strands')
    #     L2s = self.list('label2_strands')
    #     for i,L1 in enumerate(L1s):
    #         ax.plot(np.arange(self.cycles), solution[L1]/self.norm, color=FAM_HEX_cmap()(0+(len(L1s)-(i+1))*0.3))
    #     for i,L2 in enumerate(L2s):
    #         ax.plot(np.arange(self.cycles), solution[L2]/self.norm, color=FAM_HEX_cmap()(1-(len(L2s)-(i+1))*0.3))
    #     return fig, ax

    # def plotTracesGrid(self,solution_list,conc_list,annotate=True):
    #     fig = plt.figure(constrained_layout=True,figsize=[16,5])
    #     N = len(solution_list)
    #     gs = fig.add_gridspec(N//3+1,6)
    #     ind_axs = []
    #     INT = self.sweep_INT
    #     for i,(solution,conc) in enumerate(zip(solution_list,conc_list)):
    #         with plt.rc_context({'axes.labelweight':'normal','font.size':14}):
    #             ind_axs.append(fig.add_subplot(gs[i//3,i%3], sharey=ind_axs[0] if i>0 else None))
    #             self.plotTraces(ax=ind_axs[i],solution=solution)
    #             plt.setp(ind_axs[i].get_yticklabels(), visible=True if i%3==0 else False)
    #             plt.setp(ind_axs[i].get_xticklabels(), visible=True if i//3+1==(N-1)//3+1 else False)
    #             if annotate in [True,'Inner','Outer']:
    #                 plt.annotate(f'{conc:.1f} logs {INT}', xy=(.025, .825), xycoords='axes fraction',fontsize=12)
    #             if (i%3==0)&(i//3+1==(N-1)//3+1):
    #                 plt.setp(ind_axs[i],**{
    #                     'ylabel' : 'Norm Signal',
    #                     'xlabel' : 'Cycles',
    #                 })
    #     return fig, gs, ind_axs

    ################################################################################
    ## Interactive configurations with simulating and plotting
    ################################################################################

    # def interactive_solve(self,**kwargs):
    #     """Set the necessary attributes from the interactive configuration, then solve with a range of initial INT values"""
    #     for idx, row in self.connections.iterrows():
    #         assert set(self.select_label1.value).intersection(set(self.select_label2.value)) == set(), 'No strands may be labeled twice'
    #         if '_'.join(idx) in self.select_label1.value:
    #             label = self.list('labels')[0]
    #         elif '_'.join(idx) in self.select_label2.value:
    #             label = self.list('labels')[1]
    #         else:
    #             label = ''
    #         self.connections.at[idx,'Label'] = label
    #     self.buildLabels()
    #     for oligo in self.oligos:
    #         self.set_rate(oligo,kwargs['r_'+str(oligo)])
    #     for EXT in self.EXTs:
    #         self.set_oligo_init(EXT,10**kwargs[str(EXT)])
    #     for p in self.primers:
    #         self.set_primer_init(p,kwargs[str(p)])
    #     self.norm = molar2copies(kwargs['norm']*10**-9)
    #     self.cycles = kwargs['cycles']
    #     self.INT_rng = self.INT_rng_widget.value
    #     self.INT_res = np.diff(self.INT_rng)[0]/8# kwargs['INT_res']
    #     if len(self.list('INTs'))>1:
    #         self.sweep_INT = self.INT_selector.value
    #         held_INTs = [INT for INT in self.list('INTs') if INT is not self.INT_selector.value]
    #         for INT,widg in zip(held_INTs,self.INT_conc_widgets):
    #             self.set_oligo_init(INT,10**widg.value)
    #     if kwargs['plt_rslt']:
    #         self.plot_INT_sweep(indiv=kwargs['indiv'], update=False)
    #     return

    # def interactive(self):
    #     ui_concentrations = ipw.interactive(self.interactive_solve, #{'manual': True},
    #              **{
    #                  str(rate):ipw.FloatSlider(min=0.1, max=1.1, step=0.05, value=rate.value,
    #                                            description=f'{str(rate)[2:]} rate', continuous_update=False)
    #                  for rate in self.rates
    #              },**{
    #                  str(p):ipw.FloatSlider(min=1, max=500, step=25, value=self.get_primer_init(p),
    #                                             description=f'nM {str(p)}', continuous_update=False)
    #                  for p in self.primers
    #              },**{
    #                  str(EXT):ipw.FloatSlider(min=0, max=10, step=0.25, value=np.log10(self.get_oligo_init(EXT)),
    #                                             description=f'logs {str(EXT)}', continuous_update=False)
    #                  for EXT in self.EXTs
    #              },**{
    #              },
    #              norm=ipw.IntSlider(min=1, max=500, step=25, value=copies2molar(self.norm)*10**9, description='Norm (nM)', continuous_update=False),
    #              #INT_res=ipw.FloatSlider(min=0.1, max=2, step=0.05, value=self.INT_res, description='resolution', continuous_update=False),
    #              cycles=ipw.IntSlider(min=10, max=100, value=self.cycles, description='cycles', continuous_update=False),
    #              plt_rslt=ipw.Checkbox(value=False, description='Plot Result'),
    #              indiv=ipw.Checkbox(value=False, description='Individual Traces'),
    #         )

    #     self.INT_rng_widget=ipw.FloatRangeSlider(min=0, max=10, step=0.1, value=self.INT_rng,
    #                                       description=f"{self.list('INTs')[0]} range", continuous_update=False)

    #     self.select_label1 = ipw.SelectMultiple(options = self.list('strands')+['None',], value = self.list('label1_strands'),
    #                                             description = f'Label {self.list("labels")[0]:}')
    #     self.select_label2 = ipw.SelectMultiple(options = self.list('strands')+['None',], value = self.list('label2_strands'),
    #                                             description = f'Label {self.list("labels")[1]:}')

    #     n_oligos = len(self.oligos)
    #     n_primers = len(self.primers)
    #     n_EXTs = len(self.EXTs)

    #     col1 = n_oligos
    #     col2 = col1+n_primers
    #     col3 = col2+n_EXTs
    #     col4 = col3+2

    #     rate_widgets = ui_concentrations.children[:col1]
    #     primer_widgets = ui_concentrations.children[col1:col2]
    #     EXT_widgets = list(ui_concentrations.children[col2:col3])
    #     INT_widgets = [self.INT_rng_widget]

    #     if len(self.list('INTs'))>1:
    #         self.INT_selector = ipw.RadioButtons(options = self.list('INTs'), description = 'Sweep:')
    #         self.INT_conc_widgets = [ipw.FloatSlider(min=0, max=10, step=0.25, value=np.log10(self.get_oligo_init(INT)),
    #                                                  description=f'logs {str(INT)}', continuous_update=False)
    #                                  for INT in self.INTs if str(INT) is not self.INT_selector.value]
    #         INT_widgets.extend(self.INT_conc_widgets)
    #         def update_INT_widgets(*args):
    #             self.INT_rng_widget.description = f'{self.INT_selector.value} range'
    #             held_INTs = [INT for INT in self.list('INTs') if INT is not self.INT_selector.value]
    #             for INT,widg in zip(held_INTs,self.INT_conc_widgets):
    #                 widg.description = f'logs {INT}'
    #                 widg.value=np.log10(self.get_oligo_init(INT))
    #         self.INT_selector.observe(update_INT_widgets,'value')
    #         INT_widgets.append(self.INT_selector)

    #     oligo_widgets = EXT_widgets + INT_widgets

    #     display(ipw.HBox([
    #         ipw.VBox(rate_widgets),
    #         ipw.VBox(primer_widgets),
    #         ipw.VBox(EXT_widgets),
    #         ipw.VBox(INT_widgets)
    #     ]))

    #     display(ipw.HBox([
    #         self.select_label1,
    #         self.select_label2,
    #         ipw.VBox(ui_concentrations.children[col3:col4]),
    #         ipw.VBox(ui_concentrations.children[col4:-1]),
    #     ]))

    #     display(ui_concentrations.children[-1])#Show the output