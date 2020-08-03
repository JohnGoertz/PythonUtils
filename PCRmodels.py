import numpy as np
import scipy.integrate as integrate
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipywidgets as ipw
from tqdm.notebook import tqdm

import shelve
import gc
import warnings


def copies2molar(copies):
    #Calculates concentration in a 10 μL volume
    return copies / 6.022e23 / (10*10**-6)

def molar2copies(moles):
    #Calculates number of copies in a 10 μL volume
    return moles * 6.022e23 * (10*10**-6)

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

    defaults = {
        'oligo_copies' : 10**5,
        'oligo_rate' : 0.9,
        'primer_nM' : 100,
        'norm_nM' : 100,
        'cycles' : 60,
        'sol' : None,
        'solution' : None,
        'integrator' : 'RK45',
    }

    def __init__(self, connections, labels=None, oligo_names=None, label_names=None, primer_names=None):
        """
        Constructs a Monod simulation of a PCR reaction system.

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
        
        if labels is None:
            self.labels = [[]]
            self.labels[0].extend([1,0] for _ in range(self.n_oligos))
        else:
            assert all(len(label)==self.n_strands for label in labels)
            self.labels = labels
        self.n_labels = len(self.labels)
        
        if oligo_names is None:
            oligo_names = [chr(ord('a')+i) for i in range(self.n_oligos)]
        else:
            assert len(oligo_names)==self.n_oligos
            
        if label_names is None:
            label_names = [f'L{i}' for i in range(self.n_primers)]
        else:
            assert len(label_names)==self.n_labels
            self.label_names = label_names
            
        if primer_names is None:
            primer_names = [f'p{i}' for i in range(self.n_primers)]
        else:
            assert len(primer_names)==self.n_primers
        
        self.oligos = [Oligo(name, self.defaults['oligo_rate'], self.defaults['oligo_copies']) for name in oligo_names]
        self.strands = []
        for oligo in self.oligos:
            self.strands.extend(oligo.strands)
        self.primers = [Primer(name, nM=self.defaults['primer_nM']) for name in primer_names]
        self.setDefaults()
        self.buildEquations()
        #self.buildLabels()
        #self.compileODE(verbose=True)

    ################################################################################
    ## Getters and Setters
    ################################################################################
    
    @property
    def initial_oligo_copies(self):
        return [oligo.copies for oligo in self.oligos]
    
    @initial_oligo_copies.setter
    def initial_oligo_copies(self,copies_list):
        for oligo,copies in zip(self.oligos,copies_list):
            oligo.copies = copies

    @property
    def initial_strand_copies(self):
        return [strand.copies for strand in self.strands]
    
    @initial_strand_copies.setter
    def initial_strand_copies(self,copies_list):
        for strand,copies in zip(self.strand,copies_list):
            strand.copies = copies
    
    @property
    def initial_primer_copies(self):
        return [primer.copies for primer in self.primers]
    
    @initial_primer_copies.setter
    def initial_primer_copies(self,copies_list):
        for primer,copies in zip(self.primers,copies_list):
            primer.copies = copies
            
    @property
    def initial_primer_nMs(self):
        return [primer.nM for primer in self.primers]
    
    @initial_primer_nMs.setter
    def initial_primer_nMs(self,nMs_list):
        for primer,nM in zip(self.primers,nMs_list):
            primer.nM = nM
    
    @property
    def initial_copies(self):
        return self.initial_strand_copies + self.initial_primer_copies
    
    @initial_copies.setter
    def initial_copies(self,copies_list):
        if len(copies_list) == self.n_strands+self.n_primers:
            self.initial_strand_copies = copies_list[:self.n_strands]
            self.initial_primer_copies = copies_list[self.n_strands:]
        elif len(copies_list) == self.n_oligos+self.n_primers:
            self.initial_oligo_copies = copies_list[:self.n_oligos]
            self.initial_primer_copies = copies_list[self.n_oligos:]
    
    @property
    def rates(self):
        return [oligo.rate for oligo in self.oligos]
    
    @rates.setter
    def rates(self,rates_list):
        for oligo,rate in zip(self.oligos, rates_list):
            oligo.rate = rate
            
    @property
    def all_parameters(self):
        log_oligo_copies = [np.log10(oligo.copies) for oligo in self.oligos]
        primer_nMs = self.initial_primer_nMs
        oligo_rates = self.rates
        return log_oligo_copies + primer_nMs + oligo_rates
    
    @all_parameters.setter
    def all_parameters(self,parameter_list):
        self.initial_oligo_copies = 10**parameter_list[:self.n_oligos]
        self.initial_primer_nMs = parameter_list[self.n_oligos:-self.n_oligos]
        self.rates = parameter_list[:-self.n_oligos]

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
        def equations(c,copies,*rates):
            # rate-limiting contribution from each primer
            mus = np.zeros(n_p)
            for p in range(n_p):
                # all oligos that utilize the primer
                targets = np.argwhere(cm[:, p] != 0).flatten()
                # strands that bind to the primer
                strands = [copies[2*oligo+(cm[oligo,p]+1)//2] for oligo in targets]
                # total concentration of all strands that bind to the primer
                strand_concs = np.sum(strands)
                # position of the primer in the copies vector
                idx = p+n_s
                mus[p] = copies[idx]/(copies[idx] + strand_concs)
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
                    mu = mus[cm[i,:]==-strand]
                    oligo_eqns.extend(rate*complement*mu)
            # concentration of each primer at the next time point
            primer_eqns = []
            for p in range(n_p):
                # all oligos that utilize the primer
                targets = np.argwhere(cm[:, p] != 0).flatten()
                # derivatives of strands that bind to the primer
                strands = [oligo_eqns[2*oligo+np.abs(cm[oligo,p]-1)//2] for oligo in targets]
                primer_eqns.append(-np.sum(strands))
            return oligo_eqns + primer_eqns
        self.eqns = equations
        return self.eqns


    ################################################################################
    ## Running a simulation
    ################################################################################


    def solveODE(self, **kwargs):
        kwargs.setdefault('dense_output',True)
        sol = integrate.solve_ivp(self.eqns, t_span=(0,self.cycles), y0=self.initial_copies, args=self.rates,
                                  method=self.integrator, **kwargs)
        self.sol = sol
        return sol
    
    def solution_at(self, c):
        if self.sol is None: self.solveODE()
        solution = self.sol.sol(c)
        # Ensure solution is an array of shape (n_s+n_p, len(c)) 
        if solution.ndim==1:
            solution = solution[:,None]
        self.solution = solution
        return self.solution
    
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
