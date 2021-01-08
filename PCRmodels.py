import numpy as onp
from scipy.optimize import minimize
import time

import jax
from jax import numpy as np
from jax.experimental.ode import odeint
from jax.config import config
config.update('jax_enable_x64', True)


# import seaborn as sns
import matplotlib.pyplot as plt
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
    roses = onp.ones((N,4))
    for i in range(3):
        roses[:,i] = onp.linspace(rosest[sat][i], 1, N)

    tealest = {
        'light' : [35/256, 135/256, 127/256],
        'mid'  : [10/256, 111/256, 103/256],
        'dark' : [0/256, 102/256, 94/256],
    }
    teals = onp.ones((N,4))
    for i in range(3):
        teals[:,i] = onp.linspace(1, tealest[sat][i], N)

    concat_colors = onp.vstack((roses,teals))
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
        'oligo_copies' : 10**5.,
        'oligo_rate' : 0.9,
        'primer_nM' : 100.,
        'norm_nM' : 100.,
        'cycles' : 60.,
        'solution' : None,
    }

    def __init__(self, connections, labels=None, oligo_names=None, label_names=None, primer_names=None, compile_eqns=True, disable_checks=False):
        """
        Constructs a Monod simulation of a PCR reaction system.

        Example:
            ```python
            OR_gate = np.array([[-1, 0, +1, 0, 0], [0, -1, +1, 0, 0], [0, 0, -1, +1, 0], [0, 0, 0, -1, +1]])
            OR_labels = np.array([[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0]])
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

        if not disable_checks:
            assert all(np.sum(connections==-1, axis=1) == 1), 'All rows of connections must have exactly one +1 and one -1 value'
            assert all(np.sum(connections==+1, axis=1) == 1), 'All rows of connections must have exactly one +1 and one -1 value'
            assert all(np.sum(connections!=0, axis=1) == 2), 'All rows of connections must have exactly two non-zero values'

        self.cm = self.connections = np.array(connections,dtype=float)
        self.s_cm = self._make_s_cm(self.cm)
        self.p_cm = self._make_p_cm(self.cm)

        self.n_oligos, self.n_primers = connections.shape
        self.n_strands = self.n_oligos*2

        # If no labels supplied, "label" the "-" strand of every oligo
        if labels is None:
            labels = [[]]
            labels[0].extend([1,0] for _ in range(self.n_oligos))
        else:
            if not disable_checks: assert all(len(label)==self.n_strands for label in labels)
        self.labels = np.array(labels,dtype=float)
        self.n_labels = len(self.labels)

        # If no oligo names are supplied, name each oligo with successive latin letters
        if oligo_names is None:
            oligo_names = [chr(ord('a')+i) for i in range(self.n_oligos)]
        else:
            if not disable_checks: assert len(oligo_names)==self.n_oligos
        self.oligo_names = oligo_names

        # If no label names are supplied, name each label L0, L1, etc
        if label_names is None:
            label_names = [f'L{i}' for i in range(self.n_labels)]
        else:
            if not disable_checks: assert len(label_names)==self.n_labels
        self.label_names = label_names

        # If no primer names are supplied, name each primer p0, p1, etc
        if primer_names is None:
            primer_names = [f'p{i}' for i in range(self.n_primers)]
        else:
            if not disable_checks: assert len(primer_names)==self.n_primers
        self.primer_names = primer_names

        # A list storing each oligo as an Oligo object
        self.oligos = [Oligo(name, self.defaults['oligo_rate'], self.defaults['oligo_copies']) for name in oligo_names]

        # A list storing each strand as an Strand object
        self.strands = []
        for oligo in self.oligos:
            self.strands.extend(oligo.strands)
        # A list storing each primer as a Primer object
        self.primers = [Primer(name, nM=self.defaults['primer_nM']) for name in primer_names]

        # Initialize
        self.setDefaults()
        if compile_eqns:
            self.compileEquations()


    def __str__(self):
        summary = {'Connections':self.connections,
                   'Labels':self.labels,
                   'Oligos (log copies)': onp.log10(self.oligo_copies),
                   'Rates (base 2)':self.rates,
                   'Primers (nM)':self.primer_nMs,
                   }
        m = max(map(len, list(summary.keys()))) + 1
        return '\n'.join([k.rjust(m) + ': ' + onp.array2string(onp.array(v),prefix=(k.rjust(m)+': '))
                  for k, v in sorted(summary.items())])

    def __repr__(self):
        return self.__str__()



    ################################################################################
    ## Configure the necessary elements for the simulations
    ################################################################################

    def setDefaults(self):
        for primer in self.primers:
            primer.nM = self.defaults['primer_nM']
        for oligo in self.oligos:
            oligo.oligo_copies = self.defaults['oligo_copies']
            oligo.rate = self.defaults['oligo_rate']

        self.norm = Primer('norm',nM=self.defaults['norm_nM'])
        self.rhs = self._rhs

        for k,v in self.defaults.items():
            if k not in ['oligo_copies','oligo_rate','norm_nM','primer_nM']:
                setattr(self,k,v)

    @staticmethod
    def _make_p_cm(cm):
        '''Primer that binds to the given strand'''
        n_o, n_p = cm.shape
        n_s = n_o*2
        return np.reshape(np.abs(np.hstack([cm-1,cm+1]))//2,[n_s,n_p])

    @staticmethod
    def _make_s_cm(cm):
        '''Primer that generates to the given strand'''
        n_o, n_p = cm.shape
        n_s = n_o*2
        return np.reshape(np.abs(np.hstack([cm+1,cm-1]))//2,[n_s,n_p])

    @staticmethod
    def _rhs(copies,c,strand_rates,s_cm,p_cm):
        n_s, n_p = s_cm.shape
        n_o = n_s//2

        strands = copies[:n_s]
        primers = copies[n_s:]

        # rate-limiting contribution from each primer
        mus = primers/(primers+np.dot(strands,p_cm))

        # concentration of the complementary strand
        complements = np.reshape(np.reshape(strands[:,None],[n_o,2])[:,::-1],[n_s,1])

        # derivative of each strand
        strands_dt = complements*strand_rates*np.dot(s_cm,mus)[:,None]

        # derivative of each primer
        primers_dt = -np.sum(strands_dt*s_cm, axis=0)
        return np.hstack([np.squeeze(strands_dt),primers_dt])

    def compileEquations(self):
        self.rhs = jax.jit(self._rhs)
        self.rhs(self.copies,1.,self.strand_rates,self.s_cm,self.p_cm)


    def _solve(self,copies,cycles,strand_rates,s_cm,p_cm):
        return odeint(self.rhs,copies,cycles,strand_rates,s_cm,p_cm)

    def solve(self,copies=None,cycles=None,strand_rates=None,s_cm=None,p_cm=None):

        copies = self.copies if copies is None else copies

        cycles = np.arange(self.cycles+1, dtype=float) if cycles is None else cycles

        strand_rates = self.strand_rates if strand_rates is None else strand_rates

        s_cm = self.s_cm if s_cm is None else s_cm

        p_cm = self.p_cm if p_cm is None else p_cm

        return self._solve(copies, cycles, strand_rates, s_cm, p_cm)

    @staticmethod
    def _plot_connections(rxn_tpl, ax=None):
        cm,labels = rxn_tpl
        n_o,n_p = cm.shape
        p_cm = onp.array(np.reshape(np.abs(np.hstack([cm-1,cm+1]))//2,[n_o*2,n_p]))
        l1 = p_cm*labels[0,:][:,None]
        l1 = np.add(*np.split(np.reshape(l1,[n_o,n_p*2]),2,axis=1))
        l2 = p_cm*labels[1,:][:,None]
        l2 = np.add(*np.split(np.reshape(l2,[n_o,n_p*2]),2,axis=1))

        if ax is None:
            fig,ax = plt.subplots(1,1)

        ax.imshow(np.abs(cm),cmap='binary',vmin=0,vmax=1)
        ax.imshow(l1-l2, alpha = np.abs(l1-l2), cmap=FAM_HEX_cmap())
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_connections(self, ax=None):
        rxn_tpl = self.cm, self.labels
        self._plot_connections(rxn_tpl,ax)


    @staticmethod
    def _simplifyRxn(rxn_tpl, n_NAT=1):
        cm,labels = rxn_tpl

        # Remove double-labeled strands
        labels *= ~(labels.sum(axis=0)==2)
        labels = labels.T

        n_o,n_p = cm.shape
        n_y = n_o-n_NAT
        n_s = n_o*2

        SYNs = cm[n_NAT:,:]
        p_cm = onp.array(np.reshape(np.abs(np.hstack([SYNs-1,SYNs+1]))//2,[n_y*2,n_p]))
        SYN_labels = onp.array(labels[n_NAT*2:,:])

        # Primer utilized by each strand
        primers = np.argwhere(p_cm)[:,1]

        # Ensure the -1 strand is on the left
        for y in range(n_y):
            old_order = [2*y,2*y+1]
            pair = primers[old_order]==np.min(primers[old_order])
            new_order = (~pair).astype(int)+2*y
            p_cm[old_order,:] = p_cm[new_order,:]
            SYN_labels[old_order,:] = SYN_labels[new_order,:]

        SYN_cm = np.subtract(*np.split(np.reshape(p_cm,[n_y,n_p*2]),2,axis=1)[::-1])

        # Ensure oligos are ordered by position of -1 strand
        primers = np.reshape(np.argwhere(SYN_cm)[:,1],[-1,2])
        first = primers[:,1].argsort(axis=0)
        second = primers[first][:,0].argsort(axis=0)
        primers = primers[first][second]
        SYN_cm = SYN_cm[first][second]
        SYN_labels = np.reshape(np.reshape(SYN_labels,[-1,4])[first,:][second,:],[-1,2])

        cm = np.vstack([cm[:n_NAT,:],SYN_cm])
        labels = np.vstack([labels[:n_NAT*2,:],SYN_labels])

        return cm,labels.T

    def simplifyRxn(self):
        rxn_tpl = self.cm, self.labels
        return self._simplifyRxn(rxn_tpl,0)


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
    def strand_rates(self):
        return np.reshape(np.tile(self.rates,[2,1]),[self.n_strands,1],order='F')

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


class CAN(PCR):

    defaults = {**PCR.defaults,**{
        'sweep_res' : 1.,
        'sweep_rng' : [0.5,9.5],
        # Learning defaults
        'obj' : None,
        'oligo_bounds': (1.,10.),
        'rate_bounds': (0.5,1.),
        'primer_bounds': (10.,500.),
        }}

    def __init__(self, INT_connections, EXT_connections, labels=None,
                 INT_names=None, EXT_names=None, label_names=None, primer_names=None,
                 setup=True, sweep_res=1., sweep_rng=[1.,9.],
                 compile_eqns=True, disable_checks=False,):
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

        self.INT_cm = INT_connections
        self.EXT_cm = EXT_connections
        self.n_INTs, self.n_primers = self.INT_cm.shape
        self.n_EXTs, _ = self.EXT_cm.shape
        self.INT_names = INT_names
        self.EXT_names = EXT_names

        if INT_names  is None:
            INT_names = [chr(ord('α')+i) for i in range(self.n_INTs)]
        if EXT_names is None:
            EXT_names = [chr(ord('a')+i) for i in range(self.n_EXTs)]
        oligo_names = INT_names + EXT_names
        connections = np.vstack([self.INT_cm, self.EXT_cm])

        super(CAN, self).__init__(connections, labels=labels, oligo_names=oligo_names,
                                  label_names=label_names, primer_names=primer_names,
                                  compile_eqns=False, disable_checks=disable_checks)

        self.INTs = self.oligos[:self.n_INTs]
        self.INT_idxs = np.arange(self.n_INTs)
        self.EXTs = self.oligos[self.n_INTs:]

        self.sweep_res = sweep_res
        self.sweep_rng = sweep_rng

        if setup:
            self.setup_solution_sweep()
        if compile_eqns:
            self.compileEquations()



    def __str__(self):
        summary = {'INT Connections':self.INT_cm,
                   'EXT Connections':self.EXT_cm,
                   'Labels':self.labels,
                   'Oligos (log copies)': onp.log10(self.oligo_copies),
                   'Rates (base 2)':self.rates,
                   'Primers (nM)':self.primer_nMs,
                   }
        m = max(map(len, list(summary.keys()))) + 1
        return '\n'.join([k.rjust(m) + ': ' + onp.array2string(onp.array(v),prefix=(k.rjust(m)+': '))
                  for k, v in summary.items()])


    ################################################################################
    ## Running a simulation
    ################################################################################

    @staticmethod
    def _setup_solution_sweep(oligos, rng, res):
        arrays = [np.arange(rng_[0], rng_[1]+res_,res_) for rng_,res_ in zip(rng,res)]
        grids = np.meshgrid(*arrays)
        pts = np.vstack([grid.ravel() for grid in grids]).T
        return arrays, grids, pts, oligos

    def setup_solution_sweep(self, oligos=None, rng=None, res=None):
        '''Build 1D arrays and multidimensional meshgrids of all evaluation points'''
        if oligos is None:
            oligos = self.INT_idxs
        if rng is None:
            rng = self.sweep_rng
        if res is None:
            res = self.sweep_res

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

        if type(res) is not list:
            # Isotropic grid, same range and resolution for every oligo
            assert all(type(el) is not list for el in rng)
            rng = [rng for _ in range(n_oligos)]
            res = [res for _ in range(n_oligos)]
        self.sweep_setup = self._setup_solution_sweep(oligos, rng, res)
        return self.sweep_setup

    def _get_diffs(self,pt,copies,cycles,strand_rates,s_cm,p_cm,update_idx,norm,labels):
        n_s,_ = s_cm.shape
        # Set the oligo concentrations accordingly
        update_vals = np.squeeze(np.reshape(10**np.vstack([pt,pt]),[2*len(pt),1],order='F'))
        copies = jax.ops.index_update(copies,update_idx,update_vals, indices_are_sorted=True, unique_indices=True)

        solution = self._solve(copies,cycles,strand_rates,s_cm,p_cm)[-1,:]
        strands = solution[:n_s]/norm
        signals = np.dot(labels,strands)
        return -np.diff(signals)

    def get_diffs(self,pt,copies=None,cycles=None,strand_rates=None,s_cm=None,p_cm=None,update_idx=None,norm=None,labels=None):
        if copies==None:
            copies=self.copies
        if cycles==None:
            cycles=np.arange(self.cycles+1,dtype=float)
        if strand_rates==None:
            strand_rates=self.strand_rates
        if s_cm==None:
            s_cm=self.s_cm
        if p_cm==None:
            p_cm=self.p_cm
        if update_idx==None:
            update_idx = []
        if norm==None:
            norm=self.norm.copies
        if labels==None:
            labels=self.labels
        return self._get_diffs(pt,copies,cycles,strand_rates,s_cm,p_cm,update_idx,norm,labels)

    def _solution_at(self, copies, cycles, strand_rates, cm, labels, pts, oligos):
        norm = self.norm.copies
        oligos = np.array(oligos)
        n_o, n_p = cm.shape
        n_s = n_o*2

        # primer that generates each strand
        s_cm = np.reshape(np.abs(np.hstack([cm-1,cm+1]))//2,[n_s,n_p])
        # primer that binds to each strand
        p_cm = np.reshape(np.abs(np.hstack([cm+1,cm-1]))//2,[n_s,n_p])

        update_idx = np.squeeze(np.reshape(np.vstack([oligos*2,oligos*2+1]),[2*len(oligos),1],order='F'))

        diffs = jax.vmap(lambda pt: self._get_diffs(pt,copies,cycles,strand_rates,s_cm,p_cm,update_idx,norm,labels))(pts)

        ## Reshape sweep_solution into a grid
        #diffs = np.array(diffs).T.reshape(*grids[0].shape)
        #if diffs.ndim>1:
        #    diffs = diffs.transpose([1,0,*np.arange(2,len(oligos))])

        return diffs

    def solution_at(self, pts, oligos=None, copies=None, cycles=None, strand_rates=None, cm=None, labels=None):
        copies = self.copies if copies is None else copies

        cycles = np.arange(self.cycles+1, dtype=float) if cycles is None else cycles

        strand_rates = self.strand_rates if strand_rates is None else strand_rates

        cm = self.connections if cm is None else cm

        labels = self.labels if labels is None else labels

        oligos = self.INT_idxs if oligos is None else oligos

        diffs = self._solution_at(copies, cycles, strand_rates, cm, labels, pts, oligos)
        self.solution = diffs
        return diffs


    def _solution_sweep(self, copies, cycles, strand_rates, cm, labels, arrays, grids, pts, oligos):
        norm = self.norm.copies
        oligos = np.array(oligos)
        n_o, n_p = cm.shape
        n_s = n_o*2

        # primer that generates each strand
        s_cm = np.reshape(np.abs(np.hstack([cm-1,cm+1]))//2,[n_s,n_p])
        # primer that binds to each strand
        p_cm = np.reshape(np.abs(np.hstack([cm+1,cm-1]))//2,[n_s,n_p])

        update_idx = np.squeeze(np.reshape(np.vstack([oligos*2,oligos*2+1]),[2*len(oligos),1],order='F'))

        diffs = jax.vmap(lambda pt: self._get_diffs(pt,copies,cycles,strand_rates,s_cm,p_cm,update_idx,norm,labels))(pts)

        # Reshape sweep_solution into a grid
        diffs = np.array(diffs).T.reshape(*grids[0].shape)
        if diffs.ndim>1:
            diffs = diffs.transpose([1,0,*np.arange(2,len(oligos))])

        return diffs

    def solution_sweep(self, copies=None, cycles=None, strand_rates=None, cm=None, labels=None, arrays=None, grids=None, pts=None, oligos=None):
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

        '''

        copies = self.copies if copies is None else copies

        cycles = np.arange(self.cycles+1, dtype=float) if cycles is None else cycles

        strand_rates = self.strand_rates if strand_rates is None else strand_rates

        cm = self.connections if cm is None else cm

        labels = self.labels if labels is None else labels

        if any(p is None for p in [arrays, grids, pts, oligos]):
            arrays, grids, pts, oligos = self.sweep_setup

        diffs = self._solution_sweep(copies, cycles, strand_rates, cm, labels, arrays, grids, pts, oligos)
        self.solution = diffs
        return diffs


    def compileEquations(self):
        super(CAN, self).compileEquations()
        self._solution_sweep = jax.jit(self._solution_sweep)
        self.solution_sweep()
        #if self.obj is not None: self.compileLoss()


    ################################################################################
    ## Solution plotting functions
    ################################################################################

    def plot_1D_solution(self, diffs=None, ax=None, crosshairs=[5,5], plot_kws=None, scatter_kws=None):

        diffs = self.solution_sweep() if diffs is None else diffs
        diffs = onp.array(diffs)
        x = np.squeeze(self.sweep_setup[2])

        fig,ax = plt.subplots(1,1) if ax is None else (ax.figure,ax)

        plot_defaults = {'color':grey(),'zorder':0}
        plot_kws = {**plot_defaults,**plot_kws} if plot_kws is not None else plot_defaults


        ax.plot(x,diffs, **plot_kws)

        scatter_defaults = {'c':diffs,'cmap':FAM_HEX_cmap(),
                          'vmin':-1,'vmax':1,
                          's':10**2, 'edgecolor':grey(),
                          'zorder':1,
                          }
        scatter_kws = {**scatter_defaults,**scatter_kws} if scatter_kws is not None else scatter_defaults

        ax.scatter(x,diffs, **scatter_kws)

        plt.setp(ax,**{
            #'ylim' : [-1.05,1.05],
            #'title' : '{:s}-{:s} after {:d} cycles'.format(self.labels[0],self.labels[1],self.cycles),
            'ylabel' : 'Signal Difference',
            'xlabel' : f'log10 '+ str(self.INTs[0]) +' copies',
        })

    def plot_2D_solution(self, diffs=None, ax=None, crosshairs=[5,5], contour=True, cbar=True, pmesh_kws=None, contour_kws=None, cbar_kws=None):
        diffs = self.solution_sweep() if diffs is None else diffs
        diffs = onp.array(diffs)
        rng = self.sweep_rng
        res = self.sweep_res
        rng = np.arange(rng[0],rng[1]+res,res)
        ext = np.ceil(onp.max([np.max(diffs),np.abs(np.min(diffs))])*2)/2
        fig,ax = plt.subplots(1,1) if ax is None else (ax.figure,ax)

        pmesh_defaults = {'cmap':FAM_HEX_cmap(),
                          'vmin':-ext,'vmax':ext,
                          'shading':'gouraud'
                          }
        pmesh_kws = {**pmesh_defaults,**pmesh_kws} if pmesh_kws is not None else pmesh_defaults

        pcm = ax.pcolormesh(rng,rng,diffs, **pmesh_kws)

        if crosshairs[0] is not None:
            ax.axvline(crosshairs[0],color='w',linestyle=':')
        if crosshairs[1] is not None:
            ax.axhline(crosshairs[1],color='w',linestyle=':')

        cbar_defaults = {'ticks':np.arange(pmesh_kws['vmin'],pmesh_kws['vmax']+0.5,0.5)}
        cbar_kws = {**cbar_defaults,**cbar_kws} if cbar_kws is not None else cbar_defaults
        cbar = plt.colorbar(pcm, ax=ax, **cbar_kws)
        #, ax = axs[1],extend=extend,ticks=np.arange(-10,10.1,2.5))
        #cbar_x0s.ax.set_ylabel('$log_{10}$ Tar/Ref Ratio\nProviding Signal Parity',fontsize=16)
        #cbar_x0s.ax.tick_params(labelsize=16)

        contour_defaults = {'colors':'k'}
        contour_kws = {**contour_defaults,**contour_kws} if contour_kws is not None else contour_defaults

        if contour:
            cntr = ax.contour(rng,rng,diffs,**contour_kws)
            plt.clabel(cntr, inline=True, fontsize=16, fmt = '%.1f');
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('log10 '+str(self.INTs[0]))
        ax.set_ylabel('log10 '+str(self.INTs[1]))
        return pcm

    def plot_solution_sweep(self,**kws):
        if self.n_INTs==1:
            return self.plot_1D_solution(**kws)
        elif self.n_INTs==2:
            return self.plot_2D_solution(**kws)

    def plot_objective(self, **kws):
        if self.n_INTs==1:
            return self.plot_1D_solution(diffs=self.obj, **kws)
        elif self.n_INTs==2:
            kws.setdefault('contour',False)
            return self.plot_2D_solution(diffs=self.obj, **kws)


    def simplifyRxn(self):
        rxn_tpl = self.cm, self.labels
        return self._simplifyRxn(rxn_tpl,self.n_INTs)

    def BDA(self, yields=None, disable_checks=False):
        return BDA(self, yields=yields, disable_checks=disable_checks)



    ##########################################################################
    ## Learning Methods
    ##########################################################################
    def Learn(self, obj, disable_checks=False, compile_eqns=True):

        if not disable_checks:
            arrays, grids, pts, oligos = self.sweep_setup
            assert obj.shape == grids[0].shape

        self.obj = obj
        self.fit_params = self.copies_rates_nMs[self.n_INTs:]
        self.fixed_params = onp.zeros(self.fit_params.shape, dtype=bool)

        if compile_eqns: self.compileEquations()

        return self

    def compileLoss(self):
        if self.obj is None: pass
        self.loss = jax.jit(self.loss)
        self.loss(self.fit_params,self.connections,self.labels)

    @staticmethod
    def normalize(val, bound):
        return (val-bound[0])/(bound[1]-bound[0])

    @staticmethod
    def unnormalize(val, bound):
        return val*(bound[1]-bound[0])+bound[0]

    def norm_params(self,params, n_i, n_o, n_p):
        if self.obj is None: pass
        return np.hstack([self.normalize(params[:n_o-n_i],self.oligo_bounds),
                          self.normalize(params[n_o-n_i:-n_p],self.rate_bounds),
                          self.normalize(params[-n_p:],self.primer_bounds)])

    def unnorm_params(self,params, n_i, n_o, n_p):
        if self.obj is None: pass
        return np.hstack([self.unnormalize(params[:n_o-n_i],self.oligo_bounds),
                          self.unnormalize(params[n_o-n_i:-n_p],self.rate_bounds),
                          self.unnormalize(params[-n_p:],self.primer_bounds)])


    def _loss_full(self,params, obj, cycles, cm, labels, arrays, grids, pts, oligos):
        oligos=np.array(oligos)
        n_o, n_p = cm.shape
        n_s = n_o*2

        n_dims = n_o*2+n_p-len(params)
        cut = n_o-n_dims

        oligo_copies = self.unnormalize(params[:cut],self.oligo_bounds)
        oligo_rates = self.unnormalize(params[cut:-n_p],self.rate_bounds)
        strand_rates = np.reshape(np.tile(oligo_rates,[2,1]),[self.n_strands,1],order='F')
        primer_nMs = self.unnormalize(params[-n_p:],self.primer_bounds)

        fixed_copies = 10**np.squeeze(np.reshape(np.tile(oligo_copies,[2,1]),[cut*2,1],order='F'))
        sweep_strands = np.squeeze(np.reshape(np.vstack([oligos*2,oligos*2+1]),[2*len(oligos),1],order='F'))
        fixed_strands = jax.ops.index_update(np.ones(n_s),sweep_strands,0)
        strand_copies = jax.ops.index_update(fixed_strands,fixed_strands.astype(bool),fixed_copies)

        primer_copies = molar2copies(primer_nMs*1e-9)
        copies = np.hstack([strand_copies,primer_copies])
        pred = self.solution_sweep(copies, cycles, strand_rates, cm, labels, arrays, grids, pts, oligos)
        return np.sqrt(np.mean(np.square(obj-pred)))

    def loss_full(self, params=None, obj=None, cycles=None, cm=None, labels=None, arrays=None, grids=None, pts=None, oligos=None):

        if self.obj is None: pass

        params = self.fit_params if params is None else params

        obj = self.obj if obj is None else obj

        cycles = np.arange(self.cycles+1, dtype=float) if cycles is None else cycles

        cm = self.connections if cm is None else cm

        labels = self.labels if labels is None else labels

        if None in [arrays, grids, pts, oligos]:
            arrays, grids, pts, oligos = self.sweep_setup

        return self._loss_full(params, obj, cycles, cm, labels, arrays, grids, pts, oligos)


    def loss(self, params, connections, labels):
        if self.obj is None: pass
        return self._loss_full(params, self.obj, np.arange(self.cycles+1, dtype=float), connections, labels, *self.sweep_setup)


    def fit(self, init_params=None, loss=None, method='L-BFGS-B', jac=True):
        if self.obj is None: pass

        init_params = self.fit_params if init_params is None else init_params
        init_params = self.norm_params(init_params, self.n_INTs, self.n_oligos, self.n_primers)

        _loss = self.loss if loss is None else loss

        loss = lambda params: _loss(params,self.connections,self.labels)

        jac = lambda x: onp.array(jax.grad(loss)(x)) if jac else None

        bounds = [(0,1) if not self.fixed_params[p] else (init_params[p],init_params[p]) for p in range(len(self.fit_params))]

        self.fit_result = minimize(loss, init_params, method=method, bounds=bounds, jac=jac)
        self.fit_result.x = self.unnorm_params(self.fit_result.x, self.n_INTs, self.n_oligos, self.n_primers)
        self.copies_rates_nMs = np.hstack([np.repeat(5,self.n_INTs),self.fit_result.x])
        return self.fit_result

    def fix_param(self,param):
        if type(param) is int: # treat as an index
            self.fixed_params[param] = True
        elif param == 'primers': # fix primer concentrations
            self.fixed_params[-self.n_primers:] = True
        elif param == 'rates': # fix oligo rates
            self.fixed_params[self.n_EXTs:-self.n_primers] = True
        elif param == 'copies': # fix oligo copies
            self.fixed_params[:self.n_EXTs] = True
        elif param == 'none': # fix everything
            self.fixed_params = onp.ones(self.fit_params.shape, dtype=bool)
        elif type(param) is list:
            # If size exactly matches number of parameters, treat as replacement for fixed_params
            if all(param.shape == self.fixed_params.shape) and all(type(p) is bool for p in param):
                self.fixed_params = param
            else: # otherwise treat as list of individual indices or names
                for p in param:
                    self.fix_param(p)

    def release_params(self,param):
        if type(param) is int: # treat as an index
            self.fixed_params[param] = False
        elif param == 'primers': # fix primer concentrations
            self.fixed_params[-self.n_primers:] = False
        elif param == 'rates': # fix oligo rates
            self.fixed_params[self.n_EXTs:-self.n_primers] = False
        elif param == 'copies': # fix oligo copies
            self.fixed_params[:self.n_EXTs] = False
        elif param == 'all': # fix everything
            self.fixed_params = onp.zeros(self.fit_params.shape, dtype=bool)
        elif type(param) is list:
            # If size exactly matches number of parameters, treat as replacement for fixed_params
            if all(param.shape == self.fixed_params.shape) and all(type(p) is bool for p in param):
                self.fixed_params = param
            else: # otherwise treat as list of individual indices or names
                for p in param:
                    self.release_params(p)

class BDA(CAN):


    def __init__(self, INT_connections, EXT_connections, labels=None, yields=None,
                 INT_names=None, EXT_names=None, label_names=None, primer_names=None,
                 setup=True, sweep_res=1., sweep_rng=[1.,9.],
                 compile_eqns=True, disable_checks=False):

        super(BDA, self).__init__(INT_connections, EXT_connections, labels=labels,
                                  INT_names=INT_names, EXT_names=EXT_names,
                                  label_names=label_names, primer_names=primer_names,
                                  setup=setup, sweep_res=sweep_res, sweep_rng=sweep_rng,
                                  compile_eqns=False, disable_checks=disable_checks)


        if (not disable_checks) & (yields is not None):
            assert yields.shape == (self.n_strands,), 'Yields must be a 1D vector of length n_strands'

        self.yields = np.ones(self.n_strands) if yields is None else yields


        if compile_eqns:
            self.compileEquations()


    def __str__(self):
        summary = {'Connections':self.connections,
                   'Labels':self.labels,
                   'Oligos (log copies)': onp.log10(self.oligo_copies),
                   'Rates (base 2)':self.rates,
                   'Primers (nM)':self.primer_nMs,
                   'Strand Yields':self.yields,
                   }
        m = max(map(len, list(summary.keys()))) + 1
        return '\n'.join([k.rjust(m) + ': ' + onp.array2string(onp.array(v),prefix=(k.rjust(m)+': '))
                  for k, v in sorted(summary.items())])

    @property
    def strand_rates(self):
        return self.yields[:,None]*np.reshape(np.tile(self.rates,[2,1]),[self.n_strands,1],order='F')




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
