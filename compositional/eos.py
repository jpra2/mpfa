import numpy as np
from typing import List
from data_manager import DataManager
from astropy import units as u
from constants import PREosConstants as prconstants
from copy import deepcopy
    

class PengRobinson:
    
    def initialize(self, text=''):
        """_summary_

        Args:
            text (str, optional): text for save properties. Defaults to ''.
            phase_type (str, optional): L: liquid, V: gas. Defaults to 'L'.
        """
        self.component_properties = DataManager()
        self.component_properties.initialize_manager(description=text + '_' + 'components_properties')
        self.mix_properties = DataManager()
        self.mix_properties.initialize_manager(description=text + '_' + 'mix_properties')
        
    
    def set_components(self, name_components: List[str]=[], vc: List[float]=[], pc: List[float]=[], mw: List[float]=[], tc: List[float]=[], bincoef: np.ndarray=[], ac: List[float]=[]):
        """_summary_

        Args:
            name (List[str]): name of components
            vc (List[float]): critical volume [m^3/mol]
            pc (List[float]): critical pressure [Pascal]
            mw (List[float]): molar weight [Kg/mol]
            tc: (List[float]): critical temperature [K]
            bin (np.ndarray): binary interaction coefficient
            ac: (List[float]): acentric factor
        """
        
        self.component_properties.update({
            'name_components': np.array(name_components),
            'vc': vc,
            'pc': pc,
            'tc': tc,
            'bincoef': bincoef,
            'ac': ac,
            'mw': mw
        })
    
    def set_mix_properties(self, p, t, molar_fraction, phase_type):
        """_summary_

        Args:
            p (_type_): pressure
            t (_type_): temperature
            molar_fraction (_type_): molar composition of components
            phase_type (_type_): L: liquid, V: gas
        """
        if phase_type[0] not in ['L', 'V']:
            raise ValueError('phase_type must be L or V')
        
        self.mix_properties.update({
            'p': p,
            't': t,
            'molar_fraction': molar_fraction,
            'phase_type': phase_type
        })
    
    def export(self):
        self.component_properties.export_to_npz()
        self.mix_properties.export_to_npz()
    
    def load(self):
        self.component_properties.load_from_npz()
        self.mix_properties.load_from_npz()
    
    @property
    def name_components(self):
        return self.component_properties['name_components']
    
    @property
    def vc(self):
        return self.component_properties['vc']
    
    @property
    def pc(self):
        return self.component_properties['pc']
    
    @property
    def tc(self):
        return self.component_properties['tc']
    
    @property
    def bincoef(self):
        return self.component_properties['bincoef']
    
    @property
    def ac(self):
        return self.component_properties['ac']
    
    @property
    def p(self):
        return self.mix_properties['p']
    
    @property
    def t(self):
        return self.mix_properties['t']
    
    @property
    def molar_fraction(self):
        return self.mix_properties['molar_fraction']
    
    @property
    def phase_type(self):
        return self.mix_properties['phase_type']
    
    @property
    def reduced_temperature(self):
        return self.t/self.tc
    
    @property
    def reduced_pressure(self):
        return self.p/self.pc
    
    @property
    def lambda_component(self):
        return 0.37464 + 1.5423*self.ac - 0.26992*(self.ac**2)
    
    @property
    def alpha_component(self):
        return (1 + self.lambda_component*(1 - np.sqrt(self.reduced_temperature)))**2
    
    @property
    def a_component(self):        
        return prconstants.Omega_a*(prconstants.R**2)*self.alpha_component*(self.tc**2)/self.pc
    
    @property
    def b_component(self):
        return prconstants.Omega_b*prconstants.R*self.tc/self.pc
    
    @property
    def n_components(self):
        return len(self.name_components)
    
    @property
    def aijsqrt(self):
        ai = self.a_component.reshape(self.n_components, 1)
        aijsqrt = np.sqrt(np.matmul(ai, ai.T))
        return aijsqrt
    
    @property
    def a_mix(self):
        molar_frac = self.molar_fraction.reshape(self.n_components, 1)
        xij = np.matmul(molar_frac, molar_frac.T)
        bincoefneg = 1 - self.bincoef
        aijsqrt = self.aijsqrt
        resp = (xij*bincoefneg*aijsqrt).sum()
        return resp
    
    @property
    def b_mix(self):
        resp = (self.molar_fraction*self.b_component).sum()       
        return resp
    
    @property
    def A_mix(self):
        return (self.a_mix*self.p)/((prconstants.R**2)*self.t**2)
    
    @property
    def B_mix(self):
        return self.b_mix*self.p/(prconstants.R*self.t)
    
    @property
    def cubic_coefs(self):
        m3 = 1
        m2 = -(1 - self.B_mix)
        m1 = self.A_mix - 2*self.B_mix - 3*self.B_mix**2
        m0 = -(self.A_mix * self.B_mix - self.B_mix**2 - self.B_mix**3) 
        resp = np.array([m3, m2.value[0], m1.value[0], m0.value[0]])
        
        return resp
    
    @property
    def root(self):
        coefs = self.cubic_coefs
        roots = np.roots(coefs)
        real_roots = np.sort(roots[np.isreal(roots)])
        if len(real_roots) == 3:
            if self.phase_type[0] == 'L':
                my_root = real_roots.min()
            elif self.phase_type[0] == 'V':
                my_root = real_roots.max()
        else:
            try:
                my_root = real_roots[0]
            except:
                import pdb; pdb.set_trace()
                
            
        return my_root
    
    
    def get_ln_of_coef_fugactiy(self, Z):
        
        b_component = self.b_component
        b_mix = self.b_mix
        B_mix = self.B_mix
        A_mix = self.A_mix
        kij_neg = 1 - self.bincoef
        aijsqrt = self.aijsqrt
        a_mix = self.a_mix
        molar_frac = self.molar_fraction
        mmatrix_comp = (molar_frac*kij_neg*aijsqrt).sum(axis=1)
        
        t0 = (b_component/b_mix)*(Z - 1) - np.log(Z - B_mix)
        k0 = -(A_mix/(2*np.sqrt(2)*B_mix))
        k1 = (2/a_mix)*(mmatrix_comp) - (b_component/b_mix)
        k2 = np.log((Z + (1 + np.sqrt(2))*B_mix)/(Z - (1 - np.sqrt(2))*B_mix))
        
        ln_coef_fug = t0 + k0*k1*k2
                
        return ln_coef_fug
    
    def get_coef_fugacity(self, Z):
        ln_coef_fugacity = self.get_ln_of_coef_fugactiy(Z)
        coef_fugacity = np.exp(ln_coef_fugacity)
        return coef_fugacity
    
    def get_fugacity(self, Z):
        coef_fugacity = self.get_coef_fugacity(Z)
        return self.p*coef_fugacity*self.molar_fraction
    
    @property
    def fugacity(self):
        Z = self.root
        return self.get_fugacity(Z)
    
    @property
    def constant_equilibrium_guess(self):
        ki = (1/self.reduced_pressure)*np.exp(5.3727*(1 + self.ac)*(1 - 1/self.reduced_temperature))
        return ki.value
    
    @property
    def coef_fugacity(self):
        Z = self.root
        return self.get_coef_fugacity(Z)
    
    @property
    def mix_properties_data(self):
        return self.mix_properties.get_data()
    
    @property
    def component_properties_data(self):
        return self.component_properties.get_data()
    
    @property
    def ln_coef_fugacity(self):
        Z = self.root
        return self.get_ln_of_coef_fugactiy(Z)
        
        
class PhasesEquilibrium:
        
    def get_equilibrium_constants(self, coef_fugL, coef_fugV):
        
        """_summary_

        Args:
            coef_fugL (_type_): fugacity coef of liquid phase
            coef_fugV (_type_): fugacity coef of gas phase

        Returns:
            _type_: _description_
        """
        k = coef_fugL/coef_fugV
        return k
    
    def get_gas_equilibrium_molar_fraction(self, liquid_phase: PengRobinson, gas_phase: PengRobinson,tol=1e-4, maxit=1000):
        
        test = True
        count = 0
        coefs_fugacityL = liquid_phase.coef_fugacity
        xL = liquid_phase.mix_properties['molar_fraction']
        
        gas_phase_properties = gas_phase.mix_properties
        
        while test:
            coefs_fugacityV = gas_phase.coef_fugacity
            k_comp = self.get_equilibrium_constants(coefs_fugacityL, coefs_fugacityV)
            xV = xL*k_comp
            xV_scaled = xV/(xV.sum())
            gas_phase_properties.update({
                'molar_fraction': xV_scaled
            })
            gas_phase.set_mix_properties(**gas_phase_properties)
            delta = np.absolute(xV - xV_scaled).max()
            if delta <= tol:
                test = False
            if count > maxit:
                raise RuntimeError('Max iteration raised')
            count += 1
            import pdb; pdb.set_trace()
    
    def stability_test(self, phaseL: PengRobinson, phaseV: PengRobinson):
        
        single_phase = False
        
        fLini = phaseL.fugacity
        fVini = phaseV.fugacity
        initial_mix_properties = phaseL.mix_properties_data
        ki = phaseL.constant_equilibrium_guess
        
        mix_properties = deepcopy(initial_mix_properties)
        
        trivial_testV = False
        conv_testV = False
        
        trivial_testL = False
        conv_testL = False
        
        
        test = True
        
        while test:
            
            molar_fraction = mix_properties['molar_fraction']
            Yi = molar_fraction*ki
            Sv = Yi.sum()
            yi = Yi/Sv
            
            mix_properties.update({
                'molar_fraction': yi
            })
            
            phaseV.set_mix_properties(**mix_properties)

            fV = phaseV.fugacity            
            Ri = (fLini/fV)*(1/Sv)
            ki = ki*Ri
    
            trivial_testV = self.trivial_tol_test(ki)
            
            if trivial_testV:
                test = False
            
            conv_testV = self.convergence_tol_test(Ri)
            
            if conv_testV:
                test = False
        
        mix_properties = deepcopy(initial_mix_properties)
        
        phaseV.set_mix_properties(**initial_mix_properties)
        
        test = True
        
        ki = phaseV.constant_equilibrium_guess
        while test:
            
            molar_fraction = mix_properties['molar_fraction']
            Yi = molar_fraction*ki
            Sl = Yi.sum()
            yi = Yi/Sl
            
            mix_properties.update({
                'molar_fraction': yi
            })
            
            phaseL.set_mix_properties(**mix_properties)

            fL = phaseL.fugacity            
            Ri = (fL/fVini)*(1/Sl)
            ki = ki*Ri
    
            trivial_testL = self.trivial_tol_test(ki)
            
            if trivial_testL:
                test = False
            
            conv_testL = self.convergence_tol_test(Ri)
            
            if conv_testL:
                test = False
        
        if Sv <= 1 and Sl <= 1:
            single_phase = True
        elif trivial_testV and trivial_testL:
            single_phase = True
        elif trivial_testV and Sl <= 1:
            single_phase = True
        elif trivial_testL and Sv <= 1:
            single_phase = True
        
        return single_phase
    
    def trivial_tol_test(self, ki):
        trivial_tol = 1e-4
        test = (np.log(ki)**2).sum()
        if test <= trivial_tol:
            return True
        else:
            return False
    
    def convergence_tol_test(self, Ri):
        convergence_tol = 1e-10
        test = ((Ri - 1)**2).sum()
        
        if test <= convergence_tol:
            return True
        else:
            return False
    
        
        
    

if __name__ == '__main__':
    phaseL = PengRobinson()
    phaseL.initialize(text='liquid')
    phaseV = PengRobinson()
    phaseV.initialize(text='gas')
    
    components_properties = {
        'name_components': ['methane', 'ethane', 'propane'],
        'tc': np.array([190.56, 305.32, 369.83])*u.K,
        'pc': np.array([4.599, 4.872, 4.248])*u.megaPascal,
        'bincoef': np.array([
            [0, 0.00340, 0.01070],
            [0.00340, 0, 0.009],
            [0.01070, 0.009, 0]
        ]),
        'ac': np.array([0.011, 0.099, 0.152])
    }
    
    mix_properties = {
        'p': np.array([0.1])*u.megaPascal, 
        't': np.array([233.2])*u.K,
        'molar_fraction': np.array(
            [0.1, 0.3, 0.6]
        ),
        'phase_type': np.array(['L'])
    }
    
    phaseL.set_components(**components_properties)
    phaseL.set_mix_properties(**mix_properties)
    
    mix_properties.update({
        'phase_type': np.array(['V'])
    })
    
    phaseV.set_components(**components_properties)
    phaseV.set_mix_properties(**mix_properties)
    
    phase_equilibrium = PhasesEquilibrium()
    print(phase_equilibrium.stability_test(phaseL, phaseV))
    
    
    
    import pdb; pdb.set_trace()
    
    
    
    
    
    # phases_equilibrium = PhasesEquilibrium()
    # phases_equilibrium.get_gas_equilibrium_molar_fraction(phase)
    
    # print(phase.molar_fraction)
    
    import pdb; pdb.set_trace()
    
    
    
        
        