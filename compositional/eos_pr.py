from constants import PREosConstants as prconstants
import numpy as np
from astropy import units as u
"""values of components must be in list type or np.ndarray



"""

def get_reduced_value(value, critical_value):
    """primary

    Args:
        value (_type_): value of phase : Pressure or temperature
        critical_value (_type_): value of component: critical pressure or critical temperature

    Returns:
        _type_: reduced value
    """
    return value/critical_value
    
def get_lambda_component(ac):
    """primary

    Args:
        ac (_type_): acentric value of component

    Returns:
        _type_: _description_
    """
    return 0.37464 + 1.5423*ac - 0.26992*(ac**2)

def get_alpha_component(reduced_temperature, lambda_component):
    """_summary_

    Args:
        reduced_temperature (_type_): value from function get_reduced_value(T, Tc)
        lambda_component (_type_): value from function get_lambda_component

    Returns:
        _type_: _description_
    """
    
    return (1 + lambda_component*(1 - np.sqrt(reduced_temperature)))**2

def get_a_component(alpha_component, Tc, Pc):
    """_summary_

    Args:
        alpha_component: value from function get_alpha_component
        Tc (_type_): critical Temperature of component
        Pc (_type_): critical Pressure of component

    Returns:
        _type_: _description_
    """
    return prconstants.Omega_a*(prconstants.R**2)*alpha_component*(Tc**2)/Pc

def get_b_component(Tc, Pc):
    """_summary_

    Args:
        Tc (_type_): critical Temperature of component
        Pc (_type_): critical Pressure of component

    Returns:
        _type_: _description_
    """
    return prconstants.Omega_b*prconstants.R*Tc/Pc

def get_n_components(component_value):
    return len(component_value)

def get_aijsqrt(a_component):
    """_summary_

    Args:
        a_component (_type_): value from function get_a_component

    Returns:
        _type_: _description_
    """
    n_components = get_n_components(a_component)
    ai = a_component.reshape(n_components, 1)
    aij = np.matmul(ai, ai.T)
    return np.sqrt(aij)

def get_a_mix(molar_fraction, n_components, bincoef, a_component):
    """_summary_

    Args:
        molar_fraction (_type_): 
        n_components (_type_): 
        bincoef (_type_): binary interaction coefs
        a_component (_type_): value from function get_a_component

    Returns:
        _type_: _description_
    """
    molar_frac = molar_fraction.reshape(n_components, 1)
    xij = np.matmul(molar_frac, molar_frac.T)
    bincoefneg = 1 - bincoef
    aijsqrt = get_aijsqrt(a_component)
    resp = (xij*bincoefneg*aijsqrt).sum()
    return resp

def get_b_mix(molar_fraction, b_component):
    """_summary_

    Args:
        molar_fraction (_type_): _description_
        b_component (_type_): value from function get_b_component

    Returns:
        _type_: _description_
    """
    resp = (molar_fraction*b_component).sum()       
    return resp

def get_A_mix(a_mix, P, T):
    """_summary_

    Args:
        a_mix (_type_): value from get_a_mix
        P (_type_): fluid Pressure
        T (_type_): fluid Temperature

    Returns:
        _type_: _description_
    """
    return (a_mix*P)/((prconstants.R**2)*(T**2))

def get_B_mix(b_mix, P, T):
    """_summary_

    Args:
        b_mix (_type_): value from function get_b_mix
        P (_type_): fluid Pressure
        T (_type_): fluid Temperature

    Returns:
        _type_: _description_
    """
    return b_mix*P/(prconstants.R*T)

def get_cubic_coefs(A_mix, B_mix):
    """_summary_

    Args:
        A_mix (_type_): value from function get_A_mix
        B_mix (_type_): value from function get_B_mix

    Returns:
        _type_: _description_
    """
    m3 = 1
    m2 = -(1 - B_mix)
    m1 = A_mix - 2*B_mix - 3*B_mix**2
    m0 = -(A_mix*B_mix - B_mix**2 - B_mix**3) 
    resp = np.array([m3, m2, m1, m0])
    return resp

def roots(cubic_coefs):
    """_summary_

    Args:
        cubic_coefs (_type_): value from function get_cubic_coefs

    Returns:
        _type_: _description_
    """
    coefs = cubic_coefs
    roots = np.roots(coefs)
    real_roots = np.sort(roots[np.isreal(roots)])
    return real_roots

def get_ln_of_coef_fugactiy(Z, A_mix, B_mix, b_component, a_mix, b_mix, bincoef, aijsqrt, molar_fraction):
    """_summary_

    Args:
        Z (_type_): roots from function roots
        A_mix (_type_): _description_
        B_mix (_type_): _description_
        b_component (_type_): _description_
        a_mix (_type_): _description_
        b_mix (_type_): _description_
        bincoef (_type_): binary interaction coef
        aijsqrt (_type_): _description_
        molar_fraction (_type_): 

    Returns:
        _type_: _description_
    """
    n_resp = len(Z)
    n_components = get_n_components(b_component)
    ln_coef_fug = np.zeros((n_resp, n_components))
    kij_neg = 1 - bincoef
    molar_frac = molar_fraction
    mmatrix_comp = (molar_frac*kij_neg*aijsqrt).sum(axis=1)
    
    k0 = -(A_mix/(2*np.sqrt(2)*B_mix))
    k1 = (2/a_mix)*(mmatrix_comp) - (b_component/b_mix)
    
    for i in range(n_resp):
        t0 = (b_component/b_mix)*(Z - 1) - np.log(Z - B_mix)
        k2 = np.log((Z + (1 + np.sqrt(2))*B_mix)/(Z - (1 - np.sqrt(2))*B_mix))
        ln_coef_fug[i][:] = t0 + k0*k1*k2
    
    return ln_coef_fug

def get_coef_fugacity(ln_coef_fugacity):
        coef_fugacity = np.exp(ln_coef_fugacity)
        return coef_fugacity

def get_fugacity(coef_fugacity, P, molar_fraction):
    return P*coef_fugacity*molar_fraction

if __name__ == '__main__':
    
    components_properties = {
        'name_components': ['methane', 'ethane', 'propane'],
        'Tc': np.array([190.56, 305.32, 369.83])*u.K,
        'Pc': np.array([4.599, 4.872, 4.248])*u.megaPascal,
        'bincoef': np.array([
            [0, 0.00340, 0.01070],
            [0.00340, 0, 0.009],
            [0.01070, 0.009, 0]
        ]),
        'ac': np.array([0.011, 0.099, 0.152])
    }
    
    mix_properties = {
        'P': np.array([0.1])*u.megaPascal, 
        'T': np.array([233.2])*u.K,
        'molar_fraction': np.array(
            [0.1, 0.3, 0.6]
        ),
        'phase_type': np.array(['L'])
    }
    
    n_components = len(components_properties['Tc'])
    
    reduced_temperature = get_reduced_value(
        mix_properties['T'], 
        components_properties['Tc']
    )
    
    reduced_pressure = get_reduced_value(
        mix_properties['P'], 
        components_properties['Pc']
    )
    
    lambda_component= get_lambda_component()
    
    alpha_component = get_alpha_component(
        reduced_temperature,
        
        
    )
    
    a_component = get_a_component(
    
    )
    
    a_mix = get_a_mix(
        mix_properties['molar_fraction'],
        n_components,
        components_properties['bincoef'],
        
        
    )
    
    
    
    
    
    import pdb; pdb.set_trace()
    


    
    