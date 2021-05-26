def mu(b, I, mu0, mu1):
    """
    Recovery rate.
    
    Parameters:
    -----------
    I
        Number of infective persons 
    b
        Hospital beds per 10,000 persons
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    """
    # recovery rate, depends on mu0, mu1, b
    mu = mu0 + (mu1 - mu0) * (b/(I+b))
    return mu

def R0(beta, d, nu, mu1):
    """
    Basic reproduction number.
    
    Parameters:
    -----------
    beta
        Average number of adequate contacts per unit time with infectious individuals
    d
        Hospital beds per 10,000 persons
    nu
        Disease induced death rate
    mu1
        Maximum recovery rate
    """
    return beta / (d + nu + mu1)

def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for bifurcations.
    
    Parameters:
    -----------
    I
        Number of infective persons 
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        Average number of adequate contacts per unit time with infectious individuals
    A
        Recruitment rate of susceptibles (e.g. birth rate)
    d
        Natural death rate
    nu
        Disease induced death rate
    b
        Hospital beds per 10,000 persons
    """
    c0 = b**2 * d * A
    c1 = b * ((mu0-mu1+2*d) * A + (beta-nu)*b*d)
    c2 = (mu1-mu0)*b*nu + 2*b*d*(beta-nu)+d*A
    c3 = d*(beta-nu)
    res = c0 + c1 * I + c2 * I**2 + c3 * I**3
    return res
    

def model(t, y, mu0, mu1, beta, A, d, nu, b):
    """
    SIR model including hospitalization and natural death.
    
    Parameters:
    -----------
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        Average number of adequate contacts per unit time with infectious individuals
    A
        Recruitment rate of susceptibles (e.g. birth rate)
    d
        Natural death rate
    nu
        Disease induced death rate
    b
        Hospital beds per 10,000 persons
    """
    S,I,R = y[:]
    m = mu(b, I, mu0, mu1)
    
    dSdt = -S # add the correct model here
    dIdt = 0
    dRdt = 0
    
    return [dSdt, dIdt, dRdt]