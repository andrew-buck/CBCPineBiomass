import math

def SP_Sabatia(dbh, x):
    exp1 = math.exp(-1.36184-(0.6517*x))
    exp2 = math.exp(-8.37006)
    exp3 = math.exp(-5.35995)
    dbh1 = dbh**2.134146
    dbh2 = dbh**(3.553264-0.11295*x)
    dbh3 = dbh**(2.153901-0.10494*x)
    x1 = exp1*dbh1
    x2 = exp2*dbh2
    x3 = exp3*dbh3
    return x1+x2+x3

def SP_Clark(dbh, h):
    # CONVERT TO METRIC, EQUATION OUTPUT IS POUNDS
    x1 = ((dbh**2)*h)**1.11886
    x2 = 10**(-1.52244)
    return x1+x2

def LP(dbh):
    x1 = dbh**2.676835
    c = 0.037403
    return c*x1

def LP_h(dbh,h):
    x1 = dbh**2.015144
    x2 = h**0.864052
    c=0.026256
    return c*x1*x2

def LP_a(dbh,h,age):
    x1 = dbh**2.082279
    x2 = h**1.081437
    x3 = age**-0.205654
    c = 0.020594
    return c*x1*x2*x3

class Biomass:
    def __init__(self, df):
        ...
    
    def calculate_biomass(data_arr):
        ...