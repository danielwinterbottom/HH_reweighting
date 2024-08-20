import math
import pandas as pd
import numpy as np

# note the widths return in this case take the widths computed for a SM-like Higgs ignoring H->gamgam and H->Zgam and the other widths are rescaled by the coupling modifiers for fermions and bososn (not H->ff widths also includes H->gg as this depends on fermion couplings as well). The partial width is also added for H->hh, but other BSM decays (H->ZA, H->AA, H->WH+/-, and H->H+H-) are ignored which is only valid for some benchmark scenarios e.g when these decay are not kinematically allowed. 

# Load the DataFrame from the .pkl file
df = pd.read_pickle('BSM_higgs_widths.pkl')

#m12sq = 5.5*10**4
#tanb = 7.5
#mH = 650.
#mh = 125.
#mA = mH
#mHpm = mH

#BM before are from https://arxiv.org/pdf/2403.14776
#BM1
#tanb = 4.
#cosbma = 0.05
#mH = 450.
#mh = 125.
#mA = 800.
#mHpm = 800.

tanb = 10.
cosbma = 0.2
mH = 500.
mh = 125.
mA = 545.
mHpm = mA

b = math.atan(tanb)
print('tanb, b = ', tanb, b)
bma = math.acos(cosbma)
a = -bma + b
m12sq = mH**2*math.cos(a)**2/tanb

print('alpha, beta =', a,b)
print('M12^2 = ', m12sq)

vev = 246.22

def interpolateWidths(xval, df, xcol, ycol):
    # compute xval as the linear interpolation of xval where df is a dataframe and
    #  df.x are the x coordinates, and df.y are the y coordinates. df.x is expected to be sorted.
    return np.interp([xval], df[xcol], df[ycol])

Hff_width = interpolateWidths(mH, df, 'mass', 'Hff_width')[0]
HVV_width = interpolateWidths(mH, df, 'mass', 'HVV_width')[0]
Hhh_width = interpolateWidths(mH, df, 'mass', 'Hhh_width')[0]

def YukawaH(cosbma,tanb):
  sinbma = math.sin(math.acos(cosbma))
  yuk = cosbma - sinbma/tanb
  return yuk

def Yukawah(cosbma,tanb):
  sinbma = math.sin(math.acos(cosbma))
  yuk = sinbma + cosbma/tanb
  return yuk

def Lambda111(cosbma, m12sq, tanb):
  b = math.atan(tanb)
  mbarsq = m12sq/(math.sin(b)*math.cos(b))
  sinbma = math.sin(math.acos(cosbma))    

  lam = 1./(2.*vev**2) * (mh**2*sinbma**3 + (3.*mh**2 - 2.*mbarsq)*cosbma**2*sinbma +2./math.tan(2.*b)*(mh**2-mbarsq)*cosbma**3)
  return lam#*vev

def Lambda112(cosbma, m12sq, tanb):
  b = math.atan(tanb)
  mbarsq = m12sq/(math.sin(b)*math.cos(b))
  sinbma = math.sin(math.acos(cosbma))

  lam = -cosbma/(2.*vev**2) * ( (2.*mh**2 + mH**2 -4.*mbarsq)*sinbma**2 + 2./math.tan(2.*b)*(2.*mh**2+mH**2-3.*mbarsq)*sinbma*cosbma - (2.*mh**2+mH**2-2.*mbarsq)*cosbma**2)
  return lam#*vev

lam_SM = 1./(2*vev**2) *mh**2 #* vev


width = None
#if cosbma == 0.1: width = 2.47253022
#if cosbma == 0.25: width =  11.8110613

yH = YukawaH(cosbma,tanb)
yh = Yukawah(cosbma,tanb)
lam111 = Lambda111(cosbma, m12sq, tanb)
lam112 = Lambda112(cosbma, m12sq, tanb)
print('lam112 = ', lam112)

print('widths (unscaled) = %g, %g, %g' % (Hff_width, HVV_width, Hhh_width))

print(yH, cosbma)

Hff_width_scaled = Hff_width*yH**2
HVV_width_scaled = HVV_width*cosbma**2
Hhh_width_scaled = (lam112/lam_SM)**2*Hhh_width

print('!!!!', lam112)

print('widths (scaled) = %g, %g, %g, %g' % (Hff_width_scaled, HVV_width_scaled, Hhh_width_scaled, Hff_width_scaled+HVV_width_scaled+Hhh_width_scaled))

width = Hff_width_scaled+HVV_width_scaled+Hhh_width_scaled

print('cosbma = %g, yH = %g, yh = %g, kaplam111 = %g, kaplam112 = %g, width = %g' %(cosbma, yH, yh, lam111/lam_SM, lam112/lam_SM, width)) 
