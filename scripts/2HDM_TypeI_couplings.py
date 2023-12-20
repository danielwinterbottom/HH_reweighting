import math

m12sq = 5.5*10**4
tanb = 7.5
mH = 650.
mh = 125.
mA = mH
mHpm = mH

vev = 246.22

def YukawaH(cosbma,tanb):
  sinbma = math.sin(math.acos(cosbma))
  yuk = cosbma + sinbma/tanb
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

for cosbma in [0.1, 0.25]:

  width = None
  if cosbma == 0.1: width = 2.47253022
  if cosbma == 0.25: width =  11.8110613

  yH = YukawaH(cosbma,tanb)
  yh = Yukawah(cosbma,tanb)
  lam111 = Lambda111(cosbma, m12sq, tanb)
  lam112 = Lambda112(cosbma, m12sq, tanb)
  print cosbma, yH, yh, lam111/lam_SM, lam112/lam_SM, width 
