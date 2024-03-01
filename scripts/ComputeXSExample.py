import math

# The example below show how to find the cross-sections for each contribution to the di-Higgs spectrum 
# The example was done for the M=600 GeV samples with relative_width = 0.01

# need to know mass and relative width for k-factor scaling
mass = 600.
rel_width = 0.01

# lo sample cross-sections
# these are the numbers that I got out of the GenXSecAnalyzer under "xsec_before" after commenting these lines: https://github.com/cms-sw/cmssw/blob/402aaa8d730b384a986e3156044f22245d7ab229/GeneratorInterface/Core/plugins/GenXSecAnalyzer.cc#L194-L198 (to prevent it returning XS=0 when it is < 0)
# note I only ran GenXSecAnalyzer for 1 file per sample, so it might be a good idea to run it for more events

xs_box_lo = 2.755e-02 # XS for box diagram
xs_Sh_lo = 3.689e-03 # XS for SM-like s-channel diagram
xs_box_Sh_int_lo = -1.808e-02 # XS for inteference between box and SM-like s-channel diagrams

xs_SH_lo = 7.448e-03 # cross-section for heavy H s-channel diagram (resonant)
xs_SH_Sh_int_lo = -7.043e-04 # XS for inteference between heavy H s-channel SM-like s-channel diagrams
xs_box_SH_int_lo = 1.827e-03 # XS for inteference between box and heavy H s-channel diagrams

# we need to scale up the inteference templates for the heavy H to account for the couplings used in the sample generation
# in the generation kappa_t for both h and H was set to 1/sqrt(2)
# the SH-Sh inteference term is proportional to kappa_t_H*kappa_t_h so we scale up by a factor of 2
xs_SH_Sh_int_lo *= 2.
# the SH-box inteference term is proportional to kappa_t_h**2 * kappa_t_H so we need to scale up by 2**1.5
xs_box_SH_int_lo *= 2**1.5

# at this point we have all the LO values of the cross-sections for all kappas=1

# now we also need to scale the NNLO xs values using the k-factors

# we use nnlo XSs from LHCHWG for computation of k-factors

# non-resonant cross sections taken from https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH?redirectedfrom=LHCPhysics.LHCHXSWGHH#Latest_recommendations_for_gluon
xs_box_nnlo = 0.0703874
xs_Sh_nnlo = 0.0110595
xs_box_Sh_int_nnlo = -0.0504111

# resonant cross-sections taken from: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBSMAt13TeV
# download using:
# wget https://twiki.cern.ch/twiki/pub/LHCPhysics/LHCHWG/Higgs_XSBR_YR4_update.xlsx
# note take numbers from "YR4 BSM 13TeV" tab in NNLO+NNLL columb - the N3LO numbers are computed in the heavy top limit so not valid ~>2*mtop!! 
# number below is for M=600

xs_SH_nnlo = 2.006
# we need to multiply this value for the H->hh BR since LO XS includes this branching fraction also
total_width = rel_width*mass
partial_width =  31.803878252**2*(1.-4*(125./mass)**2)**.5/(8*math.pi*mass) # partial width for lambda_112 = lambda_hhh_SM (as is used in sample generation)
BR = partial_width/total_width
xs_SH_nnlo *= BR

k_box = xs_box_nnlo/xs_box_lo
k_Sh = xs_Sh_nnlo/xs_Sh_lo
k_box_Sh_int = xs_box_Sh_int_nnlo/xs_box_Sh_int_lo
k_SH = xs_SH_nnlo/xs_SH_lo

# since we do not have NNLO for the heavy resonance inteference term we compute these following the Ansatz that they are equal to sqrt(k_i*k_SH), wfor k_i = k_box, k_Sh 
# I ran this by Tania previously and she didn't disagree with it, but it may change in the future after further discussions
k_box_SH_int = (k_SH*k_box)**.5
k_SH_Sh_int = (k_SH*k_Sh)**.5

print 'LO cross-sections (pb):'

print 'box =', xs_box_lo
print 'Sh =', xs_Sh_lo
print 'box_Sh_int =', xs_box_Sh_int_lo
print 'SH =', xs_SH_lo
print 'box_SH_int =', xs_box_SH_int_lo
print 'SH_Sh_int =', xs_SH_Sh_int_lo
print '\n'

print 'K-factors (LO->NNLO):'

print 'k_box =', k_box
print 'k_Sh =', k_Sh
print 'k_box_Sh_int =', k_box_Sh_int
print 'k_SH =', k_SH
print 'k_box_SH_int =', k_box_SH_int
print 'k_SH_Sh_int =', k_SH_Sh_int
print '\n'

print 'NNLO cross-sections (pb):'

print 'box =', k_box*xs_box_lo
print 'Sh =', k_Sh*xs_Sh_lo
print 'box_Sh_int =', k_box_Sh_int*xs_box_Sh_int_lo
print 'SH =', k_SH*xs_SH_lo
print 'box_SH_int =', k_box_SH_int*xs_box_SH_int_lo
print 'SH_Sh_int =', k_SH_Sh_int*xs_SH_Sh_int_lo
