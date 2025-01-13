import ROOT
import numpy as np
from scipy.optimize import minimize

def objective(vars, P_Z, P_taupvis, P_taunvis):
    mtau = 1.775538 
    nubar_x, nubar_y, nubar_z, nu_x, nu_y, nu_z = vars
    P_taupnu = ROOT.TLorentzVector(nubar_x, nubar_y, nubar_z, (nubar_x**2+nubar_y**2+nubar_z**2)**.5) 
    P_taunnu = ROOT.TLorentzVector(nu_x, nu_y, nu_z, (nu_x**2+nu_y**2+nu_z**2)**.5) 
    # tau mass constraints
    eq1 = (P_taupvis+P_taupnu).M() - mtau 
    eq2 = (P_taunvis+P_taunnu).M() - mtau

    # Z boson momentum constraints
    mom_sum = P_taupvis+P_taunvis+P_taupnu+P_taunnu
    eq3 = mom_sum.T() - P_Z.T()
    eq4 = mom_sum.X() - P_Z.X()
    eq5 = mom_sum.Y() - P_Z.Y()
    eq6 = mom_sum.Z() - P_Z.Z()

    return eq1**2 + eq2**2 + eq3**2 + eq4**2 + eq5**2 + eq6**2

def ReconstructTau(P_Z, P_taupvis, P_taunvis, verbose=False):
    initial_guess = [P_taupvis.X(), P_taupvis.Y(), P_taupvis.Z(), P_taunvis.X(), P_taunvis.Y(), P_taunvis.Z()]

    # Call the optimizer
    result = minimize(objective, initial_guess, args=(P_Z, P_taupvis, P_taunvis))

    if verbose:
        if result.success:
            print("Optimization succeeded!")
        else:
            print("Optimization failed.")
 
    P_taupnu_reco = ROOT.TLorentzVector(result.x[0], result.x[1], result.x[2], (result.x[0]**2+result.x[1]**2+result.x[2]**2)**.5)
    P_taunnu_reco = ROOT.TLorentzVector(result.x[3], result.x[4], result.x[5], (result.x[3]**2+result.x[4]**2+result.x[5]**2)**.5)
    P_taup_reco =  P_taupvis + P_taupnu_reco
    P_taun_reco =  P_taunvis + P_taunnu_reco

    return P_taup_reco, P_taun_reco

if __name__ == '__main__':

    f = ROOT.TFile('pythia_output_ee_To_pipinunu_no_entanglementMG.root')
    tree = f.Get('tree')
    for i in range(1,10):
        print('\nEvent %i' %i)
        tree.GetEntry(i)
        P_taup_true = ROOT.TLorentzVector(tree.taup_px, tree.taup_py, tree.taup_pz, tree.taup_e) 
        P_taun_true = ROOT.TLorentzVector(tree.taun_px, tree.taun_py, tree.taun_pz, tree.taun_e) 
    
        P_taupvis = ROOT.TLorentzVector(tree.taup_pi1_px, tree.taup_pi1_py, tree.taup_pi1_pz, tree.taup_pi1_e)
        P_taunvis = ROOT.TLorentzVector(tree.taun_pi1_px, tree.taun_pi1_py, tree.taun_pi1_pz, tree.taun_pi1_e)
        P_Z = P_taup_true+P_taun_true
        #P_Z = ROOT.TLorentzVector(0.,0.,0.,91.188)
   
        P_taup_reco, P_taun_reco = ReconstructTau(P_Z, P_taupvis, P_taunvis, verbose=False)
    
        print('True taus:')
        print('tau+:', P_taup_true.X(), P_taup_true.Y(), P_taup_true.Z(), P_taup_true.T(), P_taup_true.M())
        print('tau-:', P_taun_true.X(), P_taun_true.Y(), P_taun_true.Z(), P_taun_true.T(), P_taun_true.M())
    
        print('Reco taus:')
        print('tau+:', P_taup_reco.X(), P_taup_reco.Y(), P_taup_reco.Z(), P_taup_reco.T(), P_taup_reco.M())
        print('tau-:', P_taun_reco.X(), P_taun_reco.Y(), P_taun_reco.Z(), P_taun_reco.T(), P_taun_reco.M())

