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

# analytical reconstruction functions (work fully working)

# Define the Levi-Civita symbol in 4D
def levi_civita_4d():
    """Returns the 4D Levi-Civita tensor."""
    epsilon = np.zeros((4, 4, 4, 4), dtype=int)
    indices = [0, 1, 2, 3]

    for perm in np.array(np.meshgrid(*[indices]*4)).T.reshape(-1, 4):
        i, j, k, l = perm
        sign = np.sign(np.linalg.det([[1 if x == y else 0 for x in indices] for y in perm]))
        epsilon[i, j, k, l] = sign

    return epsilon

def compute_q(M2, p_a, p_b, p_c):
    """
    Computes p^mu = ε_{μνρσ} p_a^ν p_b^ρ p_c^σ / M2
    """
    epsilon = levi_civita_4d()
    p_mu = np.zeros(4)

    # Loop over μ, ν, r, s
    for m in range(4):
        for n in range(4):
            for r in range(4):
                for s in range(4):
                    # Contribution from ε_{μνρσ} * p_a^ν * p_b^ρ * p_c^ρσ
                    p_mu[m] += epsilon[m, n, r, s] * p_a[n] * p_b[r] * p_c[s]

    return ROOT.TLorentzVector(p_mu[0]/M2, p_mu[1]/M2, p_mu[2]/M2, p_mu[3]/M2)

def ReconstructTauAnalytically(P_Z, P_taupvis, P_taunvis):
    '''
    Reconstuct tau lepton 4-momenta with 2-fold degeneracy
    following formulation in Appendix C of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.107.093002
    '''

    m_tau = 1.775538
    m_pi = 0.13957018

    m_taupvis = P_taupvis * P_taupvis
    m_taunvis = P_taunvis * P_taunvis

    q = compute_q(P_Z*P_Z,P_Z, P_taupvis, P_taunvis)

    #print('check q:') # these should all equal 0:
    #print(q.Dot(P_Z))
    #print(q.Dot(P_taupvis))
    #print(q.Dot(P_taunvis))

    x = P_Z * P_taupvis
    y = P_Z * P_taunvis
    z = P_taupvis * P_taunvis

    m_Z_sq = P_Z*P_Z

    # note the matrix method below relies on the approximation that the masses of the visible tau decay product in the tau->Xnu are the same for tau+ and tau-
    # which won't be true if taus decay into different channels and is also approximate for rho and a1 decays since these are broad resonances.
    # it is the third row of the matrix that is only correct under this assumption
    M = np.array([[-x,     P_taupvis*P_taupvis, -z     ],
                  [y,      -z,      P_taunvis*P_taunvis],
                  [m_Z_sq, -x,      y]])

    lamx = m_tau**2 + P_taupvis*P_taupvis -x
    lamy = m_tau**2 + P_taunvis*P_taunvis -y

    L = np.array([[lamx],
                  [lamy],
                  [0.]])

    M_inv = np.linalg.inv(M)

    v = np.dot(M_inv, L)

    a = v[0][0]
    b = v[1][0]
    c = v[2][0]

    #dsq = 1./(-4*q*q) * ( (1+a**2)*m_Z_sq + (b**2+c**2)*m_pi**2 - 4*m_tau**2 + 2*(a*c*y - a*b*x - b*c*z) )
    dsq = 1./(-4*q*q) * ( (1+a**2)*m_Z_sq + (b**2+c**2)*(P_taupvis*P_taupvis+P_taunvis*P_taunvis)/2 - 4*m_tau**2 + 2*(a*c*y - a*b*x - b*c*z) )
    d = dsq**.5

    solutions = []

    for i, d in enumerate([d,-d]):
        taup = (1.-a)/2*P_Z + b/2*P_taupvis - c/2*P_taunvis + d*q
        taun = (1.+a)/2*P_Z - b/2*P_taupvis + c/2*P_taunvis - d*q
        solutions.append((taup, taun))
    return solutions


if __name__ == '__main__':

    f = ROOT.TFile('pythia_output_ee_To_pipinunu_no_entanglementMG.root')
    tree = f.Get('tree')
    for i in range(1,10):
        print('\n---------------------------------------')
        print('Event %i' %i)
        tree.GetEntry(i)
        P_taup_true = ROOT.TLorentzVector(tree.taup_px, tree.taup_py, tree.taup_pz, tree.taup_e) 
        P_taun_true = ROOT.TLorentzVector(tree.taun_px, tree.taun_py, tree.taun_pz, tree.taun_e) 
    
        P_taupvis = ROOT.TLorentzVector(tree.taup_pi1_px, tree.taup_pi1_py, tree.taup_pi1_pz, tree.taup_pi1_e)
        P_taunvis = ROOT.TLorentzVector(tree.taun_pi1_px, tree.taun_pi1_py, tree.taun_pi1_pz, tree.taun_pi1_e)
        P_Z = P_taup_true+P_taun_true
        #P_Z = ROOT.TLorentzVector(0.,0.,0.,91.188) # assuming we don't know ISR and have to assume momentum is balanced
   
        P_taup_reco, P_taun_reco = ReconstructTau(P_Z, P_taupvis, P_taunvis, verbose=False)
    
        print('\nTrue taus:')
        print('tau+:', P_taup_true.X(), P_taup_true.Y(), P_taup_true.Z(), P_taup_true.T(), P_taup_true.M())
        print('tau-:', P_taun_true.X(), P_taun_true.Y(), P_taun_true.Z(), P_taun_true.T(), P_taun_true.M())
    
        print('\nReco taus (numerically):')
        print('tau+:', P_taup_reco.X(), P_taup_reco.Y(), P_taup_reco.Z(), P_taup_reco.T(), P_taup_reco.M())
        print('tau-:', P_taun_reco.X(), P_taun_reco.Y(), P_taun_reco.Z(), P_taun_reco.T(), P_taun_reco.M())
        
        solutions = ReconstructTauAnalytically(P_Z, P_taupvis, P_taunvis)
        print('\nReco taus (analytically):')
        print('solution 1:')
        print('tau+:', solutions[0][0].X(), solutions[0][0].Y(), solutions[0][0].Z(), solutions[0][0].T(), solutions[0][0].M())
        print('tau-:', solutions[0][1].X(), solutions[0][1].Y(), solutions[0][1].Z(), solutions[0][1].T(), solutions[0][1].M())
        print('solution 2:')
        print('tau+:', solutions[1][0].X(), solutions[1][0].Y(), solutions[1][0].Z(), solutions[1][0].T(), solutions[1][0].M())
        print('tau-:', solutions[1][1].X(), solutions[1][1].Y(), solutions[1][1].Z(), solutions[1][1].T(), solutions[1][1].M())

        print('\nboost to tau rest frames')
        P_taupvis_1 = P_taupvis.Clone()
        P_taupvis_2 = P_taupvis.Clone()
        P_taupvis_3 = P_taupvis.Clone()
        P_taupvis_1.Boost(-P_taup_true.BoostVector())
        P_taupvis_2.Boost(-solutions[0][0].BoostVector())
        P_taupvis_3.Boost(-solutions[1][0].BoostVector())
       
        P_taunvis_1 = P_taunvis.Clone()
        P_taunvis_2 = P_taunvis.Clone()
        P_taunvis_3 = P_taunvis.Clone()
        P_taunvis_1.Boost(-P_taun_true.BoostVector())
        P_taunvis_2.Boost(-solutions[0][1].BoostVector())
        P_taunvis_3.Boost(-solutions[1][1].BoostVector())       
    
        print('true:')
        print('tau+:', P_taupvis_1.X(), P_taupvis_1.Y(), P_taupvis_1.Z(), P_taupvis_1.T(), P_taupvis_1.M())
        print('tau-:', P_taunvis_1.X(), P_taunvis_1.Y(), P_taunvis_1.Z(), P_taunvis_1.T(), P_taunvis_1.M())
        print('solution 1:')
        print('tau+:', P_taupvis_2.X(), P_taupvis_2.Y(), P_taupvis_2.Z(), P_taupvis_2.T(), P_taupvis_2.M())
        print('tau-:', P_taunvis_2.X(), P_taunvis_2.Y(), P_taunvis_2.Z(), P_taunvis_2.T(), P_taunvis_2.M())
        print('solution 2:')
        print('tau+:', P_taupvis_3.X(), P_taupvis_3.Y(), P_taupvis_3.Z(), P_taupvis_3.T(), P_taupvis_3.M())        
        print('tau-:', P_taunvis_3.X(), P_taunvis_3.Y(), P_taunvis_3.Z(), P_taunvis_3.T(), P_taunvis_3.M())        
