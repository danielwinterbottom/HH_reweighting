import ROOT
import numpy as np
from scipy.optimize import minimize
import math

def ChangeFrame(taun, taunvis, vec):
    '''
    Rotate the coordinate axis as follows:
    z-direction should be direction of the tau-
    xz-axis should adjust the direction so that the h- lies on the x-z plane, in the +ve x-direction
    '''

    vec_new = vec.Clone()
    taunvis_copy = taunvis.Clone()
    # Step 1: Define the rotation to align the tau- with the z-axis
    theta = taun.Theta()
    phi = taun.Phi() 
    vec_new.RotateZ(-phi)
    vec_new.RotateY(-theta)

    # Step 2: first need to rotate the the taunvis vector as was done above 
    taunvis_copy.RotateZ(-phi)
    taunvis_copy.RotateY(-theta)

    # Step3: now rotate so that the the taunvis 
    phi2 = taunvis_copy.Phi()
    vec_new.RotateZ(-phi2)

    # the h- should be in the +ve x-direction, add a check for this
    if taunvis == vec and vec_new.X() < 0:
        raise ValueError("h- not pointing in the +ve x direction")
    return vec_new


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

def compare_lorentz_pairs(pair1, pair2):
    """
    Compare two pairs of TLorentzVectors by calculating the sum of squares of differences
    between their x, y, and z components.
    
    Parameters:
    - pair1: tuple of two TLorentzVectors (vector1, vector2)
    - pair2: tuple of two TLorentzVectors (vector3, vector4)
    
    Returns:
    - float: Sum of the squared differences for the x, y, and z components.
    """
    # Extract TLorentzVectors from pairs
    vec1, vec2 = pair1
    vec3, vec4 = pair2

    # Compute the squared differences for each component
    dx = (vec1.X() - vec3.X())**2 + (vec2.X() - vec4.X())**2
    dy = (vec1.Y() - vec3.Y())**2 + (vec2.Y() - vec4.Y())**2
    dz = (vec1.Z() - vec3.Z())**2 + (vec2.Z() - vec4.Z())**2

    # Return the sum of squared differences
    return dx + dy + dz

def GetOpeningAngle(tau, h):
    beta = tau.P()/tau.E()
    if beta >= 1:
        raise ValueError("Beta is >= 1, invalid for physical particles.")
    gamma = 1. / (1. - beta**2)**0.5
    x = h.E()/tau.E()
    r = h.M()/tau.M()

    costheta = (gamma*x - (1.+r**2)/(2*gamma))/(beta*(gamma**2*x**2-r**2)**.5)
    sintheta = (((1.-r**2)**2/4 - (x-(1.+r**2)/2)**2/beta**2)/(gamma**2*x**2-r**2))**.5
    if round(math.acos(costheta), 3) != round(math.asin(sintheta), 3):    
        raise ValueError("theta angles do not match", math.acos(costheta), math.asin(sintheta))

    return math.acos(costheta)

def IPConstraints(taup, taun, hp, hn):
    '''
    Following this paper: https://arxiv.org/pdf/hep-ph/9307269
    '''

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
    d = dsq**.5 if dsq > 0 else 0.

    solutions = []

    for i, d in enumerate([d,-d]):
        taup = (1.-a)/2*P_Z + b/2*P_taupvis - c/2*P_taunvis + d*q
        taun = (1.+a)/2*P_Z - b/2*P_taupvis + c/2*P_taunvis - d*q
        solutions.append((taup, taun))
    return tuple(solutions)

def GetGenImpactParam(primary_vtx, secondary_vtx, part_vec):
    sv = secondary_vtx - primary_vtx # secondary vertex wrt primary vertex    
    unit_vec = part_vec.Unit() # direction of particle / track
    ip = (sv - sv.Dot(unit_vec)*unit_vec)
    return ip    

if __name__ == '__main__':

    f = ROOT.TFile('pythia_output_ee_To_pipinunu_no_entanglementMG.root')
    tree = f.Get('tree')
    count_total = 0
    count_correct = 0
    for i in range(1,10):
        count_total+=1
        tree.GetEntry(i)
        P_taup_true = ROOT.TLorentzVector(tree.taup_px, tree.taup_py, tree.taup_pz, tree.taup_e) 
        P_taun_true = ROOT.TLorentzVector(tree.taun_px, tree.taun_py, tree.taun_pz, tree.taun_e) 
    
        P_taupvis = ROOT.TLorentzVector(tree.taup_pi1_px, tree.taup_pi1_py, tree.taup_pi1_pz, tree.taup_pi1_e)
        P_taunvis = ROOT.TLorentzVector(tree.taun_pi1_px, tree.taun_pi1_py, tree.taun_pi1_pz, tree.taun_pi1_e)
        P_Z = P_taup_true+P_taun_true
        #P_Z = ROOT.TLorentzVector(0.,0.,0.,91.188) # assuming we don't know ISR and have to assume momentum is balanced
  
        # compute IPs from SVs

        # note that the below assuems that taus are produced at 0,0,0 which might not be true for some MC samples! 
        VERTEX_taup = ROOT.TVector3(tree.taup_pi1_vx, tree.taup_pi1_vy, tree.taup_pi1_vz) # in mm
        VERTEX_taun = ROOT.TVector3(tree.taun_pi1_vx, tree.taun_pi1_vy, tree.taun_pi1_vz) # in mm

        l_true = abs((VERTEX_taup-VERTEX_taun).Mag())

        IP_taup = GetGenImpactParam(ROOT.TVector3(0.,0.,0.),VERTEX_taup, P_taupvis.Vect())
        IP_taun = GetGenImpactParam(ROOT.TVector3(0.,0.,0.),VERTEX_taun, P_taunvis.Vect())

        #TODO: this is wrong as you need to get minimum distance between 2 tracks not difference between bin distance between tracks and PV!!!
        d_min_reco = IP_taup - IP_taun

        P_taup_reco, P_taun_reco = ReconstructTau(P_Z, P_taupvis, P_taunvis, verbose=False)
   
        solutions = ReconstructTauAnalytically(P_Z, P_taupvis, P_taunvis)

        d_sol1 = compare_lorentz_pairs((P_taup_reco,P_taun_reco),solutions[0])
        d_sol2 = compare_lorentz_pairs((P_taup_reco,P_taun_reco),solutions[1])
        if d_sol1 < d_sol2: matched_solution = solutions[0]
        else: matched_solution = solutions[1]
        
        d_sol1 = compare_lorentz_pairs((P_taup_true,P_taun_true),solutions[0])
        d_sol2 = compare_lorentz_pairs((P_taup_true,P_taun_true),solutions[1])        
        if d_sol1 < d_sol2: correct_solution = solutions[0]
        else: correct_solution = solutions[1]

        FoundCorrectSolution = (matched_solution == correct_solution)
        if FoundCorrectSolution: count_correct+=1

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

        if i < 10 and False: # only print a few events

            print('\n---------------------------------------')
            print('Event %i' %i)
            print('\nTrue taus:')
            print('tau+:', P_taup_true.X(), P_taup_true.Y(), P_taup_true.Z(), P_taup_true.T(), P_taup_true.M())
            print('tau-:', P_taun_true.X(), P_taun_true.Y(), P_taun_true.Z(), P_taun_true.T(), P_taun_true.M())
    
            print('\nReco taus (numerically):')
            print('tau+:', P_taup_reco.X(), P_taup_reco.Y(), P_taup_reco.Z(), P_taup_reco.T(), P_taup_reco.M())
            print('tau-:', P_taun_reco.X(), P_taun_reco.Y(), P_taun_reco.Z(), P_taun_reco.T(), P_taun_reco.M())
    
            print('\nReco taus (analytically):')
            print('solution 1:')
            print('tau+:', solutions[0][0].X(), solutions[0][0].Y(), solutions[0][0].Z(), solutions[0][0].T(), solutions[0][0].M())
            print('tau-:', solutions[0][1].X(), solutions[0][1].Y(), solutions[0][1].Z(), solutions[0][1].T(), solutions[0][1].M())
            print('solution 2:')
            print('tau+:', solutions[1][0].X(), solutions[1][0].Y(), solutions[1][0].Z(), solutions[1][0].T(), solutions[1][0].M())
            print('tau-:', solutions[1][1].X(), solutions[1][1].Y(), solutions[1][1].Z(), solutions[1][1].T(), solutions[1][1].M())
    
            print('\nboost to tau rest frames')
            print('True:')
            print('tau+:', P_taupvis_1.X(), P_taupvis_1.Y(), P_taupvis_1.Z(), P_taupvis_1.T(), P_taupvis_1.M())
            print('tau-:', P_taunvis_1.X(), P_taunvis_1.Y(), P_taunvis_1.Z(), P_taunvis_1.T(), P_taunvis_1.M())
            print('solution 1:')
            print('tau+:', P_taupvis_2.X(), P_taupvis_2.Y(), P_taupvis_2.Z(), P_taupvis_2.T(), P_taupvis_2.M())
            print('tau-:', P_taunvis_2.X(), P_taunvis_2.Y(), P_taunvis_2.Z(), P_taunvis_2.T(), P_taunvis_2.M())
            print('solution 2:')
            print('tau+:', P_taupvis_3.X(), P_taupvis_3.Y(), P_taupvis_3.Z(), P_taupvis_3.T(), P_taupvis_3.M())        
            print('tau-:', P_taunvis_3.X(), P_taunvis_3.Y(), P_taunvis_3.Z(), P_taunvis_3.T(), P_taunvis_3.M())       
            
            print('correct solution found?', FoundCorrectSolution)

        #GetOpeningAngle(P_taup_true, P_taupvis)
        #P_taunvis_new =  ChangeFrame(P_taun_true, P_taunvis, P_taunvis)
        #P_taupvis_new =  ChangeFrame(P_taup_true, P_taupvis, P_taupvis)
        #P_taun_true_new =  ChangeFrame(P_taun_true, P_taunvis, P_taun_true)
        #P_taup_true_new =  ChangeFrame(P_taup_true, P_taupvis, P_taup_true)

        def PredictDmin(P_taunvis, P_taupvis):
            d_over_l = ROOT.TVector3(0.,0.,-1.) # by definition in the rotated coordinate frame
            n_n = P_taunvis.Vect().Unit()
            n_p = P_taupvis.Vect().Unit()
            fact1 = (d_over_l.Dot(n_p)*n_n.Dot(n_p) - d_over_l.Dot(n_n))/(1.-(n_n.Dot(n_p))**2)
            fact2 = (d_over_l.Dot(n_n)*n_n.Dot(n_p) - d_over_l.Dot(n_p))/(1.-(n_n.Dot(n_p))**2)
            term1 = n_n*fact1
            term2 = n_p*fact2
            d_min = d_over_l + (term1 - term2)
            return d_min

        for i in range(0,2):
            print ('solution %i' % (i+1))

#            #d_min = d_over_l + ( (d_over_l.Dot(n_p)*n_n.Dot(n_p) - d_over_l.Dot(n_n))*n_n - (d_over_l.Dot(n_n)*n_n.Dot(n_p) - d_over_l.Dot(n_p))*n_p )#*1./(1.-(n_n.Dot(n_p))**2)
#            d_min_reco_new = ChangeFrame(solutions[i][1], P_taunvis, d_min_reco) # rotate reco d_min to same frame
            P_taunvis_new = ChangeFrame(solutions[i][1], P_taunvis, P_taunvis)
            P_taupvis_new = ChangeFrame(solutions[i][1], P_taunvis, P_taupvis)
#            P_taun_new = ChangeFrame(solutions[i][1], P_taunvis, solutions[i][1])
#            P_taup_new = ChangeFrame(solutions[i][1], P_taunvis, solutions[i][0])
#            print("pi+:", P_taupvis_new.X(), P_taupvis_new.Y(), P_taupvis_new.Z())
#            print("pi-:", P_taunvis_new.X(), P_taunvis_new.Y(), P_taunvis_new.Z())
#            d_min = PredictDmin(P_taunvis_new, P_taupvis_new)

#            theta_p = GetOpeningAngle(P_taup_new, P_taupvis_new)
#            theta_n = GetOpeningAngle(P_taun_new, P_taunvis_new)
            print('orig true tau-:', P_taun_true.X(), P_taun_true.Y(), P_taun_true.Z())
            print('orig tau-:', solutions[i][1].X(), solutions[i][1].Y(), solutions[i][1].Z())
            print('orig pi-:', P_taunvis.X(), P_taunvis.Y(), P_taunvis.Z())
#            print('tau-:', P_taun_new.X(), P_taun_new.Y(), P_taun_new.Z())
#            print('pi-:', P_taunvis_new.X(), P_taunvis_new.Y(), P_taunvis_new.Z())
#            print('sintheta- = ', math.sin(theta_n))
#            print('sintheta+ = ', math.sin(theta_p))
#            print('pi-_x = ', P_taunvis_new.Vect().Unit().X())
#            print('pi-_y = ', P_taunvis_new.Vect().Unit().Y())
#            print('pi-_z = ', P_taunvis_new.Vect().Unit().Z())
#            n_p = P_taupvis_new.Vect().Unit()
#            print(d_min_reco.Dot(P_taupvis.Vect().Unit().Cross(P_taunvis.Vect().Unit())))
#            #sinphi = n_p.Y()/math.sin(theta)
#            #print ('sinphi =', sinphi)

    print('Found correct solution for %i / %i events = %.1f%%' % (count_correct, count_total, count_correct/count_total*100))
