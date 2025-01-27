import ROOT
import numpy as np
from scipy.optimize import minimize
import math

class TauReconstructor:
    def __init__(self, mtau=1.775538):
        """
        Initialize the TauReconstructor class.

        Args:
            mtau (float): Mass of the tau particle in GeV.
        """
        self.mtau = mtau
        self.d_min = 0.

    def reconstruct_tau(self, P_Z, P_taupvis, P_taunvis, d_min_reco=None, verbose=False):
        """
        Reconstruct the tau momenta and direction at the point of closest approach.

        Args:
            P_Z (ROOT.TLorentzVector): Momentum of the Z boson.
            P_taupvis (ROOT.TLorentzVector): Visible momentum of the positive tau.
            P_taunvis (ROOT.TLorentzVector): Visible momentum of the negative tau.
            d_min_reco (ROOT.TVector3, optional): Reconstructed d_min direction.
            verbose (bool, optional):

        Returns:
            tuple: Reconstructed positive tau, negative tau, and d_min direction.
        """
        # use analytical results to define initial guesses
        analytic_solutions = ReconstructTauAnalytically(P_Z, P_taupvis, P_taunvis)
        solutions = []

        for i, solution in enumerate(analytic_solutions):
            if verbose:
                print('Running optimization for solution %i' % (i))
            nu_p =  solution[0] - P_taupvis
            nu_n = solution[1] - P_taunvis
            initial_guess = [
                nu_p.X(), nu_p.Y(), nu_p.Z(),
                nu_n.X(), nu_n.Y(), nu_n.Z()
            ]
            #initial_guess = [
            #    0., 0., 0.,
            #    0., 0., 0.
            #]

            # Call the optimizer
            result = minimize(
                self._objective, initial_guess,
                args=(P_Z, P_taupvis, P_taunvis, d_min_reco)
            )

            if verbose:
                if result.success:
                    print("Optimization succeeded!")
                else:
                    print("Optimization failed.")

                print('best dmin dot product = ', self.d_min.Dot(d_min_reco.Unit()))

            solutions.append(result)

        #nubar_x, nubar_y, nubar_z, nu_x, nu_y, nu_z = vars
    
        obj0 = self._objective(solutions[0].x, P_Z, P_taupvis, P_taunvis, d_min_reco)
        obj1 = self._objective(solutions[1].x, P_Z, P_taupvis, P_taunvis, d_min_reco)

        if verbose:
            print('objectives for solutions 0,1:', obj0, obj1)

        if obj0 > obj1:
            result = solutions[1]
        else:
            result = solutions[0]   

        P_taupnu_reco = ROOT.TLorentzVector(
            result.x[0], result.x[1], result.x[2],
            (result.x[0]**2 + result.x[1]**2 + result.x[2]**2)**0.5
        )
        P_taunnu_reco = ROOT.TLorentzVector(
            result.x[3], result.x[4], result.x[5],
            (result.x[3]**2 + result.x[4]**2 + result.x[5]**2)**0.5
        )

        P_taup_reco = P_taupvis + P_taupnu_reco
        P_taun_reco = P_taunvis + P_taunnu_reco

        P_taunvis_new = ChangeFrame(P_taun_reco, P_taunvis, P_taunvis)
        P_taupvis_new = ChangeFrame(P_taun_reco, P_taunvis, P_taupvis)

        d_min = PredictDmin(
            P_taunvis_new, P_taupvis_new, ROOT.TVector3(0.0, 0.0, -1.0)
        ).Unit()
        d_min = ChangeFrame(P_taun_reco, P_taunvis, d_min, reverse=True)

        return P_taup_reco, P_taun_reco, d_min

    def _objective(self, vars, P_Z, P_taupvis, P_taunvis, d_min_reco=None):
        """
        Objective function for tau reconstruction.

        Args:
            vars (list): Variables to optimize [nubar_x, nubar_y, nubar_z, nu_x, nu_y, nu_z].
            P_Z (ROOT.TLorentzVector): Momentum of the Z boson.
            P_taupvis (ROOT.TLorentzVector): Visible momentum of the positive tau.
            P_taunvis (ROOT.TLorentzVector): Visible momentum of the negative tau.
            d_min_reco (ROOT.TVector3, optional): Reconstructed d_min direction.

        Returns:
            float: Value of the objective function.
        """
        nubar_x, nubar_y, nubar_z, nu_x, nu_y, nu_z = vars

        P_taupnu = ROOT.TLorentzVector(
            nubar_x, nubar_y, nubar_z, (nubar_x**2 + nubar_y**2 + nubar_z**2)**0.5
        )
        P_taunnu = ROOT.TLorentzVector(
            nu_x, nu_y, nu_z, (nu_x**2 + nu_y**2 + nu_z**2)**0.5
        )

        # Tau mass constraints
        eq1 = (P_taupvis + P_taupnu).M() - self.mtau
        eq2 = (P_taunvis + P_taunnu).M() - self.mtau

        # Z boson momentum constraints
        mom_sum = P_taupvis + P_taunvis + P_taupnu + P_taunnu
        eq3 = mom_sum.T() - P_Z.T()
        eq4 = mom_sum.X() - P_Z.X()
        eq5 = mom_sum.Y() - P_Z.Y()
        eq6 = mom_sum.Z() - P_Z.Z()

        if d_min_reco:
            # Implement IP constraint
            P_taun = P_taunvis + P_taunnu
            P_taunvis_new = ChangeFrame(P_taun, P_taunvis, P_taunvis)
            P_taupvis_new = ChangeFrame(P_taun, P_taunvis, P_taupvis)

            d_min = PredictDmin(
                P_taunvis_new, P_taupvis_new, ROOT.TVector3(0.0, 0.0, -1.0)
            ).Unit()
            #print('P_taunnu:', P_taunnu.X(), P_taunnu.Y(), P_taunnu.Z(), P_taunnu.T(), P_taunnu.M())
            #print('P_taun:', P_taun.X(), P_taun.Y(), P_taun.Z(), P_taun.T(), P_taun.M())
            #print('d_min:', d_min.X(), d_min.Y(), d_min.Z())
            d_min = ChangeFrame(P_taun, P_taunvis, d_min, reverse=True)
            #print('d_min (rotated back):', d_min.X(), d_min.Y(), d_min.Z())
            #print('d_min_reco:', d_min_reco.Unit().X(), d_min_reco.Unit().Y(), d_min_reco.Unit().Z())
            dot_product = d_min.Unit().Dot(d_min_reco.Unit())
            eq7 = (1.0 - dot_product)
            #print('dot_product:', dot_product)
            self.d_min = d_min

            return eq1**2 + eq2**2 + eq3**2 + eq4**2 + eq5**2 + eq6**2 + eq7**2

        else: 
            return eq1**2 + eq2**2 + eq3**2 + eq4**2 + eq5**2 + eq6**2

def ChangeFrame(taun, taunvis, vec, reverse=False):
    '''
    Rotate the coordinate axis as follows:
    z-direction should be direction of the tau-
    xz-axis should adjust the direction so that the h- lies on the x-z plane, in the +ve x-direction.

    Args:
        taun (TVector3): The tau- vector.
        taunvis (TVector3): The visible tau- vector (h-).
        vec (TVector3): The vector to rotate.
        reverse (bool): If True, applies the inverse transformation to return to the original frame.

    Returns:
        TVector3: The rotated vector.
    '''
    vec_new = vec.Clone()
    taunvis_copy = taunvis.Clone()
    
    # Define the rotation angles to allign with tau- direction
    theta = taun.Theta()
    phi = taun.Phi()

    # Rotate taunvis to get second phi angle for rotation
    taunvis_copy.RotateZ(-phi)
    taunvis_copy.RotateY(-theta)
    phi2 = taunvis_copy.Phi()  # This is the phi angle of the rotated taunvis vector 

    if reverse:
        # Reverse transformation: Undo the rotations in reverse order

        vec_new.RotateZ(phi2)
        vec_new.RotateY(theta)
        vec_new.RotateZ(phi)
    else:
        # Forward transformation: Apply the rotations

        vec_new.RotateZ(-phi)
        vec_new.RotateY(-theta)
        vec_new.RotateZ(-phi2)

        # Check that h- is in the +ve x-direction
        if taunvis == vec and vec_new.X() < 0:
            raise ValueError("h- not pointing in the +ve x direction")

    return vec_new

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


# analytical reconstruction functions 

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

def FindDMin(p1, d1, p2, d2):
    """
    Find the vector pointing from the closest point on Line 1 to the closest point on Line 2 using ROOT classes.

    Args:
        p1 (TVector3): A point on Line 1.
        d1 (TVector3): Direction vector of Line 1.
        p2 (TVector3): A point on Line 2.
        d2 (TVector3): Direction vector of Line 2.

    Returns:
        TVector3: Vector pointing from the closest point on Line 1 to the closest point on Line 2.
    """
    # Normalize direction vectors
    d1 = d1.Unit()
    d2 = d2.Unit()

    # Compute cross product and its magnitude squared
    cross_d1_d2 = d1.Cross(d2)
    denom = cross_d1_d2.Mag2()

    # Handle parallel lines (cross product is zero)
    if denom == 0:
        raise ValueError("The lines are parallel or nearly parallel.")

    # Compute t1 and t2 using determinant approach
    dp = p2 - p1
    t1 = (dp.Dot(d2.Cross(cross_d1_d2))) / denom
    t2 = (dp.Dot(d1.Cross(cross_d1_d2))) / denom

    # Closest points on each line
    pca1 = p1 + d1 * t1
    pca2 = p2 + d2 * t2

    # Vector pointing from closest point on Line 1 to Line 2
    return pca2 - pca1


def PredictDmin(P_taunvis, P_taupvis, d=None):
    if d is None: d_over_l = ROOT.TVector3(0.,0.,-1.) # by definition in the rotated coordinate frame
    else: d_over_l = d
    n_n = P_taunvis.Vect().Unit()
    n_p = P_taupvis.Vect().Unit()
    fact1 = (d_over_l.Dot(n_p)*n_n.Dot(n_p) - d_over_l.Dot(n_n))/(1.-(n_n.Dot(n_p))**2)
    fact2 = (d_over_l.Dot(n_n)*n_n.Dot(n_p) - d_over_l.Dot(n_p))/(1.-(n_n.Dot(n_p))**2)
    term1 = n_n*fact1
    term2 = n_p*fact2

    d_min = d_over_l + (term1 + term2)
    return d_min

class Smearing():
    """
    Class to smear the energy and angular resolution of tracks.
    """

    def __init__(self):
        self.E_smearing = ROOT.TF1("E_smearing","TMath::Gaus(x,1,[0])",0,2)
        self.Angular_smearing = ROOT.TF1("Angular_smearing","TMath::Gaus(x,0,0.001)",-1,1) # approximate guess for now (this number was quote for electromagnetic objects but is probably better for tracks)
        self.IP_z_smearing = ROOT.TF1("IP_z_smearing","TMath::Gaus(x,0,0.042)",-1,1) # number from David's thesis, note unit is mm
        self.IP_xy_smearing = ROOT.TF1("IP_xy_smearing","TMath::Gaus(x,0,0.023)",-1,1) # number from David's thesis (number for r * sqrt(1/2)), note unit is mm

    def SmearTrack(self,track):
        E_res = 0.0006*track.P() # use p dependent resolution from David's thesis
        self.E_smearing.SetParameter(0,E_res)
        rand_E = self.E_smearing.GetRandom()
        rand_dphi = self.Angular_smearing.GetRandom()
        rand_deta = self.Angular_smearing.GetRandom()

        track_smeared = ROOT.TLorentzVector(track)

        phi = track_smeared.Phi()
        new_phi = ROOT.TVector2.Phi_mpi_pi(phi + rand_dphi)
        new_eta = track_smeared.Eta() + rand_deta
        new_theta = 2 * math.atan(math.exp(-new_eta))
        track_smeared.SetPhi(new_phi)
        track_smeared.SetTheta(new_theta)
        track_smeared *= rand_E

        return track_smeared

    def SmearDmin(self, dmin):
        dmin_smeared = ROOT.TVector3(dmin)

        rand_z = self.IP_z_smearing.GetRandom()
        rand_x = self.IP_xy_smearing.GetRandom()
        rand_y = self.IP_xy_smearing.GetRandom()

        dmin_smeared.SetX(dmin_smeared.X() + rand_x)
        dmin_smeared.SetY(dmin_smeared.Y() + rand_y)
        dmin_smeared.SetZ(dmin_smeared.Z() + rand_z)

        return dmin_smeared

if __name__ == '__main__':

    smearing = Smearing()
    apply_smearing = True
    # some representative smearing functions (exact resolutions to be taken with pinch of salt)
    E_smearing = ROOT.TF1("E_smearing","TMath::Gaus(x,1,0.02)",0,2)
    f = ROOT.TFile('pythia_output_ee_To_pipinunu_no_entanglementMG.root')
    tree = f.Get('tree')
    count_total = 0
    count_correct = 0
    d_min_reco_ave = 0.
    for i in range(1,1000):
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
        d_true = VERTEX_taup-VERTEX_taun

        IP_taup = GetGenImpactParam(ROOT.TVector3(0.,0.,0.),VERTEX_taup, P_taupvis.Vect())
        IP_taun = GetGenImpactParam(ROOT.TVector3(0.,0.,0.),VERTEX_taun, P_taunvis.Vect())
    


        d_min_reco = FindDMin(VERTEX_taun, P_taunvis.Vect().Unit(), VERTEX_taup, P_taupvis.Vect().Unit())

        P_taupvis_E_before = P_taupvis.E()

        if apply_smearing:
            P_taupvis = smearing.SmearTrack(P_taupvis)
            P_taunvis = smearing.SmearTrack(P_taunvis)

            d_min_reco = smearing.SmearDmin(d_min_reco)

        P_taupvis_E_after = P_taupvis.E() 

        reconstructor = TauReconstructor()
        P_taup_reco, P_taun_reco, d_min_numeric = reconstructor.reconstruct_tau(P_Z, P_taupvis, P_taunvis, d_min_reco)


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

        if i < 10 and True: # only print a few events

            print('\n---------------------------------------')
            print('Event %i' %i)
            print('\nTrue taus:')
            print('tau+:', P_taup_true.X(), P_taup_true.Y(), P_taup_true.Z(), P_taup_true.T(), P_taup_true.M())
            print('tau-:', P_taun_true.X(), P_taun_true.Y(), P_taun_true.Z(), P_taun_true.T(), P_taun_true.M())
    
            print('\nReco taus (numerically):')
            print('tau+:', P_taup_reco.X(), P_taup_reco.Y(), P_taup_reco.Z(), P_taup_reco.T(), P_taup_reco.M())
            print('tau-:', P_taun_reco.X(), P_taun_reco.Y(), P_taun_reco.Z(), P_taun_reco.T(), P_taun_reco.M())
            print('d_min constraint:', d_min_numeric.Unit().Dot(d_min_reco.Unit()))
    
            print('\nReco taus (analytically):')
            print('solution 1:')
            print('tau+:', solutions[0][0].X(), solutions[0][0].Y(), solutions[0][0].Z(), solutions[0][0].T(), solutions[0][0].M())
            print('tau-:', solutions[0][1].X(), solutions[0][1].Y(), solutions[0][1].Z(), solutions[0][1].T(), solutions[0][1].M())
            print('solution 2:')
            print('tau+:', solutions[1][0].X(), solutions[1][0].Y(), solutions[1][0].Z(), solutions[1][0].T(), solutions[1][0].M())
            print('tau-:', solutions[1][1].X(), solutions[1][1].Y(), solutions[1][1].Z(), solutions[1][1].T(), solutions[1][1].M())
    
#            print('\nboost to tau rest frames')
#            print('True:')
#            print('tau+:', P_taupvis_1.X(), P_taupvis_1.Y(), P_taupvis_1.Z(), P_taupvis_1.T(), P_taupvis_1.M())
#            print('tau-:', P_taunvis_1.X(), P_taunvis_1.Y(), P_taunvis_1.Z(), P_taunvis_1.T(), P_taunvis_1.M())
#            print('solution 1:')
#            print('tau+:', P_taupvis_2.X(), P_taupvis_2.Y(), P_taupvis_2.Z(), P_taupvis_2.T(), P_taupvis_2.M())
#            print('tau-:', P_taunvis_2.X(), P_taunvis_2.Y(), P_taunvis_2.Z(), P_taunvis_2.T(), P_taunvis_2.M())
#            print('solution 2:')
#            print('tau+:', P_taupvis_3.X(), P_taupvis_3.Y(), P_taupvis_3.Z(), P_taupvis_3.T(), P_taupvis_3.M())        
#            print('tau-:', P_taunvis_3.X(), P_taunvis_3.Y(), P_taunvis_3.Z(), P_taunvis_3.T(), P_taunvis_3.M())       
            

            print('d_min_reco = ', d_min_reco.Mag(), d_min_reco.Unit().X(), d_min_reco.Unit().Y(), d_min_reco.Unit().Z())
            print('d_min_predicted =', d_min_numeric.Mag(), d_min_numeric.Unit().X(), d_min_numeric.Unit().Y(), d_min_numeric.Unit().Z())

            d_min = PredictDmin(P_taunvis, P_taupvis, d_true)
            print('d_min_predict (true) = ', d_min.Mag(), d_min.Unit().X(), d_min.Unit().Y(), d_min.Unit().Z())


            P_taunvis_new = ChangeFrame(P_taun_true, P_taunvis, P_taunvis)
            P_taupvis_new = ChangeFrame(P_taun_true, P_taunvis, P_taupvis)

            for i in range(0,2):
                print ('solution %i' % (i+1))

                P_taunvis_new = ChangeFrame(solutions[i][1], P_taunvis, P_taunvis)
                P_taupvis_new = ChangeFrame(solutions[i][1], P_taunvis, P_taupvis)

                d_min = PredictDmin(P_taunvis_new, P_taupvis_new, ROOT.TVector3(0.,0.,-1.))
                print('d_min_predict_solution %i = ' % (i+1), d_min.Mag(), d_min.Unit().X(), d_min.Unit().Y(), d_min.Unit().Z())
                d_min = P_taupvis_new = ChangeFrame(solutions[i][1], P_taunvis, d_min, reverse=True)
                print('d_min_predict_solution %i (undo rotation) = ' % (i+1), d_min.Mag(), d_min.Unit().X(), d_min.Unit().Y(), d_min.Unit().Z())

                print(d_min_reco.Unit().Dot(d_min.Unit()))
            print('correct solution found?', FoundCorrectSolution)

        d_min_reco_ave+=d_min_reco.Mag()
    d_min_reco_ave/=count_total

    print('ave d_min = %g' % d_min_reco_ave)
    print('Found correct solution for %i / %i events = %.1f%%' % (count_correct, count_total, count_correct/count_total*100))
