import ROOT
import numpy as np
from scipy.optimize import minimize
import math
from iminuit import Minuit
import gc
import random

class TauReconstructor:
    def __init__(self, mtau=1.777):
        """
        Initialize the TauReconstructor class.

        Args:
            mtau (float): Mass of the tau particle in GeV.
        """
        self.mtau = mtau
        self.d_min = 0.

    def reconstruct_tau(self, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, d_min_reco=None, verbose=False, mode=1):
        """
        Reconstruct the tau momenta and direction at the point of closest approach.

        Args:
            P_Z (ROOT.TLorentzVector): Momentum of the Z boson.
            P_taupvis (ROOT.TLorentzVector): Visible momentum of the positive tau.
            P_taunvis (ROOT.TLorentzVector): Visible momentum of the negative tau.
            P_taup_pi1 (ROOT.TLorentzVector): Momentum of the positive tau leading charged pion decay product.
            P_taun_pi1 (ROOT.TLorentzVector): Momentum of the negative tau sub-leading charged pion decay product.
            d_min_reco (ROOT.TVector3, optional): Reconstructed d_min direction.
            verbose (bool, optional):

        Returns:
            tuple: Reconstructed positive tau, negative tau, and d_min direction.
        """
        
        if mode == 1:
            # use analytical results to define initial guesses
            analytic_solutions = ReconstructTauAnalytically(P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1)
            nu_p_1 = analytic_solutions[0][0] - P_taupvis
            nu_n_1 = analytic_solutions[0][1] - P_taunvis
            nu_p_2 = analytic_solutions[1][0] - P_taupvis
            nu_n_2 = analytic_solutions[1][1] - P_taunvis
            initial_guesses = [
                [nu_p_1.X(), nu_p_1.Y(), nu_p_1.Z(), nu_n_1.X(), nu_n_1.Y(), nu_n_1.Z()], 
                [nu_p_2.X(), nu_p_2.Y(), nu_p_2.Z(), nu_n_2.X(), nu_n_2.Y(), nu_n_2.Z()]
                ]
        else:
            # use initial guess of zeros, perform initial minimisation, 
            # then infer the alternative second minimum by adjusting nubar momentum and start second minimisation from this initial guess
            initial_guesses = [
                [0., 0., 0., 0., 0., 0.], 
                ]    
        solutions = []

        #for i, initial_guess in enumerate(initial_guesses):
        i = 0
        while i < len(initial_guesses):  # Dynamically checks list length
            initial_guess = initial_guesses[i] 
            if verbose:
                print('Running optimization for solution %i' % (i))

            def objective(nubar_x, nubar_y, nubar_z, nu_x, nu_y, nu_z):
                vars = [nubar_x, nubar_y, nubar_z, nu_x, nu_y, nu_z]
                return self._objective(vars, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, d_min_reco)

            # Call iMinuit
            minuit = Minuit(
                objective,
                nubar_x=initial_guess[0],
                nubar_y=initial_guess[1],
                nubar_z=initial_guess[2],
                nu_x=initial_guess[3],
                nu_y=initial_guess[4],
                nu_z=initial_guess[5]
            )

            #minuit.limits = {
            #    "nubar_x": (-50, 50),
            #    "nubar_y": (-50, 50),
            #    "nubar_z": (-50, 50),
            #    "nu_x": (-50, 50),
            #    "nu_y": (-50, 50),
            #    "nu_z": (-50, 50),
            #}

            # Perform the minimization
            #if not (i == 1 and mode == 2): minuit.migrad() # temp: for now dont run minimization for second guess
            minuit.migrad()

            if verbose:
                if minuit.fmin.is_valid:
                    print("Optimization succeeded!")
                else:
                    print("Optimization failed.")

                print('best dmin dot product = ', self.d_min.Dot(d_min_reco.Unit()))

            solutions.append(minuit)

            if i == 0 and mode == 2:
                # define a new guess by changing the sign of the nubar Y momentum in the frame aligned with the tau- direction
                P_taupnu_reco = ROOT.TLorentzVector(
                    minuit.values[0], minuit.values[1], minuit.values[2],
                    (minuit.values[0]**2 + minuit.values[1]**2 + minuit.values[2]**2)**0.5
                )
                P_taunnu_reco = ROOT.TLorentzVector(
                    minuit.values[3], minuit.values[4], minuit.values[5],
                    (minuit.values[3]**2 + minuit.values[4]**2 + minuit.values[5]**2)**0.5 
                )
                P_taun_reco = P_taunnu_reco + P_taunvis

                # rotate nubar to the correct frame
                P_taupnu_reco_new = ChangeFrame(P_taun_reco, P_taunvis, P_taupnu_reco)
                # now the alternative solution should correspond to the negative of the y component of nubar so add this to the initial guesses
                P_taupnu_reco_new.SetY(-P_taupnu_reco_new.Y())
                P_taupnu_reco_new = ChangeFrame(P_taun_reco, P_taunvis, P_taupnu_reco_new, reverse=True)
                initial_guesses.append([P_taupnu_reco_new.X(), P_taupnu_reco_new.Y(), P_taupnu_reco_new.Z(), P_taunnu_reco.X(), P_taunnu_reco.Y(), P_taunnu_reco.Z()])
                #solution2 = minuit
                #solution2.values.update({'nubar_x': P_taupnu_reco_new.X(), 'nubar_y': P_taupnu_reco_new.Y(), 'nubar_z': P_taupnu_reco_new.Z(), 'nu_x': P_taunnu_reco.X(), 'nu_y': P_taunnu_reco.Y(), 'nu_z': P_taunnu_reco.Z()})
                #solutions.append(solution2)
            i+=1
    

        if len(solutions) > 1:

            obj0 = self._objective(solutions[0].values, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, d_min_reco)
            obj1 = self._objective(solutions[1].values, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, d_min_reco)

            if verbose or True:
                print('objectives for solutions 0,1:', obj0, obj1)

                for i, solution in enumerate(solutions):
                    print('solution %i:' % (i+1))
                    taup = ROOT.TLorentzVector(
                        solution.values[0], solution.values[1], solution.values[2],
                        (solution.values[0]**2 + solution.values[1]**2 + solution.values[2]**2)**0.5
                    ) + P_taupvis
                    taun = ROOT.TLorentzVector(
                        solution.values[3], solution.values[4], solution.values[5],
                        (solution.values[3]**2 + solution.values[4]**2 + solution.values[5]**2)**0.5
                    ) + P_taunvis   
                    print('tau+:', taup.X(), taup.Y(), taup.Z(), taup.M())
                    print('tau-:', taun.X(), taun.Y(), taun.Z(), taun.M())

            if obj0 > obj1:
                result = solutions[1]
            else:
                result = solutions[0]   
        else:
            result = solutions[0]

        P_taupnu_reco = ROOT.TLorentzVector(
            result.values[0], result.values[1], result.values[2],
            (result.values[0]**2 + result.values[1]**2 + result.values[2]**2)**0.5
        )
        P_taunnu_reco = ROOT.TLorentzVector(
            result.values[3], result.values[4], result.values[5],
            (result.values[3]**2 + result.values[4]**2 + result.values[5]**2)**0.5
        )

        P_taup_reco = P_taupvis + P_taupnu_reco
        P_taun_reco = P_taunvis + P_taunnu_reco

        P_taun_pi1_new = ChangeFrame(P_taun_reco, P_taun_pi1, P_taun_pi1)
        P_taup_pi1_new = ChangeFrame(P_taun_reco, P_taun_pi1, P_taup_pi1)

        d_min = PredictDmin(
            P_taun_pi1_new, P_taup_pi1_new, ROOT.TVector3(0.0, 0.0, -1.0)
        ).Unit()
        d_min = ChangeFrame(P_taun_reco, P_taun_pi1, d_min, reverse=True)
  
        #P_taunvis_new = ChangeFrame(P_taun_reco, P_taunvis, P_taunvis)
        #P_taupvis_new = ChangeFrame(P_taun_reco, P_taunvis, P_taupvis)

        #d_min = PredictDmin(
        #    P_taunvis_new, P_taupvis_new, ROOT.TVector3(0.0, 0.0, -1.0)
        #).Unit()
        #d_min = ChangeFrame(P_taun_reco, P_taunvis, d_min, reverse=True)


        return P_taup_reco, P_taun_reco, d_min
    

    def _objective(self, vars, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, d_min_reco=None):
        """
        Objective function for tau reconstruction.

        Args:
            vars (list): Variables to optimize [nubar_x, nubar_y, nubar_z, nu_x, nu_y, nu_z].
            P_Z (ROOT.TLorentzVector): Momentum of the Z boson.
            P_taupvis (ROOT.TLorentzVector): Visible momentum of the positive tau.
            P_taunvis (ROOT.TLorentzVector): Visible momentum of the negative tau.
            P_taup_pi1 (ROOT.TLorentzVector): Momentum of the positive tau leading charged pion decay product.
            P_taun_pi1 (ROOT.TLorentzVector): Momentum of the negative tau sub-leading charged pion decay product.
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

            P_taup_pi1_new = ChangeFrame(P_taun, P_taun_pi1, P_taup_pi1)
            P_taun_pi1_new = ChangeFrame(P_taun, P_taun_pi1, P_taun_pi1)

            d_min = PredictDmin(
                P_taun_pi1_new, P_taup_pi1_new, ROOT.TVector3(0.0, 0.0, -1.0)
            ).Unit()
            d_min = ChangeFrame(P_taun, P_taun_pi1, d_min, reverse=True)

            #P_taunvis_new = ChangeFrame(P_taun, P_taunvis, P_taunvis)
            #P_taupvis_new = ChangeFrame(P_taun, P_taunvis, P_taupvis)

            #d_min = PredictDmin(
            #    P_taunvis_new, P_taupvis_new, ROOT.TVector3(0.0, 0.0, -1.0)
            #).Unit()
            #d_min = ChangeFrame(P_taun, P_taunvis, d_min, reverse=True)

            dot_product = d_min.Unit().Dot(d_min_reco.Unit())
            eq7 = (1.0 - dot_product)
            self.d_min = d_min

            return eq1**2 + eq2**2 + eq3**2 + eq4**2 + eq5**2 + eq6**2 + eq7**2 

        else: 
            return eq1**2 + eq2**2 + eq3**2 + eq4**2 + eq5**2 + eq6**2

    def reconstruct_tau_alt(self, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, np_point=ROOT.TVector3(), nn_point=ROOT.TVector3(), d_min_reco=None, verbose=False,no_minimisation=False):
        """
        """

        if d_min_reco: 
            sign_from_d_min_reco = d_min_reco.Dot(P_taupvis.Vect().Unit().Cross(P_taunvis.Vect().Unit()))
            sign_from_d_min_reco = sign_from_d_min_reco/abs(sign_from_d_min_reco)
        else: sign_from_d_min_reco = 0

        # use analytical solution as initial guess
        analytic_solutions = ReconstructTauAnalytically(P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, np_point, nn_point, return_values=True)
        # sort solutions based on predicted d_min direction
        #sorted_solutions = sorted(analytic_solutions, key=lambda x: x[2].Unit().Dot(d_min_reco.Unit()), reverse=True)
        sorted_solutions = sorted(analytic_solutions, key=lambda x: (x[2].X()-d_min_reco.X())**2 + (x[2].Y()-d_min_reco.Y())**2 + (x[2].Z()-d_min_reco.Z())**2)
        # take random solution if dot product is equal
        #if sorted_solutions[0][2].Unit().Dot(d_min_reco.Unit()) == sorted_solutions[1][2].Unit().Dot(d_min_reco.Unit()):
        #    np.random.shuffle(sorted_solutions)

        if (sorted_solutions[0][2].X()-d_min_reco.X())**2 + (sorted_solutions[0][2].Y()-d_min_reco.Y())**2 + (sorted_solutions[0][2].Z()-d_min_reco.Z())**2 == (sorted_solutions[1][2].X()-d_min_reco.X())**2 + (sorted_solutions[1][2].Y()-d_min_reco.Y())**2 + (sorted_solutions[1][2].Z()-d_min_reco.Z())**2:
            np.random.shuffle(sorted_solutions)      
        _, _, _, a, b, c, d = sorted_solutions[0]    

        #initial_guess = [0., 0., 0., 0.]
        initial_guess = [a, b, c, d]

        def objective(a, b, c, d):
            vars = [a, b, c, d]
            return self._objective_alt(vars, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, d_min_reco, np_point, nn_point)

        # Call iMinuit
        minuit = Minuit(
            objective,
            a=initial_guess[0],
            b=initial_guess[1],
            c=initial_guess[2],
            d=initial_guess[3]
        )

        # Perform the minimization
        if not no_minimisation: minuit.migrad()

        if verbose:
            if minuit.fmin.is_valid:
                print("Optimization succeeded!")
            else:
                print("Optimization failed.")

        q = compute_q(P_Z*P_Z,P_Z, P_taupvis, P_taunvis)

        solutions = []

        for sign in [-1, 1]:

            P_taup_reco = (1.-minuit.values[0])/2*P_Z + minuit.values[1]/2*P_taupvis - minuit.values[2]/2*P_taunvis + sign*minuit.values[3]*q
            P_taun_reco = (1.+minuit.values[0])/2*P_Z - minuit.values[1]/2*P_taupvis + minuit.values[2]/2*P_taunvis - sign*minuit.values[3]*q     

            # set tau masses equal to the expected value since minimization does not strictly enforce this 
            P_taup_reco.SetE((P_taup_reco.P()**2+self.mtau**2)**.5)
            P_taun_reco.SetE((P_taun_reco.P()**2+self.mtau**2)**.5)

            P_taun_pi1_new = ChangeFrame(P_taun_reco, P_taun_pi1, P_taun_pi1)
            P_taup_pi1_new = ChangeFrame(P_taun_reco, P_taun_pi1, P_taup_pi1)

            # old method for doing this
            #d_min = PredictDmin(
            #    P_taun_pi1_new, P_taup_pi1_new, ROOT.TVector3(0.0, 0.0, -1.0)
            #).Unit()
            #d_min = ChangeFrame(P_taun_reco, P_taun_pi1, d_min, reverse=True)

            # the new version gets the intersection between tau+ and tau- and the h+ and h- tracks
            P_intersection, N_intersection, d_min  = find_intersections(P_taup_reco, P_taun_reco, np_point, P_taup_pi1.Vect().Unit(), nn_point, P_taun_pi1.Vect().Unit())

            solutions.append((P_taup_reco, P_taun_reco, d_min))

        # sort solutions by the dot product of the d_min direction with the reconstructed d_min direction
        # this should be 1 for the correct solution and -1 for the alternative solution (for pipi channel - might not always have different signs for rho channels)
        #if d_min_reco: sorted_solutions = sorted(solutions, key=lambda x: x[2].Unit().Dot(d_min_reco.Unit()), reverse=True)
        if d_min_reco: sorted_solutions = sorted(solutions, key=lambda x: x[2].Unit().Dot(d_min_reco.Unit()), reverse=True)
        # sort solutions based on how how well the d_mins are reconstructed
        sorted_solutions = sorted(solutions, key=lambda x: (x[2].X()-d_min_reco.X())**2 + (x[2].Y()-d_min_reco.Y())**2 + (x[2].Z()-d_min_reco.Z())**2)
        # if the dot product is equal then randomly sort the solutions
        #if (not d_min_reco) or sorted_solutions[0][2].Unit().Dot(d_min_reco.Unit()) == sorted_solutions[1][2].Unit().Dot(d_min_reco.Unit()):
        #    np.random.shuffle(sorted_solutions) 

        if (not d_min_reco) or (sorted_solutions[0][2].X()-d_min_reco.X())**2 + (sorted_solutions[0][2].Y()-d_min_reco.Y())**2 + (sorted_solutions[0][2].Z()-d_min_reco.Z())**2 == (sorted_solutions[1][2].X()-d_min_reco.X())**2 + (sorted_solutions[1][2].Y()-d_min_reco.Y())**2 + (sorted_solutions[1][2].Z()-d_min_reco.Z())**2:
            np.random.shuffle(sorted_solutions)     

        if verbose:
            print('numerical solution 1:')
            print('tau+:', sorted_solutions[0][0].X(), sorted_solutions[0][0].Y(), sorted_solutions[0][0].Z(), sorted_solutions[0][0].M())
            print('tau-:', sorted_solutions[0][1].X(), sorted_solutions[0][1].Y(), sorted_solutions[0][1].Z(), sorted_solutions[0][1].M())
            print('d_min:', sorted_solutions[0][2].X(), sorted_solutions[0][2].Y(), sorted_solutions[0][2].Z())
            print('d_min constraint:', sorted_solutions[0][2].Unit().Dot(d_min_reco.Unit()))

            print('numerical solution 2:')
            print('tau+:', sorted_solutions[1][0].X(), sorted_solutions[1][0].Y(), sorted_solutions[1][0].Z(), sorted_solutions[1][0].M())
            print('tau-:', sorted_solutions[1][1].X(), sorted_solutions[1][1].Y(), sorted_solutions[1][1].Z(), sorted_solutions[1][1].M())
            print('d_min:', sorted_solutions[1][2].X(), sorted_solutions[1][2].Y(), sorted_solutions[1][2].Z())
            print('d_min constraint:', sorted_solutions[1][2].Unit().Dot(d_min_reco.Unit()))

        return sorted_solutions[0]

    def _objective_alt(self, vars, P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, d_min_reco=None, np_point=None, nn_point=None):

        a, b, c, d = vars

        q = compute_q(P_Z*P_Z,P_Z, P_taupvis, P_taunvis)

        x = P_Z * P_taupvis
        y = P_Z * P_taunvis
        z = P_taupvis * P_taunvis

        m_Z_sq = P_Z.Mag2()
        m_pvis_sq = P_taupvis.Mag2()
        m_nvis_sq = P_taunvis.Mag2()
        q_sq = q*q

        eq1 = self.mtau**2 + m_pvis_sq -x +a*x -b*m_pvis_sq +c*z
        eq2 = self.mtau**2 + m_nvis_sq -y -a*y + b*z -c*m_nvis_sq
        eq3 = a*m_Z_sq -b*x + c*y + (b**2+c**2)/4*(m_nvis_sq - m_pvis_sq)
        eq4 = (1.+a**2)/2*m_Z_sq + (b**2+c**2)/4*(m_pvis_sq + m_nvis_sq) +2*d**2*q_sq -a*b*x + a*c*y - b*c*z - 2*self.mtau**2

        P_taup = (1.-a)/2*P_Z + b/2*P_taupvis - c/2*P_taunvis + d*q
        P_taun = (1.+a)/2*P_Z - b/2*P_taupvis + c/2*P_taunvis - d*q 

        if d_min_reco and np_point and nn_point:
            eq5 = 0.
            _, _, d_min  = find_intersections(P_taup, P_taun, np_point, P_taup_pi1.Vect().Unit(), nn_point, P_taun_pi1.Vect().Unit())
            eq5 = (d_min.X()-d_min_reco.X())**2 + (d_min.Y()-d_min_reco.Y())**2 + (d_min.Z()-d_min_reco.Z())**2
            return eq1**2 + eq2**2 + eq3**2 + eq4**2 #+ eq5*1000
        else: 
            return eq1**2 + eq2**2 + eq3**2 + eq4**2

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
    vec1, vec2    = pair1
    vec3, vec4, _ = pair2

    # Compute the squared differences for each component
    dx = (vec1.X() - vec3.X())**2 + (vec2.X() - vec4.X())**2
    dy = (vec1.Y() - vec3.Y())**2 + (vec2.Y() - vec4.Y())**2
    dz = (vec1.Z() - vec3.Z())**2 + (vec2.Z() - vec4.Z())**2

    # Return the sum of squared differences
    return dx + dy + dz

def GetOpeningAngle(tau, h):
    beta = tau.P()/tau.E()
    if beta >= 1:
        print('Warning: Beta is >= 1, invalid for physical particles (Beta = %g). Recomputing using the expected tau mass.' % beta)
        beta = tau.P()/(tau.P()**2 + 1.777**2)**0.5 # if beta is unphysical then recalculate it using the expected mass of the tau lepton
        print('recalculated beta:', beta)
    gamma = 1. / (1. - beta**2)**0.5
    x = h.E()/tau.E()
    r = h.M()/tau.M()

    costheta = min((gamma*x - (1.+r**2)/(2*gamma))/(beta*(gamma**2*x**2-r**2)**.5),1.)
    sintheta = (((1.-r**2)**2/4 - (x-(1.+r**2)/2)**2/beta**2)/(gamma**2*x**2-r**2))**.5
    # if sintheta is complex then set to 0
    if sintheta.imag != 0:
        sintheta = 0.
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

def ReconstructTauAnalytically(P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, np_point=ROOT.TVector3(), nn_point=ROOT.TVector3(), verbose=False, return_values=False):
    
    '''
    Reconstuct tau lepton 4-momenta with 2-fold degeneracy
    following formulation in Appendix C of https://journals.aps.org/prd/pdf/10.1103/PhysRevD.107.093002
    '''

    m_tau = 1.777
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
    #print('dsq:', dsq)
    d = dsq**.5 if dsq > 0 else 0.

    solutions = []

    for i, d in enumerate([d,-d]):
        taup = (1.-a)/2*P_Z + b/2*P_taupvis - c/2*P_taunvis + d*q
        taun = (1.+a)/2*P_Z - b/2*P_taupvis + c/2*P_taunvis - d*q

        ############################
        # old version of this:
        #This version gets dot product equal to 0 or 1 but not one of each for the 2 solutions
        #compute predicted d_min
        #P_taun_pi1_new = ChangeFrame(taun, P_taun_pi1, P_taun_pi1)
        #P_taup_pi1_new = ChangeFrame(taun, P_taun_pi1, P_taup_pi1)

        #d_min = PredictDmin(
        #    P_taun_pi1_new, P_taup_pi1_new, ROOT.TVector3(0.0, 0.0, -1.0)
        #).Unit()
        #d_min = ChangeFrame(taun, P_taun_pi1, d_min, reverse=True)

        # the new version gets the intersection between tau+ and tau- and the h+ and h- tracks
        P_intersection, N_intersection, d_min  = find_intersections(taup, taun, np_point, P_taup_pi1.Vect().Unit(), nn_point, P_taun_pi1.Vect().Unit())
 

        if return_values: solutions.append((taup, taun, d_min, a, b, c, d))
        else: solutions.append((taup, taun, d_min))
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

def PredictD(P_taun, P_taup, P_taunvis, P_taupvis, d_min):
  d = ROOT.TVector3()

  n_n = P_taunvis.Vect().Unit()
  n_p = P_taupvis.Vect().Unit()

  theta_n = GetOpeningAngle(P_taun, P_taunvis)
  theta_p = GetOpeningAngle(P_taup, P_taupvis)

  sin_n = math.sin(theta_n)
  sin_p = math.sin(theta_p)
  cos_n = math.cos(theta_n)
  cos_p = math.cos(theta_p)


  proj = d_min.Dot(n_p.Cross(n_n))

  if sin_n*sin_p==0.: cosphi=1.
  else: cosphi = (n_n.Dot(n_p) + cos_n*cos_p)/(sin_n*sin_p)
  cosphi = max(-1.0, min(1.0, cosphi))
  sinphi = math.sin(math.acos(cosphi))

  if sin_n*sin_p*sinphi == 0: l = 1 # TODO: this might need a better fix
  else: l = abs(proj/(sin_n*sin_p*sinphi)) # this l seems to be close but slightly different to the true l even when inputting gen quantities, need to figure out why

  #print('l_reco:', l)

  d_min_over_l = d_min * (1./l)

  fact1 = (cos_p*n_p.Dot(n_n) + cos_n) / (1.-(n_n.Dot(n_p))**2)
  fact2 = (-cos_n*n_p.Dot(n_n) - cos_p) / (1.-(n_n.Dot(n_p))**2)

  term1 = n_n*fact1
  term2 = n_p*fact2
  d_over_l = d_min_over_l - term1 - term2

  #print('d_unit_reco:', d_over_l.X(), d_over_l.Y(), d_over_l.Z())
  #print('d_reco:', d_over_l.X()*l_true, d_over_l.Y()*l_true, d_over_l.Z()*l_true)

  d.SetX(d_over_l.X()*l)
  d.SetY(d_over_l.Y()*l)
  d.SetZ(d_over_l.Z()*l)

  return d

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

def find_intersections(taup, taun, np_point, np_dir, nn_point, nn_dir):
    # Convert TVector3 inputs to NumPy arrays for computation
    taup_arr = np.array([taup.X(), taup.Y(), taup.Z()])
    taun_arr = np.array([taun.X(), taun.Y(), taun.Z()])
    np_point_arr = np.array([np_point.X(), np_point.Y(), np_point.Z()])
    np_dir_arr = np.array([np_dir.X(), np_dir.Y(), np_dir.Z()])
    nn_point_arr = np.array([nn_point.X(), nn_point.Y(), nn_point.Z()])
    nn_dir_arr = np.array([nn_dir.X(), nn_dir.Y(), nn_dir.Z()])
    
    # Solve for lambda_p, lambda_n, t_p, t_n
    A = np.array([
        [taup_arr[0], -np_dir_arr[0], 0, 0],
        [taup_arr[1], -np_dir_arr[1], 0, 0],
        [taup_arr[2], -np_dir_arr[2], 0, 0],
        [0, 0, taun_arr[0], -nn_dir_arr[0]],
        [0, 0, taun_arr[1], -nn_dir_arr[1]],
        [0, 0, taun_arr[2], -nn_dir_arr[2]]
    ])
    
    b = np.array([
        np_point_arr[0], np_point_arr[1], np_point_arr[2],
        nn_point_arr[0], nn_point_arr[1], nn_point_arr[2]
    ])
    
    try:
        solution = np.linalg.lstsq(A, b, rcond=None)[0]
        lambda_p, tp, lambda_n, tn = solution
    except np.linalg.LinAlgError:
        print("No unique solution exists.")
        return None
    
    # Compute intersection points
    P_intersection_arr = taup_arr * lambda_p
    N_intersection_arr = taun_arr * lambda_n
    
    # Convert back to TVector3
    P_intersection = ROOT.TVector3(*P_intersection_arr)
    N_intersection = ROOT.TVector3(*N_intersection_arr)
    
    d_min_pred = FindDMin(N_intersection, nn_dir.Unit(), P_intersection, np_dir.Unit())

    return P_intersection, N_intersection, d_min_pred

def closest_distance(P1, P2, P):
    v = P2 - P1  # Direction vector of the line
    w = P - P1   # Vector from P1 to the point P
    
    t = w.Dot(v) / v.Mag2()  # Projection factor

    P_closest = P1 + t * v  # Closest point on the line

    return (P - P_closest).Mag()  # Distance between P and P_closest  

class Smearing():
    """
    Class to smear the energy and angular resolution of tracks.
    """

    def __init__(self):
        self.E_smearing = ROOT.TF1("E_smearing","TMath::Gaus(x,1,[0])",0,2)
        self.Angular_smearing = ROOT.TF1("Angular_smearing","TMath::Gaus(x,0,0.001)",-1,1) # approximate guess for now (this number was quoted for electromagnetic objects but is probably better for tracks)
        self.IP_z_smearing = ROOT.TF1("IP_z_smearing","TMath::Gaus(x,0,0.042)",-1,1) # number from David's thesis, note unit is mm
        self.IP_xy_smearing = ROOT.TF1("IP_xy_smearing","TMath::Gaus(x,0,0.023)",-1,1) # number from David's thesis (number for r * sqrt(1/2)), note unit is mm

        # pi0 numbers taken from here: https://cds.cern.ch/record/272484/files/ppe-94-170.pdf
        self.Pi0_Angular_smearing = ROOT.TF1("Angular_smearing","TMath::Gaus(x,0,10*(2.5/sqrt([0])+0.25)/1000.)",-1,1) # approximate guess for now, took the numbers from the paper for ECAL showers and conservatively scaled by 10
        self.Pi0_E_smearing = ROOT.TF1("Pi0_E_smearing","TMath::Gaus(x,1,0.065)",0,2) # approximate guess for now
        self.Q_smearing = ROOT.TF1("Q_smearing","TMath::Gaus(x,0,TMath::Max(1.,[0]*0.1))",-100,100) # approximate guess for now - assume 10% resolution or 1 GeV, whichever is largest

    def SmearPi0(self,pi0):
        E = pi0.E()
        self.Pi0_Angular_smearing.SetParameter(0,E)
        rand_E = self.Pi0_E_smearing.GetRandom()
        rand_dphi = self.Pi0_Angular_smearing.GetRandom()
        rand_dtheta = self.Pi0_Angular_smearing.GetRandom()

        pi0_smeared = ROOT.TLorentzVector(pi0)
        phi = pi0_smeared.Phi()
        new_phi = ROOT.TVector2.Phi_mpi_pi(phi + rand_dphi)

        theta = pi0_smeared.Theta()
        new_theta = theta + rand_dtheta

        pi0_smeared.SetPhi(new_phi)
        pi0_smeared.SetTheta(new_theta)
        pi0_smeared *= rand_E

        return pi0_smeared


    def SmearTrack(self,track):
        E_res = 0.0006*track.P() # use p dependent resolution from David's thesis
        self.E_smearing.SetParameter(0,E_res)
        rand_E = self.E_smearing.GetRandom()
        rand_dphi = self.Angular_smearing.GetRandom()
        rand_dtheta = self.Angular_smearing.GetRandom()

        track_smeared = ROOT.TLorentzVector(track)

        phi = track_smeared.Phi()
        new_phi = ROOT.TVector2.Phi_mpi_pi(phi + rand_dphi)

        theta = track_smeared.Theta()
        new_theta = theta + rand_dtheta

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

    def SmearQ(self, Q):
        # assume we know the momentum of the radiated photon(s)
        Q_smeared = ROOT.TLorentzVector(Q)
        self.Q_smearing.SetParameter(0,Q.X())
        rand_x = self.Q_smearing.GetRandom()
        self.Q_smearing.SetParameter(0,Q.Y())
        rand_y = self.Q_smearing.GetRandom()
        self.Q_smearing.SetParameter(0,Q.Z())
        rand_z = self.Q_smearing.GetRandom()
        Q_smeared = ROOT.TLorentzVector(Q)
        x = Q.X()+rand_x
        y = Q.Y()+rand_y
        z = Q.Z()+rand_z
        t = Q.T()
        Q_smeared.SetXYZT(x,y,z,t)

        return Q_smeared

if __name__ == '__main__':

    smearing = Smearing()
    apply_smearing = False
    #f = ROOT.TFile('pythia_output_ee_To_pipinunu_no_entanglementMG.root')
    f = ROOT.TFile('pythia_output_rhorhoMG.root')
    #f = ROOT.TFile('pythia_output_pipiMG.root')
    tree = f.Get('tree')
    count_total = 0
    count_correct = 0
    d_min_reco_ave = 0.
    print('starting...')
    for i in range(1,10):
        count_total+=1
        tree.GetEntry(i)
        P_taup_true = ROOT.TLorentzVector(tree.taup_px, tree.taup_py, tree.taup_pz, tree.taup_e) 
        P_taun_true = ROOT.TLorentzVector(tree.taun_px, tree.taun_py, tree.taun_pz, tree.taun_e) 

        P_taup_pi1 = ROOT.TLorentzVector(tree.taup_pi1_px, tree.taup_pi1_py, tree.taup_pi1_pz, tree.taup_pi1_e)
        P_taun_pi1 = ROOT.TLorentzVector(tree.taun_pi1_px, tree.taun_pi1_py, tree.taun_pi1_pz, tree.taun_pi1_e)

        #P_taupvis = ROOT.TLorentzVector(tree.taup_pi1_px, tree.taup_pi1_py, tree.taup_pi1_pz, tree.taup_pi1_e)
        #P_taunvis = ROOT.TLorentzVector(tree.taun_pi1_px, tree.taun_pi1_py, tree.taun_pi1_pz, tree.taun_pi1_e)

        P_taup_pizero1 = ROOT.TLorentzVector(tree.taup_pizero1_px, tree.taup_pizero1_py, tree.taup_pizero1_pz, tree.taup_pizero1_e)
        P_taun_pizero1 = ROOT.TLorentzVector(tree.taun_pizero1_px, tree.taun_pizero1_py, tree.taun_pizero1_pz, tree.taun_pizero1_e)

        if tree.taup_pizero1_e > 0:
            P_taupvis = P_taup_pi1+P_taup_pizero1
        else:
            P_taupvis = P_taup_pi1    
        if tree.taun_pizero1_e > 0:
            P_taunvis = P_taun_pi1+P_taun_pizero1   
        else:
            P_taunvis = P_taun_pi1
    
        P_Z = P_taup_true+P_taun_true
        #P_Z = ROOT.TLorentzVector(0.,0.,0.,91.188) # assuming we don't know ISR and have to assume momentum is balanced
  
        # compute IPs from SVs

        # note that the below assuems that taus are produced at 0,0,0 which might not be true for some MC samples! 
        VERTEX_taup = ROOT.TVector3(tree.taup_pi1_vx, tree.taup_pi1_vy, tree.taup_pi1_vz) # in mm
        VERTEX_taun = ROOT.TVector3(tree.taun_pi1_vx, tree.taun_pi1_vy, tree.taun_pi1_vz) # in mm

        VERTEX_taup += ROOT.TVector3(10.,5.,-7)
        VERTEX_taun += ROOT.TVector3(10.,5,-7)

        l_true = abs((VERTEX_taup-VERTEX_taun).Mag())
        d_true = VERTEX_taup-VERTEX_taun

        d_min_reco = FindDMin(VERTEX_taun, P_taun_pi1.Vect().Unit(), VERTEX_taup, P_taup_pi1.Vect().Unit())
        print('!!!!', d_min_reco.Mag(), d_min_reco.X(), d_min_reco.Y(), d_min_reco.Z())
        rand1 = random.uniform(-100,100)
        rand2 = random.uniform(-100,100)
        VERTEX_taun_rand = VERTEX_taun + P_taun_pi1.Vect().Unit()*rand1
        VERTEX_taup_rand = VERTEX_taup + P_taup_pi1.Vect().Unit()*rand2
        P_intersection, N_intersection, d_min_pred = find_intersections(P_taup_true, P_taun_true, VERTEX_taup_rand, P_taup_pi1.Vect().Unit(), VERTEX_taun_rand, P_taun_pi1.Vect().Unit())
        closest_distance(P_intersection, N_intersection, ROOT.TVector3(0.,0.,0.))
        print('vertex_taup:', VERTEX_taup.X(), VERTEX_taup.Y(), VERTEX_taup.Z())
        print('vertex_taun:', VERTEX_taun.X(), VERTEX_taun.Y(), VERTEX_taun.Z())
        print('P_intersection:', P_intersection.X(), P_intersection.Y(), P_intersection.Z())
        print('N_intersection:', N_intersection.X(), N_intersection.Y(), N_intersection.Z())
        print('closest_distance:', closest_distance(P_intersection, N_intersection, ROOT.TVector3(0.,0.,0.)))
        print('d_min_pred:', d_min_pred.Mag(), d_min_pred.X(), d_min_pred.Y(), d_min_pred.Z())

        if apply_smearing:
            # TODO: update smearing for case of pi + pi0 decays!
            P_taupvis = smearing.SmearTrack(P_taupvis)
            P_taunvis = smearing.SmearTrack(P_taunvis)

            P_taup_pi1 = smearing.SmearTrack(P_taup_pi1)
            P_taun_pi1 = smearing.SmearTrack(P_taun_pi1)

            d_min_reco = smearing.SmearDmin(d_min_reco)


        reconstructor = TauReconstructor()
        P_taup_reco, P_taun_reco, d_min_numeric = reconstructor.reconstruct_tau_alt(P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, np_point=VERTEX_taup_rand, nn_point=VERTEX_taun_rand, d_min_reco=d_min_reco)

        solutions = ReconstructTauAnalytically(P_Z, P_taupvis, P_taunvis, P_taup_pi1, P_taun_pi1, VERTEX_taup_rand, VERTEX_taun_rand, verbose=True)

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

        if i < 30 and True: # only print a few events

            print('\n---------------------------------------')
            print('Event %i' %i)
            print('\nTrue taus:\n')
            print('tau+:', P_taup_true.X(), P_taup_true.Y(), P_taup_true.Z(), P_taup_true.T(), P_taup_true.M())
            print('tau-:', P_taun_true.X(), P_taun_true.Y(), P_taun_true.Z(), P_taun_true.T(), P_taun_true.M())

            print('\nReco taus (numerically):\n')
            print('tau+:', P_taup_reco.X(), P_taup_reco.Y(), P_taup_reco.Z(), P_taup_reco.T(), P_taup_reco.M())
            print('tau-:', P_taun_reco.X(), P_taun_reco.Y(), P_taun_reco.Z(), P_taun_reco.T(), P_taun_reco.M())
            P_intersection, N_intersection, d_min_pred  = find_intersections(P_taup_reco, P_taun_reco, VERTEX_taup_rand, P_taup_pi1.Vect().Unit(), VERTEX_taun_rand, P_taun_pi1.Vect().Unit())
            # find clostest distance between N_intersection - P_intersection and 0,0,0

            print('P_intersection:', P_intersection.X(), P_intersection.Y(), P_intersection.Z())
            print('N_intersection:', N_intersection.X(), N_intersection.Y(), N_intersection.Z())
            closest_distance(P_intersection, N_intersection, ROOT.TVector3(0.,0.,0.))
            print('closest_distance:', closest_distance(P_intersection, N_intersection, ROOT.TVector3(0.,0.,0.)))
            print('d_min_numeric:', d_min_numeric.Mag(), d_min_numeric.X(), d_min_numeric.Y(), d_min_numeric.Z())
            print('d_min_reco:', d_min_reco.Mag(), d_min_reco.X(), d_min_reco.Y(), d_min_reco.Z())
            print('d_min_reco (unit):', d_min_reco.Unit().X(), d_min_reco.Unit().Y(), d_min_reco.Unit().Z())
            print('d_min constraint:', d_min_numeric.Unit().Dot(d_min_reco.Unit()))
            d_min_diff = (d_min_reco.X()-d_min_numeric.X())**2 + (d_min_reco.Y()-d_min_numeric.Y())**2 + (d_min_reco.Z()-d_min_numeric.Z())**2
            print('d_min_diff:', d_min_diff)

            print('d_true:', d_true.Mag(), d_true.X(), d_true.Y(), d_true.Z())
            d_reco = PredictD(P_taun_reco, P_taup_reco, P_taunvis, P_taupvis, d_min_reco)
            print('d_reco:', d_reco.Mag(), d_reco.X(), d_reco.Y(), d_reco.Z())
    
            print('\nReco taus (analytically):')
            print('\nsolution 1:')
            print('tau+:', solutions[0][0].X(), solutions[0][0].Y(), solutions[0][0].Z(), solutions[0][0].T(), solutions[0][0].M())
            print('tau-:', solutions[0][1].X(), solutions[0][1].Y(), solutions[0][1].Z(), solutions[0][1].T(), solutions[0][1].M())
            print('d_min_reco:', d_min_reco.Mag(), d_min_reco.X(), d_min_reco.Y(), d_min_reco.Z())
            print('d_min_prec:', solutions[0][2].Mag(), solutions[0][2].X(), solutions[0][2].Y(), solutions[0][2].Z())
            print('d_min constraint:', solutions[0][2].Unit().Dot(d_min_reco.Unit()))
            d_min_diff = (d_min_reco.X()-solutions[0][2].X())**2 + (d_min_reco.Y()-solutions[0][2].Y())**2 + (d_min_reco.Z()-solutions[0][2].Z())**2
            print('d_min_diff:', d_min_diff)
            print('d_true:', d_true.Mag(), d_true.X(), d_true.Y(), d_true.Z())
            #d_reco = PredictD(solutions[0][1], solutions[0][0], P_taunvis, P_taupvis, d_min_reco)
            #print('d_reco:', d_reco.Mag(), d_reco.X(), d_reco.Y(), d_reco.Z())

            P_intersection, N_intersection, d_min_pred  = find_intersections(solutions[0][0], solutions[0][1], VERTEX_taup, P_taup_pi1.Vect().Unit(), VERTEX_taun, P_taun_pi1.Vect().Unit())
            print('P_intersection:', P_intersection.X(), P_intersection.Y(), P_intersection.Z())
            print('N_intersection:', N_intersection.X(), N_intersection.Y(), N_intersection.Z())
            closest_distance(P_intersection, N_intersection, ROOT.TVector3(0.,0.,0.))
            print('closest_distance:', closest_distance(P_intersection, N_intersection, ROOT.TVector3(0.,0.,0.)))
            print('d_min_pred:', d_min_pred.Mag(), d_min_pred.X(), d_min_pred.Y(), d_min_pred.Z())

            print('\nsolution 2:')
            print('tau+:', solutions[1][0].X(), solutions[1][0].Y(), solutions[1][0].Z(), solutions[1][0].T(), solutions[1][0].M())
            print('tau-:', solutions[1][1].X(), solutions[1][1].Y(), solutions[1][1].Z(), solutions[1][1].T(), solutions[1][1].M())
            print('d_min_reco:',d_min_reco.Mag(), d_min_reco.X(), d_min_reco.Y(), d_min_reco.Z())
            print('d_min_pred:', solutions[1][2].Mag(), solutions[1][2].X(), solutions[1][2].Y(), solutions[1][2].Z())
            print('d_min constraint:', solutions[1][2].Unit().Dot(d_min_reco.Unit()))
            d_min_diff = (d_min_reco.X()-solutions[1][2].X())**2 + (d_min_reco.Y()-solutions[1][2].Y())**2 + (d_min_reco.Z()-solutions[1][2].Z())**2
            print('d_min_diff:', d_min_diff)
            print('d_true:', d_true.Mag(), d_true.X(), d_true.Y(), d_true.Z())
            #d_reco = PredictD(solutions[1][1], solutions[1][0], P_taunvis, P_taupvis, d_min_reco)
            #print('d_reco:', d_reco.Mag(), d_reco.X(), d_reco.Y(), d_reco.Z())    

            P_intersection, N_intersection, d_min_pred  = find_intersections(solutions[1][0], solutions[1][1], VERTEX_taup, P_taup_pi1.Vect().Unit(), VERTEX_taun, P_taun_pi1.Vect().Unit())
            print('P_intersection:', P_intersection.X(), P_intersection.Y(), P_intersection.Z())
            print('N_intersection:', N_intersection.X(), N_intersection.Y(), N_intersection.Z())
            closest_distance(P_intersection, N_intersection, ROOT.TVector3(0.,0.,0.))
            print('closest_distance:', closest_distance(P_intersection, N_intersection, ROOT.TVector3(0.,0.,0.)))
            print('d_min_pred:', d_min_pred.Mag(), d_min_pred.X(), d_min_pred.Y(), d_min_pred.Z())
            print('correct solution found?', FoundCorrectSolution)
            print('\n')

        d_min_reco_ave+=d_min_reco.Mag()
    d_min_reco_ave/=count_total

    print('ave d_min = %g' % d_min_reco_ave)
    print('Found correct solution for %i / %i events = %.1f%%' % (count_correct, count_total, count_correct/count_total*100))
