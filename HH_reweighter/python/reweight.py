import imp
import math
import copy

class HHReweight:

    def __init__(self, params_map = {}, E='13TeV'):
        self.rw = imp.load_module('allmatrix2py', *imp.find_module('../rwgt_%s/rw_me/SubProcesses/allmatrix2py' % E))
        self.rw.set_madloop_path('rwgt_%s/SubProcesses/MadLoop5_resources' % E)
        self.params_map = params_map    

    def invert_momenta(self,p):
        """ fortran/C-python do not order table in the same order"""
        return [[p[j][i] for j in range(len(p))] for i in range(len(p[0]))]
    
    
    # code to do lorentz boosts taken from: https://github.com/MatthewDKnight/nanoAOD-tools/blob/eventIDSkimming/python/postprocessing/modules/reweighting/standalone_reweight.py

    def rotZ(self, p, angle):
        p[1], p[2] = p[1]*math.cos(angle)-p[2]*math.sin(angle),  p[1]*math.sin(angle)+p[2]*math.cos(angle)
        return p

    def rotY(self, p, angle):
        p[1], p[3] = p[1]*math.cos(angle)+p[3]*math.sin(angle),  -p[1]*math.sin(angle)+p[3]*math.cos(angle)
        return p

    def allboost(self, p, pboost):
        """
        Strategy here is to rotate pboost such that it lies along the z axis. Then boost pboost 
        so that it is in its rest frame. Perform the same transformations to p, and then undo
        the rotations at the end.
        """
        p, pboost = copy.copy(p), copy.copy(pboost) #force pass by value
    
        #rotate around z axis such that there is no y component
        if pboost[1]!=0:
            z_rot_angle = math.atan2(pboost[2],pboost[1])
        elif pboost[2]>0:
            z_rot_angle = math.pi/2
        else:
            z_rot_angle = -math.pi/2
    
        p = self.rotZ(p, -z_rot_angle)
        pboost = self.rotZ(pboost, -z_rot_angle)
    
        #rotate around y axis so pboost lies along z axis
        r = math.sqrt(pboost[1]**2+pboost[2]**2+pboost[3]**2)
        y_rot_angle = math.acos(pboost[3]/r)
    
        p = self.rotY(p, -y_rot_angle)
        pboost = self.rotY(pboost, -y_rot_angle)
    
        #perform Lorentz boost
        p = self.zboost(p, pboost)
        pboost = self.zboost(pboost, pboost)
    
        #undo rotations
        p = self.rotY(p, y_rot_angle)
        pboost = self.rotY(pboost, y_rot_angle)
        p = self.rotZ(p, z_rot_angle)
        pboost = self.rotZ(pboost, z_rot_angle)
    
        if abs(p[1]) < 1e-6 * p[0]:
            p[1] = 0
        if abs(p[2]) < 1e-6 * p[0]:
            p[2] = 0
        if abs(p[3]) < 1e-6 * p[0]:
            p[3] = 0
        if abs(pboost[1]) < 1e-6 * pboost[0]:
            pboost[1] = 0
        if abs(pboost[2]) < 1e-6 * pboost[0]:
            pboost[2] = 0
        if abs(pboost[3]) < 1e-6 * pboost[0]:
            pboost[3] = 0
    
        #check that transformations put pboost in rest frame
        assert pboost[1]==pboost[2]==pboost[3]==0
    
        return p
    
    
    def zboost(self, part, pboost=[]):
        """Both momenta should be in the same frame.
           The boost perform correspond to the boost required to set pboost at
           rest (only z boost applied).
        """
        E = pboost[0]
        pz = pboost[3]
    
        #beta = pz/E
        gamma = E / math.sqrt(E**2-pz**2)
        gammabeta = pz  / math.sqrt(E**2-pz**2)
    
        out =  [gamma * part[0] - gammabeta * part[3],
                            part[1],
                            part[2],
                            gamma * part[3] - gammabeta * part[0]]
    
        if abs(out[3]) < 1e-6 * out[0]:
            out[3] = 0
        return out

    def ReweightEvent(self, parts, alphas=0.137, nhel=-1):

        """
        parts is the list of particles where each particle is specified like [pdgid, E, px, py, pz],
        """

        # sort the particles by pdgid so that gluons are given first then higgs'
        parts = sorted(parts, key=lambda x: x[0])
        mode = -1
  
        if len(parts) == 4 and parts[0][0] == 21 and parts[1][0] == 21 and parts[2][0] == 25 and parts[3][0] == 25: 
            # if 2 gluons and 2 higgs bosons are included then this is likely reweighting LO events with the full information about the incoming and outgoing particles
            # TODO: add checks for E/momentum conservation between incoming and outgoing, if momentum is not conserved then switch to mode 2
            mode = 1

        if mode != 1 and sum(1 for part in parts if part[0] == 25) == 2:
            # if gluons are not provided then we default to approximate method (useful for NLO or LO with missing generator information is missing)
            # in this case we boost to the di-Higgs rest frame, construct incoming gluons in this rest frame, and sum over helicity states 

            mode = 2
            higgs = [part for part in parts if part[0] == 25]    
 
            pboost = [higgs[0][i] + higgs[1][i] for i in range(1,5)] 
    
            higgs_boosted_4vec = []
            for part in higgs:
                if (pboost[1]!=0) or (pboost[2]!=0): # if non-zero pt boost do boost in all directions
                    higgs_boosted_4vec.append(self.allboost(part[1:], pboost))
                else: # otherwise just boost along z
                    higgs_boosted_4vec.append(self.zboost(part[1:], pboost))
      
            # now we construct our 'artificial' gluons
            g1 = [21, higgs_boosted_4vec[0][0], 0., 0., higgs_boosted_4vec[0][0]] 
            g2 = [21, higgs_boosted_4vec[0][0], 0., 0., -higgs_boosted_4vec[0][0]] 
           
            h1 = [25] + higgs_boosted_4vec[0]
            h2 = [25] + higgs_boosted_4vec[1]

            parts = [g1, g2, h1, h2]
            nhel = -1

        if mode not in [1,2]:
            raise Exception("The particle content should include at least two Higgs bosons")

        pdgids = []
        part_4vecs = []
        for part in parts:
            pdgids.append(part[0])
            part_4vecs.append(part[1:])
        
        #TODO: Shift all higgs masses to 125 GeV to match the value used in the ME code 
        part_4vecs = self.invert_momenta(part_4vecs)
    
        scale2=0.

        me_vals = {}

        for name, params in params_map.items():
            self.rw.initialise(params) 
            val = self.rw.smatrixhel(pdgids,part_4vecs,alphas,scale2,nhel)
            me_vals[name] = val[0]   

        weights = {}
        if 'ref' not in me_vals:
            raise Exception("No reference ME was specified for the reweighting")
        
        for name, val in me_vals.items():
            if name == 'ref': continue
            weights[name] = val/me_vals['ref']

        return weights
 
if __name__ == '__main__':

    alphas = 1.01467400e-01
    
    parts = [
      [21, 1.4861514428e+03, 0.0000000000e+00, 0.0000000000e+00, 1.4861514428e+03],
      [21, 4.1482521106e+01, 0.0000000000e+00, 0.0000000000e+00, -4.1482521106e+01],
      [25, 4.2120989418e+02, 1.3960712742e+02, -1.1253971723e+02, 3.6005199204e+02],
      [25, 1.1064240698e+03, -1.3960712742e+02, 1.1253971723e+02, 1.0846169297e+03]
    ]
 
    params_map = {
        'ref': 'param_card_sm.dat', 
        'box': 'param_card_box.dat', 
    }

    rw = HHReweight(params_map)
    weights = rw.ReweightEvent(parts,alphas)
    print weights
