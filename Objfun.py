import numpy as np
from Glob_Vars import Glob_Vars
from Model_Proposed import Model_Proposed


def Objfun(Soln):
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))

    for i in range(v):
        soln = np.array(Soln)
        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        Feat = Glob_Vars.Feat
        Target = Glob_Vars.Target
        [Precision, Recall, F1Score] = Model_Proposed(Feat, sol, Target)
        Fitn[i] = 1/np.mean(Precision)+np.mean(Recall)
    return Fitn

