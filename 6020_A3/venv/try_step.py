# try_step of Newton-hook. Programmed in lectures 13 -- ...
import numpy as np
from copy import copy


def try_step(x, dx, f, res):
    z = x + dx
    new_r = f(z)
    new_res = np.linalg.norm(new_r, 2)
    if new_res <= res:
        accept = 1
    else:
        accept = 0
    return accept
