# import required modules
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.ticker as ticker

import mdlab as mdl

cnd = "contrast"
dur = ""
fn = "_".join([cnd, dur]) + ".npz"
# pdb.set_trace()
mdl.batch_mat2npz(fn)
print("completed")
print("completed")
print("completed")
