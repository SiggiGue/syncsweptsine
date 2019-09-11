import numpy as np
from syncsweptsine import SyncSweep
from syncsweptsine import HigherHarmonicImpulseResponse
from syncsweptsine import HammersteinModel
import matplotlib.pyplot as plt

sweep = SyncSweep(startfreq=16, stopfreq=16000, durationappr=10, samplerate=96000)

def nonlinear_system(sig):
    return 1.0 * sig + 0.25 * sig**2 + 0.125 * sig**3

outsweep = nonlinear_system(np.array(sweep))
# hhir = HigherHarmonicImpulseResponse.from_sweeps(sweep, outsweep)
#hm = HammersteinModel.from_higher_harmonic_impulse_response(
#    hhir, 2048, orders=(1, 2, 3), delay=0)
hm = HammersteinModel.from_sweeps(sweep, outsweep, orders=(1, 2, 3), regularize=False)
for kernel, order in zip(hm.kernels, hm.orders):
    plt.plot(abs(kernel.frf))
    print('Coefficient estimate of nonlinear system:', 
            np.round(np.max(abs(kernel.frf[500:1000])), 3), 
            'Order', 
            order)

plt.show()
