"""Example 1, using the example system from Novak et al. 2015"""
from pylab import *

from syncsweptsine import IirFilterKernel
from syncsweptsine import HammersteinModel
from syncsweptsine import SyncSweep
from syncsweptsine import HigherHarmonicImpulseResponse


nfft = 2**15
samplerate = 96000
# sweep params:
f1 = 2.4 
f2 = 24_000
dursec = 10

# Filter kernels for theoretical hammerstein model:
# the ARMA filters definition (ARMA order = 2, number of filters = N = 4)
A = [
    [1.0, -1.8996, 0.9025],
    [1.0, -1.9075, 0.9409],
    [1.0, -1.8471, 0.8649],
    ]
B = [
    [1.0, -1.9027, 0.9409],
    [1.0, -1.8959, 0.9025],
    [0.5, -0.9176, 0.4512],
    ]
orders = [1, 2, 3]
kernels_theo = [IirFilterKernel(*ba) for ba in zip(B, A)]
hm_theo = HammersteinModel(kernels_theo, orders)


# system identification of the theoretical system
sweep = SyncSweep(f1, f2, dursec, samplerate)
sweep_sig = sweep.get_windowed_signal(1024, 1024, pausestart=0, pausestop=512)
outsweep = hm_theo.filter(sweep_sig)
hhir = HigherHarmonicImpulseResponse.from_sweeps(sweep, outsweep, regularize=False)
hm_identified = HammersteinModel.from_higher_harmonic_impulse_response(
    hhir=hhir,
    length=nfft,
    orders=orders,
    delay=0,
    window=True,
)

# bode diagram of the theoretical and identification results
figure()
for theo, kernel, order in zip(hm_theo.kernels, hm_identified.kernels, orders):
    freq = kernel.freq
    G_kernel = kernel.frf
    freq_theo, G_kernel_theo = theo.freqz(nfft)

    ax = subplot(len(orders), 1, order )
    l0 = ax.semilogx(
        freq_theo/pi*samplerate/2, 
        20*log10(abs(G_kernel_theo)), 
        '-',
        color='cornflowerblue',
        label=f'|H| Theor. (order={order})'
        )
    l1 = ax.semilogx(
        freq, 
        20*log10(abs(G_kernel)),
        '--',
        color='darkblue', 
        label=f'|H| Estimate (order={order})'
        )
    xlim(4*f1, f2/2)
    ylim(-35, 35)
    ylabel('$|H|$ / dB')
    if order < max(orders): xticks([])
    grid()
    
    for ytlabel in ax.get_yticklabels(): ytlabel.set_color('b')

    ax2 = gca().twinx()
    ylim(-pi, pi)
    l2 = ax2.semilogx(
        freq_theo/pi*samplerate/2, 
        unwrap(angle(G_kernel_theo)), 
        '-',
        color='limegreen',
        label=f'$\\phi$ Theor. (order={order})'
        )
    phi_est = (angle(G_kernel*exp(-1j*freq*pi*nfft/hhir.samplerate)))
    l3 = ax2.semilogx(
        freq, 
        phi_est, 
        '--',
        color='darkgreen',
        label=f'$\\phi$ Estimate (order={order})'
        )
    for ytlabel in ax2.get_yticklabels(): ytlabel.set_color('g')
    ylabel('$\\phi$ / rad')
    grid()
    lines = l0 + l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    legend(lines, labels)
    xlabel('Frequency $f$ / Hz')

show()