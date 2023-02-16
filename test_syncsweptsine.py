"""Test suite for `syncsweptsine.py`"""
import pytest
import numpy as np
from scipy import signal

from pytest import assume
from syncsweptsine import SyncSweep
from syncsweptsine import invert_spectrum_reg
from syncsweptsine import InvertedSyncSweepSpectrum
from syncsweptsine import HigherHarmonicImpulseResponse
from syncsweptsine import FrfFilterKernel
from syncsweptsine import IirFilterKernel
from syncsweptsine import HammersteinModel
from syncsweptsine import LinearModel
from syncsweptsine import hannramp, get_hann_win_flanks
from syncsweptsine import spectrum_to_minimum_phase


def test__typed_property():
    from syncsweptsine import _typed_property
    class T(object):
        prop_str = _typed_property(
            name='prop_str',
            expected_type=str,
            desc='This is a string property.')
        prop_scalar = _typed_property(
            name='prop_scalar',
            expected_type=(int, float),
            desc='This is a scalar type, accepting int and float.')

    tinst = T()
    TESTSTR = 'test123'
    tinst.prop_str = TESTSTR
    assume(tinst.prop_str == TESTSTR)
    assume(tinst._prop_str == TESTSTR)
    with pytest.raises(TypeError):
        tinst.prop_str = 1
    with pytest.raises(TypeError):
        tinst.prop_str = 0.1
    with pytest.raises(TypeError):
        tinst.prop_str = False
    TESTSCALAR = 19878215
    tinst.prop_scalar = TESTSCALAR
    assume(tinst.prop_scalar == TESTSCALAR)
    assume(tinst._prop_scalar == TESTSCALAR)
    tinst.prop_scalar = TESTSCALAR*3.1415
    with pytest.raises(TypeError):
        tinst.prop_scalar = '0.45234'

    assume(hasattr(tinst, '_typed_property_was_changed'))


def test_hannramp():
    nph = np.hanning(32)
    hramp = hannramp(np.ones(32), 16)
    assume(np.allclose(nph, hramp))
    hramp = hannramp(np.ones(32), 16, 16)
    assume(np.allclose(nph, hramp))
    hramp = hannramp(np.ones(32+16), 16, 32)
    nphr = np.hanning(64)
    assume(np.allclose(nph[:16], hramp[:16]))
    assume(np.allclose(nphr[32:], hramp[16:]))


@pytest.mark.parametrize(
    'startfreq,stopfreq,durationappr,samplerate', (
        (20000, 20, 5, 44100),
        (-20, 20000, 5, 48000),
        (20, -20000, 5, 48000),
        (20, 20000, -1, 48000),
        (20, 20000, 5, -10),
        (16, 16000, 5, 1000)
        ))
def test_sync_sweep_instantiation_errors(
        startfreq, stopfreq, durationappr, samplerate):
    with pytest.raises(ValueError):
        SyncSweep(startfreq, stopfreq, durationappr, samplerate)


def test_sync_sweep_instantiation():
    SyncSweep(20, 20000, 4, 44100)
    SyncSweep(16, 16000, 5, 48000)
    with pytest.raises(TypeError):
        # must not be callable without arguments.
        SyncSweep()



@pytest.mark.parametrize(
    'startfreq,stopfreq,durationappr,samplerate', (
        (20, 20000, 5, 44100),
        (20, 20000, 3.5, 48000),
        (20, 20000, 2, 96000),
        (20, None, 2, 1000),
        (20, None, 1, 580),
        ))
def test_sync_sweep(startfreq, stopfreq, durationappr, samplerate):
    sweep = SyncSweep(startfreq, stopfreq, durationappr, samplerate)

    assume(sweep.__repr__().startswith('SyncSweep('))
    assume(sweep.startfreq == startfreq)

    if stopfreq is None:
        assume(sweep.stopfreq == samplerate / 2)
    else:
        assume(sweep.stopfreq == stopfreq)

    assume(sweep.durationappr == durationappr)
    assume(sweep.samplerate == samplerate)
    assume(sweep.duration > 0)  # better test is welcome

    # error between dur and approx dur is smaller than 5%:
    assume((sweep.duration/sweep.durationappr-1) < 0.05)

    assume(sweep.sweepperiod > 0)

    assume(len(sweep.signal) == int(sweep.duration*sweep.samplerate)+1)
    assume(len(sweep.time) == len(sweep.signal))

    assume(abs(sweep.signal[0]) < 1e-12 ) # first sample is zero
    assume(abs(sweep.signal.mean()) < 1e-2)  # average signal converges to zero
    assume(abs(abs(sweep.signal).max()-1) < 1e-6)  # amplitude is one.
    assume(np.all(sweep.signal[::-1] == sweep[::-1]))
    wsig = sweep.get_windowed_signal(
        left=128,
        right=128,
        pausestart=1024,
        pausestop=2048,
        amplitude=0.5)
    assume(len(wsig)==(len(sweep.signal)+1024+2048))
    assume(np.allclose(max(abs(wsig)), 0.5))

def test_invert_spectrum_reg():
    x = np.random.randn(128)
    X = np.fft.rfft(x)

    beta = 1.0
    expected = X.conj() / (X*X.conj() + beta)
    assume(np.all(expected == invert_spectrum_reg(X, beta)))

    beta = np.random.rand(len(X)) + 0.1
    expected = X.conj() / (X*X.conj() + beta)
    assume(np.all(expected == invert_spectrum_reg(X, beta)))

def test_spectrum_to_minimum_phase():
    b = [1]
    a = [1, 0.8]
    _, H = signal.freqz(b, a, whole=True)
    min_phase = spectrum_to_minimum_phase(H)
    assume(np.allclose(np.unwrap(np.angle(H)), min_phase))

def test_inverted_sync_sweep_spectrum():
    FFTLEN = 1024
    sweep = SyncSweep(10, 1000, 2, 2000)
    ispec = InvertedSyncSweepSpectrum.from_sweep(sweep, fftlen=FFTLEN)
    assume(ispec.fftlen == FFTLEN)
    assume(len(ispec.freq) == (FFTLEN//2+1))
    assume(len(ispec.spectrum) == (FFTLEN//2+1))
    assume(ispec.spectrum[0] == 0j)
    expected_invspec = np.zeros_like(ispec.spectrum)
    expected_invspec[0] = 0j
    expected_invspec[1:] = (
        2*np.sqrt(ispec.freq[1:]/sweep.sweepperiod)
        *np.exp(-2j*np.pi*ispec.freq[1:]*sweep.sweepperiod
        *(1-np.log(ispec.freq[1:]/sweep.startfreq))
        +1j*np.pi/4)
        )
    assume(np.allclose(ispec.spectrum, expected_invspec))

    ispec.fftlen = 2*FFTLEN
    assume(ispec.fftlen == 2*FFTLEN)
    assume(len(ispec.freq) == (2*FFTLEN//2+1))
    assume(len(ispec.spectrum) == (2*FFTLEN//2+1))
    assume(len(ispec) == len(ispec.spectrum))
    assume(np.all(ispec.spectrum[::-1] == ispec[::-1]))
    assume(np.array(ispec, dtype='complex64').dtype == np.complex64)
    assume(ispec.__repr__().startswith('InvertedSyncSweepSpectrum('))


def test_higher_harmonic_impulse_response():
    sweep = SyncSweep(10, 10000, 5, 20000)
    hhir = HigherHarmonicImpulseResponse.from_sweeps(sweep, sweep)
    assume(np.all(np.array(hhir) == hhir.hhir))
    hir = hhir.harmonic_impulse_response(order=1, length=1024, delay=-512, window=True)
    assume(type(hir) == np.ndarray)
    assume(len(hir) == 1024)
    assume(np.argmax(hir)==512)
    hir = hhir.harmonic_impulse_response(order=1, length=1024, delay=-512, window=np.linspace(0, 1, 1024))
    assume(type(hir) == np.ndarray)
    assume(len(hir) == 1024)
    assume(np.argmax(hir)==512)


def test_frf_filter_kernel():
    x = np.random.randn(128)*1e-3
    X = np.fft.rfft(x)
    freq = np.fft.rfftfreq(len(x))
    fk = FrfFilterKernel(freq, X)
    y = np.zeros(128)
    y[0] = 1
    fk.filter(y)
    assume(np.all(fk.freq == freq))
    assume(np.all(fk.frf == np.array(fk)))
    assume(fk.__repr__().startswith('FrfFilterKernel('))
    with pytest.raises(ValueError):
        FrfFilterKernel(np.zeros(32), np.zeros(16))

    fkir = FrfFilterKernel.from_ir(fk.ir, 1)
    assume(np.allclose(fkir.frf, fk.frf))


def test_iir_filter_kernel():
    from scipy.signal import lfilter, freqz
    bcoeff = [1.0, -1.8996, 0.9025]
    acoeff = [1.0, -1.9027, 0.9409]
    fk = IirFilterKernel(bcoeff, acoeff)
    fk.filter([1, 0, 0, 0])
    x = [1]+127*[0]
    assume(np.all(fk._bcoeff == bcoeff))
    assume(np.all(fk._acoeff == acoeff))
    assume(np.all(fk.filter(x) == lfilter(bcoeff, acoeff, x)))
    freq, frf = fk.freqz(128)
    f, r = freqz(bcoeff, acoeff, 128//2+1)
    assume(np.all(freq == f))
    assume(np.all(frf == r))
    frk = fk.to_frf_filter_kernel(128)
    assume(np.all(frk.frf == frf))
    assume(np.all(frk.freq == freq))
    assume(fk.__repr__().startswith('IirFilterKernel('))

def test_hammerstein_model():

    sweep = SyncSweep(startfreq=16, stopfreq=16000, durationappr=10, samplerate=96000)

    def nonlinear_system(sig):
        return 1.0 * sig + 0.25 * sig**2 + 0.125 * sig**3

    outsweep = nonlinear_system(np.array(sweep))
    hhir = HigherHarmonicImpulseResponse.from_sweeps(sweep, outsweep, regularize=False)
    hm = HammersteinModel.from_higher_harmonic_impulse_response(
        hhir, 2048, orders=(1, 2, 3), delay=-1024)  # -delay since irs are acausal in this case.
    assume(hm.orders==(1, 2, 3))
    x = np.array([1]+127*[0])
    hm.filter(x)
    assume(hm.__repr__().startswith('HammersteinModel('))
    expectedcoeffs = [1.0, 0.25, 0.125]
    for kernel, expc in zip(hm.kernels, expectedcoeffs):
        print(abs(np.median(abs(kernel.frf[100:200]))), expc)
        assume(abs(np.median(abs(kernel.frf[100:200]))-expc) < 0.01)

    with pytest.raises(ValueError):
        hm = HammersteinModel.from_higher_harmonic_impulse_response(
            hhir, hhir.max_hir_length(1)+1, orders=(1, 2, 3), delay=-1024)
    lm = LinearModel.from_hammerstein_model(hm)


def test_linear_model():
    sweep = SyncSweep(startfreq=16, stopfreq=16000, durationappr=10, samplerate=96000)

    def nonlinear_system(sig):
        return 0.5 * sig + 0.25 * sig**2 + 0.125 * sig**3

    outsweep = nonlinear_system(np.array(sweep))
    hhir = HigherHarmonicImpulseResponse.from_sweeps(sweep, outsweep)
    lm = LinearModel.from_higher_harmonic_impulse_response(hhir, length=1024)
    x = np.array([1]+127*[0])
    y = lm.filter(x)
    with pytest.raises(ValueError):
        hm = LinearModel.from_higher_harmonic_impulse_response(
            hhir, hhir.max_hir_length(1)+1, delay=0)


if __name__ == "__main__":
    pytest.main()