"""This module implements the Synchronized Swept Sine Method according to Nowak et al. 2015 as reusable python module.

It can be used for the system identification of linear and nonlinear systems.
The identification results can be represented as Hammerstein models (Diagonal Volterra Series).
Furthermore simple regularization is provided as optional feature.


Classes
-------

High level classes:

- :class:`SyncSweep`: defines the synchronized sweep model

- :class:`HigherHarmonicImpulseResponse`: defines the Higher harmonic impulse response
  e.g. by deconvolution of the reference SyncSweep instance and
  the actual measured sweep signal at the output of the system under test.

- :class:`HammersteinModel`: defines the generalized hammerstein model based on a list of kernels and corresponding nonlinearity orders.

Low level classes:

- :class:`InvertedSyncSweepSpectrum`: defines the inverted spectrum of a synchronized sweep.

- :class:`FrfFilterKernel`: defines a filter kernel based on a frequency response function.

- :class:`IirFilterKernel`: defines a filter kernel based on IIR filter coefficients.


Examples
--------

Estimating the coefficients of a simple nonlinear system:

.. code::

    import numpy as np
    from syncsweptsine import SyncSweep
    from syncsweptsine import HigherHarmonicImpulseResponse
    from syncsweptsine import HammersteinModel

    sweep = SyncSweep(startfreq=16, stopfreq=16000, durationappr=10, samplerate=96000)

    def nonlinear_system(sig):
        return 1.0 * sig + 0.25 * sig**2 + 0.125 * sig**3

    outsweep = nonlinear_system(np.array(sweep))
    hhir = HigherHarmonicImpulseResponse.from_sweeps(sweep, outsweep)
    hm = HammersteinModel.from_higher_harmonic_impulse_response(
        hhir, 2048, orders=(1, 2, 3), delay=0)
    for kernel, order in zip(hm.kernels, hm.orders):
        print('Coefficient estimate of nonlinear system:',
                np.round(np.percentile(abs(kernel.frf), 95), 3),
                'Order',
                order)

    Out[7]:
    Coefficient estimate of nonlinear system: 1.009 Order 1
    Coefficient estimate of nonlinear system: 0.25 Order 2
    Coefficient estimate of nonlinear system: 0.125 Order 3


Estimating the Hammerstein model of a theoretically created Hammerstein model usin IIR kernels:

.. code::

    from pylab import *

    from syncsweptsine import IirFilterKernel
    from syncsweptsine import HammersteinModel
    from syncsweptsine import SyncSweep
    from syncsweptsine import HigherHarmonicImpulseResponse


    nfft = 1024
    samplerate = 96000
    # sweep params:
    f1 = 1.2
    f2 = 16_000
    dursec = 30

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
    hhir = HigherHarmonicImpulseResponse.from_sweeps(sweep, outsweep)
    hm_identified = HammersteinModel.from_higher_harmonic_impulse_response(
        hhir=hhir,
        length=nfft,
        orders=orders,
        delay=0,
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
            'b-',
            label=f'|H| Theor. (order={order})'
            )
        l1 = ax.semilogx(
            freq,
            20*log10(abs(G_kernel)),
            '--',
            color='skyblue',
            label=f'|H| Estimate (order={order})'
            )
        xlim(4*f1, f2)
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
            'g-',
            label=f'$\\phi$ Theor. (order={order})'
            )
        phi_theo = unwrap(angle(G_kernel*exp(-1j*freq*pi*nfft/hhir.samplerate)))
        l3 = ax2.semilogx(
            freq,
            phi_theo,
            '--',
            color='lightgreen',
            label=f'$\\phi$ Estimate (order={order})'
            )
        for ytlabel in ax2.get_yticklabels(): ytlabel.set_color('g')
        ylabel('$\\phi$ / rad')
        grid()
        lines = l0 + l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        legend(lines, labels)
        xlabel('Frequency $f$ / Hz')

"""

from textwrap import dedent as _dedent
from scipy.special import binom as _binom
import scipy.signal as _spsignal
import numpy as _np


def _typed_property(name, expected_type, desc=None):
    """Returns a type checked property

    Parameters
    ----------
    name : str
    expected_type : type or tuple of types
    desc : str
        Description for this property.

    Returns
    -------
    prop : property

    Example
    -------
    >>> class Apple:
    >>>     color = typed_property('color', str, 'Color of the apple.')
    >>>     def __init__(self, color):
    >>>         self.color = color

    >>> Apple('red')
    Out[6]: <__main__.Apple at 0x7f92b84913c8>

    >>> badapple = Apple('brown')
    >>> badapple.color = 1
    TypeError: color must be of type <class 'str'> not <class 'int'>

    """
    storage_name = ''.join(('_' , name))

    @property
    def prop(self):
        return getattr(self, storage_name)

    @prop.setter
    def prop(self, value):
        if isinstance(value, expected_type) or value is None:
            setattr(self, storage_name, value)
            self._typed_property_was_changed = True
        else:
            raise TypeError('{} must be of type {} not {}'.format(
                name, expected_type, type(value)))

    if desc is not None:
        prop.__doc__ = str(desc)

    return prop


def get_hann_win_flanks(left, right=None):
    if right is None:
        right = left
    navg = left + left - 1
    winleft = _np.sin((_np.pi/navg) * _np.arange(left))**2
    if left == right:
        winright = winleft[::-1]
    else:
        navg = right + right - 1
        winright = _np.sin((_np.pi/navg) * _np.arange(right, right+right))**2
    return winleft, winright


def hannramp(sig, left, right=None):
    """Retruns faded signal faded with hanning flanks.

    Parameter
    ---------
    sig: ndarray
    left: int
        Number of samples to fade left.
    right: int
        Number of samples to fade right.

    Returns
    -------
    sigfaded: ndarray
        Signal faded with hanning ramps.

    Warning
    -------
    The input signal is modified. If you don't want your providing signal
    variable be modified, pleas create a copy of the signal e.g. np.array(sig).

    """
    winleft, winright = get_hann_win_flanks(left, right)
    right = right or left
    sig[:left] = sig[:left] * winleft
    sig[-right:] = sig[-right:] * winright
    return sig


class SyncSweep(object):
    """Synchronized Swept Sine Signal Model

    Parameters
    ----------
    startfreq : scalar
        Start frequency of sweep in Hz
    stopfreq : scalar
        Stop frequency of sweep in Hz
    durationappr : scalar
        Approximate duration in seconds
    samplerate : scalar
        Samplerate of the signal in Hz.

    Returns
    -------
    sweep : SyncSweep

    Examples
    --------
    >>> sweep = SyncSweep(16, 16000, 5, 44100)

    .. plot::

        import matplotlib.pyplot as plt
        from syncsweptsine import SyncSweep
        plt.subplot(211)
        plt.plot(SyncSweep(16, 64, 1, 44100))
        plt.title('Example Sweep')
        plt.xlim([0, 43000])
        plt.ylabel('amplitude')
        plt.subplot(212)
        plt.specgram(SyncSweep(200, 20050, 1, 44100), NFFT=512, noverlap=256, Fs=1);
        plt.ylabel('frequency')
        plt.xlabel('sample')
        plt.xlim([0, 43000])
        plt.show()



    """
    _typed_property_was_changed = True
    startfreq = _typed_property(
        name='startfreq',
        expected_type=(float, int),
        desc='Start frequency in Hz')
    stopfreq = _typed_property(
        name='stopfreq',
        expected_type=(float, int),
        desc='Stop frequency in Hz')
    durationappr = _typed_property(
        name='durationappr',
        expected_type=(float, int),
        desc='Approximate/planned duration in seconds.')
    samplerate = _typed_property(
        name='samplerate',
        expected_type=(float, int),
        desc='Sample rate of the signal in Hz.')


    def __init__(self,
                 startfreq,
                 stopfreq,
                 durationappr,
                 samplerate):
        SyncSweep._check_parameters(startfreq, stopfreq, durationappr, samplerate)
        self.startfreq = startfreq
        self.stopfreq = stopfreq
        self.durationappr = durationappr
        self.samplerate = samplerate
        self._logfreqratio = None
        self._kappa = None
        self._duration = None
        self._sweepperiod = None
        self._time = None
        self._phi = None
        self._signal = None
        self._update()

    @property
    def signal(self):
        """Returns the sweep time signal."""
        self._update()
        return self._signal

    @property
    def duration(self):
        """Actual duration of the sweep."""
        self._update()
        return self._duration

    @property
    def sweepperiod(self):
        """Returns the sweep period
        according to symbol $L$ in the paper.
        """
        self._update()
        return self._sweepperiod

    @property
    def time(self):
        """Time vector
        relating to given samplerate and actual duration."""
        self._update()
        return self._time

    def _update(self):
        """Updates the sweep if properties were changed."""
        if self._typed_property_was_changed:
            self._calculate_sweep()
            self._typed_property_was_changed = False

    def _calculate_sweep(self):
        """This method calculates the actual sweep
        using current state of input parameters.

        Some interim results variables will be available
        as readonly properties.

        """
        self.stopfreq = SyncSweep._limit_stopfreq(self.stopfreq, self.samplerate)
        startfreq = self.startfreq
        stopfreq = self.stopfreq
        durationappr = self.durationappr
        samplerate = self.samplerate
        SyncSweep._check_parameters(startfreq, stopfreq, durationappr, samplerate)
        logfreqratio = _np.log(stopfreq/startfreq)  # ln(f2/f1)

        # symbol $k$, eq. 32 from paper
        kappa = _np.round(startfreq*durationappr/logfreqratio)

        # symbol $T$ in paper
        duration = kappa * logfreqratio / startfreq

        # symbol L in paper
        sweepperiod = kappa / startfreq
        dt = 1.0 / samplerate
        time = _np.arange(0, duration, dt)
        # eq. 33 from paper
        phi = 2*_np.pi*startfreq*sweepperiod*_np.exp(time/sweepperiod)
        sweep = _np.sin(phi)

        # keep as private attributes
        self._logfreqratio = logfreqratio
        self._kappa = kappa

        # make accessible through readonly properties
        self._sweepperiod = sweepperiod
        self._duration = duration
        self._time = time
        self._signal = sweep

    def get_windowed_signal(self, left, right, pausestart=0, pausestop=0, amplitude=1):
        """Returns windowd sweep signal

        The sweep time signal will be faded in and out by hanning ramps.

        Parameters
        ----------
        left : int
            Number of samples for fade in hanning ramp at start of the sweep.
        right : int
            Number of samples for fade out hanning ramp at end of the sweep.
        pausestart : int
            Number of samples for pause befor windowed sweep starts. default is 0.
        pausestop : int
            Number of samples for pause after windowed sweep stopps. default is 0.
        amplitude : scalar
            Cahnge the amplitude of the sweep. default is 1

        """
        return _np.concatenate((
            _np.zeros(pausestart),
            hannramp(self.signal, left, right)*amplitude,
            _np.zeros(pausestop)
            ))

    @staticmethod
    def _limit_stopfreq(value, samplerate):
        """Returns a value that is <= nyquist frequency."""
        return value or 0.5*samplerate

    @staticmethod
    def _check_parameters(startfreq, stopfreq, durationappr, samplerate):
        """Checks the parameters for a synchronized sweep, raises exceptions if neccessary."""
        stopfreq = SyncSweep._limit_stopfreq(stopfreq, samplerate)
        if startfreq < 0 or stopfreq < 0:
            raise ValueError(
                '`startfreq` (={}) and `stopfreq` (={}) must be bigger than 0.'.format(
                startfreq, stopfreq))
        if stopfreq and startfreq >= stopfreq:
            raise ValueError(
                '`startfreq` (={}) must be smaller than `stopfreq` (={}).'.format(
                startfreq, stopfreq))
        if durationappr <= 0:
            raise ValueError('`durationappr` ', durationappr, ' must be bigger than 0.')
        if samplerate <= 0:
            raise ValueError('samplerate must be bigger than 0.')
        if samplerate < 2*max(startfreq, stopfreq):
            raise ValueError(
                '`samplerate` must be at least twice as big as '
                '`startfreq` and `stopfreq`.')

    def __getitem__(self, index):
        """Allows slicing of the SyncSweep instance"""
        return self.signal[index]

    def __array__(self, dtype=None):
        """Support ndarray casting."""
        if dtype:
            return self.signal.astype(dtype)
        return self.signal

    def __len__(self):
        """Returns the length of the time signal."""
        return len(self.signal)

    def __repr__(self):
        """Nice reprint of the instance with actual parameters."""
        return ("SyncSweep(\n"
            "    startfreq={},\n"
            "    stopfreq={},\n"
            "    duration={},\n"
            "    samplerate={})\n"
            ).format(
                self.startfreq,
                self.stopfreq,
                self.duration,
                self.samplerate)


def invert_spectrum_reg(spec, beta):
    """Returns inverse spec with regularization by beta

    Parameters
    ----------
    spec : ndarray
        Complex spectrum.
    beta : ndarray or scalar
        Regularization parameter.
        Either of same size as spec or a scalar value.

    Returns
    -------
    invspec : ndarray

    """
    return spec.conj() / (spec*spec.conj() + beta)


def spectrum_to_minimum_phase(spec):
    """Returns a minimum-phase spectrum for given complex `spec`

    Parameters
    ----------
    spec : ndarray
        Spectrum (must be twosided)

    Returns
    -------
    minphase : ndarray

    """
    return _np.unwrap(-_np.imag(_spsignal.hilbert(_np.log(_np.abs(spec)))))


class InvertedSyncSweepSpectrum(object):
    """Inverted Spectrum of Synchronized Swept Sine Signal Model
    Creates the analytical solution of the spectrum according to eq. 43.

    Parameters
    ----------
    samplerate : scalar
        Sample rate of the sweep signal.
    sweepperiod : scalar
        Sweep period of the sweep signal.
    startfreq : scalar
        Start frequency of the sweep signal.
    stopfreq : scalar
        Stop frequency of the sweep signal.
    fftlen : int
        Number of spectral bins.

    Notes
    -----
    If you want to invert a SyncSweep instance use the :func:`InvertedSyncSweepSpectrum.from_sweep()`.

    Returns
    -------
    ispec : InvertedSyncSweepSpectrum instance

    Examples
    --------
    >>> sweep = SyncSweep(16, 16000, 5, 44100)
    >>> inv_sweep = InvertedSyncSweepSpectrum.from_sweep(sweep)


    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from syncsweptsine import SyncSweep, InvertedSyncSweepSpectrum
        inv_spec = InvertedSyncSweepSpectrum.from_sweep(SyncSweep(16, 16000, 1, 44100), 8192)
        plt.subplot(211)
        plt.plot(inv_spec.freq, abs(inv_spec.spectrum))
        plt.ylabel('magnitude')
        plt.subplot(212)
        plt.plot(inv_spec.freq, np.angle(inv_spec.spectrum))
        plt.ylabel('phase')
        plt.xlabel('frequency')
        plt.show()
        
    See also
    --------
    :func:`InvertedSyncSweepSpectrum.from_sweep`

    :class:`SyncSweep`

    """

    def __init__(self,
                 samplerate,
                 sweepperiod,
                 startfreq,
                 stopfreq,
                 fftlen):
        self._samplerate = samplerate
        self._sweepperiod = sweepperiod
        self._startfreq = startfreq
        self._stopfreq = stopfreq
        self._fftlen = fftlen
        self._spectrum = None
        self._freq = None
        self._changes = True
        self._update()

    @classmethod
    def from_sweep(cls, syncsweep, fftlen):
        """Returns a InvertedSyncSweepSpectrum instance for given syncsweep.
        Creates the analytical solution of the spectrum according to eq. 43.

        Parameters
        ----------
        syncsweep : SyncSweep
            Instance of a SyncSweep.from_syncsweep
        fftlen : int
            Length of fft for spectrum creation
        """
        return cls(
            samplerate=syncsweep.samplerate,
            sweepperiod=syncsweep.sweepperiod,
            startfreq=syncsweep.startfreq,
            stopfreq=syncsweep.stopfreq,
            fftlen=fftlen)

    @property
    def spectrum(self):
        """The inverted spectrum."""
        self._update()
        return self._spectrum

    @property
    def freq(self):
        """Frequency vector for the spectrum"""
        self._update()
        return self._freq

    @property
    def fftlen(self):
        """Number of fft bins."""
        return self._fftlen

    @fftlen.setter
    def fftlen(self, value):
        """Set the number of fft bins."""
        self._changes = True
        self._fftlen = int(value)

    def _update(self):
        if self._changes:
            self._calculate_inverted_sweep_spectrum()
            self._changes = False

    def _calculate_inverted_sweep_spectrum(self):
        samplerate = self._samplerate
        sweepperiod = self._sweepperiod
        startfreq = self._startfreq
        freq = _np.linspace(0, samplerate/2, int(_np.round(self.fftlen/2+1)))
        spectrum = _np.zeros_like(freq, dtype=_np.complex_)
        # eq. 43 definition of the inverse spectrum in frequency domain
        spectrum[1:] = (
            2*_np.sqrt(freq[1:]/sweepperiod) 
            * _np.exp(-2j * _np.pi*freq[1:]*sweepperiod
                * (1 - _np.log(freq[1:]/startfreq))
                + 1j * _np.pi/4))
        self._spectrum = spectrum
        self._freq = freq

    def __getitem__(self, index):
        """Allow slicing of the spectrum."""
        return self.spectrum[index]

    def __array__(self, dtype=None):
        """Allows creation of ndarrays, returning the spectrum."""
        if dtype:
            return self.spectrum.astype(dtype)
        else:
            return self.spectrum

    def __len__(self):
        """Returns the length of the spectrum."""
        return len(self.spectrum)

    def __repr__(self):
        """Nice reprinting with actual parameters."""
        return (
            """InvertedSyncSweepSpectrum(
                fftlen={}, samplerate={}, sweepperiod={}, startfreq={})
            """.format(
                self.fftlen,
                self._samplerate,
                self._sweepperiod,
                self._startfreq)
            )


class HigherHarmonicImpulseResponse(object):
    """Higher Harmonic Impulse Response
    Signal containing Impulse responsens for all harmonics.

    To create a HigherHarmonicImpulseResponse from sweep input and output signals,
    use the HigherHarmonicImpulseResponse.from_sweeps() class method.

    Parameters
    ----------
    hhir : ndarray
        Higher Harmonic Impulse Response array.
    hhfrf : ndarray
        Higher Harmonic Frequency Response Function array. Optional.
        Will be available if .from_sweeps() method is used.
    sweepperiod : scalar
        Sweep period of the used sweep.
        Needed for calculation of time position of harmonic impulse responses.
    samplerate : scalar

    Returns
    -------
    hhir : HigherHarmonicImpulseResponse

    Notes
    -----
    To create a HigherHarmonicImpulseResponse from sweep input and output signals,
    use the HigherHarmonicImpulseResponse.from_sweeps() class method.

    Examples
    --------
    >>> sweep = SyncSweep(16, 16000, 5, 44100)
    >>> sig = sweep.get_windowed_signal(4096, 4096, 2*8192, 4*8192)
    >>> measured = sig + 0.5*sig**2 + 0.25*sig**3
    >>> hhir = HigherHarmonicImpulseResponse.from_sweeps(sweep, measured)

    .. plot::

        import matplotlib.pyplot as plt
        from syncsweptsine import SyncSweep, HigherHarmonicImpulseResponse
        sweep = SyncSweep(10, 10000, 5, 80000)
        sig = sweep.get_windowed_signal(4096, 4096, 2*8192, 4*8192)
        measured = sig + 0.5*sig**2 + 0.25*sig**3
        hhir = HigherHarmonicImpulseResponse.from_sweeps(sweep, measured)
        plt.plot(hhir)
        plt.xlim([0, len(hhir.hhir)])
        plt.xlabel('sample')
        plt.ylabel('amplitude')
        plt.title('Higher Harmonic Impulse Response')
        plt.show()

    See also
    --------
    :func:`HigherHarmonicImpulseResponse.from_sweeps`

    :func:`HigherHarmonicImpulseResponse.from_spectra`

    :class:`SyncSweep`

    :class:`HammersteinModel`

    """
    def __init__(self, hhir=None, hhfrf=None, sweepperiod=None, samplerate=None):
        self.hhir = hhir
        self.hhfrf = hhfrf
        self._sweepperiod = sweepperiod
        self._samplerate = samplerate

    @property
    def samplerate(self):
        """Returns the Samplerate of the impulse response."""
        return self._samplerate

    def hir_time_position(self, order):
        """Returns the time delay for the harmonic impulse response of `order`."""
        if order == 1:
            return 0
        else: 
            return len(self.hhir)/self._samplerate - self._sweepperiod * _np.log(order)

    def hir_sample_position(self, order):
        """Returns the sample delay for the harmonic impulse response of `order`."""
        return int(self.hir_time_position(order) * self._samplerate)

    def hir_index(self, order, length, delay=0):
        """Returns the index the harmonic impulse response of `order` and `length`.

        Parameters
        ----------
        order : int
            Order of required harmonic impulse response.
        length : int
            Length of required harmonic impulse response.
        delay : int
            Delay of system under test the hhir was derived from.

        """
        return _np.arange(length, dtype=int)+int(delay)+self.hir_sample_position(order)

    def max_hir_length(self, order):
        """Returns the maximum length of mpulse responses for given orders.

        Parameters
        ----------
        order: int

        Returns
        -------
        maxlength : int

        Notes
        -----
        The HHIR contains all harmonic impulse responses (HIR).
        For slicing one specific HIR there is a maximum number of
        samples around this HIR. A bigger slice may contain parts
        of neighbouring HIRs.
        Depending on the highest order there is a maximum length.

        """
        if order == 1:
            return len(self.hhir)//2-1
        else:
            distance = self.hir_sample_position(order) - self.hir_sample_position(order + 1)
        return distance

    def harmonic_impulse_response(self, order, length=None, delay=0, window=None):
        """Returns the harmonic impulse response of `order` and `length`

        Parameters
        ----------
        order : int
            Order of required harmonic impulse response.
        length : int
            Length of required harmonic impulse response.
        delay : int
            Delay of system under test the hhir was derived from.

        """
        length = length or self.max_hir_length(order)
        sig = _np.take(
            self.hhir,
            self.hir_index(order, length, delay),
            mode='wrap')
        if _np.any(window):
            if isinstance(window, (int, _np.integer)):
                if window == True:
                    window = length//2
                sig = hannramp(sig, left=window, right=window)
            elif (type(window) == _np.ndarray) and len(window):
                sig = sig*window
            else:   
                raise ValueError(f'could not interpret window input {window}')
        return sig

    @classmethod
    def from_sweeps(cls, syncsweep, measuredsweep, fftlen=None, regularize=1e-6):
        """Returns  Higher Harmonic Impulse Response instance for given sweep signals.

        Parameters
        ----------
        syncsweep : SyncSweep
            A SyncSweep instance.
        measuredsweep : ndarray
            Measured sweep.
            Must be the output signal of the system under test excited with the provided `syncsweep`.
            Besides it must be sampled at the same samplerate as the provided syncsweep.
        fftlen : int
            Length of the calculated ffts. fftlen will be guessed from measuredsweep length if fftlen is None.

        """
        fftlen = fftlen or int(2**_np.ceil(1+_np.log2(len(measuredsweep))))
        rspec = _np.fft.rfft(measuredsweep, fftlen)
        rinvspec = InvertedSyncSweepSpectrum.from_sweep(syncsweep, fftlen=fftlen).spectrum
        freq = _np.fft.rfftfreq(fftlen, 1/syncsweep.samplerate)
        if regularize is not False and regularize is not None:
            sweepspec = _np.fft.rfft(syncsweep, fftlen)
            if _np.isscalar(regularize):
                regu = _np.ones_like(rinvspec)*regularize
                regu[freq<=syncsweep.startfreq] = 1/regularize
                regu[freq>=syncsweep.stopfreq] = 1/regularize
                regularize = regu
            reguspec = invert_spectrum_reg(rinvspec*sweepspec, beta=regularize)
            rinvspec[1:] = rinvspec[1:]*reguspec[1:]
        else:
            rspec /= syncsweep.samplerate
        return cls.from_spectra(
            rspec=rspec,
            rinvspec=rinvspec,
            sweepperiod=syncsweep.sweepperiod,
            samplerate=syncsweep.samplerate)

    @classmethod
    def from_spectra(cls, rspec, rinvspec, sweepperiod, samplerate):
        """Returns  Higher Harmonic Response instance

        Parameters
        ----------
        rspec : ndarray
            rfft spectrum from measured sweep.
        rinvspec : ndarray
            rfft spectrum from inverted reference sweep.
        sweepperiod : scalar
            The parameter L from the paper to calculate the time delays for hhir decomposition.

        """
        hhfrf = _np.array(rspec) * _np.array(rinvspec)
        hhir = _np.fft.irfft(hhfrf)
        return cls(hhir=hhir, hhfrf=hhfrf, sweepperiod=sweepperiod, samplerate=samplerate)

    def __array__(self):
        return self.hhir


class FrfFilterKernel(object):
    """Returns a FRF-FilterKernel

    Parameters
    ----------
    freq : ndarray
        Frequency vector (positive frequencies)
    frf : ndarray
        Frequency responce function (onesided spectrum)
    ir : ndarray
        Impulse response (optional)
        If you just have an impulse response use the `FrfFilterKernel.from_ir()` classmethod.

    See also
    --------
    :class:`HammersteinModel`

    """
    def __init__(self, freq, frf, ir=None):
        if len(freq) != len(frf):
            raise ValueError('`freq` and `frf` must have the same length, not ', len(freq), len(frf))
        self._frf = frf
        self._freq = freq
        if ir is None:
            self._ir = _np.fft.irfft(frf)
        else:
            self._ir = ir

    @classmethod
    def from_ir(cls, ir, samplerate, startfreq=None, stopfreq=None):
        freq = _np.fft.rfftfreq(len(ir), 1/samplerate)
        frf = _np.fft.rfft(ir)
        if startfreq:
            frf[freq<startfreq] = 0j
        if stopfreq:
            frf[freq>stopfreq] = 0j
        if startfreq or stopfreq:
            ir = _np.fft.irfft(frf)
        return cls(freq=freq, frf=frf, ir=ir)

    @property
    def freq(self):
        """Returns the frequency vector."""
        return self._freq

    @property
    def frf(self):
        """Returns the frequency response function (FRF)"""
        return self._frf

    @property
    def ir(self):
        """Returns the impulse response (IR)"""
        return self._ir

    def filter(self, x):
        """Returns the convolved signal `x`."""
        return _spsignal.convolve(self._ir, x)

    def as_minimum_phase(self):
        """Returns a filter kernel with minimum phase response."""
        frf = _np.array(self.frf)
        frf_min_phase = _np.abs(frf) * _np.exp(1j*spectrum_to_minimum_phase(frf))
        return FrfFilterKernel(
            freq=self.freq,
            frf=frf_min_phase)

    def __array__(self):
        return self._frf

    def __repr__(self):
        return 'FrfFilterKernel(len(freq)={}, len(frf)={})'.format(
            len(self._freq), len(self._frf))


class IirFilterKernel(object):
    """Returns a IIR-FilterKernel

    Parameters
    ----------
    bcoeff : ndarray
        Filter coefficients of the numerator.
    acoeff : ndarray
        Filter coefficients of the denominator.

    See also
    --------
    :class:`HammersteinModel`

    """
    def __init__(self, bcoeff, acoeff):
        self._bcoeff = bcoeff
        self._acoeff = acoeff

    def filter(self, x, axis=-1, zi=None):
        """Returns the filtered signal `x`.
        For more info see help of `scipy.signal.lfilter`.

        """
        return _spsignal.lfilter(self._bcoeff, self._acoeff, x, axis, zi)

    def freqz(self, nfft):
        """Returns the frequency response for the IIR Filter Kernel.

        Parameters
        ----------
        nfft : int
            Number of bins.

        """
        return _spsignal.freqz(self._bcoeff, self._acoeff, int(1 + 0.5*nfft))

    def to_frf_filter_kernel(self, nfft):
        """Returns a FrfFilterKernel instance

        Parameters
        ----------
        nfft : int
            Number of bins.

        """
        return FrfFilterKernel(*self.freqz(nfft))

    def __repr__(self):
        return 'IirFilterKernel(bcoeff={}, acoeff={})'.format(self._bcoeff, self._acoeff)


class HammersteinModel(object):
    """Hammerstein Model

    .. math::

        y = f(x) = \\sum_n^N x^n * h_n

    A Hammerstein model can be created from a :class:`HigherHarmonicImpulseResponse` by
    using the method :func:`HammersteinModel.from_higher_harmonic_impulse_response()`.

    Parameters
    ----------
    kernels: iterable
        Contains Kernels with a .filter() method.
    orders: iterable
        Denotes the nonlinearity order for the kernels. Must be of same length as `kernels`.
        The linear kernel order is 1 (x**1), the second order kernel is 2 (x**2) ...

    See also
    --------
    :class:`HigherHarmonicImpulseResponse`

    :class:`FrfFilterKernel`

    :class:`IirFilterKernel`

    """
    def __init__(self, kernels, orders):
        self._kernels = kernels
        self._orders = orders

    @classmethod
    def from_sweeps(cls, syncsweep, measuredsweep, orders, delay=0, irlength=None, window=None, fftlen=None, regularize=1e-6):
        """Returns a HammersteinModel for given sweeps

        Parameters
        ----------
        syncsweep : SyncSweep
            A SyncSweep instance.
        measuredsweep : ndarray
            Measured sweep.
            Must be the output signal of the system under test excited with the provided `syncsweep`.
            Besides it must be sampled at the same samplerate as the provided syncsweep.
        orders : iterable of int
            The orders of hammerstein kernels to compute.
            Linear kernel is order 1 (x**1), quadratic kernel is order 2 (x**2), ...            
        delay : int
            delay of the system under test, needed for correct slicing of harmonic impulse responses.
        irlength : int
            length of the harmonic impulse response to compute the kernels from.
        window : bool, int or ndarray(length)
            Linear kernel is order 1 (x**1), quadratic kernel is order 2 (x**2), ...
        fftlen : int
            Length of the calculated ffts. fftlen will be guessed from measuredsweep length if fftlen is None.
        regularize : scalar or False
            Regularizes the system so if measuredsweep would be equal to the syncsweep signal, identity is ensured.
            
        """
        hhir = HigherHarmonicImpulseResponse.from_sweeps(
            syncsweep=syncsweep, 
            measuredsweep=measuredsweep, 
            fftlen=fftlen, 
            regularize=regularize)
        instance = cls.from_higher_harmonic_impulse_response(
            hhir=hhir,
            length=irlength,
            orders=orders,
            delay=delay,
            window=window
        )
        instance._hhir = hhir
        return instance

    @classmethod
    def from_higher_harmonic_impulse_response(cls, hhir, length, orders, delay=0, window=None):
        """Returns a HammersteinModel for given HigherHarmonicImpulseResponse

        Parameters
        ----------
        hhir : HigherHarmonicImpulseResponse
        length : int
            length of the harmonic impulse responses to compute hammerstein kernels from.
            The hammerstein kernels will have the same length.
        orders : iterable of int
            The orders of hammerstein kernels to compute.
            Linear kernel is order 1 (x**1), quadratic kernel is order 2 (x**2), ...
        delay : int
            delay of the system under test, needed for correct slicing of harmonic impulse responses.
        window : bool, int or ndarray(length)

        """
        maxlength = hhir.max_hir_length(max(orders))
        length = length or maxlength
        if length > maxlength:
            raise ValueError(
                f'Given `length` {length} must not be bigger than {maxlength}.'
                f' Otherwise other harmonic impulse responses will corrupt your kernels.')
        freq = _np.fft.rfftfreq(length, 1/hhir.samplerate)
        transformation_matrix = cls.create_kernel_to_hhfrf_transformation_matrix(orders)
        # slice the harmonic impulse responses and calculate harmonic frequency responses.
        hirs = [hhir.harmonic_impulse_response(o, length, delay, window) for o in orders]
        hhfrfs = (_np.fft.rfft(hir-_np.mean(hir), length) for hir in hirs)
        # create a matrix
        hhfrf_matrix = _np.array(list(hhfrfs))
        # transform higher harmonic frequency responses to hammerstein kernels by using the transformation matrix
        kernels_matrix = _np.dot(_np.linalg.inv(transformation_matrix), hhfrf_matrix)
        # create filter kernels
        kernels = [FrfFilterKernel(freq=freq, frf=frf) for frf in kernels_matrix]
        # return the hammerstein instance
        return cls(kernels, orders)

    @property
    def kernels(self):
        """Returns the hammerstein kernels."""
        return self._kernels

    @property
    def orders(self):
        """Returns the orders for the hammerstein kernels."""
        return self._orders

    @staticmethod
    def create_kernel_to_hhfrf_transformation_matrix(orders):
        """Returns a transformation matrix for combining kernels to higher harmonic frequency response functions

        Parameters
        ----------
        orders : int
            Orders of the kernels.

        Returns
        -------
        transformation_matrix : ndarray

        """
        count = len(orders)
        transformation_matrix = _np.zeros((count, count), dtype=_np.complex128)
        for idxn, n in enumerate(orders):
            for idxm, m in enumerate(orders):
                if (n >= m) and ((n+m)%2 == 0):
                    transformation_matrix[idxm, idxn] = (((-1 + 0j)**(2*(n) - (m-1)/2))) / (2**(n-1)) * _binom(n,(n-m)/2)
        return transformation_matrix

    def gen_filtered_signal_cascade(self, sig):
        """Yields hammerstein cascade filtered signals."""
        for kernel, order in zip(self._kernels, self._orders):
            if order > 1:
                cursig = sig**order
            else:
                cursig = sig

            yield kernel.filter(cursig)

    def filter(self, sig):
        """Returns nonlinear filtered signal by this hammerstein model cascade."""
        return sum(fsig for fsig in self.gen_filtered_signal_cascade(sig))

    def __repr__(self):
        return 'HammersteinModel(kernels={}, orders={})'.format(self._kernels, self._orders)


class LinearModel(object):
    """Returns a LinearModel

    Parameters
    ----------
    kernel : FilterKernel
        A kernel instance with a filter method


    See also
    --------
    :func:`LinearModel.from_higher_harmonic_impulse_response`
    :func:`LinearModel.from_hammerstein_model`

    """
    def __init__(self, kernel):
        self._kernel = kernel

    @property
    def kernel(self):
        return self._kernel

    @classmethod
    def from_sweeps(cls, syncsweep: SyncSweep, measuredsweep, delay=0, irlength=None, window=None, fftlen=None, regularize=1e-6, bandpass=True):
        """Returns a LinerModel for given sweeps

        Parameters
        ----------
        syncsweep : SyncSweep
            A SyncSweep instance.
        measuredsweep : ndarray
            Measured sweep.
            Must be the output signal of the system under test excited with the provided `syncsweep`.
            Besides it must be sampled at the same samplerate as the provided syncsweep.
        delay : int
            delay of the system under test, needed for correct slicing of harmonic impulse responses.
        irlength : int
            length of the harmonic impulse response to compute the kernel from.
        window : bool, int or ndarray(length)
            Linear kernel is order 1 (x**1), quadratic kernel is order 2 (x**2), ...
        fftlen : int
            Length of the calculated ffts. fftlen will be guessed from measuredsweep length if fftlen is None.
        regularize : scalar or False
            Regularizes the system so if measuredsweep would be equal to the syncsweep signal, identity is ensured.

        """
        hhir = HigherHarmonicImpulseResponse.from_sweeps(
            syncsweep=syncsweep, 
            measuredsweep=measuredsweep, 
            fftlen=fftlen, 
            regularize=regularize)
        instance = cls.from_higher_harmonic_impulse_response(
            hhir=hhir,
            length=irlength,
            delay=delay,
            window=window,
            startfreq=syncsweep.startfreq if bandpass else None,
            stopfreq=syncsweep.stopfreq if bandpass else None
        )
        instance._hhir = hhir
        return instance

    @classmethod
    def from_higher_harmonic_impulse_response(
            cls, 
            hhir: HigherHarmonicImpulseResponse, 
            length=None, 
            delay=0, 
            window=None, 
            startfreq=None, 
            stopfreq=None):
        """Returns a LinerModel for given HigherHarmonicImpulseResponse

        Parameters
        ----------
        hhir : HigherHarmonicImpulseResponse
        length : int
            length of the harmonic impulse compute the kernel from.
        orders : iterable of int
            The orders of hammerstein kernels to compute.
            Linear kernel is order 1 (x**1), quadratic kernel is order 2 (x**2), ...
        delay : int
            delay of the system under test, needed for correct slicing of harmonic impulse responses.
        window : bool, int or ndarray(length)
        startfreq : scalar or None
            Frequency window in spectrum will be applied.
        stopfreq : scalar or None
            Frequency window in spectrum will be applied.
        
        """
        maxlength = hhir.max_hir_length(order=1)
        length = length or maxlength
        if length > maxlength:
            raise ValueError(
                f'Given `length` {length} must not be bigger than {maxlength}.'
                f' Otherwise other harmonic impulse responses will corrupt your linear model kernel.')
        # slice the harmonic impulse responses and calculate the linear filter kernel.
        kernel = FrfFilterKernel.from_ir(
            ir=hhir.harmonic_impulse_response(order=1, length=length, delay=delay, window=window),
            samplerate=hhir.samplerate,
            startfreq=startfreq,
            stopfreq=stopfreq
        )
        return cls(kernel)

    @classmethod
    def from_hammerstein_model(cls, hmodel):
        """Returns a LinearModel of the given HammersteinModel.

        Parameters
        ----------
        hmodel : HmmersteinModel

        Returns
        -------
        lmodel : LinearModel

        """
        return cls(hmodel.kernels[0])

    def filter(self, sig):
        """Returns linear filtered `sig`."""
        return self.kernel.filter(sig)
