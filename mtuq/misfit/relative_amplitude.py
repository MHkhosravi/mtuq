
"""
Polarity + relative P-amplitude moment-tensor misfit.

Implements the inversion of Ding et al. (2026, BSSA, doi:10.1785/0120250267):
full moment tensors are constrained by P first-motion *polarity* and *relative*
P-wave amplitudes. Because geophone instrument responses were unavailable, the
seismic moment M0 cannot be recovered; instead each event's P amplitudes are
normalized by the maximum absolute value across sensors (separately for observed
and synthetic), which cancels a common instrument response and leaves a
scale-free amplitude pattern that constrains fault geometry and source type.

This module provides PolarityRelPampMisfit supporting polarity, relative P-wave
amplitude, and an optional S/P amplitude-ratio constraint.
"""

import numpy as np

import mtuq
from mtuq.util import AttribDict, warn
from mtuq.util.signal import m_to_deg
from mtuq.misfit.waveform.level2 import _to_array, _type
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees


_ALLOWED_METHODS = ('taup', 'precomputed', 'FK_metadata', 'CPS_metadata')


class GeometryList(list):
    """ Container for station geometry without Green's function waveforms.

    Holds per-station geometry entries (each an :class:`mtuq.util.AttribDict`)
    with keys:
        - origin: an :class:`mtuq.event.Origin`
        - station: an :class:`mtuq.util.AttribDict` (network, station, location,
          latitude, longitude, elevation_in_m)
        - azimuth: float, degrees (source -> station)
        - distance_in_m: float, epicentral distance
        - takeoff_angle: (optional) float, deg from downward vertical at source (>90 for up-going)
        - incidence_angle: (optional) float, deg from vertical at receiver
        - ray_length_in_m: (optional) float, geometric ray-path length
        - vp_over_vs: (optional) float, Vp/Vs ratio
    """

    def __init__(self, entries=()):
        super(GeometryList, self).__init__(entries)

    def select(self, selector):
        """ Selects geometry entries that match the given Origin. """
        if not isinstance(selector, mtuq.Origin):
            raise TypeError("Bad selector: %s" % type(selector).__name__)
        selected = self.__class__([entry for entry in self if entry.origin == selector])
        if len(selected) == 0 and len(self) > 0:
            import warnings
            warnings.warn("Nothing found matching given selector (%s)\n"
                          % type(selector).__name__)
        return selected

    def get_stations(self):
        """ List of station AttribDicts in order. """
        return [entry.station for entry in self]

    @classmethod
    def from_arrays(cls, origin, stations, azimuth, distance_in_m,
                    takeoff_angle=None, incidence_angle=None,
                    ray_length_in_m=None, vp_over_vs=None):
        """ Constructs a GeometryList from coordinate and parameter arrays. """
        n = len(stations)
        az = np.asarray(azimuth, dtype=np.float64)
        dist = np.asarray(distance_in_m, dtype=np.float64)
        if az.size != n:
            raise ValueError("Length mismatch: len(stations)=%d, len(azimuth)=%d"
                             % (n, az.size))
        if dist.size != n:
            raise ValueError("Length mismatch: len(stations)=%d, len(distance_in_m)=%d"
                             % (n, dist.size))

        to_arr = None
        if takeoff_angle is not None:
            to_arr = np.asarray(takeoff_angle, dtype=np.float64)
            if to_arr.size != n:
                raise ValueError("Length mismatch: len(stations)=%d, len(takeoff_angle)=%d"
                                 % (n, to_arr.size))

        inc_arr = None
        if incidence_angle is not None:
            inc_arr = np.asarray(incidence_angle, dtype=np.float64)
            if inc_arr.size != n:
                raise ValueError("Length mismatch: len(stations)=%d, len(incidence_angle)=%d"
                                 % (n, inc_arr.size))

        rl_arr = None
        if ray_length_in_m is not None:
            rl_arr = np.asarray(ray_length_in_m, dtype=np.float64)
            if rl_arr.size != n:
                raise ValueError("Length mismatch: len(stations)=%d, len(ray_length_in_m)=%d"
                                 % (n, rl_arr.size))

        vov_arr = None
        if vp_over_vs is not None:
            v_raw = np.asarray(vp_over_vs, dtype=np.float64)
            if v_raw.ndim == 0 or v_raw.size == 1:
                vov_arr = np.full(n, float(v_raw.ravel()[0]))
            elif v_raw.size == n:
                vov_arr = v_raw.ravel()
            else:
                raise ValueError("Length mismatch: len(stations)=%d, len(vp_over_vs)=%d"
                                 % (n, v_raw.size))

        entries = []
        for i in range(n):
            s = stations[i]
            if isinstance(s, AttribDict):
                sta_attr = s.copy()
            elif isinstance(s, dict):
                sta_attr = AttribDict(s)
            else:
                sta_attr = AttribDict()
                for key in ('network', 'station', 'location', 'latitude', 'longitude', 'elevation_in_m'):
                    if hasattr(s, key):
                        sta_attr[key] = getattr(s, key)

            defaults = {
                'network': '',
                'station': '',
                'location': '',
                'latitude': None,
                'longitude': None,
                'elevation_in_m': None,
            }
            for k, v in defaults.items():
                if k not in sta_attr:
                    sta_attr[k] = v

            entry = AttribDict({
                'origin': origin,
                'station': sta_attr,
                'azimuth': float(az.ravel()[i]),
                'distance_in_m': float(dist.ravel()[i]),
            })
            if to_arr is not None:
                entry.takeoff_angle = float(to_arr.ravel()[i])
            if inc_arr is not None:
                entry.incidence_angle = float(inc_arr.ravel()[i])
            if rl_arr is not None:
                entry.ray_length_in_m = float(rl_arr.ravel()[i])
            if vov_arr is not None:
                entry.vp_over_vs = float(vov_arr[i])

            entries.append(entry)

        return cls(entries)


class PolarityRelPampMisfit(object):
    """ Joint misfit for P-wave polarity, relative P-wave amplitude, and S/P ratio.

    .. rubric:: Usage

    Two-step, like the other MTUQ misfits::

        misfit = PolarityRelPampMisfit(**parameters)
        values = misfit(data, greens, sources)     # -> (len(sources), 1)

    ``data`` is an observation bundle ``(pol_obs, pamp_obs, w_pol, w_pamp)`` or
    ``(pol_obs, pamp_obs, sprat_obs, w_pol, w_pamp, w_sprat)`` of equal-length arrays
    (see :func:`relative_pamp_polarity_from_dict` and
    :func:`relative_pol_pamp_spamp_from_dict`), or a plain polarity array/list for
    backward-compatible polarity-only use.

    ``greens`` is a :class:`~mtuq.greens_tensor.GreensTensorList` or
    :class:`~mtuq.misfit.relative_amplitude.GeometryList`.

    .. rubric:: Parameters

    ``method`` (`str`): ``'taup'`` (default), ``'precomputed'``, ``'FK_metadata'``,
    ``'CPS_metadata'``. For ``'precomputed'``, takeoff angle, incidence angle, and
    optionally ray length and Vp/Vs are read directly from geometry entries.

    ``taup_model`` (`str`): ObsPy TauP model name or path (default ``'ak135'``).

    ``lambda_pol`` (`float`): weight of the polarity term (default 1.0).

    ``lambda_pamp`` (`float`): weight of the relative-P-amplitude term (default 1.0).

    ``lambda_sprat`` (`float`): weight of the S/P amplitude ratio term (default 0.0).

    ``norm_pol`` / ``norm_pamp`` / ``norm_sprat`` (`str`): ``'L1'`` (default, paper) or ``'L2'``.

    ``project_to_vertical`` (`bool`): if True, multiply the predicted P amplitude
    by ``cos(incidence angle at receiver)`` so the synthetic is the true vertical
    component. Default False (scalar along-ray amplitude, matching the sibling
    implementation; the constant ``1/(4 pi rho alpha**3)`` cancels under
    max-abs normalization).

    ``distance_mode`` (`str`): ``'epicenter'`` (default) or ``'ray_path'`` for the
    ``1/r`` geometric-spreading factor.

    ``sp_obs_is_log10`` (`bool`): if True, observed S/P ratios are already in log10
    space; if False (default), log10 is applied to positive observations.

    ``sprat_offset`` (`None`, `'fit'`, or `float`): baseline offset for log10(S/P).
    If float `c`, `+c` is added to predicted log10(S/P). If ``'fit'``, per-source
    optimal offset is fitted (weighted median for L1, weighted mean for L2) to
    minimize the residual. Default None.

    ``sprat_max_incidence`` (`float` or `None`): if set (degrees), stations whose
    receiver incidence angle exceeds this value are excluded from the S/P term only.
    Default None.

    ``rho`` / ``vp`` / ``vs`` (`float` or `None`): optional medium override;
    otherwise taken from the TauP model at source depth or precomputed geometry.

    .. rubric:: Presets and Methods

    :meth:`paper_preset`: Classmethod returning an instance configured like
    Ding et al. (2026).
    :meth:`ding_error`: Computes Ding et al. (2026) Eq. 6 misfit values per source.

    .. note::

        *Convention*: positive vertical first motions are +1, negative -1, and
        unpicked/indeterminate 0.
    """

    def __init__(self,
        method='taup',
        taup_model='ak135',
        lambda_pol=1.0,
        lambda_pamp=1.0,
        lambda_sprat=0.0,
        norm_pol='L1',
        norm_pamp='L1',
        norm_sprat='L1',
        project_to_vertical=False,
        distance_mode='epicenter',
        sp_obs_is_log10=False,
        sprat_offset=None,
        sprat_max_incidence=None,
        rho=None,
        vp=None,
        vs=None,
        **parameters):

        # validate cheap parameters first
        self.norm_pol = _validate_norm(norm_pol)
        self.norm_pamp = _validate_norm(norm_pamp)
        self.norm_sprat = _validate_norm(norm_sprat)

        if distance_mode not in ('epicenter', 'ray_path'):
            raise ValueError("distance_mode must be 'epicenter' or 'ray_path'")
        self.distance_mode = distance_mode

        if not method:
            raise Exception('Undefined parameter: method')
        if method not in _ALLOWED_METHODS:
            raise TypeError('Bad parameter: method=%r' % method)
        self.method = method

        if sprat_offset is not None and sprat_offset != 'fit':
            if isinstance(sprat_offset, bool) or not isinstance(sprat_offset, (int, float)):
                raise ValueError(
                    "sprat_offset must be None, 'fit', or a float; got %r" % (sprat_offset,))
            self.sprat_offset = float(sprat_offset)
        else:
            self.sprat_offset = sprat_offset

        if sprat_max_incidence is not None:
            if (isinstance(sprat_max_incidence, bool) or
                    not isinstance(sprat_max_incidence, (int, float)) or
                    sprat_max_incidence < 0):
                raise ValueError(
                    "sprat_max_incidence must be None or a non-negative float (degrees); got %r"
                    % (sprat_max_incidence,))
            self.sprat_max_incidence = float(sprat_max_incidence)
        else:
            self.sprat_max_incidence = None

        self.taup_model = taup_model
        self.lambda_pol = float(lambda_pol)
        self.lambda_pamp = float(lambda_pamp)
        self.lambda_sprat = float(lambda_sprat)
        self.project_to_vertical = bool(project_to_vertical)
        self.sp_obs_is_log10 = bool(sp_obs_is_log10)
        self._rho_user = rho
        self._vp_user = vp
        self._vs_user = vs

        if self.method == 'taup':
            assert self.taup_model is not None
            self._taup = TauPyModel(self.taup_model)
        else:
            # precomputed / FK_metadata / CPS_metadata
            self._taup = None

    @classmethod
    def paper_preset(cls, **overrides):
        """ Returns an instance configured like Ding et al. (2026).

        Default configuration:
            method='taup', taup_model='ak135',
            lambda_pol=0.0, lambda_pamp=1.0, lambda_sprat=0.0,
            norm_pamp='L1', project_to_vertical=True,
            distance_mode='ray_path'

        Any keyword argument in ``overrides`` replaces the default.
        """
        defaults = dict(
            method='taup',
            taup_model='ak135',
            lambda_pol=0.0,
            lambda_pamp=1.0,
            lambda_sprat=0.0,
            norm_pamp='L1',
            project_to_vertical=True,
            distance_mode='ray_path',
        )
        defaults.update(overrides)
        return cls(**defaults)

    def ding_error(self, data, greens, sources):
        """ Ding et al. (2026) Eq. 6 misfit value per source.

        Returns array of shape (n_mt,):
            e = 0.5 * sum_{l in valid} w_l * |Pobs_l - Pcalc_l|
        where Pobs and Pcalc are max-abs normalized over valid stations.
        Not divided by sum of weights.
        """
        _, pamp_obs, _, _, w_pamp, _ = self.get_observed(data)
        _, pamp_theo, _ = self._predict(greens, sources)
        n_mt = pamp_theo.shape[0]

        vamp = _observed_mask(pamp_obs, w_pamp, require_nonzero=False)
        if not np.any(vamp):
            return np.zeros(n_mt, dtype=np.float64)

        pobs_n = _max_abs_normalize(pamp_obs, vamp)
        pth_n = _max_abs_normalize(pamp_theo, vamp)

        residual = np.abs(pth_n[:, vamp] - pobs_n[vamp][None, :])
        w = w_pamp[vamp]
        e = 0.5 * np.sum(w[None, :] * residual, axis=1)
        return e

    def get_observed(self, data):
        """ Extract observations as ``(pol, pamp, sprat, w_pol, w_pamp, w_sprat)``.

        Accepts a 6-tuple bundle ``(pol, pamp, sprat, w_pol, w_pamp, w_sprat)``,
        a 4-tuple bundle ``(pol, pamp, w_pol, w_pamp)`` (S/P set to zero), or a
        plain polarity array/list (only the polarity term active). ``sprat`` is
        the observed S/P amplitude ratio (linear, or log10 if
        ``sp_obs_is_log10``).
        """
        if _is_observation_bundle(data, 6):
            return tuple(np.asarray(data[i], dtype=np.float64)
                         for i in range(6))

        if _is_observation_bundle(data, 4):
            pol = np.asarray(data[0], dtype=np.float64)
            pamp = np.asarray(data[1], dtype=np.float64)
            w_pol = np.asarray(data[2], dtype=np.float64)
            w_pamp = np.asarray(data[3], dtype=np.float64)
            z = np.zeros(pol.size, dtype=np.float64)
            return pol, pamp, z.copy(), w_pol, w_pamp, z.copy()

        pol_obs = np.asarray(data, dtype=np.float64).ravel()
        n = pol_obs.size
        w_pol = (pol_obs != 0).astype(np.float64)
        z = np.zeros(n, dtype=np.float64)
        return pol_obs, z.copy(), z.copy(), w_pol, z.copy(), z.copy()

    def _predict(self, greens, sources, force_sprat=False):
        """ Predicted (pol_theo, pamp_theo, sprat_theo) for each source/station.

        ``sprat_theo`` (log10 S/P) is computed when ``lambda_sprat != 0`` or
        ``force_sprat=True``; otherwise it is ``None``. P amplitude includes
        ``1/r`` spreading and, if ``project_to_vertical``, a ``cos(incidence)``
        factor. The constant ``1/(4 pi rho alpha**3)`` is omitted (it cancels
        under normalization).
        """
        mt = self._sources_to_use_array(sources)

        takeoff, incidence = self._takeoff_incidence(greens)
        azimuth = self._azimuths(greens)
        r = self._distances_m(greens)
        spreading = 1.0 / np.maximum(r, np.finfo(np.float64).tiny)
        vertical_cos = (np.cos(np.deg2rad(incidence))
                        if self.project_to_vertical else None)

        pol_theo, pamp_theo = _pol_pamp_mt(
            mt, takeoff, azimuth,
            spreading=spreading, vertical_cos=vertical_cos)

        sprat_theo = None
        if self.lambda_sprat != 0.0 or force_sprat:
            sprat_theo = _s_over_p_log10_mt(
                mt, takeoff, azimuth, self._vp_over_vs(greens))
            if isinstance(self.sprat_offset, (int, float)) and not isinstance(self.sprat_offset, bool):
                sprat_theo = sprat_theo + float(self.sprat_offset)

        return pol_theo, pamp_theo, sprat_theo

    def get_predicted(self, greens, sources, return_all=False, return_sprat=False):
        """ Predicted P polarity (and amplitude, S/P ratio) for each source and station.

        Parameters
        ----------
        greens : GreensTensorList or GeometryList
        sources : MomentTensor, Grid, or array-like
        return_all : bool
            If True and ``return_sprat=False``, returns ``(pol_theo, pamp_theo)``.
        return_sprat : bool
            If True, returns ``(pol_theo, pamp_theo, sprat_theo)`` with S/P ratio
            computed even when ``lambda_sprat == 0``.

        Returns
        -------
        pol_theo : array (n_mt, n_sta)
            If ``return_all=False`` and ``return_sprat=False`` (default).
        (pol_theo, pamp_theo) : tuple of arrays
            If ``return_all=True`` and ``return_sprat=False``.
        (pol_theo, pamp_theo, sprat_theo) : tuple of arrays
            If ``return_sprat=True``.
        """
        pol_theo, pamp_theo, sprat_theo = self._predict(
            greens, sources, force_sprat=return_sprat)
        if return_sprat:
            return pol_theo, pamp_theo, sprat_theo
        if return_all:
            return pol_theo, pamp_theo
        return pol_theo

    def get_terms(self, data, greens, sources):
        """ Per-source polarity, relative-P-amplitude, and S/P misfit terms. """
        (pol_obs, pamp_obs, sprat_obs,
         w_pol, w_pamp, w_sprat) = self.get_observed(data)
        pol_theo, pamp_theo, sprat_theo = self._predict(greens, sources)
        n_mt = pol_theo.shape[0]

        if self.lambda_pol != 0.0:
            vpol = _observed_mask(pol_obs, w_pol, require_nonzero=True)
            phi_pol = _weighted_misfit_term(
                pol_obs[vpol], pol_theo[:, vpol], w_pol[vpol],
                norm=self.norm_pol, scale=0.5)
        else:
            phi_pol = np.zeros(n_mt, dtype=np.float64)

        if self.lambda_pamp != 0.0:
            vamp = _observed_mask(pamp_obs, w_pamp, require_nonzero=False)
            # relative (max-abs) normalization over valid stations, obs and
            # synthetics separately (paper Eq. 5-6)
            pobs_n = _max_abs_normalize(pamp_obs, vamp)
            pth_n = _max_abs_normalize(pamp_theo, vamp)
            phi_pamp = _weighted_misfit_term(
                pobs_n[vamp], pth_n[:, vamp], w_pamp[vamp],
                norm=self.norm_pamp)
        else:
            phi_pamp = np.zeros(n_mt, dtype=np.float64)

        if self.lambda_sprat != 0.0:
            # S/P is compared in log10 space (already per-station
            # instrument- and spreading-independent; no cross-sensor norm)
            if self.sp_obs_is_log10:
                vsp = _observed_mask(sprat_obs, w_sprat)
                obs_log = sprat_obs
            else:
                vsp = _observed_mask(sprat_obs, w_sprat, require_positive=True)
                obs_log = np.log10(np.maximum(
                    sprat_obs, np.finfo(np.float64).tiny))

            if self.sprat_max_incidence is not None:
                _, incidence = self._takeoff_incidence(greens)
                vsp = vsp & (np.asarray(incidence) <= float(self.sprat_max_incidence))

            if not np.any(vsp):
                phi_sprat = np.zeros(n_mt, dtype=np.float64)
            elif self.sprat_offset == 'fit':
                diff = obs_log[vsp][None, :] - sprat_theo[:, vsp]
                w_vsp = w_sprat[vsp]
                if self.norm_sprat == 'L1':
                    c = _weighted_median_rows(diff, w_vsp)
                else:  # 'L2'
                    c = np.sum(diff * w_vsp[None, :], axis=1) / np.sum(w_vsp)
                phi_sprat = _weighted_misfit_term(
                    obs_log[vsp], sprat_theo[:, vsp] + c[:, None], w_vsp,
                    norm=self.norm_sprat)
            else:
                phi_sprat = _weighted_misfit_term(
                    obs_log[vsp], sprat_theo[:, vsp], w_sprat[vsp],
                    norm=self.norm_sprat)
        else:
            phi_sprat = np.zeros(n_mt, dtype=np.float64)

        return phi_pol, phi_pamp, phi_sprat

    def __call__(self, data, greens, sources, progress_handle=None,
                 set_attributes=False):
        phi_pol, phi_pamp, phi_sprat = self.get_terms(data, greens, sources)
        values = (self.lambda_pol * phi_pol +
                  self.lambda_pamp * phi_pamp +
                  self.lambda_sprat * phi_sprat)
        return values.reshape(-1, 1)

    def count_observations(self, data):
        """ Number of usable polarity, P-amplitude, and S/P observations. """
        (pol_obs, pamp_obs, sprat_obs,
         w_pol, w_pamp, w_sprat) = self.get_observed(data)
        return {
            'pol': int(np.count_nonzero(
                _observed_mask(pol_obs, w_pol, require_nonzero=True))),
            'pamp': int(np.count_nonzero(
                _observed_mask(pamp_obs, w_pamp, require_nonzero=False))),
            'sprat': int(np.count_nonzero(_observed_mask(
                sprat_obs, w_sprat,
                require_positive=not self.sp_obs_is_log10))),
        }

    def collect_attributes(self, data, greens):
        """ Per-station attributes (azimuth, takeoff, polarity) for beachballs. """
        takeoff, _ = self._takeoff_incidence(greens)
        observed = self.get_observed(data)[0]

        attrs_list = []
        for i, g in enumerate(greens):
            attrs = AttribDict()
            try:
                attrs.azimuth = g.azimuth
                attrs.distance_in_m = g.distance_in_m
            except Exception:
                pass
            try:
                attrs.network = g.station.network
                attrs.station = g.station.station
                attrs.location = g.station.location
                attrs.latitude = g.station.latitude
                attrs.longitude = g.station.longitude
            except Exception:
                pass
            try:
                attrs.takeoff_angle = takeoff[i]
            except Exception:
                pass
            try:
                attrs.polarity = observed[i]
            except Exception:
                pass
            attrs_list.append(attrs)
        return attrs_list

    def description(self):
        return '\n'.join([
            '    Misfit function type:\n    %s\n' % type(self).__name__,
            '    Misfit function method:\n    %s\n' % self.method,
        ])

    #
    # geometry helpers
    #

    def _sources_to_use_array(self, sources):
        if isinstance(sources, np.ndarray):
            return sources.reshape((-1, 6))
        if isinstance(sources, mtuq.MomentTensor):
            return sources.as_vector().reshape((1, 6))
        if isinstance(sources, mtuq.Force):
            raise NotImplementedError('Force sources are not supported')
        try:
            dims = sources.dims
        except AttributeError:
            raise TypeError('Unsupported sources type: %r' % type(sources))
        if _type(dims) == 'MomentTensor':
            return _to_array(sources)
        raise NotImplementedError('Only MomentTensor sources are supported')

    def _takeoff_incidence(self, greens):
        if self.method == 'precomputed':
            n = len(greens)
            takeoff = np.zeros(n, dtype=np.float64)
            incidence = np.zeros(n, dtype=np.float64)
            for i, g in enumerate(greens):
                sta = getattr(g, 'station', None)
                sta_name = getattr(sta, 'station', str(i)) if sta is not None else str(i)
                to = getattr(g, 'takeoff_angle', None)
                if to is None and isinstance(g, dict):
                    to = g.get('takeoff_angle')
                if to is None:
                    raise ValueError("Missing 'takeoff_angle' for station %s" % sta_name)

                inc = getattr(g, 'incidence_angle', None)
                if inc is None and isinstance(g, dict):
                    inc = g.get('incidence_angle')
                if inc is None:
                    raise ValueError("Missing 'incidence_angle' for station %s" % sta_name)

                takeoff[i] = float(to)
                incidence[i] = float(inc)
            return takeoff, incidence
        elif self.method == 'taup':
            n = len(greens)
            takeoff = np.zeros(n, dtype=np.float64)
            incidence = np.zeros(n, dtype=np.float64)
            for i, g in enumerate(greens):
                depth_km = g.origin.depth_in_m / 1000.0
                deg = _distance_in_degrees(g)
                takeoff[i], incidence[i] = _takeoff_incidence_taup(
                    self._taup, depth_km, deg)
            return takeoff, incidence
        else:
            raise NotImplementedError("Only method='taup' or 'precomputed' is implemented")

    def _azimuths(self, greens):
        return np.array([g.azimuth for g in greens], dtype=np.float64)

    def _distances_m(self, greens):
        if self.method == 'precomputed':
            if self.distance_mode == 'ray_path':
                out = np.zeros(len(greens), dtype=np.float64)
                for i, g in enumerate(greens):
                    sta = getattr(g, 'station', None)
                    sta_name = getattr(sta, 'station', str(i)) if sta is not None else str(i)
                    rl = getattr(g, 'ray_length_in_m', None)
                    if rl is None and isinstance(g, dict):
                        rl = g.get('ray_length_in_m')
                    if rl is None:
                        raise ValueError("Missing 'ray_length_in_m' for station %s" % sta_name)
                    out[i] = float(rl)
                return out
            else:
                return np.array([g.distance_in_m for g in greens], dtype=np.float64)

        dist = np.array([g.distance_in_m for g in greens], dtype=np.float64)
        if self.distance_mode == 'ray_path' and self.method == 'taup':
            try:
                return _ray_path_lengths_m_taup(self._taup, greens)
            except Exception as exc:
                warn('ray-path length failed (%s); using epicentral distance.'
                     % exc)
                return dist
        return dist

    def _vp_over_vs(self, greens):
        """ vp/vs at each source depth (user override, else AK135). """
        if self._vp_user is not None and self._vs_user is not None:
            return np.full(len(greens),
                           float(self._vp_user) / float(self._vs_user))
        if self.method == 'precomputed':
            out = np.zeros(len(greens), dtype=np.float64)
            for i, g in enumerate(greens):
                sta = getattr(g, 'station', None)
                sta_name = getattr(sta, 'station', str(i)) if sta is not None else str(i)
                vov = getattr(g, 'vp_over_vs', None)
                if vov is None and isinstance(g, dict):
                    vov = g.get('vp_over_vs')
                if vov is None:
                    raise ValueError("Missing 'vp_over_vs' for station %s" % sta_name)
                out[i] = float(vov)
            return out
        if self.method != 'taup':
            raise NotImplementedError(
                "vp/vs requires method='taup' or explicit vp and vs")
        out = np.zeros(len(greens), dtype=np.float64)
        for i, g in enumerate(greens):
            _, vp, vs = _medium_from_taup_at_depth(
                self._taup, g.origin.depth_in_m / 1000.0)
            out[i] = vp / vs
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _weighted_median_rows(values_2d, weights_1d):
    """ Vectorized weighted median of each row of values_2d with weights_1d.

    Parameters
    ----------
    values_2d : array-like, shape (n_rows, n_cols)
        Values for which the weighted median is computed per row.
    weights_1d : array-like, shape (n_cols,)
        Non-negative 1D weights corresponding to columns.

    Returns
    -------
    medians : ndarray, shape (n_rows,)
        Lower weighted median of each row (where cumulative weight reaches or
        exceeds 50% of the total weight).

    Notes
    -----
    If weights sum to zero or n_cols == 0, returns zeros of shape (n_rows,).
    Uses the lower weighted median convention when the threshold falls exactly
    on an accumulated weight step.
    """
    vals = np.asarray(values_2d, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals.reshape((1, -1))
    n_rows, n_cols = vals.shape

    w = np.asarray(weights_1d, dtype=np.float64).ravel()
    tot_w = np.sum(w)
    if n_cols == 0 or tot_w <= 0.0:
        return np.zeros(n_rows, dtype=np.float64)

    sort_idx = np.argsort(vals, axis=1)
    sorted_vals = np.take_along_axis(vals, sort_idx, axis=1)
    sorted_w = np.take_along_axis(w[None, :], sort_idx, axis=1)

    cum_w = np.cumsum(sorted_w, axis=1)
    threshold = 0.5 * tot_w
    mask = cum_w >= threshold
    first_idx = np.argmax(mask, axis=1)

    return sorted_vals[np.arange(n_rows), first_idx]


def _s_over_p_log10_mt(mt_array_use, takeoff, azimuth, vp_over_vs):
    """ Far-field log10(S/P) amplitude ratio for USE moment tensors.

    Uses the SAME takeoff/azimuth as the P wave (per design decision), computing
    ``S/P = |S| / |P|`` from the far-field radiation patterns. The 1/r geometric
    spreading cancels in the ratio; the velocity factor does not, giving

        log10(S/P) = log10(|S_rad| / |P_rad|) + 3 * log10(vp/vs)

    The SV/SH radiation bases (in NED, contracting with
    [NN, EE, DD, NE, ND, ED]) are ported from the reviewed
    ``PolarityPampSPampRatioMisfit`` implementation. ``vp_over_vs`` is a scalar
    or a length-``n_sta`` array. Numerator and denominator are floored by machine
    tiny so isotropic / nodal sources yield a large finite value, not +/-inf.
    """
    mt = np.asarray(mt_array_use, dtype=np.float64)
    if mt.ndim == 1:
        mt = mt.reshape((1, 6))
    n_sta = len(takeoff)

    ned = _use_to_ned(mt)
    th = np.deg2rad(np.asarray(takeoff, dtype=np.float64))
    ph = np.deg2rad(np.asarray(azimuth, dtype=np.float64))

    vov = np.asarray(vp_over_vs, dtype=np.float64)
    if vov.size == 1:
        vov = np.full(n_sta, float(vov.ravel()[0]))

    out = np.zeros((mt.shape[0], n_sta), dtype=np.float64)
    eps = np.finfo(np.float64).tiny

    for i in range(n_sta):
        st, ct = np.sin(th[i]), np.cos(th[i])
        sf, cf = np.sin(ph[i]), np.cos(ph[i])
        s2t, c2t = np.sin(2.0 * th[i]), np.cos(2.0 * th[i])
        s2f, c2f = np.sin(2.0 * ph[i]), np.cos(2.0 * ph[i])

        rp_basis = np.array([
            st * st * cf * cf, st * st * sf * sf, ct * ct,
            st * st * s2f, s2t * cf, s2t * sf], dtype=np.float64)
        rsv_basis = np.array([
            0.5 * s2t * cf * cf, 0.5 * s2t * sf * sf, -0.5 * s2t,
            0.5 * s2t * s2f, c2t * cf, c2t * sf], dtype=np.float64)
        rsh_basis = np.array([
            -0.5 * st * s2f, 0.5 * st * s2f, 0.0,
            st * c2f, -ct * sf, ct * cf], dtype=np.float64)

        Rp = np.dot(ned, rp_basis)
        Rsv = np.dot(ned, rsv_basis)
        Rsh = np.dot(ned, rsh_basis)
        s_mag = np.sqrt(Rsv ** 2 + Rsh ** 2)

        ratio = np.maximum(s_mag, eps) / np.maximum(np.abs(Rp), eps)
        out[:, i] = np.log10(ratio) + 3.0 * np.log10(vov[i])

    return out


def _medium_from_taup_at_depth(taup, depth_km):
    """ Density, P and S velocity (SI units) at ``depth_km`` from a TauP model. """
    layers = taup.model.s_mod.v_mod.layers
    for layer in layers:
        z1 = float(layer['top_depth'])
        z2 = float(layer['bot_depth'])
        if z1 <= depth_km <= z2:
            frac = 0.0 if z2 == z1 else (depth_km - z1) / (z2 - z1)
            vp = (float(layer['top_p_velocity']) +
                  frac * (float(layer['bot_p_velocity']) -
                          float(layer['top_p_velocity'])))
            vs = (float(layer['top_s_velocity']) +
                  frac * (float(layer['bot_s_velocity']) -
                          float(layer['top_s_velocity'])))
            rho = (float(layer['top_density']) +
                   frac * (float(layer['bot_density']) -
                           float(layer['top_density'])))
            return rho * 1000.0, vp * 1000.0, vs * 1000.0
    raise ValueError('Depth %s km is outside the TauP model range' % depth_km)


def _validate_norm(value):
    norm = str(value).upper()
    if norm not in ('L1', 'L2'):
        raise ValueError("Invalid norm type. Use 'L1' or 'L2'.")
    return norm


def _use_to_ned(mt_array):
    """ Convert MTUQ USE-convention moment tensors to NED 6-vectors.

    USE order [m_uu, m_ss, m_ee, m_us, m_ue, m_se] ->
    NED order [m_NN, m_EE, m_DD, m_NE, m_ND, m_ED]
            = [m_ss, m_ee, m_uu, -m_se, m_us, -m_ue].
    """
    mt_array = np.asarray(mt_array, dtype=np.float64)
    if mt_array.ndim == 1:
        mt_array = mt_array.reshape((1, 6))
    return np.column_stack([
        mt_array[:, 1],
        mt_array[:, 2],
        mt_array[:, 0],
        -mt_array[:, 5],
        mt_array[:, 3],
        -mt_array[:, 4],
    ])


def _observed_mask(values, weights, require_nonzero=False,
                   require_positive=False):
    """ Stations with finite observations and strictly positive weight. """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if require_positive:
        valid &= values > 0.0
    elif require_nonzero:
        valid &= values != 0.0
    return valid


def _distance_in_degrees(greens_tensor):
    try:
        return locations2degrees(
            greens_tensor.origin.latitude,
            greens_tensor.origin.longitude,
            greens_tensor.station.latitude,
            greens_tensor.station.longitude)
    except Exception:
        return m_to_deg(greens_tensor.distance_in_m)


def _takeoff_incidence_taup(taup, depth_in_km, distance_in_deg):
    """ Takeoff angle (at source) and incidence angle (at receiver), degrees. """
    arrivals = taup.get_travel_times(
        depth_in_km, distance_in_deg, phase_list=['p', 'P'])
    if len(arrivals) == 0:
        raise Exception('No p/P arrival for depth=%s km, distance=%s deg'
                        % (depth_in_km, distance_in_deg))
    arrival = sorted(arrivals, key=lambda a: a.time)[0]
    return arrival.takeoff_angle, arrival.incident_angle


def _ray_path_lengths_m_taup(taup, greens):
    """ Geometric P ray-path length (m) per station, integrated along the ray. """
    out = np.zeros(len(greens), dtype=np.float64)
    for i, g in enumerate(greens):
        dep_km = g.origin.depth_in_m / 1000.0
        deg = _distance_in_degrees(g)
        paths = taup.get_ray_paths(
            source_depth_in_km=dep_km,
            distance_in_degree=deg,
            receiver_depth_in_km=0.0,
            phase_list=['p', 'P'])
        if len(paths) == 0:
            raise ValueError('No p/P ray path for depth=%s km, distance=%s deg'
                             % (dep_km, deg))
        path = paths[0].path
        radius_km = 6371.0 - path['depth']
        theta = path['dist']
        segments = np.sqrt(
            radius_km[:-1] ** 2 + radius_km[1:] ** 2 -
            2.0 * radius_km[:-1] * radius_km[1:] *
            np.cos(theta[1:] - theta[:-1]))
        out[i] = float(np.sum(segments) * 1000.0)
    return out


def _is_observation_bundle(data, length):
    if not isinstance(data, (tuple, list)) or len(data) != length:
        return False
    return all(np.ndim(item) > 0 for item in data)


def _max_abs_normalize(arr, valid):
    """ Divide by the max absolute value over the valid stations (last axis).

    ``arr`` may be 1D (observed, shape (n_sta,)) or 2D (synthetic, shape
    (n_mt, n_sta)); 2D rows are each normalized by their own max. Only stations
    with ``valid=True`` contribute to the max; all entries are divided. The
    denominator is floored by machine tiny to stay finite.
    """
    arr = np.asarray(arr, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    eps = np.finfo(np.float64).tiny
    absarr = np.abs(arr)

    if arr.ndim == 1:
        m = np.max(absarr[valid]) if valid.any() else 0.0
        return arr / max(m, eps)

    masked = np.where(valid[None, :], absarr, 0.0)
    m = np.maximum(masked.max(axis=1), eps)
    return arr / m[:, None]


def _weighted_misfit_term(obs, syn, weights, norm='L1', scale=1.0):
    """ Weighted L1/L2 residual term, normalized by the sum of weights.

    ``obs`` shape (n_sta,), ``syn`` shape (n_mt, n_sta), ``weights`` shape
    (n_sta,). Returns shape (n_mt,). If the total weight is zero (no valid
    stations), returns zeros.
    """
    obs = np.asarray(obs, dtype=np.float64)
    syn = np.asarray(syn, dtype=np.float64)
    if syn.ndim == 1:
        syn = syn.reshape((1, -1))
    weights = np.asarray(weights, dtype=np.float64)

    n_mt = syn.shape[0]
    denom = np.sum(weights)
    if denom <= 0.0 or obs.size == 0:
        return np.zeros(n_mt, dtype=np.float64)

    residual = scale * (syn - obs[None, :])
    if norm == 'L1':
        numer = np.sum(weights[None, :] * np.abs(residual), axis=1)
    else:  # 'L2'
        numer = np.sum(weights[None, :] * residual ** 2, axis=1)
    return numer / denom


def _pol_pamp_mt(mt_array_use, takeoff, azimuth, spreading=None,
                 vertical_cos=None):
    """ Far-field P polarity and (relative) P amplitude for USE moment tensors.

    Parameters
    ----------
    mt_array_use : array (n_mt, 6) or (6,)
        Moment tensors in MTUQ USE convention.
    takeoff, azimuth : array (n_sta,)
        Takeoff angle (from downward vertical) and azimuth, in degrees.
    spreading : array (n_sta,) or None
        Optional per-station geometric-spreading factor (e.g. ``1/r``). A
        constant prefactor cancels under max-abs normalization, so only the
        station-varying part matters.
    vertical_cos : array (n_sta,) or None
        Optional per-station ``cos(incidence angle at receiver)`` to project the
        along-ray P amplitude onto the vertical component.

    Returns
    -------
    polarities : array (n_mt, n_sta)   # +1 / -1 / 0, sign of the P radiation
    p_amp      : array (n_mt, n_sta)   # signed P amplitude (spreading/projection applied)

    Note: SV/SH radiation and S/P amplitude ratios are computed in :func:`_s_over_p_log10_mt`.
    """
    mt = np.asarray(mt_array_use, dtype=np.float64)
    if mt.ndim == 1:
        mt = mt.reshape((1, 6))
    if mt.shape[1] != 6:
        raise ValueError('moment tensors must have 6 components (USE)')

    n_mt = mt.shape[0]
    n_sta = len(takeoff)
    if len(azimuth) != n_sta:
        raise ValueError('takeoff and azimuth must have equal length')

    ned = _use_to_ned(mt)                       # (n_mt, 6): [NN,EE,DD,NE,ND,ED]
    th = np.deg2rad(np.asarray(takeoff, dtype=np.float64))
    ph = np.deg2rad(np.asarray(azimuth, dtype=np.float64))

    polarities = np.zeros((n_mt, n_sta), dtype=np.float64)
    p_amp = np.zeros((n_mt, n_sta), dtype=np.float64)

    for i in range(n_sta):
        st, ct = np.sin(th[i]), np.cos(th[i])
        sf, cf = np.sin(ph[i]), np.cos(ph[i])
        s2t = np.sin(2.0 * th[i])
        s2f = np.sin(2.0 * ph[i])

        # rp_basis contracts with NED [NN,EE,DD,NE,ND,ED] to give gamma.M.gamma
        # gamma_N=st*cf, gamma_E=st*sf, gamma_D=ct
        rp_basis = np.array([
            st * st * cf * cf,     # gamma_N^2
            st * st * sf * sf,     # gamma_E^2
            ct * ct,               # gamma_D^2
            st * st * s2f,         # 2 gamma_N gamma_E
            s2t * cf,              # 2 gamma_N gamma_D
            s2t * sf,              # 2 gamma_E gamma_D
        ], dtype=np.float64)

        Rp = np.dot(ned, rp_basis)             # (n_mt,)

        amp = Rp.copy()
        if spreading is not None:
            amp = amp * float(spreading[i])
        if vertical_cos is not None:
            amp = amp * float(vertical_cos[i])

        polarities[Rp > 0, i] = +1
        polarities[Rp < 0, i] = -1
        p_amp[:, i] = amp

    return polarities, p_amp


def _obs_and_weight(value_dict, weight_dict, station):
    """ (value, weight) for a station; absent -> (0, 0); weight_dict None -> 1
    where the value is present, else 0. """
    _MISSING = object()
    v = _lookup_station_value(value_dict, station, default=_MISSING)
    present = v is not _MISSING
    val = float(v) if present else 0.0
    if weight_dict is None:
        w = 1.0 if present else 0.0
    else:
        w = float(_lookup_station_value(weight_dict, station, default=0.0))
    return val, w


def relative_pol_pamp_spamp_from_dict(
        dict_polarity, dict_p_amp, dict_sp_ratio,
        dict_polarity_weights, dict_p_amp_weights, dict_sp_ratio_weights,
        stations):
    """ Aligned observation bundle including S/P amplitude ratio.

    Returns ``(pol, pamp, sprat, w_pol, w_pamp, w_sprat)``, each a
    length-``len(stations)`` array. Absent stations get value 0 and weight 0;
    a None weights dict defaults to weight 1 where the value is present.
    """
    n = len(stations)
    pol = np.zeros(n, dtype=np.float64)
    pamp = np.zeros(n, dtype=np.float64)
    sprat = np.zeros(n, dtype=np.float64)
    w_pol = np.zeros(n, dtype=np.float64)
    w_pamp = np.zeros(n, dtype=np.float64)
    w_sprat = np.zeros(n, dtype=np.float64)

    for i, station in enumerate(stations):
        pol[i], w_pol[i] = _obs_and_weight(
            dict_polarity, dict_polarity_weights, station)
        pamp[i], w_pamp[i] = _obs_and_weight(
            dict_p_amp, dict_p_amp_weights, station)
        sprat[i], w_sprat[i] = _obs_and_weight(
            dict_sp_ratio, dict_sp_ratio_weights, station)

    return pol, pamp, sprat, w_pol, w_pamp, w_sprat


def _lookup_station_value(mapping, station, default=None):
    if mapping is None:
        return default
    station_name = getattr(station, 'station', None)
    network = getattr(station, 'network', None)
    if station_name is None:
        return default
    if network:
        network_station = '{}.{}'.format(network, station_name)
        if network_station in mapping:
            return mapping[network_station]
    if station_name in mapping:
        return mapping[station_name]
    return default


def relative_pamp_polarity_from_dict(
        dict_polarity, dict_p_amp,
        dict_polarity_weights, dict_p_amp_weights,
        stations):
    """ Build aligned observation arrays from per-station dictionaries.

    Returns ``(pol_obs, pamp_obs, w_pol, w_pamp)``, each a length-``len(stations)``
    array. Absent stations get value 0 and weight 0. When a weights dict is None,
    the weight defaults to 1 where the corresponding value is present, else 0.
    """
    n = len(stations)
    pol_obs = np.zeros(n, dtype=np.float64)
    pamp_obs = np.zeros(n, dtype=np.float64)
    w_pol = np.zeros(n, dtype=np.float64)
    w_pamp = np.zeros(n, dtype=np.float64)

    _MISSING = object()

    for i, station in enumerate(stations):
        pol = _lookup_station_value(dict_polarity, station, default=_MISSING)
        amp = _lookup_station_value(dict_p_amp, station, default=_MISSING)

        pol_present = pol is not _MISSING
        amp_present = amp is not _MISSING

        pol_obs[i] = float(pol) if pol_present else 0.0
        pamp_obs[i] = float(amp) if amp_present else 0.0

        if dict_polarity_weights is None:
            w_pol[i] = 1.0 if pol_present else 0.0
        else:
            w_pol[i] = float(_lookup_station_value(
                dict_polarity_weights, station, default=0.0))

        if dict_p_amp_weights is None:
            w_pamp[i] = 1.0 if amp_present else 0.0
        else:
            w_pamp[i] = float(_lookup_station_value(
                dict_p_amp_weights, station, default=0.0))

    return pol_obs, pamp_obs, w_pol, w_pamp
