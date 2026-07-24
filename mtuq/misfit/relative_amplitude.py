
"""
Polarity + relative P-amplitude moment-tensor misfit.

Implements the inversion of Ding et al. (2026, BSSA, doi:10.1785/0120250267):
full moment tensors are constrained by P first-motion *polarity* and *relative*
P-wave amplitudes. Because geophone instrument responses were unavailable, the
seismic moment M0 cannot be recovered; instead each event's P amplitudes are
normalized by the maximum absolute value across sensors (separately for observed
and synthetic), which cancels a common instrument response and leaves a
scale-free amplitude pattern that constrains fault geometry and source type.

This is a fresh, dedicated implementation for polarity + relative-P-amplitude.
An S/P amplitude-ratio term is intentionally NOT implemented yet; hooks are noted
where SV/SH radiation and an ``lambda_sprat`` term would slot in later.
"""

import numpy as np

import mtuq
from mtuq.util import AttribDict, warn
from mtuq.util.signal import m_to_deg
from mtuq.misfit.waveform.level2 import _to_array, _type
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees


_ALLOWED_METHODS = ('taup', 'FK_metadata', 'CPS_metadata')


class PolarityRelPampMisfit(object):
    """ Joint misfit for P-wave polarity and relative P-wave amplitude.

    .. rubric:: Usage

    Two-step, like the other MTUQ misfits::

        misfit = PolarityRelPampMisfit(**parameters)
        values = misfit(data, greens, sources)     # -> (len(sources), 1)

    ``data`` is an observation bundle ``(pol_obs, pamp_obs, w_pol, w_pamp)`` of
    equal-length arrays (see :func:`relative_pamp_polarity_from_dict`), or a plain
    polarity array/list for backward-compatible polarity-only use.

    .. rubric:: Parameters

    ``method`` (`str`): ``'taup'`` (default), ``'FK_metadata'``, ``'CPS_metadata'``.
    Only ``'taup'`` is implemented.

    ``taup_model`` (`str`): ObsPy TauP model name or path (default ``'ak135'``).

    ``lambda_pol`` (`float`): weight of the polarity term (default 1.0).

    ``lambda_pamp`` (`float`): weight of the relative-P-amplitude term (default 1.0).

    ``norm_pol`` / ``norm_pamp`` (`str`): ``'L1'`` (default, paper) or ``'L2'``.

    ``project_to_vertical`` (`bool`): if True, multiply the predicted P amplitude
    by ``cos(incidence angle at receiver)`` from the TauP model so the synthetic is
    the true vertical component. Default False (scalar along-ray amplitude, matching
    the sibling implementation; the constant ``1/(4 pi rho alpha**3)`` cancels under
    max-abs normalization).

    ``distance_mode`` (`str`): ``'epicenter'`` (default) or ``'ray_path'`` for the
    ``1/r`` geometric-spreading factor.

    ``rho`` / ``vp`` (`float` or `None`): optional medium override; otherwise taken
    from the TauP model at source depth. Only affects a constant prefactor, which
    cancels under normalization.

    .. note::

        *Convention*: positive vertical first motions are +1, negative -1, and
        unpicked/indeterminate 0.
    """

    def __init__(self,
        method='taup',
        taup_model='ak135',
        lambda_pol=1.0,
        lambda_pamp=1.0,
        norm_pol='L1',
        norm_pamp='L1',
        project_to_vertical=False,
        distance_mode='epicenter',
        rho=None,
        vp=None,
        **parameters):

        # validate cheap parameters first
        self.norm_pol = _validate_norm(norm_pol)
        self.norm_pamp = _validate_norm(norm_pamp)

        if distance_mode not in ('epicenter', 'ray_path'):
            raise ValueError("distance_mode must be 'epicenter' or 'ray_path'")
        self.distance_mode = distance_mode

        if not method:
            raise Exception('Undefined parameter: method')
        if method not in _ALLOWED_METHODS:
            raise TypeError('Bad parameter: method=%r' % method)
        self.method = method

        self.taup_model = taup_model
        self.lambda_pol = float(lambda_pol)
        self.lambda_pamp = float(lambda_pamp)
        self.project_to_vertical = bool(project_to_vertical)
        self._rho_user = rho
        self._vp_user = vp

        if self.method == 'taup':
            assert self.taup_model is not None
            self._taup = TauPyModel(self.taup_model)
        else:
            # FK_metadata / CPS_metadata: geometry sources not implemented yet
            self._taup = None

    def get_observed(self, data):
        """ Extract observed (pol, pamp, w_pol, w_pamp) arrays from ``data``.

        Accepts either a 4-tuple bundle ``(pol_obs, pamp_obs, w_pol, w_pamp)``
        of equal-length arrays, or a plain polarity array/list (backward
        compatible: amplitude weights are 0 so only the polarity term is active).
        """
        if _is_observation_bundle(data, 4):
            return (np.asarray(data[0], dtype=np.float64),
                    np.asarray(data[1], dtype=np.float64),
                    np.asarray(data[2], dtype=np.float64),
                    np.asarray(data[3], dtype=np.float64))

        pol_obs = np.asarray(data, dtype=np.float64).ravel()
        n = pol_obs.size
        w_pol = (pol_obs != 0).astype(np.float64)
        zeros = np.zeros(n, dtype=np.float64)
        return pol_obs, zeros.copy(), w_pol, zeros.copy()

    def get_predicted(self, greens, sources, return_all=False):
        """ Predicted P polarity (and amplitude) for each source and station.

        Returns ``pol_theo`` (shape ``(n_sources, n_sta)``) by default, or
        ``(pol_theo, pamp_theo)`` when ``return_all=True``. The predicted P
        amplitude includes ``1/r`` geometric spreading and, if
        ``project_to_vertical``, a ``cos(incidence)`` vertical projection. The
        constant ``1/(4 pi rho alpha**3)`` is omitted because it cancels under the
        max-abs normalization applied in :meth:`get_terms`.
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

        if return_all:
            return pol_theo, pamp_theo
        return pol_theo

    def get_terms(self, data, greens, sources):
        """ Return per-source polarity and relative-P-amplitude misfit terms. """
        pol_obs, pamp_obs, w_pol, w_pamp = self.get_observed(data)
        pol_theo, pamp_theo = self.get_predicted(greens, sources, return_all=True)
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

        return phi_pol, phi_pamp

    def __call__(self, data, greens, sources, progress_handle=None,
                 set_attributes=False):
        phi_pol, phi_pamp = self.get_terms(data, greens, sources)
        values = self.lambda_pol * phi_pol + self.lambda_pamp * phi_pamp
        return values.reshape(-1, 1)

    def count_observations(self, data):
        """ Number of usable polarity and amplitude observations. """
        pol_obs, pamp_obs, w_pol, w_pamp = self.get_observed(data)
        return {
            'pol': int(np.count_nonzero(
                _observed_mask(pol_obs, w_pol, require_nonzero=True))),
            'pamp': int(np.count_nonzero(
                _observed_mask(pamp_obs, w_pamp, require_nonzero=False))),
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
        if self.method != 'taup':
            raise NotImplementedError("Only method='taup' is implemented")
        n = len(greens)
        takeoff = np.zeros(n, dtype=np.float64)
        incidence = np.zeros(n, dtype=np.float64)
        for i, g in enumerate(greens):
            depth_km = g.origin.depth_in_m / 1000.0
            deg = _distance_in_degrees(g)
            takeoff[i], incidence[i] = _takeoff_incidence_taup(
                self._taup, depth_km, deg)
        return takeoff, incidence

    def _azimuths(self, greens):
        return np.array([g.azimuth for g in greens], dtype=np.float64)

    def _distances_m(self, greens):
        dist = np.array([g.distance_in_m for g in greens], dtype=np.float64)
        if self.distance_mode == 'ray_path' and self.method == 'taup':
            try:
                return _ray_path_lengths_m_taup(self._taup, greens)
            except Exception as exc:
                warn('ray-path length failed (%s); using epicentral distance.'
                     % exc)
                return dist
        return dist


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _observed_mask(values, weights, require_nonzero=False):
    """ Stations with finite observations and strictly positive weight. """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if require_nonzero:
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

    Hook: SV/SH radiation and an S/P-ratio term would be added here in future.
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
