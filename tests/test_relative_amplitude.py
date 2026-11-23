"""
Data-free unit tests for mtuq.misfit.relative_amplitude

Implements the Ding et al. (2026) polarity + relative-P-amplitude MT misfit.
These tests require only numpy + obspy (TauP ak135); no waveform data or
Green's functions are downloaded.
"""

import numpy as np
import pytest

from types import SimpleNamespace

from mtuq.misfit.relative_amplitude import (
    PolarityRelPampMisfit,
    GeometryList,
    relative_pamp_polarity_from_dict,
    _use_to_ned,
    _pol_pamp_mt,
    _max_abs_normalize,
    _weighted_misfit_term,
    _weighted_median_rows,
    _s_over_p_log10_mt,
    _medium_from_taup_at_depth,
    _takeoff_incidence_taup,
    _ray_path_lengths_m_taup,
)
from mtuq.event import Origin
from mtuq.util import AttribDict
from mtuq.grid.moment_tensor import DoubleCoupleGridRegular, _semiregular
from mtuq.util.math import to_delta, to_gamma, to_mij
from obspy.taup import TauPyModel
from obspy.geodetics import kilometers2degrees
from obspy.imaging.beachball import aux_plane


def _station(name, network='XX'):
    return SimpleNamespace(station=name, network=network)


# ---------------------------------------------------------------------------
# USE -> NED conversion
# ---------------------------------------------------------------------------

def test_use_to_ned_known_values():
    # USE order: [m_uu, m_ss, m_ee, m_us, m_ue, m_se]
    # NED order: [m_NN, m_EE, m_DD, m_NE, m_ND, m_ED]
    #          = [m_ss, m_ee, m_uu, -m_se, m_us, -m_ue]
    use = np.array([[1., 2., 3., 4., 5., 6.]])
    ned = _use_to_ned(use)
    np.testing.assert_allclose(ned[0], [2., 3., 1., -6., 4., -5.])


def test_use_to_ned_accepts_1d():
    ned = _use_to_ned(np.array([1., 2., 3., 4., 5., 6.]))
    assert ned.shape == (1, 6)


# ---------------------------------------------------------------------------
# Observation input helper
# ---------------------------------------------------------------------------

def test_from_dict_aligns_to_station_order():
    stations = [_station('AAA'), _station('BBB'), _station('CCC')]
    pol = {'AAA': +1, 'BBB': -1, 'CCC': +1}
    pamp = {'AAA': 0.5, 'BBB': 0.2, 'CCC': 0.9}
    pol_obs, pamp_obs, w_pol, w_pamp = relative_pamp_polarity_from_dict(
        pol, pamp, None, None, stations)
    np.testing.assert_allclose(pol_obs, [1., -1., 1.])
    np.testing.assert_allclose(pamp_obs, [0.5, 0.2, 0.9])


def test_from_dict_absent_station_gets_zero_value_and_weight():
    stations = [_station('AAA'), _station('BBB')]
    pol = {'AAA': +1}          # BBB absent
    pamp = {'AAA': 0.5}
    pol_obs, pamp_obs, w_pol, w_pamp = relative_pamp_polarity_from_dict(
        pol, pamp, None, None, stations)
    assert pol_obs[1] == 0.0 and pamp_obs[1] == 0.0
    assert w_pol[1] == 0.0 and w_pamp[1] == 0.0
    # present station defaults to weight 1
    assert w_pol[0] == 1.0 and w_pamp[0] == 1.0


def test_from_dict_explicit_weights_used():
    stations = [_station('AAA'), _station('BBB')]
    pol = {'AAA': +1, 'BBB': -1}
    pamp = {'AAA': 0.5, 'BBB': 0.2}
    wpol = {'AAA': 2.0, 'BBB': 0.0}
    wpamp = {'AAA': 3.0, 'BBB': 1.0}
    _, _, w_pol, w_pamp = relative_pamp_polarity_from_dict(
        pol, pamp, wpol, wpamp, stations)
    np.testing.assert_allclose(w_pol, [2.0, 0.0])
    np.testing.assert_allclose(w_pamp, [3.0, 1.0])


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_constructor_defaults():
    m = PolarityRelPampMisfit(method='taup')
    assert m.lambda_pol == 1.0
    assert m.lambda_pamp == 1.0
    assert m.project_to_vertical is False
    assert m.norm_pol == 'L1' and m.norm_pamp == 'L1'


def test_constructor_rejects_bad_method():
    with pytest.raises((TypeError, ValueError, Exception)):
        PolarityRelPampMisfit(method='nonsense')


def test_constructor_rejects_bad_norm():
    with pytest.raises(ValueError):
        PolarityRelPampMisfit(method='taup', norm_pamp='L3')


# ---------------------------------------------------------------------------
# Forward model: P radiation, polarity, amplitude
# ---------------------------------------------------------------------------

def test_isotropic_source_all_positive_polarity():
    # isotropic USE [m_uu, m_ss, m_ee, ...] = [1,1,1,0,0,0] -> NED diag = 1
    # Rp = gamma.M.gamma = 1 for every ray direction
    iso = np.array([[1., 1., 1., 0., 0., 0.]])
    takeoff = np.array([10., 45., 80., 120.])
    azimuth = np.array([0., 90., 180., 270.])
    pol, amp = _pol_pamp_mt(iso, takeoff, azimuth)
    np.testing.assert_allclose(pol[0], [1., 1., 1., 1.])
    np.testing.assert_allclose(amp[0], [1., 1., 1., 1.], atol=1e-12)


def test_dc_radiation_matches_analytic_sin2theta_cosphi():
    # USE [0,0,0,1,0,0] -> NED has only M_ND = 1 (index 4).
    # Rp = 2 * M_ND * gamma_N * gamma_D = 2 sin(th) cos(th) cos(phi)
    #    = sin(2*th) * cos(phi)
    mt = np.array([[0., 0., 0., 1., 0., 0.]])
    takeoff = np.array([45., 45., 30.])
    azimuth = np.array([0., 180., 0.])
    expected = np.sin(np.deg2rad(2 * takeoff)) * np.cos(np.deg2rad(azimuth))
    pol, amp = _pol_pamp_mt(mt, takeoff, azimuth)
    np.testing.assert_allclose(amp[0], expected, atol=1e-12)
    np.testing.assert_allclose(pol[0], np.sign(expected))


def test_spreading_scales_amplitude_not_polarity():
    iso = np.array([[1., 1., 1., 0., 0., 0.]])
    takeoff = np.array([45., 45.])
    azimuth = np.array([0., 90.])
    spreading = np.array([2.0, 0.5])
    pol, amp = _pol_pamp_mt(iso, takeoff, azimuth, spreading=spreading)
    np.testing.assert_allclose(amp[0], [2.0, 0.5])
    np.testing.assert_allclose(pol[0], [1., 1.])


def test_vertical_projection_scales_amplitude():
    iso = np.array([[1., 1., 1., 0., 0., 0.]])
    takeoff = np.array([45., 45.])
    azimuth = np.array([0., 90.])
    vcos = np.array([0.5, 1.0])
    pol, amp = _pol_pamp_mt(iso, takeoff, azimuth, vertical_cos=vcos)
    np.testing.assert_allclose(amp[0], [0.5, 1.0])
    np.testing.assert_allclose(pol[0], [1., 1.])


# ---------------------------------------------------------------------------
# S/P amplitude-ratio radiation (log10 space)
# ---------------------------------------------------------------------------

def test_medium_from_taup_positive_and_vp_gt_vs():
    from obspy.taup import TauPyModel
    rho, vp, vs = _medium_from_taup_at_depth(TauPyModel('ak135'), 5.0)
    assert rho > 0 and vp > 0 and vs > 0 and vp > vs


def test_isotropic_sp_ratio_is_finite_and_very_negative():
    # isotropic sources radiate NO S -> |S|=0 -> floored -> large finite negative
    iso = np.array([[1., 1., 1., 0., 0., 0.]])
    sprat = _s_over_p_log10_mt(
        iso, np.array([30., 60.]), np.array([20., 200.]), np.array([1.7, 1.7]))
    assert np.all(np.isfinite(sprat[0]))
    assert np.all(sprat[0] < -5.0)


def test_sp_velocity_factor_is_additive_in_log():
    mt = np.array([[0.2, -1., 0.5, 0.3, -0.2, 0.1]])
    takeoff = np.array([40., 70.])
    azimuth = np.array([15., 190.])
    s1 = _s_over_p_log10_mt(mt, takeoff, azimuth, np.array([1.7, 1.7]))
    s2 = _s_over_p_log10_mt(mt, takeoff, azimuth, np.array([1.9, 1.9]))
    np.testing.assert_allclose(s2[0] - s1[0], 3.0 * np.log10(1.9 / 1.7))


def test_sp_ratio_finite_when_p_is_nodal():
    # M_ND=1: Rp = sin(2th) cos(phi); at th=45, phi=90 -> Rp ~ 0
    mt = np.array([[0., 0., 0., 1., 0., 0.]])
    sprat = _s_over_p_log10_mt(
        mt, np.array([45.]), np.array([90.]), np.array([1.7]))
    assert np.isfinite(sprat[0, 0])


# ---------------------------------------------------------------------------
# Misfit math: max-abs normalization + weighted L1/L2 term
# ---------------------------------------------------------------------------

def test_max_abs_normalize_1d():
    np.testing.assert_allclose(
        _max_abs_normalize(np.array([2., -4.]), np.array([True, True])),
        [0.5, -1.0])


def test_max_abs_normalize_uses_only_valid_stations():
    x = np.array([10., 2., -4.])          # 10 is invalid (excluded from max)
    valid = np.array([False, True, True])
    out = _max_abs_normalize(x, valid)
    np.testing.assert_allclose(out[1:], [0.5, -1.0])   # max over valid = 4


def test_max_abs_normalize_2d_per_row():
    syn = np.array([[1., -2.], [4., 2.]])
    out = _max_abs_normalize(syn, np.array([True, True]))
    np.testing.assert_allclose(out, [[0.5, -1.0], [1.0, 0.5]])


def test_weighted_term_l1_zero_when_identical():
    obs = np.array([1., -1.])
    syn = np.array([[1., -1.]])
    assert _weighted_misfit_term(obs, syn, np.array([1., 1.]), 'L1')[0] == 0.0


def test_weighted_term_l1_value():
    obs = np.array([1., -1.])
    syn = np.array([[-1., 1.]])           # residual [-2, 2]
    # (1*2 + 1*2) / (1+1) = 2.0
    np.testing.assert_allclose(
        _weighted_misfit_term(obs, syn, np.array([1., 1.]), 'L1')[0], 2.0)


def test_weighted_term_l2_value():
    obs = np.array([1., -1.])
    syn = np.array([[-1., 1.]])           # residual^2 [4, 4]
    np.testing.assert_allclose(
        _weighted_misfit_term(obs, syn, np.array([1., 1.]), 'L2')[0], 4.0)


def test_weighted_term_scale_half_for_polarity():
    obs = np.array([1., -1.])
    theo = np.array([[-1., -1.]])         # one flip, one match
    # 0.5*(|1-(-1)|, |-1-(-1)|) = (1, 0); weighted sum 1 / Sum w 2 = 0.5
    np.testing.assert_allclose(
        _weighted_misfit_term(obs, theo, np.array([1., 1.]), 'L1', scale=0.5)[0],
        0.5)


def test_weighted_term_zero_weight_station_excluded():
    obs = np.array([1., -1., 5.])
    syn = np.array([[1., -1., -99.]])     # huge residual at station 3
    w = np.array([1., 1., 0.])            # but weight 0
    np.testing.assert_allclose(
        _weighted_misfit_term(obs, syn, w, 'L1')[0], 0.0)


# ---------------------------------------------------------------------------
# get_observed
# ---------------------------------------------------------------------------

def test_get_observed_bundle_roundtrip():
    m = PolarityRelPampMisfit(method='taup')
    # 4-tuple (polarity + P amp) input still accepted -> S/P defaults to zero
    bundle = (np.array([1., -1.]), np.array([0.5, 0.2]),
              np.array([1., 1.]), np.array([1., 1.]))
    pol, pamp, sprat, wp, wa, ws = m.get_observed(bundle)
    np.testing.assert_allclose(pol, [1., -1.])
    np.testing.assert_allclose(pamp, [0.5, 0.2])
    np.testing.assert_allclose(wp, [1., 1.])
    np.testing.assert_allclose(wa, [1., 1.])
    np.testing.assert_allclose(sprat, [0., 0.])
    np.testing.assert_allclose(ws, [0., 0.])


def test_get_observed_6tuple_roundtrip():
    m = PolarityRelPampMisfit(method='taup')
    bundle = (np.array([1., -1.]), np.array([0.5, 0.2]), np.array([2.0, 3.0]),
              np.array([1., 1.]), np.array([1., 1.]), np.array([1., 0.]))
    pol, pamp, sprat, wp, wa, ws = m.get_observed(bundle)
    np.testing.assert_allclose(sprat, [2.0, 3.0])
    np.testing.assert_allclose(ws, [1., 0.])


def test_get_observed_polarity_only_backward_compatible():
    m = PolarityRelPampMisfit(method='taup')
    pol, pamp, sprat, wp, wa, ws = m.get_observed([1, -1, 0])
    np.testing.assert_allclose(pol, [1., -1., 0.])
    np.testing.assert_allclose(wp, [1., 1., 0.])   # picked stations weight 1
    np.testing.assert_allclose(wa, [0., 0., 0.])   # no amplitude data
    np.testing.assert_allclose(ws, [0., 0., 0.])   # no S/P data


# ---------------------------------------------------------------------------
# Integration: get_predicted / __call__ over stub station geometry
# ---------------------------------------------------------------------------

def _make_greens(distances_m, azimuths_deg, depth_m=1500.0):
    """ Minimal stub Green's-tensor list: only station geometry is used. """
    g = []
    for k, (d, az) in enumerate(zip(distances_m, azimuths_deg)):
        g.append(SimpleNamespace(
            origin=SimpleNamespace(depth_in_m=float(depth_m)),
            station=SimpleNamespace(station='S%d' % k, network='XX'),
            azimuth=float(az),
            distance_in_m=float(d)))
    return g


def test_get_predicted_isotropic_all_positive():
    m = PolarityRelPampMisfit(method='taup')
    greens = _make_greens([500., 1500., 3000., 4500.], [10., 100., 190., 280.])
    iso = np.array([[1., 1., 1., 0., 0., 0.]])
    pol, amp = m.get_predicted(greens, iso, return_all=True)
    assert (pol[0] == 1).all()


def test_project_to_vertical_reduces_or_equals_amplitude():
    greens = _make_greens([500., 3000.], [30., 210.])
    mt = np.array([[1., -1., 0., 0.3, 0.2, -0.1]])
    m0 = PolarityRelPampMisfit(method='taup', project_to_vertical=False)
    m1 = PolarityRelPampMisfit(method='taup', project_to_vertical=True)
    _, a0 = m0.get_predicted(greens, mt, return_all=True)
    _, a1 = m1.get_predicted(greens, mt, return_all=True)
    assert not np.allclose(a0, a1)                       # projection changes amp
    assert np.all(np.abs(a1) <= np.abs(a0) + 1e-12)       # cos(incidence) <= 1


def test_call_perfect_fit_is_zero():
    m = PolarityRelPampMisfit(method='taup')
    greens = _make_greens([500., 1500., 3000., 4500., 6000.],
                          [10., 80., 150., 220., 300.])
    true = np.array([[1., -1., 0., 0.5, 0.2, -0.3]])
    pol_t, amp_t = m.get_predicted(greens, true, return_all=True)
    obs = (pol_t[0], amp_t[0], np.ones(5), np.ones(5))
    e = m(obs, greens, true)
    assert e.shape == (1, 1)
    assert e[0, 0] < 1e-9


def test_self_recovery_argmin_is_true_source():
    m = PolarityRelPampMisfit(method='taup')
    greens = _make_greens([500., 1500., 3000., 4500., 6000., 2000.],
                          [10., 80., 150., 220., 300., 45.])
    true = np.array([1., -1., 0., 0.5, 0.2, -0.3])
    pol_t, amp_t = m.get_predicted(greens, true.reshape(1, 6), return_all=True)
    obs = (pol_t[0], amp_t[0], np.ones(6), np.ones(6))
    candidates = np.array([
        [2., 0., 1., 0., 0., 0.],
        true,                              # index 1 == truth
        [0., 0., 0., 1., 0., 0.],
        -true,
        [1., 0., -1., 0.2, -0.4, 0.1],
    ])
    e = m(obs, greens, candidates)
    assert e.shape == (5, 1)
    assert int(np.argmin(e[:, 0])) == 1
    assert e[1, 0] < 1e-9


def test_zero_weight_station_does_not_change_misfit():
    m = PolarityRelPampMisfit(method='taup')
    g5 = _make_greens([500., 1500., 3000., 4500., 6000.],
                      [10., 80., 150., 220., 300.])
    true = np.array([[1., -0.5, 0.3, 0.4, -0.2, 0.1]])
    pol_t, amp_t = m.get_predicted(g5, true, return_all=True)
    cand = np.array([[0.2, -1., 0.5, 0.1, 0.3, -0.4]])

    obs5 = (pol_t[0], amp_t[0], np.ones(5), np.ones(5))
    e5 = m(obs5, g5, cand)

    g6 = g5 + _make_greens([9999.], [42.])           # extra junk station
    pol6 = np.append(pol_t[0], 9.0)                   # garbage obs at station 6
    amp6 = np.append(amp_t[0], 9.0)
    obs6 = (pol6, amp6, np.append(np.ones(5), 0.0), np.append(np.ones(5), 0.0))
    e6 = m(obs6, g6, cand)

    np.testing.assert_allclose(e5, e6)


def test_all_zero_weights_is_finite_zero():
    m = PolarityRelPampMisfit(method='taup')
    g = _make_greens([500., 1500.], [10., 200.])
    obs = (np.array([1., -1.]), np.array([0.5, 0.2]),
           np.zeros(2), np.zeros(2))
    e = m(obs, g, np.array([[1., -1., 0., 0.2, 0.1, -0.1]]))
    assert np.isfinite(e[0, 0]) and e[0, 0] == 0.0


def test_single_station_is_finite():
    m = PolarityRelPampMisfit(method='taup')
    g = _make_greens([1500.], [77.])
    true = np.array([[1., -1., 0., 0.5, 0.2, -0.3]])
    pol_t, amp_t = m.get_predicted(g, true, return_all=True)
    obs = (pol_t[0], amp_t[0], np.ones(1), np.ones(1))
    e = m(obs, g, np.array([[0.1, 0.2, -0.3, 0.4, 0.5, 0.6]]))
    assert np.isfinite(e[0, 0])


def test_ray_path_distance_mode_finite_positive():
    m = PolarityRelPampMisfit(method='taup', distance_mode='ray_path')
    g = _make_greens([1500., 4000.], [30., 200.])
    r = m._distances_m(g)
    assert np.all(np.isfinite(r)) and np.all(r > 0)


def test_count_observations():
    m = PolarityRelPampMisfit(method='taup')
    obs = (np.array([1., -1., 0.]), np.array([0.5, 0.2, 0.9]),
           np.array([2., 3., 4.]),
           np.array([1., 1., 0.]), np.array([1., 0., 1.]),
           np.array([1., 1., 0.]))
    c = m.count_observations(obs)
    assert c['pol'] == 2
    assert c['pamp'] == 2
    assert c['sprat'] == 2


def test_description_returns_str():
    m = PolarityRelPampMisfit(method='taup')
    assert isinstance(m.description(), str)
    assert 'PolarityRelPampMisfit' in m.description()


def test_public_import_paths():
    # exported from the misfit subpackage
    from mtuq.misfit import (
        PolarityRelPampMisfit as _A,
        relative_pamp_polarity_from_dict as _B,
        relative_pol_pamp_spamp_from_dict as _C,
    )
    # re-exported at the top level
    import mtuq
    assert hasattr(mtuq, 'PolarityRelPampMisfit')


def test_constructor_sprat_defaults_and_validation():
    m = PolarityRelPampMisfit(method='taup')
    assert m.lambda_sprat == 0.0
    assert m.norm_sprat == 'L1'
    assert m.sp_obs_is_log10 is False
    with pytest.raises(ValueError):
        PolarityRelPampMisfit(method='taup', norm_sprat='L9')


def test_from_dict_spamp_aligns_and_weights():
    stations = [_station('AAA'), _station('BBB')]
    from mtuq.misfit.relative_amplitude import relative_pol_pamp_spamp_from_dict
    pol = {'AAA': 1, 'BBB': -1}
    pamp = {'AAA': 0.5, 'BBB': 0.2}
    sprat = {'AAA': 2.0}                      # BBB absent -> 0 value, 0 weight
    out = relative_pol_pamp_spamp_from_dict(
        pol, pamp, sprat, None, None, None, stations)
    pol_o, pamp_o, sprat_o, wp, wa, ws = out
    np.testing.assert_allclose(sprat_o, [2.0, 0.0])
    np.testing.assert_allclose(ws, [1.0, 0.0])


def test_lambda_sprat_zero_ignores_sp_observations():
    m = PolarityRelPampMisfit(method='taup')  # lambda_sprat == 0
    greens = _make_greens([500., 1500., 3000.], [10., 120., 240.])
    true = np.array([[1., -1., 0., 0.5, 0.2, -0.3]])
    pol_t, amp_t = m.get_predicted(greens, true, return_all=True)
    cand = np.array([[0.2, -1., 0.5, 0.1, 0.3, -0.4]])
    e4 = m((pol_t[0], amp_t[0], np.ones(3), np.ones(3)), greens, cand)
    # add garbage S/P with unit weight; must be ignored because lambda_sprat=0
    e6 = m((pol_t[0], amp_t[0], np.array([9., 9., 9.]),
            np.ones(3), np.ones(3), np.ones(3)), greens, cand)
    np.testing.assert_allclose(e4, e6)


def test_self_recovery_with_sprat_term():
    m = PolarityRelPampMisfit(
        method='taup', lambda_pol=1.0, lambda_pamp=1.0, lambda_sprat=1.0,
        sp_obs_is_log10=True)
    greens = _make_greens(
        [500., 1500., 3000., 4500., 6000., 2500., 3500.],
        [10., 60., 120., 180., 240., 300., 340.])
    true = np.array([1., -1., 0.2, 0.5, 0.2, -0.3])
    pol_t, pamp_t, sprat_t = m._predict(greens, true.reshape(1, 6))
    n = len(greens)
    obs = (pol_t[0], pamp_t[0], sprat_t[0],
           np.ones(n), np.ones(n), np.ones(n))
    candidates = np.array([
        [2., 0., 1., 0., 0., 0.],
        true,                             # index 1 == truth
        [0., 0., 0., 1., 0., 0.],
        [1., 0., -1., 0.2, -0.4, 0.1],
    ])
    e = m(obs, greens, candidates)
    assert int(np.argmin(e[:, 0])) == 1
    assert e[1, 0] < 1e-9


def test_sp_obs_is_log10_flag_matches_linear_input():
    greens = _make_greens([500., 1500., 3000., 4500.], [10., 100., 190., 280.])
    true = np.array([[0.4, -1., 0.6, 0.3, -0.2, 0.1]])
    base = PolarityRelPampMisfit(
        method='taup', lambda_pol=0., lambda_pamp=0., lambda_sprat=1.,
        sp_obs_is_log10=True)
    _, _, sprat_t = base._predict(greens, true)
    n = len(greens)
    logobs = sprat_t[0]

    m_log = base
    m_lin = PolarityRelPampMisfit(
        method='taup', lambda_pol=0., lambda_pamp=0., lambda_sprat=1.,
        sp_obs_is_log10=False)
    z = np.zeros(n)
    obs_log = (z, z, logobs, z, z, np.ones(n))
    obs_lin = (z, z, 10.0 ** logobs, z, z, np.ones(n))

    cands = np.array([true[0],
                      [1., 0., -1., 0.2, -0.4, 0.1],
                      [0.2, -1., 0.5, 0.1, 0.3, -0.4]])
    e_log = m_log(obs_log, greens, cands)
    e_lin = m_lin(obs_lin, greens, cands)
    np.testing.assert_allclose(e_log, e_lin, atol=1e-9)
    assert e_log[0, 0] < 1e-9              # perfect fit at truth


def test_collect_attributes_has_polarity_and_takeoff():
    m = PolarityRelPampMisfit(method='taup')
    g = _make_greens([1500., 3000.], [30., 210.])
    obs = (np.array([1., -1.]), np.array([0.5, 0.2]),
           np.ones(2), np.ones(2))
    attrs = m.collect_attributes(obs, g)
    assert len(attrs) == 2
    assert attrs[0].polarity == 1.0
    assert attrs[1].polarity == -1.0
    assert attrs[0].azimuth == 30.0
    assert np.isfinite(attrs[0].takeoff_angle)


def test_end_to_end_real_dc_grid_recovers_truth():
    """ Evaluate the misfit over a real MTUQ DoubleCouple grid (offline). """
    from mtuq.grid import DoubleCoupleGridRegular

    m = PolarityRelPampMisfit(method='taup', lambda_pol=1.0, lambda_pamp=1.0)
    greens = _make_greens(
        [500., 1500., 3000., 4500., 6000., 2500., 3500.],
        [10., 60., 120., 180., 240., 300., 340.])

    grid = DoubleCoupleGridRegular(magnitudes=[4.5], npts_per_axis=6)
    idx0 = 100
    true_mt = grid.get(idx0).as_vector().reshape(1, 6)

    pol_t, amp_t = m.get_predicted(greens, true_mt, return_all=True)
    n = len(greens)
    obs = (pol_t[0], amp_t[0], np.ones(n), np.ones(n))

    e = m(obs, greens, grid)                    # misfit over the whole grid
    assert e.shape[0] == grid.size
    assert np.min(e[:, 0]) < 1e-9               # perfect fit exists in the grid
    assert e[idx0, 0] < 1e-9                     # ... at the true source


def test_end_to_end_fixed_fault_lune_grid_recovers_truth():
    """ Stage 2 of the example: fixed-fault Lune grid over source type. """
    from mtuq.grid import Grid, DoubleCoupleGridRegular
    from mtuq.grid.moment_tensor import to_mt
    from mtuq.util.math import open_interval

    m = PolarityRelPampMisfit(method='taup', lambda_pol=1.0, lambda_pamp=1.0)
    greens = _make_greens(
        [500., 1500., 3000., 4500., 6000., 2500., 3500.],
        [10., 60., 120., 180., 240., 300., 340.])

    # fix a fault geometry from a DC grid point, then vary v, w (Lune)
    dc = DoubleCoupleGridRegular(magnitudes=[4.5], npts_per_axis=6)
    d = dc.get_dict(50)
    v = open_interval(-1. / 3., 1. / 3., 8)
    w = open_interval(-3. / 8. * np.pi, 3. / 8. * np.pi, 8)
    lune = Grid(
        dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        coords=(np.array([d['rho']]), v, w,
                np.array([d['kappa']]), np.array([d['sigma']]),
                np.array([d['h']])),
        callback=to_mt)

    idx0 = 20
    true_mt = lune.get(idx0).as_vector().reshape(1, 6)
    pol_t, amp_t = m.get_predicted(greens, true_mt, return_all=True)
    n = len(greens)
    obs = (pol_t[0], amp_t[0], np.ones(n), np.ones(n))

    e = m(obs, greens, lune)
    assert e.shape[0] == lune.size
    assert np.all(np.isfinite(e))
    assert e[idx0, 0] < 1e-9


# ---------------------------------------------------------------------------
# C6.1: GeometryList tests
# ---------------------------------------------------------------------------

def test_geometry_list_select_from_arrays_and_get_stations():
    orig1 = Origin({'latitude': 54.0, 'longitude': -120.0, 'depth_in_m': 2500.0})
    orig2 = Origin({'latitude': 54.0, 'longitude': -120.0, 'depth_in_m': 3500.0})
    stations = [_station('STA1'), _station('STA2'), _station('STA3')]
    az = [45., 135., 225.]
    dist = [1000., 2000., 3000.]

    # length validation
    with pytest.raises(ValueError):
        GeometryList.from_arrays(orig1, stations, [45.], dist)
    with pytest.raises(ValueError):
        GeometryList.from_arrays(orig1, stations, az, [1000.])
    with pytest.raises(ValueError):
        GeometryList.from_arrays(orig1, stations, az, dist, takeoff_angle=[100.])
    with pytest.raises(ValueError):
        GeometryList.from_arrays(orig1, stations, az, dist, incidence_angle=[10.])
    with pytest.raises(ValueError):
        GeometryList.from_arrays(orig1, stations, az, dist, ray_length_in_m=[1000.])
    with pytest.raises(ValueError):
        GeometryList.from_arrays(orig1, stations, az, dist, vp_over_vs=[1.7, 1.8])

    # valid creation
    gl = GeometryList.from_arrays(
        orig1, stations, az, dist,
        takeoff_angle=[150., 140., 130.],
        incidence_angle=[30., 40., 50.],
        ray_length_in_m=[1200., 2200., 3200.],
        vp_over_vs=1.73)
    assert len(gl) == 3

    # get_stations order and fields
    stas = gl.get_stations()
    assert len(stas) == 3
    assert [s.station for s in stas] == ['STA1', 'STA2', 'STA3']
    assert [s.network for s in stas] == ['XX', 'XX', 'XX']

    # append entry with orig2
    entry4 = AttribDict({
        'origin': orig2,
        'station': AttribDict({'station': 'STA4', 'network': 'XX'}),
        'azimuth': 315.,
        'distance_in_m': 4000.,
    })
    gl.append(entry4)
    assert len(gl) == 4

    # select only entries matching orig1
    sel1 = gl.select(orig1)
    assert isinstance(sel1, GeometryList)
    assert len(sel1) == 3
    assert [e.station.station for e in sel1] == ['STA1', 'STA2', 'STA3']
    assert all(e.origin == orig1 for e in sel1)

    # select only entries matching orig2
    sel2 = gl.select(orig2)
    assert isinstance(sel2, GeometryList)
    assert len(sel2) == 1
    assert sel2[0].station.station == 'STA4'

    # non-Origin selector raises TypeError
    with pytest.raises(TypeError):
        gl.select("not_an_origin")
    with pytest.raises(TypeError):
        gl.select(12345)
    with pytest.raises(TypeError):
        gl.select({'depth_in_m': 2500.0})


# ---------------------------------------------------------------------------
# C6.2: Precomputed method matches TauP
# ---------------------------------------------------------------------------

def test_precomputed_method_matches_taup():
    orig = Origin({'latitude': 56.0, 'longitude': -122.0, 'depth_in_m': 2500.0})
    greens = _make_greens([500., 1200., 2500., 4000.], [30., 110., 210., 315.], depth_m=2500.0)

    m_taup = PolarityRelPampMisfit(
        method='taup', taup_model='ak135',
        lambda_pol=1.0, lambda_pamp=1.0, lambda_sprat=1.0,
        project_to_vertical=True, distance_mode='ray_path')

    takeoff, incidence = m_taup._takeoff_incidence(greens)
    ray_len = m_taup._distances_m(greens)
    vov = m_taup._vp_over_vs(greens)

    geom_list = GeometryList.from_arrays(
        orig, [g.station for g in greens],
        [g.azimuth for g in greens], [g.distance_in_m for g in greens],
        takeoff_angle=takeoff, incidence_angle=incidence,
        ray_length_in_m=ray_len, vp_over_vs=vov)

    m_pre = PolarityRelPampMisfit(
        method='precomputed',
        lambda_pol=1.0, lambda_pamp=1.0, lambda_sprat=1.0,
        project_to_vertical=True, distance_mode='ray_path')
    assert m_pre._taup is None

    sources = np.array([
        [1., -1., 0.2, 0.5, 0.2, -0.3],
        [0.3, 0.4, -0.7, 0.1, -0.5, 0.2],
        [-0.5, 0.2, 0.3, -0.2, 0.4, -0.1],
    ])

    pol_taup, pamp_taup, sp_taup = m_taup.get_predicted(greens, sources, return_sprat=True)
    pol_pre, pamp_pre, sp_pre = m_pre.get_predicted(geom_list, sources, return_sprat=True)

    np.testing.assert_allclose(pol_taup, pol_pre)
    np.testing.assert_allclose(pamp_taup, pamp_pre, rtol=1e-12)
    np.testing.assert_allclose(sp_taup, sp_pre, rtol=1e-12)

    obs = (pol_taup[0], pamp_taup[0], sp_taup[0],
           np.ones(4), np.ones(4), np.ones(4))
    e_taup = m_taup(obs, greens, sources)
    e_pre = m_pre(obs, geom_list, sources)
    np.testing.assert_allclose(e_taup, e_pre, atol=1e-12)

    terms_taup = m_taup.get_terms(obs, greens, sources)
    terms_pre = m_pre.get_terms(obs, geom_list, sources)
    for tt, tp in zip(terms_taup, terms_pre):
        np.testing.assert_allclose(tt, tp, atol=1e-12)


# ---------------------------------------------------------------------------
# C6.3: Paper preset values and overrides
# ---------------------------------------------------------------------------

def test_paper_preset_defaults_and_overrides():
    m = PolarityRelPampMisfit.paper_preset()
    assert m.method == 'taup'
    assert m.taup_model == 'ak135'
    assert m.lambda_pol == 0.0
    assert m.lambda_pamp == 1.0
    assert m.lambda_sprat == 0.0
    assert m.norm_pamp == 'L1'
    assert m.project_to_vertical is True
    assert m.distance_mode == 'ray_path'

    m_ov = PolarityRelPampMisfit.paper_preset(
        method='precomputed', lambda_sprat=1.0, sp_obs_is_log10=True,
        norm_sprat='L2')
    assert m_ov.method == 'precomputed'
    assert m_ov.lambda_sprat == 1.0
    assert m_ov.sp_obs_is_log10 is True
    assert m_ov.norm_sprat == 'L2'
    assert m_ov.lambda_pol == 0.0
    assert m_ov.lambda_pamp == 1.0
    assert m_ov.project_to_vertical is True
    assert m_ov.distance_mode == 'ray_path'


# ---------------------------------------------------------------------------
# C6.4: TauP straight-ray relation in homogeneous crust
# ---------------------------------------------------------------------------

def test_taup_straight_ray_in_ak135_homogeneous_crust():
    model = TauPyModel('ak135')
    depth_km = 2.5
    for d_km in [0.3, 1.0, 2.0, 4.0]:
        d_deg = kilometers2degrees(d_km)
        arrs = model.get_travel_times(depth_km, d_deg, phase_list=['p', 'P'])
        arr = sorted(arrs, key=lambda a: a.time)[0]

        expected_inc = np.degrees(np.arctan2(d_km, depth_km))
        expected_to = 180.0 - expected_inc

        # Check empirical takeoff and incidence against straight ray
        assert abs(arr.takeoff_angle - expected_to) < 0.1
        assert abs(arr.incident_angle - expected_inc) < 0.1

        # Check ray path length against hypocentral distance
        g = [SimpleNamespace(
            origin=SimpleNamespace(depth_in_m=depth_km * 1000.0),
            station=SimpleNamespace(station='S', network='XX'),
            azimuth=0.0,
            distance_in_m=d_km * 1000.0,
        )]
        rl = _ray_path_lengths_m_taup(model, g)[0]
        hyp = np.sqrt((d_km * 1000.0) ** 2 + (depth_km * 1000.0) ** 2)
        assert abs(rl - hyp) / hyp < 0.005


# ---------------------------------------------------------------------------
# C6.5: project_to_vertical factor equals x3/r under straight rays
# ---------------------------------------------------------------------------

def test_project_to_vertical_factor_equals_x3_over_r():
    depth_m = 2500.0
    distances = np.array([300., 1000., 2000., 4000.])
    azimuths = np.array([0., 90., 180., 270.])

    inc = np.degrees(np.arctan2(distances, depth_m))
    to = 180.0 - inc
    hyp = np.sqrt(distances ** 2 + depth_m ** 2)

    orig = Origin({'latitude': 0., 'longitude': 0., 'depth_in_m': depth_m})
    stations = [_station('S%d' % i) for i in range(len(distances))]

    geom = GeometryList.from_arrays(
        orig, stations, azimuths, distances,
        takeoff_angle=to, incidence_angle=inc,
        ray_length_in_m=hyp, vp_over_vs=1.73)

    m_vert = PolarityRelPampMisfit(method='precomputed', project_to_vertical=True)
    m_slant = PolarityRelPampMisfit(method='precomputed', project_to_vertical=False)

    mt = np.array([[1., -1., 0.2, 0.5, 0.2, -0.3]])
    _, pamp_vert = m_vert.get_predicted(geom, mt, return_all=True)
    _, pamp_slant = m_slant.get_predicted(geom, mt, return_all=True)

    ratio = pamp_vert / pamp_slant
    expected_factor = depth_m / hyp
    np.testing.assert_allclose(ratio[0], expected_factor, rtol=1e-12)


# ---------------------------------------------------------------------------
# C6.6: sprat_offset='fit' and weighted median helper
# ---------------------------------------------------------------------------

def test_weighted_median_rows_random_reference():
    def brute_force_weighted_median(values, weights):
        idx = np.argsort(values)
        s_vals = values[idx]
        s_w = weights[idx]
        cum_w = np.cumsum(s_w)
        tot_w = np.sum(s_w)
        k = np.where(cum_w >= 0.5 * tot_w)[0][0]
        return s_vals[k]

    np.random.seed(123)
    n_rows, n_cols = 25, 40
    val = np.random.randn(n_rows, n_cols)
    w = np.random.uniform(0.1, 5.0, n_cols)

    vec_medians = _weighted_median_rows(val, w)
    bf_medians = np.array([brute_force_weighted_median(val[i], w) for i in range(n_rows)])
    np.testing.assert_allclose(vec_medians, bf_medians)


def test_sprat_offset_fit_invariance_and_norms():
    greens = _make_greens([1000., 2000., 3000.], [30., 120., 240.])
    sources = np.array([
        [1., -1., 0., 0.5, 0.2, -0.3],
        [0.2, 0.4, -0.6, 0.1, -0.2, 0.3],
    ])
    weights = np.array([1., 2., 3.])

    # L1 norm test
    m_l1 = PolarityRelPampMisfit(
        method='taup', lambda_pol=0.0, lambda_pamp=0.0, lambda_sprat=1.0,
        norm_sprat='L1', sp_obs_is_log10=True, sprat_offset='fit')

    _, _, sp_syn = m_l1.get_predicted(greens, sources, return_sprat=True)
    obs_sp = sp_syn[0] + 0.4
    obs1 = (np.zeros(3), np.zeros(3), obs_sp,
            np.zeros(3), np.zeros(3), weights)
    obs2 = (np.zeros(3), np.zeros(3), obs_sp + 4.8,
            np.zeros(3), np.zeros(3), weights)

    e_l1_1 = m_l1(obs1, greens, sources)
    e_l1_2 = m_l1(obs2, greens, sources)
    np.testing.assert_allclose(e_l1_1, e_l1_2, atol=1e-12)

    # L2 norm test
    m_l2 = PolarityRelPampMisfit(
        method='taup', lambda_pol=0.0, lambda_pamp=0.0, lambda_sprat=1.0,
        norm_sprat='L2', sp_obs_is_log10=True, sprat_offset='fit')

    e_l2_1 = m_l2(obs1, greens, sources)
    e_l2_2 = m_l2(obs2, greens, sources)
    np.testing.assert_allclose(e_l2_1, e_l2_2, atol=1e-12)

    # Verify L2 uses weighted mean
    diff = obs_sp[None, :] - sp_syn
    c_expected = np.sum(diff * weights[None, :], axis=1) / np.sum(weights)
    res = sp_syn + c_expected[:, None] - obs_sp[None, :]
    expected_l2 = np.sum(weights[None, :] * res ** 2, axis=1) / np.sum(weights)
    np.testing.assert_allclose(e_l2_1[:, 0], expected_l2, atol=1e-12)


# ---------------------------------------------------------------------------
# C6.7: sprat_max_incidence excludes high-incidence stations
# ---------------------------------------------------------------------------

def test_sprat_max_incidence_excludes_high_incidence():
    orig = Origin({'latitude': 50., 'longitude': -120., 'depth_in_m': 2500.})
    geom = GeometryList.from_arrays(
        orig, [_station('S1'), _station('S2'), _station('S3')],
        [10., 100., 200.], [1000., 2000., 3000.],
        takeoff_angle=[165., 150., 125.],
        incidence_angle=[15., 30., 55.],
        ray_length_in_m=[1200., 2200., 3500.],
        vp_over_vs=1.73)

    m_filt = PolarityRelPampMisfit(
        method='precomputed', lambda_pol=1.0, lambda_pamp=1.0, lambda_sprat=1.0,
        sp_obs_is_log10=True, sprat_max_incidence=35.0)

    obs_sp_clean = np.array([0.5, 0.8, 1.2])
    obs_sp_dirty = np.array([0.5, 0.8, 999.0])  # extreme value at station 3 (inc 55 > 35)

    data_clean = (np.array([1., 1., 1.]), np.array([0.5, 0.3, 0.2]),
                  obs_sp_clean, np.ones(3), np.ones(3), np.ones(3))
    data_dirty = (np.array([1., 1., 1.]), np.array([0.5, 0.3, 0.2]),
                  obs_sp_dirty, np.ones(3), np.ones(3), np.ones(3))

    sources = np.array([[1., -1., 0., 0.5, 0.2, -0.3]])

    t_clean = m_filt.get_terms(data_clean, geom, sources)
    t_dirty = m_filt.get_terms(data_dirty, geom, sources)
    # S/P term must be completely unaffected by station 3
    np.testing.assert_allclose(t_clean[2], t_dirty[2], atol=1e-12)

    # Without filter, station 3 blows up the misfit
    m_nofilt = PolarityRelPampMisfit(
        method='precomputed', lambda_pol=1.0, lambda_pamp=1.0, lambda_sprat=1.0,
        sp_obs_is_log10=True, sprat_max_incidence=None)
    t_nofilt = m_nofilt.get_terms(data_dirty, geom, sources)
    assert t_nofilt[2][0] > 100.0

    # Polarity and P-amplitude at station 3 are NOT excluded
    data_pol_flipped = (np.array([1., 1., -1.]), np.array([0.5, 0.3, 0.2]),
                        obs_sp_clean, np.ones(3), np.ones(3), np.ones(3))
    t_pol = m_filt.get_terms(data_pol_flipped, geom, sources)
    assert t_pol[0][0] != t_clean[0][0]


# ---------------------------------------------------------------------------
# C6.8: Ding published nodes in grids
# ---------------------------------------------------------------------------

def test_ding_published_nodes_in_grids():
    ding_gammas = [-3.55, -2.13, -9.25, -10.68, -7.82, -12.12, -16.5]
    ding_colats = [88.04, 90.0, 93.9, 91.9, 88.0, 82.2]

    v_grid, w_grid = _semiregular(40, 81, 0.8, 0.8)
    gamma_grid = to_gamma(v_grid)
    colat_grid = 90.0 - to_delta(w_grid)

    for g in ding_gammas:
        diff = np.min(np.abs(gamma_grid - g))
        assert diff < 0.1, "gamma=%f not within 0.1 deg of grid" % g

    for c in ding_colats:
        diff = np.min(np.abs(colat_grid - c))
        assert diff < 0.1, "colatitude=%f not within 0.1 deg of grid" % c

    grid_dc = DoubleCoupleGridRegular(npts_per_axis=40)
    k_idx = grid_dc.dims.index('kappa')
    s_idx = grid_dc.dims.index('sigma')
    h_idx = grid_dc.dims.index('h')

    # Ding DC: strike=139.5, dip=24.1, rake=78.75
    diff_k = np.min(np.abs(grid_dc.coords[k_idx] - 139.5))
    diff_s = np.min(np.abs(grid_dc.coords[s_idx] - 78.75))
    dips = np.degrees(np.arccos(grid_dc.coords[h_idx]))
    diff_dip = np.min(np.abs(dips - 24.1))

    assert diff_k < 1e-6
    assert diff_s < 1e-6
    assert diff_dip < 0.1


# ---------------------------------------------------------------------------
# C6.9: DC nodal planes produce identical moment tensor
# ---------------------------------------------------------------------------

def test_both_dc_nodal_planes_produce_same_moment_tensor():
    s1, d1, r1 = 143.4, 30.9, 93.8
    s2, d2, r2 = aux_plane(s1, d1, r1)

    mt1 = to_mij(1.0, 0.0, 0.0, s1, r1, np.cos(np.deg2rad(d1)))
    mt2 = to_mij(1.0, 0.0, 0.0, s2, r2, np.cos(np.deg2rad(d2)))

    np.testing.assert_allclose(mt1, mt2, rtol=1e-6, atol=1e-12)


# ---------------------------------------------------------------------------
# C6.10: ding_error formula and scaling
# ---------------------------------------------------------------------------

def test_ding_error_formula_and_scaling():
    m = PolarityRelPampMisfit(method='taup', lambda_pamp=1.0)
    greens = _make_greens([1000., 2000., 3000., 4000.], [30., 120., 210., 300.])
    true_mt = np.array([[1., -1., 0.2, 0.5, 0.2, -0.3]])
    cand_mt = np.array([
        [0.5, 0.2, -0.7, 0.1, -0.3, 0.4],
        true_mt[0],
    ])

    pol_t, amp_t = m.get_predicted(greens, true_mt, return_all=True)
    w_unit = np.ones(4)
    obs_unit = (pol_t[0], amp_t[0], w_unit, w_unit)

    err_unit = m.ding_error(obs_unit, greens, cand_mt)
    # At true MT, ding_error is 0
    assert err_unit[1] < 1e-12

    # For candidate 0, verify formula 0.5 * sum |Pobs_n - Pcalc_n|
    _, pth_all = m.get_predicted(greens, cand_mt, return_all=True)
    pobs_n = _max_abs_normalize(amp_t[0], np.ones(4, dtype=bool))
    pth_n = _max_abs_normalize(pth_all[0], np.ones(4, dtype=bool))
    expected_e0 = 0.5 * np.sum(np.abs(pobs_n - pth_n))
    np.testing.assert_allclose(err_unit[0], expected_e0, atol=1e-12)

    # Scaling weights by 2.5 scales ding_error by 2.5
    w_scaled = 2.5 * np.ones(4)
    obs_scaled = (pol_t[0], amp_t[0], w_unit, w_scaled)
    err_scaled = m.ding_error(obs_scaled, greens, cand_mt)
    np.testing.assert_allclose(err_scaled, 2.5 * err_unit, atol=1e-12)


# ---------------------------------------------------------------------------
# Additional coverage: validation and return_sprat
# ---------------------------------------------------------------------------

def test_sprat_arg_validation():
    with pytest.raises(ValueError):
        PolarityRelPampMisfit(sprat_offset='unrecognized')
    with pytest.raises(ValueError):
        PolarityRelPampMisfit(sprat_offset=True)

    with pytest.raises(ValueError):
        PolarityRelPampMisfit(sprat_max_incidence=-10.0)
    with pytest.raises(ValueError):
        PolarityRelPampMisfit(sprat_max_incidence='bad')
    with pytest.raises(ValueError):
        PolarityRelPampMisfit(sprat_max_incidence=False)


def test_get_predicted_return_sprat_when_lambda_sprat_zero():
    m = PolarityRelPampMisfit(method='taup', lambda_sprat=0.0)
    greens = _make_greens([1000., 2000.], [45., 135.])
    sources = np.array([[1., -1., 0., 0.5, 0.2, -0.3]])

    # return_sprat=False (default) returns pol
    pol = m.get_predicted(greens, sources)
    assert isinstance(pol, np.ndarray) and pol.shape == (1, 2)

    # return_all=True returns (pol, pamp)
    pol, pamp = m.get_predicted(greens, sources, return_all=True)
    assert pol.shape == (1, 2) and pamp.shape == (1, 2)

    # return_sprat=True returns (pol, pamp, sprat) with sprat computed
    pol, pamp, sprat = m.get_predicted(greens, sources, return_sprat=True)
    assert pol.shape == (1, 2) and pamp.shape == (1, 2)
    assert sprat is not None and sprat.shape == (1, 2)
    assert np.all(np.isfinite(sprat))


def test_precomputed_missing_attributes_raise_value_error():
    orig = Origin({'latitude': 50., 'longitude': -120., 'depth_in_m': 2500.})
    m = PolarityRelPampMisfit(method='precomputed', distance_mode='ray_path')

    # Missing takeoff
    g_no_to = [AttribDict({'origin': orig, 'station': _station('S1'),
                           'azimuth': 0., 'distance_in_m': 1000.,
                           'incidence_angle': 20., 'ray_length_in_m': 1200.,
                           'vp_over_vs': 1.73})]
    with pytest.raises(ValueError, match="takeoff_angle"):
        m._takeoff_incidence(g_no_to)

    # Missing incidence
    g_no_inc = [AttribDict({'origin': orig, 'station': _station('S1'),
                            'azimuth': 0., 'distance_in_m': 1000.,
                            'takeoff_angle': 160., 'ray_length_in_m': 1200.,
                            'vp_over_vs': 1.73})]
    with pytest.raises(ValueError, match="incidence_angle"):
        m._takeoff_incidence(g_no_inc)

    # Missing ray_length_in_m
    g_no_rl = [AttribDict({'origin': orig, 'station': _station('S1'),
                           'azimuth': 0., 'distance_in_m': 1000.,
                           'takeoff_angle': 160., 'incidence_angle': 20.,
                           'vp_over_vs': 1.73})]
    with pytest.raises(ValueError, match="ray_length_in_m"):
        m._distances_m(g_no_rl)

    # Missing vp_over_vs
    g_no_vov = [AttribDict({'origin': orig, 'station': _station('S1'),
                            'azimuth': 0., 'distance_in_m': 1000.,
                            'takeoff_angle': 160., 'incidence_angle': 20.,
                            'ray_length_in_m': 1200.})]
    with pytest.raises(ValueError, match="vp_over_vs"):
        m._vp_over_vs(g_no_vov)
