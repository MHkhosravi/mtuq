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
    relative_pamp_polarity_from_dict,
    _use_to_ned,
    _pol_pamp_mt,
    _max_abs_normalize,
    _weighted_misfit_term,
    _s_over_p_log10_mt,
    _medium_from_taup_at_depth,
)


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
