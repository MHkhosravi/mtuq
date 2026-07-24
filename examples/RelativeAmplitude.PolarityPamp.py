#!/usr/bin/env python

"""
Sequential polarity + relative-P-amplitude moment-tensor inversion.

Reproduces the procedure of Ding et al. (2026, BSSA, doi:10.1785/0120250267):

  Stage 1  grid search over strike/dip/rake for the best double-couple (DC)
           fault plane, using P first-motion polarity and *relative* (max-abs
           normalized) P-wave amplitude;

  Stage 2  with the fault geometry fixed to the Stage-1 result, grid search
           over the Lune longitude/colatitude (source type) to resolve the
           non-DC (CLVD/ISO) components.

Because the instrument response is assumed unknown, only *relative* amplitudes
are used, so seismic moment / Mw are NOT recovered.

USAGE
    python examples/RelativeAmplitude.PolarityPamp.py

NOTE
    Like the other MTUQ examples, this downloads Green's functions from IRIS
    syngine on first run to obtain station geometry (the misfit itself uses only
    station geometry, not the Green's-function waveforms). The observed
    polarities and P amplitudes here are SYNTHESIZED from a known "true" moment
    tensor so the example is self-contained and demonstrates source recovery.
"""

import numpy as np

from mtuq import read, download_greens
from mtuq.event import Origin
from mtuq.graphics import (
    plot_beachball, plot_polarities, plot_misfit_dc, plot_misfit_lune)
from mtuq.grid import Grid, DoubleCoupleGridRegular
from mtuq.grid.moment_tensor import to_mt
from mtuq.grid_search import grid_search
from mtuq.misfit import PolarityRelPampMisfit
from mtuq.util import fullpath
from mtuq.util.cap import parse_station_codes
from mtuq.util.math import open_interval


if __name__ == '__main__':

    model = 'ak135'
    event_id = '20090407201255351'
    path_data = fullpath('data/examples/%s/*.[zrt]' % event_id)
    path_weights = fullpath('data/examples/%s/weights.dat' % event_id)

    origin = Origin({
        'time': '2009-04-07T20:12:55.000000Z',
        'latitude': 61.454200744628906,
        'longitude': -149.7427978515625,
        'depth_in_m': 33033.599853515625,
        })

    #
    # The misfit: joint polarity + relative-P-amplitude, both L1 (paper default),
    # with per-station weights supplied through the observation bundle.
    #
    misfit = PolarityRelPampMisfit(
        method='taup',
        taup_model=model,
        lambda_pol=1.0,
        lambda_pamp=1.0,
        norm_pol='L1',
        norm_pamp='L1',
        project_to_vertical=False,   # scalar along-ray P amplitude (see docstring)
        )

    # To additionally constrain the source with S/P amplitude ratios (beyond
    # the paper), set lambda_sprat > 0 and supply per-station S/P observations:
    #
    #   from mtuq.misfit import relative_pol_pamp_spamp_from_dict
    #   misfit = PolarityRelPampMisfit(method='taup', taup_model=model,
    #       lambda_pol=1.0, lambda_pamp=1.0, lambda_sprat=1.0,
    #       sp_obs_is_log10=False)
    #   obs = relative_pol_pamp_spamp_from_dict(
    #       pol_dict, pamp_dict, sprat_dict,
    #       w_pol_dict, w_pamp_dict, w_sprat_dict, stations)
    #
    # (S/P is compared absolutely in log10 space, not max-abs normalized.)

    #
    # Read station metadata and download Green's functions for their geometry
    #
    print('Reading data / station geometry...\n')
    station_id_list = parse_station_codes(path_weights)
    data = read(path_data, format='sac', event_id=event_id,
                station_id_list=station_id_list,
                tags=['units:m', 'type:velocity'])
    data.sort_by_distance()
    stations = data.get_stations()

    print('Downloading Green\'s functions (geometry only)...\n')
    greens = download_greens(stations, origin, model)
    greens = greens.select(origin)

    #
    # Synthesize "observed" polarity + P amplitude from a known true MT so the
    # example is self-contained. (In a real study these come from picks:
    #   obs = relative_pamp_polarity_from_dict(pol, pamp, w_pol, w_pamp, stations))
    #
    grid_dc = DoubleCoupleGridRegular(magnitudes=[4.5], npts_per_axis=40)
    true_mt = grid_dc.get(20000)                      # an arbitrary DC as "truth"

    pol_true, pamp_true = misfit.get_predicted(
        greens, true_mt, return_all=True)
    n = len(greens)
    obs = (pol_true[0], pamp_true[0], np.ones(n), np.ones(n))   # unit weights

    #
    # Stage 1: best DC fault plane (strike/dip/rake) over a 40-per-axis grid
    #
    print('Stage 1: DC grid search (strike/dip/rake)...\n')
    results_dc = grid_search(obs, greens, misfit, origin, grid_dc)
    idx = results_dc.source_idxmin()
    best_dc = grid_dc.get(idx)
    best_dict = grid_dc.get_dict(idx)

    #
    # Stage 2: fix the fault geometry, grid search the Lune (source type)
    #
    print('Stage 2: Lune grid search (source type, fault fixed)...\n')
    v = open_interval(-1. / 3., 1. / 3., 40)
    w = open_interval(-3. / 8. * np.pi, 3. / 8. * np.pi, 40)
    grid_lune = Grid(
        dims=('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
        coords=(np.array([best_dict['rho']]), v, w,
                np.array([best_dict['kappa']]),
                np.array([best_dict['sigma']]),
                np.array([best_dict['h']])),
        callback=to_mt)
    results_lune = grid_search(obs, greens, misfit, origin, grid_lune)
    best_full = grid_lune.get(results_lune.source_idxmin())

    #
    # Figures: DC and Lune misfit surfaces, beachball, and polarity fit
    #
    print('Generating figures...\n')
    plot_misfit_dc(event_id + '_RelPamp_misfit_dc.png', results_dc,
                   title='Polarity + relative-P-amplitude misfit')
    plot_misfit_lune(event_id + '_RelPamp_misfit_lune.png', results_lune,
                     title='Source-type misfit (fault fixed)')
    plot_beachball(event_id + '_RelPamp_beachball.png',
                   best_full, stations, origin)

    predicted = misfit.get_predicted(greens, best_dc)
    attrs = misfit.collect_attributes(obs, greens)
    plot_polarities(event_id + '_RelPamp_polarities.png',
                    obs[0], predicted, attrs, origin, best_dc)

    print('\nFinished\n')
