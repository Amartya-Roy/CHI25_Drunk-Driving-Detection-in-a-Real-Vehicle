# emacs: -*- mode: python; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 noet:
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the remodnavlad package for the
#   copyright and license terms.
#
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
__version__ = '1.0'

import logging
lgr = logging.getLogger('remodnav')

import pandas as pd

from .clf import (
    EyegazeClassifier,
    events2bids_events_tsv,
)

help = {
    'dilate_nan': """duration for which to replace data by missing data
    markers on either side of a signal-loss window (in seconds)""",
    'lowpass_cutoff_freq': """cut-off frequency of a Butterworth low-pass
    filter applied to determine drift velocities in a pursuit event
    candidate (in Hz)""",
    'max_initial_saccade_freq': """maximum saccade frequency for initial
    detection of major saccades, initial data chunking is stopped if this
    frequency is reached (should be smaller than an expected (natural)
    saccade frequency in a particular context) (in Hz)""",
    'max_pso_duration': """maximum duration of a post-saccadic oscillation
    (glissade) candidate (in seconds)""",
    'max_vel': """maximum velocity threshold, will replace value with
    maximum, and issue warning if exceeded to inform about potentially
    inappropriate filter settings""",
    'median_filter_length': """smoothing median-filter size (for initial
    data chunking only) (in seconds)""",
    'min_blink_duration': """missing data windows shorter than this
    duration will not be considered for --dilate-nan (in seconds)""",
    'min_fixation_duration': """minimum duration of a fixation event
    candidate (in seconds)""",
    'min_intersaccade_duration': """no saccade detection is performed in
    windows shorter than twice this value, plus minimum saccade and PSO
    duration (in seconds)""",
    'min_pursuit_duration': """minimum duration of a pursuit event
    candidate (in seconds)""",
    'min_saccade_duration': """minimum duration of a saccade event
    candidate (in seconds)""",
    'noise_factor': """adaptive saccade onset threshold velocity is the
    median absolute deviation of velocities in the window of interest,
    times this factor (peak velocity threshold is twice the onset velocity);
    increase for noisy data to reduce false positives""",
    'pursuit_velthresh': """fixed drift velocity threshold to distinguish
    periods of pursuit from periods of fixation; higher than natural ocular
    drift velocities during fixations""",
    'saccade_context_window_length': """size of a window centered on any
    velocity peak for adaptive determination of saccade velocity thresholds
    (for initial data chunking only) (in seconds)""",
    'savgol_length': """size of Savitzky-Golay filter for noise reduction
    (in seconds)""",
    'savgol_polyord': """polynomial order of Savitzky-Golay filter for noise
    reduction""",
    'velthresh_startvelocity': """start value for adaptive velocity threshold
    algorithm, should be larger than any conceivable minimum saccade velocity
    (in deg/s)""",
}


def remodnav(data, args):
    import argparse
    import inspect
    kwargs = {}
    for func in (EyegazeClassifier.__init__, EyegazeClassifier.preproc):
        # pull kwargs and their defaults out of the function definitions
        argspec = inspect.getfullargspec(func)
        kwargs.update(zip(argspec.args[::-1], argspec.defaults[::-1]))

    parser = argparse.ArgumentParser(
        prog='remodnav',
        description='{}'.format(
            EyegazeClassifier.__doc__,
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'outfile', metavar='<eventfile>',
        help="""Output file name. This file will contain information on all
        detected eye movement events in BIDS events.tsv format.""")
    parser.add_argument(
        'px2deg', type=float, metavar='<PX2DEG>',
        help="""Factor to convert pixel coordinates to visual degrees, i.e.
        the visual angle of a single pixel. Pixels are assumed to be square.
        This will typically be a rather small value.""")
    parser.add_argument(
        'sampling_rate', type=float, metavar='<SAMPLING RATE>',
        help="""Sampling rate of the data in Hertz. Only data with dense
        regular sampling are supported.""")
    parser.add_argument(
        'input_type', type=str,
        help="""To specify whether the input file is in px or deg Important for later calculations.
        Can either be 'deg' or 'px'.""")
    parser.add_argument(
        'plot_figure', type=str,
        help="""Set to True to plot a figure of the events, speed and coordinates/ angles.""")
    parser.add_argument(
        '--log-level', choices=('debug', 'info', 'warn', 'error'),
        metavar='level', default='warn',
        help="""debug|info|warn|error. 'info' and 'debug' levels enable output
        of increasing levels of detail on the algorithm steps and decision
        making. Default: warn""")

    for argname, default in sorted(kwargs.items(), key=lambda x: x[0]):
        parser.add_argument(
            '--{}'.format(argname.replace('_', '-')),
            dest=argname,
            metavar='<float>' if argname != 'savgol-polyord' else '<int>',
            type=float if argname != 'savgol-polyord' else int,
            default=default,
            help=help[argname] + ' [default: {}]'.format(default))

    args = parser.parse_args(args[1:])

    logging.basicConfig(
        format='%(levelname)s:%(message)s',
        level=getattr(logging, args.log_level.upper()))

    # lgr.info('Read %i samples', len(data))

    clf = EyegazeClassifier(
        **{k: getattr(args, k) for k in (
            'px2deg', 'sampling_rate', 'velthresh_startvelocity',
            'min_intersaccade_duration', 'min_saccade_duration',
            'min_pursuit_duration', 'pursuit_velthresh',
            'max_initial_saccade_freq', 'saccade_context_window_length',
            'max_pso_duration', 'min_fixation_duration', 'lowpass_cutoff_freq',
            'noise_factor', 'input_type')}
    )

    # Add specifically named rows for remodnav algorithm
    data['time_rem'] = data.index
    data['x'] = data['azimuth']
    data['y'] = data['elevation']
    # Data is filtered and preprocessed data for event classification is calculated

    data, pp = clf.preproc(
        data,
        **{k: getattr(args, k) for k in (
            'min_blink_duration', 'dilate_nan', 'median_filter_length',
            'savgol_length', 'savgol_polyord', 'max_vel')}
    )

    # Assign the filtered x, y data to the azimuth and elevation angle
    data['azimuth'] = data['x']
    data['elevation'] = data['y']

    events = clf(pp, classify_isp=True, sort_events=True)

    import pytz
    data_events = pd.DataFrame(events)
    data_events = data_events.drop(columns=['id'])
    data_events.index = pd.to_datetime(data_events['start_time'])
    data_events.index = data_events.index.tz_localize(pytz.utc).tz_convert(pytz.timezone('CET'))

    # events2bids_events_tsv(events, args.outfile)

    if args.plot_figure == 'True':
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as pl
        import matplotlib.dates as mdates
        from matplotlib.dates import DateFormatter
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()

        # Convert angles from radians to degree to plot
        import numpy as np
        data['x'] = data['x'] * 180 / np.pi
        data['y'] = data['y'] * 180 / np.pi
        pp['x'] = pp['x'] * 180 / np.pi
        pp['y'] = pp['y'] * 180 / np.pi

        # one inch per second, or as big as PNG software/browsers can handle
        duration = float(len(data)) / args.sampling_rate
        pl.figure(figsize=(min(duration, 400), 3), dpi=100)
        clf.show_gaze(data=data, pp=pp, events=events, show_vels=True)
        dt_fmt = DateFormatter("%H:%M:%S", tz=data.index.tz)
        seclocator = mdates.SecondLocator()
        pl.gca().xaxis.set_major_formatter(dt_fmt)
        pl.gca().xaxis.set_major_locator(seclocator)
        pl.tick_params(labelrotation=45)
        pl.title('Detected eye movement events, parameters: {}'.format(
            ', '.join([
                '{}={}'.format(k, getattr(args, k))
                for k in sorted((
                    'px2deg', 'sampling_rate', 'velthresh_startvelocity',
                    'min_intersaccade_duration', 'min_saccade_duration',
                    'max_initial_saccade_freq', 'saccade_context_window_length',
                    'max_pso_duration', 'min_fixation_duration',
                    'min_pursuit_duration', 'pursuit_velthresh',
                    'min_blink_duration', 'dilate_nan',
                    'median_filter_length', 'savgol_length', 'savgol_polyord',
                    'max_vel', 'lowpass_cutoff_freq', 'noise_factor'))
            ])
        ))
        pl.ylim((-50, 200))
        if args.input_type == 'deg':
            pl.ylabel('angles (azimuth and elevation)')
        elif args.input_type == 'px':
            pl.ylabel('coordinates (pixel)')
        pl.xlabel('time')
        pl.savefig(
            '{}.png'.format(
                args.outfile[:-4] if args.outfile.endswith('.tsv')
                else args.outfile),
            bbox_inches='tight', format='png', dpi=100)

    # Drop columns that were only created for remodnav algorithm
    data.drop(columns=['time_rem', 'x', 'y'], inplace=True)

    return data, data_events

    # import pandas as pd
    # events = pd.DataFrame(events)
    #
    # saccades = events[events['label'] == 'SACC']
    # isaccades = events[events['label'] == 'ISAC']
    # hvpso = events[(events['label'] == 'HPSO') | (events['label'] == 'IHPS')]
    # lvpso = events[(events['label'] == 'LPSO') | (events['label'] == 'ILPS')]
    #
    # pl.figure(figsize=(6,4))
    # for ev, sym, color, label in (
    #        (saccades, '.', 'black', 'saccades'),
    #        (isaccades, '.', 'xkcd:green teal', '"minor" saccades'),
    #        (hvpso, '+', 'xkcd:burnt sienna', 'fast PSOs'),
    #        (lvpso, '+', 'xkcd:azure', 'slow PSOs'))[::-1]:
    #     pl.loglog(ev['amp'], ev['peak_vel'], sym, color=color,
    #              alpha=.2, lw=1, label=label)
    #
    # pl.ylim((10.0, args.max_vel))
    # pl.xlim((0.01, 40.0))
    # pl.legend(loc=4)
    # pl.ylabel('peak velocities (deg/s)')
    # pl.xlabel('amplitude (deg)')
    # pl.savefig(
    #    '{}_mainseq.svg'.format(
    #        args.outfile[:-4] if args.outfile.endswith('.tsv')
    #        else args.outfile),
    #    bbox_inches='tight', format='svg')
