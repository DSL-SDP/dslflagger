"""Plot time or frequency slices.

Inheritance diagram
-------------------

.. inheritance-diagram:: Plot
   :parts: 2

"""

from datetime import datetime, timedelta
import numpy as np
from caput import memh5, mpiutil
from dslpipe.pipeline.pipeline import OneAndOne
from dslpipe.utils.path_util import input_path, output_path
from dslpipe.utils.logging import get_logger
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

logger = get_logger(__name__)


class Plot(OneAndOne):
    """Plot time or frequency slices.

    This task plots a given number of time (or frequency) slice of a TOD.
    """

    params_init = {
                    'plot_type': 'time', # or 'freq'
                    'sat_incl': 'all', # or a list of include (sat1, sat2)
                    'sat_excl': [],
                    'flag_mask': False,
                    'slices': 10, # number of slices to plot
                    'fig_name': 'slice/slice',
                    'rotate_xdate': False, # True to rotate xaxis date ticks, else half the number of date ticks
                  }

    prefix = 'psl_'

    def process(self, input):

        plot_type = self.params['plot_type']
        slices = self.params['slices']
        flag_mask = self.params['flag_mask']
        fig_prefix = self.params['fig_name']
        rotate_xdate = self.params['rotate_xdate']

        # TODO: Implement MPI parallel in the future
        if mpiutil.rank0:
            logger.info(f'Plot {slices} {plot_type} of the input TOD')

            lsts = np.asarray(input['time'])
            freqs = np.asarray(input['freq'])

            mask = None
            has_mask = False
            if 'mask' in input:
                mask = input['mask']
                has_mask = True

            n_pol = input['tod'].shape[2]
            n_sat = input['tod'].shape[3]

            for pi in range(n_pol):
                for si in range(n_sat):
                    tfslice = input['tod'][:, :, pi, si]
                    sat_no = input['sat_no'][si]

                    if plot_type == 'time':
                        nt = tfslice.shape[0]
                        c = nt//2
                        s = max(0, c-slices//2)
                        e = min(nt, s+slices)
                        if flag_mask and has_mask:
                            tfslice1 = np.ma.array(tfslice[s:e], mask=mask[s:e, :, pi, si])
                        else:
                            tfslice1 = tfslice[s:e, :]

                        o = c - s
                        shift = 0.1 * np.ma.max(np.abs(tfslice1[o]))

                        ax_val = freqs[:]
                        xlabel = r'$\nu$ / MHz'
                    elif plot_type == 'freq':
                        nfreq = tfslice.shape[1]
                        c = nfreq//2
                        s = max(0, c-slices//2)
                        e = min(nfreq, s+slices)
                        if flag_mask and has_mask:
                            tfslice1 = np.ma.array(tfslice[:, s:e], mask=mask[:, s:e, pi, si])
                        else:
                            tfslice1 = tfslice[:, s:e]

                        o = c - s
                        shift = 0.1 * np.ma.max(np.abs(tfslice1[:, o]))

                        ax_val = lsts[:]
                        xlabel = r'$t$ / Second'
                    else:
                        raise ValueError('Unknown plot_type %s, must be either time or freq' % plot_type)

                    plt.figure()
                    f, axarr = plt.subplots(3, sharex=True)
                    for i in range(e - s):
                        if plot_type == 'time':
                            axarr[0].plot(ax_val, tfslice1[i].real + (i - o)*shift, label='real')
                        elif plot_type == 'freq':
                            axarr[0].plot(ax_val, tfslice1[:, i].real + (i - o)*shift, label='real')
                        if i == 0:
                            axarr[0].legend()

                        if plot_type == 'time':
                            axarr[1].plot(ax_val, tfslice1[i].imag + (i - o)*shift, label='imag')
                        elif plot_type == 'freq':
                            axarr[1].plot(ax_val, tfslice1[:, i].imag + (i - o)*shift, label='imag')
                        if i == 0:
                            axarr[1].legend()

                        if plot_type == 'time':
                            axarr[2].plot(ax_val, np.abs(tfslice1[i]) + (i - o)*shift, label='abs')
                        elif plot_type == 'freq':
                            axarr[2].plot(ax_val, np.abs(tfslice1[:, i]) + (i - o)*shift, label='abs')
                        if i == 0:
                            axarr[2].legend()

                    if plot_type == 'freq':
                        duration = (ax_val[-1] - ax_val[0])
                        dt = duration / len(ax_val)
                        ext = max(0.05*duration, 5*dt)
                        axarr[2].set_xlim([ax_val[0]-ext, ax_val[-1]+ext])
                        axarr[2].xaxis_date()
                        # date_format = mdates.DateFormatter('%H:%M')
                        # axarr[2].xaxis.set_major_formatter(date_format)
                        if rotate_xdate:
                            # set the x-axis tick labels to diagonal so it fits better
                            f.autofmt_xdate()
                        else:
                            # half the number of date ticks so they do not overlap
                            # axarr[2].set_xticks(axarr[2].get_xticks()[::2])
                            # reduce the number of tick locators
                            locator = MaxNLocator(nbins=6)
                            axarr[2].xaxis.set_major_locator(locator)
                            axarr[2].xaxis.set_minor_locator(AutoMinorLocator(2))
                    elif plot_type == 'time':
                        bw = (ax_val[-1] - ax_val[0])
                        df = bw / len(ax_val)
                        ext = max(0.05*bw, df)
                        axarr[2].set_xlim([ax_val[0]-ext, ax_val[-1]+ext])

                    axarr[2].set_xlabel(xlabel)

                    fig_name = '%s_%s_%d_%d.png' % (fig_prefix, plot_type, pi, sat_no)
                    fig_name = output_path(fig_name)
                    plt.savefig(fig_name)
                    plt.close()

        return super().process(input)

    def read_input(self):
        """Override to implement reading inputs from disk."""

        # TODO: Implement MPI parallel in the future
        if mpiutil.rank0:
            return memh5.MemDiskGroup.from_file(input_path(self.params['input_files']))

    def write_output(self, output):
        """Override to implement writing outputs to disk."""

        # TODO: Implement MPI parallel in the future
        if mpiutil.rank0:
            return output.save(output_path(self.params['output_files']), mode='w')
