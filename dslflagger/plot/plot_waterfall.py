"""Plot waterfall images.

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
    """Waterfall plot for Timestream.

    This task plots the waterfall (i.e., TOD as a function of time and frequency).
    """

    params_init = {
                    'sat_incl': 'all', # or a list of include (sat1, sat2)
                    'sat_excl': [],
                    'flag_mask': False,
                    'y_axis': 'time', # or 'jul_date', or 'ra'
                    'use_utc': False, # True to use UTC time, else Beijing time
                    'plot_abs': False,
                    'abs_only': False,
                    'gray_color': False,
                    'color_flag': False,
                    'flag_color': 'yellow',
                    'transpose': False, # now only for abs plot
                    'fig_name': 'wf/wf',
                    'rotate_xdate': False, # True to rotate xaxis date ticks, else half the number of date ticks
                  }

    prefix = 'pwf_'

    def process(self, input):
        flag_mask = self.params['flag_mask']
        plot_abs = self.params['plot_abs']
        abs_only = self.params['abs_only']
        gray_color = self.params['gray_color']
        color_flag = self.params['color_flag']
        flag_color = self.params['flag_color']
        transpose = self.params['transpose']
        fig_prefix = self.params['fig_name']
        rotate_xdate = self.params['rotate_xdate']

        # TODO: Implement MPI parallel in the future
        if mpiutil.rank0:
            logger.info(f'Plot waterfall of the input TOD')

            lsts = np.asarray(input['time'])
            freqs = np.asarray(input['freq'])
            freq_extent = [freqs[0], freqs[-1]]
            time_extent = [lsts[0], lsts[-1]]
            extent = freq_extent + time_extent

            x_label = r'$\nu$ / MHz'
            y_label = r'$t$ / Second'

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

                    if flag_mask and has_mask:
                        tfslice1 = np.ma.array(tfslice, mask=mask[:, :, pi, si])
                    else:
                        tfslice1 = tfslice

                    plt.figure()

                    if gray_color:
                        # cmap = 'gray'
                        cmap = plt.cm.gray
                        if color_flag:
                            cmap.set_bad(flag_color)
                    else:
                        cmap = None

                    if abs_only:
                        if transpose:
                            tfslice1 = tfslice1.T
                            x_label, y_label = y_label, x_label
                            extent = time_extent + freq_extent

                        fig, ax = plt.subplots()
                        tfslice_abs = np.abs(tfslice1)
                        im = ax.imshow(tfslice_abs, extent=extent, origin='lower', aspect='auto', cmap=cmap)
                        if transpose:
                            if rotate_xdate:
                                # set the x-axis tick labels to diagonal so it fits better
                                fig.autofmt_xdate()
                            else:
                                # reduce the number of tick locators
                                locator = MaxNLocator(nbins=6)
                                ax.xaxis.set_major_locator(locator)
                                ax.xaxis.set_minor_locator(AutoMinorLocator(2))

                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        plt.colorbar(im)
                    else:
                        if plot_abs:
                            fig, axarr = plt.subplots(1, 3, sharey=True)
                        else:
                            fig, axarr = plt.subplots(1, 2, sharey=True)
                        im = axarr[0].imshow(tfslice1.real, extent=extent, origin='lower', aspect='auto', cmap=cmap)
                        axarr[0].set_xlabel(x_label)
                        axarr[0].set_ylabel(y_label)
                        plt.colorbar(im, ax=axarr[0])
                        im = axarr[1].imshow(tfslice1.imag, extent=extent, origin='lower', aspect='auto', cmap=cmap)
                        axarr[1].set_xlabel(x_label)
                        plt.colorbar(im, ax=axarr[1])
                        if plot_abs:
                            im = axarr[2].imshow(np.abs(tfslice1), extent=extent, origin='lower', aspect='auto', cmap=cmap)
                            axarr[2].set_xlabel(x_label)
                            plt.colorbar(im, ax=axarr[2])

                    fig_name = '%s_%d_%d.png' % (fig_prefix, pi, sat_no)
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
