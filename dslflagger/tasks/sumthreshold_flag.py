"""RFI flagging with the SumThreshold method.

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import numpy as np
from caput import memh5, mpiutil
from dslpipe.pipeline.pipeline import OneAndOne
from dslpipe.utils.path_util import input_path, output_path
from dslpipe.utils.logging import get_logger
from dslflagger.rfi import interpolate
from dslflagger.rfi import gaussian_filter
from dslflagger.rfi import sum_threshold

logger = get_logger(__name__)


class Flag(OneAndOne):
    """RFI flagging with the SumThreshold method.
    """

    params_init = {
                    'first_threshold': 12.0,
                    'exp_factor': 1.5,
                    'distribution': 'Rayleigh',
                    'max_threshold_len': 1024,
                    'sensitivity': 1.0,
                    'min_connected': 1,
                    'flag_direction': ('time', 'freq'),
                    'tk_size': 1.0, # 128.0 for dish
                    'fk_size': 3.0, # 2.0 for dish
                    'threshold_num': 2, # number of threshold
                  }

    prefix = 'stf_'

    def process(self, input):

        # TODO: Implement MPI parallel in the future
        if mpiutil.rank0:
            logger.info('RFI flagging with the SumThreshold method.')

            if 'mask' not in input:
                input.create_dataset('mask', data=np.zeros_like(input['tod'], dtype=bool))

            n_pol = input['tod'].shape[2]
            n_sat = input['tod'].shape[3]

            for pi in range(n_pol):
                for si in range(n_sat):
                    tfslice = input['tod'][:, :, pi, si]
                    tfmask = input['mask'][:, :, pi, si]
                    self.flag(tfslice.astype(np.float32), tfmask, 0, 0, None, None)

        return super().process(input)

    def flag(self, tfslice, tfmask, li, gi, tf, ts, **kwargs):
        """Function that does the actual flag."""

        # if all have been masked, no need to flag again
        if tfmask.all():
            return

        first_threshold = self.params['first_threshold']
        exp_factor = self.params['exp_factor']
        distribution = self.params['distribution']
        max_threshold_len = self.params['max_threshold_len']
        sensitivity = self.params['sensitivity']
        min_connected = self.params['min_connected']
        flag_direction = self.params['flag_direction']
        tk_size = self.params['tk_size']
        fk_size = self.params['fk_size']
        threshold_num = max(0, int(self.params['threshold_num']))

        tf_abs = np.abs(tfslice) # operate only on the amplitude

        # first round
        # first complete masked vals due to ns by interpolate
        itp = interpolate.Interpolate(tf_abs, tfmask)
        background = itp.fit()
        # Gaussian fileter
        gf = gaussian_filter.GaussianFilter(background, time_kernal_size=tk_size, freq_kernal_size=fk_size, filter_direction=flag_direction)
        background = gf.fit()
        # sum-threshold
        tf_diff = tf_abs - background
        # an initial run of N = 1 only to remove extremely high amplitude RFI
        st = sum_threshold.SumThreshold(tf_diff, tfmask, first_threshold, exp_factor, distribution, 1, min_connected)
        st.execute(sensitivity, flag_direction)

        # if all have been masked, no need to flag again
        if st.vis_mask.all():
            tfmask[:] = st.vis_mask
            return

        # next rounds
        for i in range(threshold_num):
            # Gaussian fileter
            gf = gaussian_filter.GaussianFilter(tf_diff, st.vis_mask, time_kernal_size=tk_size, freq_kernal_size=fk_size, filter_direction=flag_direction)
            background = gf.fit()
            # sum-threshold
            tf_diff = tf_diff - background
            st = sum_threshold.SumThreshold(tf_diff, st.vis_mask, first_threshold, exp_factor, distribution, max_threshold_len, min_connected)
            st.execute(sensitivity, flag_direction)

            # if all have been masked, no need to flag again
            if st.vis_mask.all():
                break

        # replace vis_mask with the flagged mask
        tfmask[:] = st.vis_mask

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
