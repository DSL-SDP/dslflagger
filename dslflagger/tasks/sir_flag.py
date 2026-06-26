"""RFI flagging by applying the SIR (Scale-Invariant Rank) operator.

Inheritance diagram
-------------------

.. inheritance-diagram:: Sir
   :parts: 2

"""

import numpy as np
from caput import memh5, mpiutil
from dslpipe.pipeline.pipeline import OneAndOne
from dslpipe.utils.path_util import input_path, output_path
from dslpipe.utils.logging import get_logger
from dslflagger.rfi import sir_operator

logger = get_logger(__name__)


class Flag(OneAndOne):
    """RFI flagging by applying the SIR (Scale-Invariant Rank) operator.

    The scale-invariant rank (SIR) operator is a one-dimensional mathematical
    morphology technique that can be used to find adjacent intervals in the
    time or frequency domain that are likely to be affected by RFI.

    """

    params_init = {
                    'eta': 0.2,
                  }

    prefix = 'sir_'

    def process(self, input):

        # TODO: Implement MPI parallel in the future
        if mpiutil.rank0:
            logger.info('RFI flagging by applying the SIR (Scale-Invariant Rank) operator.')

            if 'mask' not in input:
                input.create_dataset('mask', data=np.zeros_like(input['tod'], dtype=bool))

            n_pol = input['tod'].shape[2]
            n_sat = input['tod'].shape[3]

            for pi in range(n_pol):
                for si in range(n_sat):
                    tfslice = input['tod'][:, :, pi, si]
                    tfmask = input['mask'][:, :, pi, si]
                    self.operate(tfslice.astype(np.float32), tfmask, 0, 0, None, None)

        return super().process(input)

    def operate(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
        """Function that does the actual operation."""

        eta = self.params['eta']

        if vis_mask.ndim == 2:
            mask = vis_mask.copy()
            mask = sir_operator.vertical_sir(mask, eta)
            vis_mask[:] = sir_operator.horizontal_sir(mask, eta)
        elif vis_mask.ndim == 3:
            # This shold be done after the combination of all pols
            mask = vis_mask[:, :, 0].copy()
            mask = sir_operator.vertical_sir(mask, eta)
            vis_mask[:] = sir_operator.horizontal_sir(mask, eta)[:, :, np.newaxis]
        else:
            raise RuntimeError('Invalid shape of vis_mask: %s' % vis_mask.shape)

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
