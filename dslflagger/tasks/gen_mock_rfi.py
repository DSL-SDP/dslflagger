"""Generate mock RFI.

Inheritance diagram
-------------------

.. inheritance-diagram:: GenMockRFI
   :parts: 2

"""

import numpy as np
from caput import memh5, mpiutil
from dslpipe.pipeline.pipeline import OneAndOne
from dslpipe.utils.path_util import input_path, output_path
from dslpipe.utils.logging import get_logger
from dslflagger.sim.rfi import Impulse, Scatter

logger = get_logger(__name__)


class GenMockRFI(OneAndOne):
    """Generate mock RFI.

    Reads a mock high-frequency spectrometer TOD from disk, synthesises
    impulse and/or scatter RFI using :class:`dslflagger.sim.rfi.Impulse`
    and :class:`dslflagger.sim.rfi.Scatter`, and writes the corrupted TOD
    back to disk.
    """

    params_init = {
                    'input_files': 'mock_high_freq_spec_tod.hdf5',
                    'output_files': 'mock_high_freq_spec_tod_with_rfi.hdf5',
                    # Switch each RFI class on or off.
                    'rfi_impulse': True,
                    'rfi_scatter': True,
                    # Impulse parameters (short bursts, broad frequency).
                    'rfi_impulse_chance': 0.01,
                    'rfi_impulse_strength': 20000.0,
                    # Scatter parameters (sparse time/frequency bins).
                    'rfi_scatter_chance': 0.01,
                    'rfi_scatter_strength': 10000.0,
                    'rfi_scatter_std': 1000.0,
                    # RNG seed for reproducible mock RFI. None -> nondeterministic.
                    'rng_seed': None,
                  }

    prefix = 'gmr_'

    def process(self, input):

        # TODO: Implement MPI parallel in the future
        if mpiutil.rank0:
            logger.info('Generate mock RFI and add it to the input TOD')

            lsts = np.asarray(input['time'])
            freqs = np.asarray(input['freq'])

            n_pol = input['tod'].shape[2]
            n_sat = input['tod'].shape[3]

            # Spawn one child RNG per satellite so each satellite sees an
            # independent realisation of the RFI. ``spawn`` is deterministic
            # in ``rng_seed`` and falls back to nondeterminism when it's None.
            parent_rng = np.random.default_rng(self.params['rng_seed'])
            sat_rngs = parent_rng.spawn(n_sat) if n_sat > 0 else []

            # ``input['sat_no']`` carries the satellite *labels* (e.g. 9),
            # not positional indices, so iterate over the leading axis of
            # the TOD directly. A 4D ``rfi`` is recorded alongside the TOD
            # for comparision
            rfi = np.zeros_like(input['tod'])
            for pi in range(n_pol):
                for si in range(n_sat):
                    rfi1 = np.zeros((lsts.size, freqs.size), dtype=np.float64)

                    if self.params['rfi_impulse']:
                        impulse = Impulse(
                            impulse_chance=self.params['rfi_impulse_chance'],
                            impulse_strength=self.params['rfi_impulse_strength'],
                            rng=sat_rngs[si],
                        )
                        rfi1 += impulse(lsts, freqs).real

                    if self.params['rfi_scatter']:
                        scatter = Scatter(
                            scatter_chance=self.params['rfi_scatter_chance'],
                            scatter_strength=self.params['rfi_scatter_strength'],
                            scatter_std=self.params['rfi_scatter_std'],
                            rng=sat_rngs[si],
                        )
                        rfi1 += scatter(lsts, freqs).real

                    input['tod'][:, :, pi, si] += rfi1
                    rfi[:, :, pi, si] = rfi1

            # 3D rfi with the same axes as ``tod``.
            input.create_dataset('rfi', data=rfi)

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
