

import numpy as np
from scipy.special import expit  # pylint: disable=no-name-in-module


def update_z(inputs, outputs,
             z, x, context_ids,
             paths, codes):

    _, max_n_senses, _ = inputs.shape
    _, path_length = paths.shape

    for y in context_ids:
        for n in range(path_length):
            if codes[y, n] != -1:

                for k in range(max_n_senses):
                    f = inputs[x, k] @ outputs[paths[y, n]]

                    z[k] += np.log(expit(f * (1 - 2*codes[y, n])))  # pylint: disable=no-member

    return np.exp(z - max(z)) / sum(z)


def inplace_update_z(vm, z, w, context):
    return update_z(vm.In, vm.Out,
                    z, w, context,
                    vm.path, vm.code)
