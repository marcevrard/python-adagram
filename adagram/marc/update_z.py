import numpy as np
from scipy.special import expit  # pylint: disable=no-name-in-module


def update_z(inputs, outputs,
             z, x, context_ids,
             paths, codes):

    for y in context_ids:
        for code, path_w in zip(codes[y], paths[y]):
            if code != -1:
                f = inputs[x] @ outputs[path_w]
                z += np.log(expit(f * (1 - 2 * code)))  # pylint: disable=no-member

    z_exp = np.exp(z - max(z))

    return z_exp / sum(z_exp)


def py_update_z(vm, z, w, context):
    return update_z(vm.In, vm.Out,
                    z, w, context,
                    vm.path, vm.code)
