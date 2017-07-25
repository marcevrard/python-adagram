import numpy as np
from scipy.special import expit  # pylint: disable=no-name-in-module

# pylint: disable=no-member

def update_z(inputs, outputs,
             z_log, x, context_ids,
             paths, codes):

    codes_ma = np.ma.masked_array(codes.copy(), [codes == -1])
    codes_ma[codes_ma == 1] = -1  # Right child of previous node
    codes_ma[codes_ma == 0] = 1   # Left ..

    for y in context_ids:
        for code, path_w in zip(codes_ma[y], paths[y]):
            f = inputs[x] @ outputs[path_w]     # Sim btw in word senses and each HS node
            z_log += np.log(expit(f * code))
            # Build softmax (log) numerator with +/- f (l or r nodes)
    z = np.exp(z_log - max(z_log))  # Normalize and take exp to get back `z` as prob ([0:1])

    return z / sum(z)               # Regularize the probability to get the Softmax


def py_update_z(vm, z_log, w, context):
    return update_z(vm.In, vm.Out,
                    z_log, w, context,
                    vm.path, vm.code)
