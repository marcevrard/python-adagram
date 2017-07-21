

import numpy as np
from scipy.special import expit  # pylint: disable=no-name-in-module


def update_z(inputs, outputs,
             z, x, context_ids,
             paths, codes):

    voc_length, max_n_senses, _dim = inputs.shape
    context_length = len(context_ids)

    for idx in range(context_length):
        y = context_ids[idx]
        sub_paths = paths[y * voc_length]
        sub_codes = codes[y * voc_length]

        for n in range(voc_length):
            if sub_codes[n] != -1:

                for k in range(max_n_senses):
                    f = inputs[x, k] @ outputs[sub_paths[n]]

                    z[k] += np.log(expit(f * (1 - 2*sub_codes[n])))  # pylint: disable=no-member

    z = np.exp(z - max(z)) / sum(z)
