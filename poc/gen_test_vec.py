import os

from vdaf_poc.test_utils import gen_test_vec_for_vdaf

from flp_pine import encode_float
from vdaf_pine import Pine64, Pine128

VERSION = int(open('VERSION').read())
TEST_VECTOR_PATH = os.environ.get('TEST_VECTOR_PATH',
                                  'test_vec/{:02}'.format(VERSION))

if __name__ == '__main__':
    num_frac_bits = 4
    l2_norm_bound = encode_float(1.0, 2**num_frac_bits)
    dimension = 20
    chunk_length = 150
    chunk_length_norm_equality = 4
    num_shares = 2

    measurements = []
    measurements.append([0.0] * dimension)
    measurements.append([0.0] * dimension)
    measurements[0][0] = 1.0
    measurements[1][1] = 1.0

    pine64 = Pine64(
        l2_norm_bound,
        num_frac_bits,
        dimension,
        chunk_length,
        chunk_length_norm_equality,
        num_shares,
    )
    gen_test_vec_for_vdaf(
        TEST_VECTOR_PATH,
        pine64,
        None,
        measurements,
        0,
    )

    pine128 = Pine128(
        l2_norm_bound,
        num_frac_bits,
        dimension,
        chunk_length,
        chunk_length_norm_equality,
        num_shares,
    )
    gen_test_vec_for_vdaf(
        TEST_VECTOR_PATH,
        pine128,
        None,
        measurements,
        0,
    )
