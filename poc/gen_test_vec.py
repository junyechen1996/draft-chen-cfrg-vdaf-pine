import os

from vdaf_poc.field import Field64
from vdaf_poc.test_utils import gen_test_vec_for_vdaf

from field32 import Field32
from flp_pine import NUM_WR_CHECKS, NUM_WR_SUCCESSES, encode_float
from vdaf_pine import (Pine, Pine32HmacSha256Aes128, Pine64,
                       Pine64HmacSha256Aes128, Pine128)
from xof_hmac_sha256_aes128 import XofHmacSha256Aes128

VERSION = int(open('VERSION').read())
TEST_VECTOR_PATH = os.environ.get('TEST_VECTOR_PATH',
                                  'test_vec/{:02}'.format(VERSION))

if __name__ == '__main__':
    num_shares = 2
    chunk_length = 150
    chunk_length_norm_equality = 4

    for (i, (num_frac_bits,
             dimension,
             num_wr_checks,
             num_wr_successes)) in enumerate([
                 (7, 20, NUM_WR_CHECKS, NUM_WR_SUCCESSES),
                 (4, 250, NUM_WR_CHECKS, NUM_WR_SUCCESSES),
                 (10, 19, 75, 75),
                 (10, 19, 75, 50)
             ]):
        l2_norm_bound = encode_float(1.0, num_frac_bits)
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
            num_wr_checks=num_wr_checks,
            num_wr_successes=num_wr_successes,
        )
        gen_test_vec_for_vdaf(
            TEST_VECTOR_PATH,
            pine64,
            None,
            measurements,
            i,
        )

        pine128 = Pine128(
            l2_norm_bound,
            num_frac_bits,
            dimension,
            chunk_length,
            chunk_length_norm_equality,
            num_shares,
            num_wr_checks=num_wr_checks,
            num_wr_successes=num_wr_successes,
        )
        gen_test_vec_for_vdaf(
            TEST_VECTOR_PATH,
            pine128,
            None,
            measurements,
            i,
        )

        pine32_hmac_sha256_aes128 = Pine32HmacSha256Aes128(
            l2_norm_bound,
            num_frac_bits,
            dimension,
            chunk_length,
            chunk_length_norm_equality,
            num_shares,
            num_wr_checks=num_wr_checks,
            num_wr_successes=num_wr_successes,
        )
        gen_test_vec_for_vdaf(
            TEST_VECTOR_PATH,
            pine32_hmac_sha256_aes128,
            None,
            measurements,
            i,
        )

        pine64_hmac_sha256_aes128 = Pine64HmacSha256Aes128(
            l2_norm_bound,
            num_frac_bits,
            dimension,
            chunk_length,
            chunk_length_norm_equality,
            num_shares,
            num_wr_checks=num_wr_checks,
            num_wr_successes=num_wr_successes,
        )
        gen_test_vec_for_vdaf(
            TEST_VECTOR_PATH,
            pine64_hmac_sha256_aes128,
            None,
            measurements,
            i,
        )

        # Custom type with Field32 and XofHmacSha256Aes128.
        pine32_custom = Pine(
            Field32,
            XofHmacSha256Aes128,
            l2_norm_bound,
            num_frac_bits,
            dimension,
            chunk_length,
            chunk_length_norm_equality,
            num_shares,
            num_proofs=4,
            num_proofs_norm_equality=2,
            num_wr_checks=num_wr_checks,
            num_wr_successes=num_wr_successes,
            vdaf_id=Pine32HmacSha256Aes128.ID,
        )
        gen_test_vec_for_vdaf(
            TEST_VECTOR_PATH,
            pine32_custom,
            None,
            measurements,
            2 * i,
        )

        # Custom type with Field64 and XofHmacSha256Aes128.
        pine64_custom = Pine(
            Field64,
            XofHmacSha256Aes128,
            l2_norm_bound,
            num_frac_bits,
            dimension,
            chunk_length,
            chunk_length_norm_equality,
            num_shares,
            num_proofs=1,
            num_proofs_norm_equality=1,
            num_wr_checks=num_wr_checks,
            num_wr_successes=num_wr_successes,
            vdaf_id=Pine64HmacSha256Aes128.ID,
        )
        gen_test_vec_for_vdaf(
            TEST_VECTOR_PATH,
            pine64_custom,
            None,
            measurements,
            2 * i + 1,
        )
