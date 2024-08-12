"""Unit tests for PINE VDAF. """

import unittest

from vdaf_poc.common import gen_rand
from vdaf_poc.field import Field64, Field128
from vdaf_poc.test_utils import TestVdaf
from vdaf_poc.vdaf_prio3 import (USAGE_JOINT_RAND_PART, USAGE_JOINT_RAND_SEED,
                                 USAGE_JOINT_RANDOMNESS, USAGE_MEAS_SHARE,
                                 USAGE_PROOF_SHARE, USAGE_PROVE_RANDOMNESS,
                                 USAGE_QUERY_RANDOMNESS)

from flp_pine import encode_float
from vdaf_pine import Pine64, Pine128


class TestDomainSeparationTag(unittest.TestCase):

    def test_usage_uniqueness(self):
        usages = [USAGE_MEAS_SHARE, USAGE_PROOF_SHARE, USAGE_JOINT_RANDOMNESS,
                  USAGE_PROVE_RANDOMNESS, USAGE_QUERY_RANDOMNESS,
                  USAGE_JOINT_RAND_SEED, USAGE_JOINT_RAND_PART]
        self.assertListEqual(usages, list(range(1, len(usages) + 1)))

    def test_length(self):
        """
        Check `Pine.domain_separation_tag` output length: 1 byte for draft
        version, 4 bytes for algorithm ID, 2 bytes for usage string.
        """
        pine = Pine64(l2_norm_bound=2**4, dimension=100, num_frac_bits=4,
                      chunk_length=30, chunk_length_norm_equality=10, num_shares=2)
        self.assertEqual(len(pine.domain_separation_tag(0)), 7)


class TestShard(unittest.TestCase):

    def test_result_share_length(self):
        """Check the result shares of `shard()` have the expected lengths. """
        pine = Pine64(l2_norm_bound=encode_float(1.0, 4),
                      num_frac_bits=4,
                      dimension=4,
                      chunk_length=150,
                      chunk_length_norm_equality=2,
                      num_shares=2)
        measurement = [0.0] * pine.valid.dimension
        nonce = gen_rand(pine.NONCE_SIZE)
        rand = gen_rand(pine.RAND_SIZE)
        (public_share, input_shares) = pine.shard(measurement, nonce, rand)
        self.assertTrue(public_share is not None)
        self.assertTrue(input_shares is not None)
        self.assertEqual(len(input_shares), pine.SHARES)

        [wr_joint_rand_parts, vf_joint_rand_parts] = public_share
        self.assertEqual(len(wr_joint_rand_parts), pine.SHARES)
        self.assertEqual(len(vf_joint_rand_parts), pine.SHARES)
        self.assertTrue(all(len(part) == pine.xof.SEED_SIZE
                        for part in wr_joint_rand_parts))
        self.assertTrue(all(len(part) == pine.xof.SEED_SIZE
                        for part in vf_joint_rand_parts))

        # Check leader share length.
        (meas_share, proofs_share, wr_joint_rand_blind, vf_joint_rand_blind) = \
            input_shares[0]
        self.assertEqual(len(meas_share), pine.MEAS_LEN)
        self.assertEqual(
            len(proofs_share),
            pine.flp_norm_equality.PROOF_LEN + pine.flp.PROOF_LEN * pine.PROOFS
        )


class TestPineVdafEndToEnd(TestVdaf):

    def setUp(self):
        self.num_frac_bits = 4
        self.l2_norm_bound = encode_float(1.0, 2**self.num_frac_bits)
        self.dimension = 20
        self.chunk_length_norm_equality = 4
        self.chunk_length = 150
        self.num_shares = 2

        self.measurements = []
        self.measurements.append([0.0] * self.dimension)
        self.measurements.append([0.0] * self.dimension)
        self.measurements[0][0] = 1.0
        self.measurements[1][1] = 1.0
        self.expected_agg_result = [
            x + y for (x, y) in zip(self.measurements[0], self.measurements[1])
        ]

    def test_64(self):
        pine = Pine64(self.l2_norm_bound,
                      self.num_frac_bits,
                      self.dimension,
                      self.chunk_length,
                      self.chunk_length_norm_equality,
                      self.num_shares)
        self.run_vdaf_test(pine,
                           None,
                           self.measurements,
                           self.expected_agg_result)

    def test_128(self):
        pine = Pine128(self.l2_norm_bound,
                       self.num_frac_bits,
                       self.dimension,
                       self.chunk_length,
                       self.chunk_length_norm_equality,
                       self.num_shares)
        self.run_vdaf_test(pine,
                           None,
                           self.measurements,
                           self.expected_agg_result)

    def test_fields(self):
        for PineType, FieldType in [(Pine64, Field64), (Pine128, Field128)]:
            pine = PineType(l2_norm_bound=self.l2_norm_bound,
                            dimension=self.dimension,
                            num_frac_bits=self.num_frac_bits,
                            chunk_length=self.chunk_length,
                            chunk_length_norm_equality=self.chunk_length_norm_equality,
                            num_shares=self.num_shares)
            self.assertEqual(pine.flp_norm_equality.field, FieldType)
            self.assertEqual(pine.flp.field, FieldType)


if __name__ == '__main__':
    unittest.main()
