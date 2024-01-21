"""Unit tests for PINE VDAF. """

import os
import sys
import unittest

# Access poc folder in submoduled VDAF draft.
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "draft-irtf-cfrg-vdaf", "poc"))
from common import TEST_VECTOR, gen_rand
from field import Field64, Field128
from vdaf import test_vdaf
from vdaf_pine import Pine, Pine128, Pine64, VERSION
from vdaf_prio3 import (
    USAGE_MEAS_SHARE, USAGE_PROOF_SHARE, USAGE_JOINT_RANDOMNESS,
    USAGE_PROVE_RANDOMNESS, USAGE_QUERY_RANDOMNESS, USAGE_JOINT_RAND_SEED,
    USAGE_JOINT_RAND_PART
)


class TestDomainSeparationTag(unittest.TestCase):

    def test_usage_uniqueness(self):
        usages = [USAGE_MEAS_SHARE, USAGE_PROOF_SHARE, USAGE_JOINT_RANDOMNESS,
                  USAGE_PROVE_RANDOMNESS, USAGE_QUERY_RANDOMNESS,
                  USAGE_JOINT_RAND_SEED, USAGE_JOINT_RAND_PART]
        self.assertListEqual(usages, list(range(1, len(usages) + 1)))

    def test_length(self):
        # Check `Pine.domain_separation_tag` output length: 1 byte for draft
        # version, 4 bytes for algorithm ID, 2 bytes for usage string.
        self.assertEqual(len(Pine.domain_separation_tag(0)), 7)


class TestShard(unittest.TestCase):

    def test_result_share_length(self):
        """Check the result shares of `shard()` have the expected lengths. """
        pine = Pine.with_params(l2_norm_bound = 1.0,
                                num_frac_bits = 4,
                                dimension = 4,
                                chunk_length = 150,
                                num_shares = 2,
                                field = Field64,
                                num_proofs = 1)
        measurement = [0.0] * pine.Flp.Valid.dimension
        nonce = gen_rand(pine.NONCE_SIZE)
        rand = gen_rand(pine.RAND_SIZE)
        (public_share, input_shares) = pine.shard(measurement, nonce, rand)
        self.assertTrue(public_share is not None)
        self.assertTrue(input_shares is not None)
        self.assertEqual(len(input_shares), pine.SHARES)

        [wr_joint_rand_parts, vf_joint_rand_parts] = public_share
        self.assertEqual(len(wr_joint_rand_parts), pine.SHARES)
        self.assertEqual(len(vf_joint_rand_parts), pine.SHARES)
        self.assertTrue(all(len(part) == pine.Flp.Valid.Xof.SEED_SIZE
                        for part in wr_joint_rand_parts))
        self.assertTrue(all(len(part) == pine.Flp.Valid.Xof.SEED_SIZE
                        for part in vf_joint_rand_parts))

        # Check leader share length.
        (meas_share, proofs_share, wr_joint_rand_blind, vf_joint_rand_blind) = \
            input_shares[0]
        self.assertEqual(len(meas_share), pine.MEAS_LEN)
        self.assertEqual(len(proofs_share), pine.Flp.PROOF_LEN * pine.PROOFS)


class TestPineVdafEndToEnd(unittest.TestCase):

    def setUp(self):
        self.l2_norm_bound = 1.0
        self.dimension = 20
        self.num_frac_bits = 4
        self.chunk_length = 150
        self.num_shares = 2

    def run_pine_vdaf(self, pine):
        measurement_1 = [0.0] * pine.Flp.Valid.dimension
        measurement_1[0] = 1.0
        measurement_2 = [0.0] * pine.Flp.Valid.dimension
        measurement_2[1] = 1.0
        expected_agg_result = [x + y for (x, y) in zip(measurement_1, measurement_2)]
        test_vdaf(
            pine,
            None,
            [measurement_1, measurement_2],
            expected_agg_result,
            print_test_vec=TEST_VECTOR,
            test_vec_instance=pine.Flp.Valid.Field.__name__
        )

    def test_field64_three_proofs(self):
        pine = Pine64.with_params(l2_norm_bound = self.l2_norm_bound,
                                  dimension = self.dimension,
                                  num_frac_bits = self.num_frac_bits,
                                  chunk_length = self.chunk_length,
                                  num_shares = self.num_shares)
        self.assertEqual(pine.Flp.Field, Field64)
        self.run_pine_vdaf(pine)

    def test_field128_one_proof(self):
        pine = Pine128.with_params(l2_norm_bound = self.l2_norm_bound,
                                   dimension = self.dimension,
                                   num_frac_bits = self.num_frac_bits,
                                   chunk_length = self.chunk_length,
                                   num_shares = self.num_shares)
        self.assertEqual(pine.Flp.Field, Field128)
        self.run_pine_vdaf(pine)


if __name__ == '__main__':
    unittest.main()
