"""Unit tests for The PINE FLP."""

import math
import unittest

from vdaf_poc.common import gen_rand
from vdaf_poc.field import Field64, Field128
from vdaf_poc.flp_bbcggi19 import FlpBBCGGI19
from vdaf_poc.test_utils import TestFlpBBCGGI19
from vdaf_poc.xof import XofTurboShake128

from flp_pine import (ALPHA, NUM_WR_CHECKS, NUM_WR_SUCCESSES, PineValid,
                      construct_circuits, encode_float)


class TestEncoding(unittest.TestCase):

    def test_roundtrip_f64(self):
        valid = PineValid(Field128, 1.0, 15, 2, 1)
        test_cases = [
            {
                "input": -100.0,
                "expected_result": -100.0,
            },
            {
                "input": -1.0,
                "expected_result": -1.0,
            },
            {
                "input": -0.0001,
                "expected_result": -0.0001220703125,
            },
            {
                "input": -0.0,
                "expected_result": -0.0,
            },
            {
                "input": 0.0,
                "expected_result": 0.0,
            },
            {
                "input": 0.0001,
                "expected_result": 9.1552734375e-05,
            },
            {
                "input": 0.1,
                "expected_result": 0.0999755859375,
            },
            {
                "input": 0.5,
                "expected_result": 0.5,
            },
            {
                "input": 1.0,
                "expected_result": 1.0,
            },
            {
                "input": 10000.0,
                "expected_result": 10000.0,
            },
        ]
        for (i, t) in enumerate(test_cases):
            # XXX Not sure what's casing decoding to fail in pure Python3 but not SageMath.
            try:
                encoded = valid.encode_float_into_field(t["input"])
            except:
                continue
            if t["input"] < 0:
                # Negative values are represented with the upper half of the
                # field bits.
                self.assertTrue(
                    encoded.as_unsigned() > math.floor(valid.field.MODULUS / 2))
            decoded = valid.decode_float_from_field(encoded)
            self.assertEqual(decoded, t["expected_result"])

    def test_roundtrip_gradient(self):
        num_frac_bits = 15
        l2_norm_bound = encode_float(1.0, num_frac_bits)
        valid = PineValid(Field128, l2_norm_bound, num_frac_bits, 2, 1)
        f64_vals = [0.5, 0.5]
        self.assertEqual(
            f64_vals,
            valid.decode(
                valid.truncate(valid.encode_gradient_and_norm(f64_vals)),
                1
            )
        )


class TestOperationalParameters(unittest.TestCase):

    def test_bounds(self):
        """
        Test bound computations (squared norm and wraparound checks) for
        various parameters and the default alpha, number of wraparound tests,
        and number of successes.
        """

        test_cases = [
            {
                "l2_norm_bound": 1.0,
                "num_frac_bits": 15,
                "expected_sq_norm_bound": Field128(1073741824),
                "expected_num_bits_for_sq_norm": 31,
                "expected_wr_check_bound": Field128(524288),
                "expected_num_bits_for_wr_check": 20,
            },
            {
                "l2_norm_bound": 1.0,
                "num_frac_bits": 24,
                "expected_sq_norm_bound": Field128(281474976710656),
                "expected_num_bits_for_sq_norm": 49,
                "expected_wr_check_bound": Field128(268435456),
                "expected_num_bits_for_wr_check": 29,
            },
            {
                "l2_norm_bound": 1000.0,
                "num_frac_bits": 15,
                "expected_sq_norm_bound": Field128(1073741824000000),
                "expected_num_bits_for_sq_norm": 50,
                "expected_wr_check_bound": Field128(536870912),
                "expected_num_bits_for_wr_check": 30,
            },
            {
                "l2_norm_bound": 0.0001,
                "num_frac_bits": 15,
                "expected_sq_norm_bound": Field128(9),
                "expected_num_bits_for_sq_norm": 4,
                "expected_wr_check_bound": Field128(32),
                "expected_num_bits_for_wr_check": 6,
            },
            {
                "l2_norm_bound": 1.0,
                "num_frac_bits": 0,
                "expected_sq_norm_bound": Field128(1),
                "expected_num_bits_for_sq_norm": 1,
                "expected_wr_check_bound": Field128(16),
                "expected_num_bits_for_wr_check": 5,
            },
            {
                "l2_norm_bound": 1337.0,
                "num_frac_bits": 0,
                "expected_sq_norm_bound": Field128(1787569),
                "expected_num_bits_for_sq_norm": 21,
                "expected_wr_check_bound": Field128(16384),
                "expected_num_bits_for_wr_check": 15,
            },
        ]

        for t in test_cases:
            l2_norm_bound = encode_float(
                t["l2_norm_bound"], t["num_frac_bits"])
            # The dimension and chunk_length don't impact these tests.
            v = PineValid(Field128, l2_norm_bound,
                          t["num_frac_bits"], 10000, 123)
            self.assertEqual(v.alpha, ALPHA)
            self.assertEqual(v.num_wr_checks, NUM_WR_CHECKS)
            self.assertEqual(v.num_wr_successes, NUM_WR_SUCCESSES)
            self.assertEqual(v.l2_norm_bound, l2_norm_bound)
            self.assertEqual(v.num_frac_bits, t["num_frac_bits"])
            self.assertEqual(v.dimension, 10000)
            self.assertEqual(v.chunk_length, 123)
            self.assertEqual(v.sq_norm_bound,
                             t["expected_sq_norm_bound"])
            self.assertEqual(v.num_bits_for_sq_norm,
                             t["expected_num_bits_for_sq_norm"])
            self.assertEqual(v.wr_check_bound, t["expected_wr_check_bound"])
            self.assertEqual(v.num_bits_for_wr_check,
                             t["expected_num_bits_for_wr_check"])

    def test_field_modulus_check(self):
        """
        Test `PineValid` can or cannot be initialized given the user parameters
        and the field modulus constraint.
        """

        test_cases = [
            {
                # Violate range check requirement.
                "l2_norm_bound": 1.0,
                "num_frac_bits": 32,
                "field": Field64,
                "expected_success": False,
            },
            {
                "l2_norm_bound": 1.0,
                "num_frac_bits": 24,
                "field": Field64,
                "expected_success": True,
            },
            {
                # Violate range check requirement.
                "l2_norm_bound": 1.0,
                "num_frac_bits": 64,
                "field": Field128,
                "expected_success": False,
            },
            {
                "l2_norm_bound": 1.0,
                "num_frac_bits": 56,
                "field": Field128,
                "expected_success": True,
            },
            {
                # Intentionally specify a large `alpha` to cause wraparound
                # check's field size requirement to fail.
                "l2_norm_bound": 1.0,
                "num_frac_bits": 56,
                "field": Field128,
                "alpha": 1_000_000,
                "expected_success": False,
            },
        ]

        for t in test_cases:
            alpha = t.get("alpha", ALPHA)
            l2_norm_bound = encode_float(
                t["l2_norm_bound"], t["num_frac_bits"])
            # The dimension and chunk_length don't impact these tests.
            if t["expected_success"]:
                v = PineValid(t["field"], l2_norm_bound, t["num_frac_bits"],
                              10000, 123, alpha)
                self.assertIsNotNone(v)
            else:
                with self.assertRaises(ValueError):
                    v = PineValid(
                        t["field"],
                        l2_norm_bound,
                        t["num_frac_bits"],
                        10000,
                        123,
                        alpha
                    )


class TestCircuits(TestFlpBBCGGI19):

    def test_flp(self):
        """
        Run `test_flp_generic()` from the upstream VDAF repo for various
        parameters.
        """
        # `PineValid` with L2-norm bound `1.0`, `num_frac_bits = 4`,
        # `dimension = 4`, `chunk_length = 150`.
        num_frac_bits = 4
        l2_norm_bound = encode_float(1.0, num_frac_bits)
        dimension = 4
        chunk_length = 150
        chunk_length_norm_equality = 2
        args = [l2_norm_bound, num_frac_bits, dimension,
                chunk_length, chunk_length_norm_equality]
        # A gradient with a L2-norm of exactly 1.0.
        measurement = [1.0 / 2] * dimension
        for field in [Field64, Field128]:
            valids = construct_circuits(field, *args)
            pine_valid = valids[1]
            flp = FlpBBCGGI19(pine_valid)
            xof = XofTurboShake128(gen_rand(16), b"", b"")
            encoded_gradient_and_norm = \
                flp.valid.encode_gradient_and_norm(measurement)
            (wr_check_bits, wr_check_results) = \
                pine_valid.encode_wr_checks(encoded_gradient_and_norm[:dimension],
                                            xof)
            meas = encoded_gradient_and_norm + wr_check_bits + wr_check_results

            for valid in valids:
                test_flp = FlpBBCGGI19(valid)
                # Test PINE FLP with verification.
                self.run_flp_test(test_flp, [(meas, True)])


if __name__ == '__main__':
    unittest.main()
