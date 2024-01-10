"""Unit tests for The PINE FLP."""

import math
import os
import sys

# Access poc folder in submoduled VDAF draft.
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "draft-irtf-cfrg-vdaf", "poc"))

from common import gen_rand
from flp_pine import PineValid, bit_chunks
from field import Field64, Field128
from flp_generic import FlpGeneric, test_flp_generic

import unittest


class TestEncoding(unittest.TestCase):

    def test_roundtrip_f64(self):
        valid = PineValid.with_field(Field128)(1.0, 15, 2, 1)
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
            encoded = valid.encode_f64_into_field(t["input"])
            if t["input"] < 0:
                # Negative values are represented with the upper half of the
                # field bits.
                self.assertTrue(
                    encoded.as_unsigned() > math.floor(valid.Field.MODULUS/2))
            decoded = valid.decode_f64_from_field(encoded)
            self.assertEqual(decoded, t["expected_result"])

    def test_roundtrip_gradient(self):
        valid = PineValid.with_field(Field128)(1.0, 15, 2, 1)
        f64_vals = [0.5, 0.5]
        self.assertEqual(f64_vals,
                         valid.decode(valid.truncate(valid.encode(f64_vals)), 1))


class TestCircuit(unittest.TestCase):

    def test_bit_chunks(self):
        """
        Test that chunking the buffer with `num_chunk_bits` bits at a time and
        joining them back together outputs the original buffer.
        """
        buf = os.urandom(16)

        # First, display `buf` as a string of bits in `bits_str`.
        bits_str = "".join(map(lambda byte: "{0:08b}".format(byte), buf))
        for num_chunk_bits in [1, 2, 4, 8]:
            bit_chunks_str = "".join(map(
                # Format each chunk as bits string, and zero fill the most
                # significant bits.
                lambda bit_chunk: format(bit_chunk, "b").zfill(num_chunk_bits),
                bit_chunks(buf, num_chunk_bits)
            ))
            assert bits_str == bit_chunks_str

    def test_flp(self):
        """
        Run `test_flp_generic()` from the upstream VDAF repo for various
        parameters.
        """
        # `PineValid` with `l2_norm_bound = 1.0`, `num_frac_bits = 4`,
        # `dimension = 4`, `chunk_length = 150`.
        l2_norm_bound = 1.0
        dimension = 4
        args = [l2_norm_bound, 4, dimension, 150]
        # A gradient with a L2-norm of exactly 1.0.
        measurement = [l2_norm_bound / 2] * dimension
        for field in [Field64, Field128]:
            pine_valid = PineValid.with_field(field)(*args)
            flp = FlpGeneric(pine_valid)

            # Test PINE FLP with verification.
            xof = PineValid.Xof(gen_rand(16), b"", b"")
            encoded_gradient = flp.encode(measurement)
            (wr_check_bits, wr_dot_prods) = \
                pine_valid.run_wr_checks(encoded_gradient, xof)
            meas = encoded_gradient + wr_check_bits + wr_dot_prods
            test_flp_generic(flp, [(meas, True)])


if __name__ == '__main__':
    unittest.main()
