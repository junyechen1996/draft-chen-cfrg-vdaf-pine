import unittest

from vdaf_poc.field import NttField

from field import Field32, Field40


# This is copied from
# https://github.com/cfrg/draft-irtf-cfrg-vdaf/blob/main/poc/tests/test_field.py#L8
# TODO: consider to refactor the above code s.t. we can reuse them.
class TestFields(unittest.TestCase):
    def run_ntt_field_test(self, cls: type[NttField]) -> None:
        # Test constructing a field element from an integer.
        self.assertTrue(cls(1337) == cls(cls.gf(1337)))

        # Test generating a zero-vector.
        vec = cls.zeros(23)
        self.assertTrue(len(vec) == 23)
        for x in vec:
            self.assertTrue(x == cls(cls.gf.zero()))

        # Test generating a random vector.
        vec = cls.rand_vec(23)
        self.assertTrue(len(vec) == 23)

        # Test arithmetic.
        x = cls(cls.gf.random_element())
        y = cls(cls.gf.random_element())
        self.assertTrue(x + y == cls(x.val + y.val))
        self.assertTrue(x - y == cls(x.val - y.val))
        self.assertTrue(-x == cls(-x.val))
        self.assertTrue(x * y == cls(x.val * y.val))
        self.assertTrue(x.inv() == cls(x.val**-1))

        # Test serialization.
        want = cls.rand_vec(10)
        got = cls.decode_vec(cls.encode_vec(want))
        self.assertTrue(got == want)

        # Test encoding integer as bit vector.
        vals = [i for i in range(15)]
        bits = 4
        for val in vals:
            encoded = cls.encode_into_bit_vector(val, bits)
            self.assertTrue(cls.decode_from_bit_vector(
                encoded).as_unsigned() == val)

        # Test generator.
        self.assertTrue(cls.gen() ** cls.GEN_ORDER == cls(1))
        self.assertTrue(cls.gen() ** int(cls.GEN_ORDER / 2) == cls(-1))

    def test_field32(self) -> None:
        self.run_ntt_field_test(Field32)

    def test_field40(self) -> None:
        self.run_ntt_field_test(Field40)
