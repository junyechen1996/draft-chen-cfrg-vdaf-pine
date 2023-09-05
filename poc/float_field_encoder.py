"""Utility methods for encoding/decoding float into/from field. """
import math
import os
import sys

# Access poc folder in submoduled VDAF draft.
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "draft-irtf-cfrg-vdaf", "poc"))
from common import Unsigned
from field import Field, Field64


def encode_f64_into_field(Field,
                          x: float,
                          num_frac_bits: Unsigned) -> Field:
    if (math.isnan(x) or not math.isfinite(x) or
        (x != 0.0 and abs(x) < sys.float_info.min)):
        # Reject NAN, infinity, and subnormal floats,
        # per {{fp-encoding}}.
        raise ValueError("f64 encoding doesn't accept NAN, "
                         "infinite, or subnormal floats.")
    x_encoded = math.floor(x * (2 ** num_frac_bits))
    if x >= 0:
        return Field(x_encoded)
    return Field(Field.MODULUS + x_encoded)

def decode_f64_from_field(field_elem: Field,
                          num_frac_bits: Unsigned) -> float:
    decoded = field_elem.as_unsigned()
    # If the aggregated field is larger than half of the field
    # size, the decoded result should be negative.
    if decoded > math.floor(field_elem.MODULUS / 2):
        # We need to take the difference between the result
        # and the field modulus, and return the result as negative.
        decoded = -(field_elem.MODULUS - decoded)
    # Divide by 2^num_frac_bits and we will get a float back.
    decoded_float = decoded / (2 ** num_frac_bits)
    return decoded_float


def test():
    field_cls = Field64
    num_frac_bits = 15

    f64_vals = [1.5, -1.5]
    for f64_val in f64_vals:
        encoded = encode_f64_into_field(field_cls, f64_val, num_frac_bits)
        decoded = decode_f64_from_field(encoded, num_frac_bits)
        assert(decoded == f64_val)


if __name__ == '__main__':
    test()
