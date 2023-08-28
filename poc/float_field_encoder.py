"""Utility methods for encoding/decoding float into/from field. """
import math
import os
import sys

# Access poc folder in submoduled VDAF draft.
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "draft-irtf-cfrg-vdaf", "poc"))
from field import Field
from common import ERR_INPUT, Unsigned


def encode_f64_into_field(Field,
                          x: float,
                          num_frac_bits: Unsigned) -> Field:
    if (math.isnan(x) or not math.isfinite(x) or
        (x != 0.0 and abs(x) < sys.float_info.min)):
        # Reject NAN, infinity, and subnormal floats,
        # per {{fp-encoding}}.
        raise ERR_INPUT
    x_encoded = math.floor(x * (2 ** num_frac_bits))
    if x >= 0:
        return Field(x_encoded)
    return Field(Field.MODULUS + x_encoded)

def decode_f64_from_field(field_elem: Field,
                          num_measurements: Unsigned,
                          num_frac_bits: Unsigned) -> float:
    # The first half of the field size is reserved for
    # positive values.
    positive_upper_bound = math.floor(field_elem.MODULUS / 2)
    decoded = field_elem.as_unsigned()
    if decoded > positive_upper_bound:
        # We need to take the difference between the result
        # and the field modulus, and return the result as negative.
        decoded = -(field_elem.MODULUS - decoded)
    # Divide by 2^num_frac_bits and we will get a float back.
    decoded_float = decoded / (2 ** num_frac_bits)
    return decoded_float
