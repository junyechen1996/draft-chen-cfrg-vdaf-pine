"""Validity circuit for PINE. """

import math
import os
import sys

# Access poc folder in submoduled VDAF draft.
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "draft-irtf-cfrg-vdaf", "poc"))
import field
from common import ERR_INPUT, Unsigned, Vec
from flp_generic import FlpGeneric, Valid, test_flp_generic
from float_field_encoder import encode_f64_into_field, decode_f64_from_field


class PineValid(Valid):
    # Operational parameters set by user.
    l2_norm_bound: float = None # Set by constructor
    num_frac_bits: Unsigned = None # Set by constructor
    dimension: Unsigned = None # Set by constructor

    # Associated types
    Measurement = Vec[float]
    AggResult = Vec[float]
    Field = field.Field128

    # Default internal operational parameters, may be changed by constructor.
    # Figure out how to fix them safely, so it doesn't negatively impact
    # soundness and completeness error. (#24)
    alpha: float = 7
    num_wr_reps: Unsigned = 135
    tau: float = 0.75
    # Other internal operational parameters, set by constructor.
    encoded_l2_norm_bound: Field = None
    encoded_sq_l2_norm_bound: Field = None

    # Associated parameters
    MEAS_LEN = None # Set by constructor
    JOINT_RAND_LEN = None # Set by constructor
    OUTPUT_LEN = None # Set by constructor
    GADGETS = [] # TODO(junyec): update
    GADGET_CALLS = None # Set by constructor

    def __init__(self,
                 l2_norm_bound: float,
                 num_frac_bits: Unsigned,
                 dimension: Unsigned):
        """
        Instantiate the `PineValid` circuit for measurements with `dimension`
        elements. Each element will be kept with `num_frac_bits` binary
        fractional bits, and the L2-norm bound of each measurement is bounded
        by `l2_norm_bound`.
        """
        if l2_norm_bound <= 0.0:
            raise ValueError("Invalid L2-norm bound, it must be positive")
        if num_frac_bits <= 0 or num_frac_bits >= 128:
            raise ValueError(
                "Invalid number of fractional bits, it must be in [0, 128)"
            )
        if dimension <= 0:
            raise ValueError("Invalid dimension, it must be positive")

        self.l2_norm_bound = l2_norm_bound
        self.num_frac_bits = num_frac_bits
        self.dimension = dimension
        self.encoded_l2_norm_bound = encode_f64_into_field(
            self.Field, l2_norm_bound, num_frac_bits
        )
        if (self.Field.MODULUS / self.encoded_l2_norm_bound.as_unsigned()
            < self.encoded_l2_norm_bound.as_unsigned()):
            # Squaring encoded norm bound overflows field size, reject.
            raise ValueError("User-specified norm bound and number of "
                             "fractional bits are too big.")
        self.encoded_sq_l2_norm_bound = \
            self.encoded_l2_norm_bound * self.encoded_l2_norm_bound

        # Set FLP parameters.
        self.MEAS_LEN = dimension # TODO(junyec): update
        self.JOINT_RAND_LEN = 0 # TODO(junyec): update
        self.OUTPUT_LEN = dimension
        self.GADGET_CALLS = [] # TODO(junyec): update

    def eval(self,
             meas: Vec[Field],
             joint_rand: Vec[Field],
             num_shares: Unsigned) -> Field:
        self.check_valid_eval(meas, joint_rand)
        # TODO(junyec): implement actual validity circuits.
        return self.Field(0)

    def encode(self, measurement: Measurement) -> Vec[Field]:
        if len(measurement) != self.dimension:
            raise ERR_INPUT
        return [encode_f64_into_field(self.Field, x, self.num_frac_bits)
                for x in measurement]

    def truncate(self, meas: Vec[Field]):
        return meas[:self.dimension]

    def decode(self,
               output: Vec[Field],
               num_measurements: Unsigned) -> AggResult:
        return [
            decode_f64_from_field(x, num_measurements, self.num_frac_bits)
            for x in output
        ]

    def test_vec_set_type_param(self, test_vec):
        test_vec["l2_norm_bound"] = self.l2_norm_bound
        test_vec["num_frac_bits"] = self.num_frac_bits
        test_vec["dimension"] = self.dimension
        return ["l2_norm_bound", "num_frac_bits", "dimension"]


def test():
    l2_norm_bound = 1.0
    num_frac_bits = 15
    dimension = 4
    flp = FlpGeneric(PineValid(l2_norm_bound, num_frac_bits, dimension))
    test_flp_generic(flp, [
        (flp.encode([l2_norm_bound, 0.0, 0.0, 0.0]), True),
        (flp.encode([l2_norm_bound / 2] * dimension), True),
    ])


if __name__ == '__main__':
    test()
