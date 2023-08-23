"""A generic FLP based on {{BBCGGI19}}, Theorem 4.3."""

import copy

import field
from common import ERR_ABORT, ERR_INPUT, Unsigned, Vec, next_power_of_2
from field import poly_eval, poly_interp, poly_mul, poly_strip
from flp import Flp, run_flp

class WraparoundAndNormBoundCheck(Valid):
    # Associated types
    Measurement = Unsigned
    AggResult = Unsigned
    Field = field.Field128

    # Associated parameters
    GADGETS = [Mul()]
    GADGET_CALLS = [1]
    MEAS_LEN = 1
    JOINT_RAND_LEN = 0
    OUTPUT_LEN = 1

    def eval(self, meas, joint_rand, _num_shares):
        # TODO Add the wraparound check, norm bound check, and bit checks here.
        self.check_valid_eval(meas, joint_rand)
        return self.GADGETS[0].eval(self.Field, [meas[0], meas[0]]) - meas[0]

    def encode(self, measurement):
        if measurement not in [0, 1]:
            raise ERR_INPUT
        return [self.Field(measurement)]

    def truncate(self, meas):
        if len(meas) != 1:
            raise ERR_INPUT
        return meas

    def decode(self, output, _num_measurements):
        return output[0].as_unsigned()
