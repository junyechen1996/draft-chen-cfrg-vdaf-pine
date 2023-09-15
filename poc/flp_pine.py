"""Validity circuit for PINE. """

import math
import os
import sys

# Access poc folder in submoduled VDAF draft.
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "draft-irtf-cfrg-vdaf", "poc"))
from common import Unsigned, front, next_power_of_2
from field import Field
from flp_generic import Mul, ParallelSum, Valid
from xof import Xof, XofTurboShake128

# TODO(junyechen1996): Pick values for these that provide sufficient soundness
# and completeness (#39).
ALPHA = 7.0
NUM_WR_CHECKS = 135
NUM_WR_SUCCESSES = math.floor(0.75 * NUM_WR_CHECKS)

class PineValid(Valid):
    # XOF for `PineValid`.
    Xof = XofTurboShake128

    # Associated types for `Valid`.
    Measurement = list[float]
    AggResult = list[float]
    Field = None  # Set by `with_field()`.

    @classmethod
    def with_field(PineValid, TheField):
        class PineValidWithField(PineValid):
            Field = TheField
        return PineValidWithField

    def __init__(self,
                 l2_norm_bound: float,
                 num_frac_bits: Unsigned,
                 dimension: Unsigned,
                 chunk_length: Unsigned,
                 alpha: float = ALPHA,
                 num_wr_checks: Unsigned = NUM_WR_CHECKS,
                 num_wr_successes: Unsigned = NUM_WR_SUCCESSES):
        """
        Instantiate the `PineValid` circuit for gradients with `dimension`
        elements. Each element will be truncated to `num_frac_bits` binary
        fractional bits, and the L2-norm bound of each gradient is bounded
        by `l2_norm_bound`.
        """
        if l2_norm_bound <= 0.0:
            raise ValueError("Invalid L2-norm bound, it must be positive")
        if num_frac_bits < 0 or num_frac_bits >= 128:
            raise ValueError(
                "Invalid number of fractional bits, it must be in [0, 128)"
            )
        if dimension <= 0:
            raise ValueError("Invalid dimension, it must be positive")

        self.l2_norm_bound = l2_norm_bound
        self.num_frac_bits = num_frac_bits
        self.dimension = dimension
        self.alpha = alpha
        self.num_wr_checks = num_wr_checks
        self.num_wr_successes = num_wr_successes
        encoded_norm_bound_unsigned = \
            self.encode_f64_into_field(l2_norm_bound).as_unsigned()
        if (self.Field.MODULUS / encoded_norm_bound_unsigned
            < encoded_norm_bound_unsigned):
            # Squaring encoded norm bound overflows field size, reject.
            raise ValueError("Invalid combination of L2-norm bound and "
                             "number of fractional bits, that causes the "
                             "encoded norm bound to be larger than "
                             "field modulus.")
        self.encoded_sq_norm_bound = self.Field(
            encoded_norm_bound_unsigned ** 2
        )
        # Number of bits to represent the squared L2-norm, which should
        # be in range `[0, encoded_sq_norm_bound]`. The total number of values
        # in this range is `encoded_sq_norm_bound + 1`, so take the `log2`
        # of this quantity.
        self.num_bits_for_sq_norm = \
            math.ceil(math.log2(self.encoded_sq_norm_bound.as_unsigned() + 1))
        # TODO(junyechen1996): check other field size requirement,
        # specifically the requirements from Figure 1 and Lemma 4.3
        # in the PINE paper.

        # Wraparound check bound, equal to the smallest power of two
        # larger than or equal to `alpha * encoded_norm_bound_unsigned
        # + 1` is a power of 2. Using a power of 2 allows us to use
        # the optimization of Remark 3.2 without degrading
        # completeness.
        self.wr_check_bound = self.Field(
            next_power_of_2(alpha * encoded_norm_bound_unsigned + 1) - 1
        )

        # Number of bits to represent each wraparound check result, which
        # should be in range `[-wr_bound, wr_bound + 1]`. The number of
        # values in this range is `2 * (wr_bound + 1)`, so take the `log2`
        # of `wr_bound + 1` and add 1 to it.
        self.num_bits_for_wr_check = 1 + math.ceil(math.log2(
            self.wr_check_bound.as_unsigned() + 1
        ))

        # Length of the encoded gradient, including the gradient
        # itself and the L2-norm check.
        self.encoded_gradient_len = dimension + 2 * self.num_bits_for_sq_norm

        # Length of the bit-checked portion of the encoded
        # measurement. This includes:
        # - The L2-norm check
        # - Each wraparound check result
        # - The success bit for each wraparound check
        self.bit_checked_len = \
            2 * self.num_bits_for_sq_norm + \
            (self.num_bits_for_wr_check + 1) * num_wr_checks

        # Set `Valid` parameters.
        # The measurement length expected by `Flp.eval()` contains the encoded
        # gradient, the expected bits, and the dot products in wraparound
        # checks.
        self.MEAS_LEN = dimension + self.bit_checked_len + num_wr_checks
        # 1 for bit check, 1 for wraparound check, 1 for final reduction.
        # Note we don't include the number of wraparound joint randomness field
        # elements in `JOINT_RAND_LEN`.
        self.JOINT_RAND_LEN = 1 + 1 + 1
        self.OUTPUT_LEN = dimension

        self.chunk_length = chunk_length
        self.GADGET_CALLS = [
            chunk_count(chunk_length, self.bit_checked_len) + \
            chunk_count(chunk_length, dimension) + \
            chunk_count(chunk_length, num_wr_checks)
        ]
        self.GADGETS = [ParallelSum(Mul(), chunk_length)]

    def eval(self,
             meas: list[Field],
             joint_rand: list[Field],
             num_shares: Unsigned) -> Field:
        """Validity circuit for PINE.

        The goal of this circuit is to recognize gradients with bounded L2-norm
        (hereafter "norm"). This is accomplished by computing the squared norm
        of the gradient and checking that its value is in the desired range. We
        also need to ensure the squared norm is not so large that it wraps
        around the field modulus. Otherwise, a squared norm that is too large
        (or too small) may appear to be in range when in fact it is not.

        Wraparound enforcement is accomplished by a sequence of probabilistic
        tests devised by [ROCT23]. A successful wraparound test indicates,
        w.h.p., that the squared norm of the gradient, as it is represented in
        the field, is a value between 0 and an upper bound that depends on the
        circuit parameters.
        """
        self.check_valid_eval(meas, joint_rand)
        shares_inv = self.Field(num_shares).inv()

        # Unpack the encoded measurement. It is composed of the following
        # components:
        #
        # - The gradient `encoded_gradient`.
        #
        # - A pair `(sq_norm_v_bits, sq_norm_u_bits)`, the bit-encoded,
        #   range-checked, squared norm of the gradient.
        #
        # - For each wraparound test, a pair `(wr_check_v_bits, wr_check_g)`:
        #   `wr_check_v_bits` is the bit-encoded, range-checked test result;
        #   and `wr_check_g` is an indication of whether the test succeeded
        #   (i.e., the result is in the specified range).
        #
        # - For each wraparound test, the result `wr_check_results`.
        rest = meas
        (encoded_gradient, rest) = front(self.dimension, rest)
        (bit_checked, rest) = front(self.bit_checked_len, rest)
        (wr_check_results, rest) = front(self.num_wr_checks, rest)
        assert len(rest) == 0

        rest = bit_checked
        (sq_norm_v_bits, rest) = front(self.num_bits_for_sq_norm, rest)
        (sq_norm_u_bits, rest) = front(self.num_bits_for_sq_norm, rest)
        wr_check_v_bits = []
        wr_check_g = []
        while len(rest) > 0:
            (v_bits, rest) = front(self.num_bits_for_wr_check, rest)
            wr_check_v_bits.append(v_bits)
            ([g], rest) = front(1, rest)
            wr_check_g.append(g)
        assert len(rest) == 0

        # Unpack the joint randomness.
        [r_bit_check, r_wr_check, r_final] = joint_rand

        bit_checks_result = self.eval_bit_checks(r_bit_check,
                                                 bit_checked,
                                                 shares_inv)

        (wr_checks_result, wr_success_count_check_result) = \
            self.eval_wr_checks(r_wr_check,
                                wr_check_v_bits,
                                wr_check_g,
                                wr_check_results,
                                shares_inv)

        (sq_norm_equality_check_result, sq_norm_range_check_result) = \
            self.eval_norm_check(encoded_gradient,
                                 sq_norm_v_bits,
                                 sq_norm_u_bits,
                                 shares_inv)

        return bit_checks_result + \
            r_final * sq_norm_equality_check_result + \
            r_final**2 * sq_norm_range_check_result + \
            r_final**3 * wr_checks_result + \
            r_final**4 * wr_success_count_check_result

    def eval_bit_checks(self, r_bit_check, bit_checked, shares_inv):
        """
        Check that each element of `bit_checked` is a 0 or 1.

        Construct a polynomial from the bits and evaluate it at `r_bit_check`.
        The polynomial is

           f(x) = B[0]*(B[0]-1) + x*B[1]*(B[1]-1) + x^2*B[2]*(B[2]-1) + ...

        where `B[i]` is the `i`-th bit. The value of `B[i](B[i]-1)` is 0 if
        and only if `B[i]` is 0 or 1. Thus if one of the bits is non-zero, then
        `f(r_bit_check)` will be non-zero w.h.p.
        """
        mul_inputs = []
        r_power = self.Field(1)
        for bit in bit_checked:
            mul_inputs += [r_power * bit, bit - shares_inv]
            r_power *= r_bit_check
        return self.parallel_sum(mul_inputs)

    def eval_norm_check(self,
                        encoded_gradient,
                        sq_norm_v_bits,
                        sq_norm_u_bits,
                        shares_inv):
        """
        Check that (1) the reported squared was computed correctly and (2) the
        squared norm is in range. The result is only valid if the bit checks
        and the wraparound checks were successful.
        """
        # Compute the squared norm.
        mul_inputs = []
        for val in encoded_gradient:
            mul_inputs += [val, val]
        computed_sq_norm = self.parallel_sum(mul_inputs)

        sq_norm_v = self.Field.decode_from_bit_vector(sq_norm_v_bits)
        sq_norm_u = self.Field.decode_from_bit_vector(sq_norm_u_bits)

        return (
            # Check that the computed squared L2-norm result matches
            # the value claimed by the Client.
            sq_norm_v - computed_sq_norm,
            # Check the squared L2-norm is in range (see [ROCT23], Figure 1).
            sq_norm_v + sq_norm_u - self.encoded_sq_norm_bound * shares_inv,
        )

    def eval_wr_checks(self,
                       r_wr_check,
                       wr_check_v_bits,
                       wr_check_g,
                       wr_check_results,
                       shares_inv):
        """
        Check two things:

        (1) For each wraparound test, (i) the reported success bit
            (`wr_test_g`) is 0 or (ii) the success bit is 1 and the reported
            result (`wr_test_v`) was computed correctly.

        (2) The number of reported successes is equal to the expected number of
            successes.

        See [ROCT23], Figure 2 for details.

        A test is only successful if the reported result is in the specified
        range. The range is chosen so that it is sufficient to bit-check the
        reported result. See [ROCT23], Remark 3.2 for details.

        These checks are only valid if the bit checks were successful.
        """
        mul_inputs = []
        wr_success_count = self.Field(0)
        r_power = self.Field(1)

        for i in range(self.num_wr_checks):
            wr_check_v = self.Field.decode_from_bit_vector(wr_check_v_bits[i])
            computed_result = wr_check_v - self.wr_check_bound * shares_inv

            # (1) For each check, we want that either (i) the Client reported
            # failure (`wr_check_g[i] == 0`) or (ii) the Client reported
            # success and the reported result was computed correctly. To
            # accomplish this, subtract the computed result from the reported
            # result and multiply by the success bit.
            #
            # Similar to the bit checks, interpret the values as coefficients
            # of a polynomial and evaluate the polynomial at a random point
            # (`r_wr_check`).
            mul_inputs += [
                r_power * (wr_check_results[i] - computed_result),
                wr_check_g[i],
            ]

            # (2) Sum up the success bits for each test.
            wr_success_count += wr_check_g[i]
            r_power *= r_wr_check

        return (
            self.parallel_sum(mul_inputs),
            wr_success_count - self.Field(self.num_wr_successes) * shares_inv
        )

    def parallel_sum(self, mul_inputs):
        """
        Split `mul_inputs` into chunks, call the gadget on each chunk, and
        return the sum of the results. If there is a partial leftover, then it
        is padded with zeros.
        """
        s = self.Field(0)
        while len(mul_inputs) >= 2*self.chunk_length:
            (chunk, mul_inputs) = front(2*self.chunk_length, mul_inputs)
            s += self.GADGETS[0].eval(self.Field, chunk)
        if len(mul_inputs) > 0:
            chunk = self.Field.zeros(2*self.chunk_length)
            for i in range(len(mul_inputs)):
                chunk[i] = mul_inputs[i]
            s += self.GADGETS[0].eval(self.Field, chunk)
        return s

    def encode_gradient(self, measurement):
        """
        Encode the gradient and the range-checked, squared L2-norm.
        """
        if len(measurement) != self.dimension:
            raise ValueError("Unexpected gradient dimension.")
        encoded_gradient = [self.encode_f64_into_field(x) for x in measurement]

        # Encode results for range check of the squared L2-norm.
        sq_norm = sum((x**2 for x in encoded_gradient), self.Field(0))
        (_, sq_norm_v, sq_norm_u) = range_check(
            sq_norm, self.Field(0), self.encoded_sq_norm_bound,
        )
        encoded_gradient += self.Field.encode_into_bit_vector(
            sq_norm_v.as_unsigned(),
            self.num_bits_for_sq_norm,
        )
        encoded_gradient += self.Field.encode_into_bit_vector(
            sq_norm_u.as_unsigned(),
            self.num_bits_for_sq_norm,
        )
        return encoded_gradient

    def run_wr_checks(self, encoded_gradient, wr_joint_rand_xof):
        """
        Run the wraparound tests. For each test, we compute the dot product of
        the gradient and a random `{-1, 0, 1}`-vector derived from the provided
        XOF instance.

        Each element of the `{-1, 0, 1}`-vector is independently distributed as
        follows:

        - `-1` occurs with probability 1/4;
        - `0` occurs with probability 1/2; and
        - `1` occurs with probability 1/4.

        To sample this distribution exactly, we will look at the bytes from
        `Xof` two bits at a time, so the number of field elements we can
        generate from each byte is 4.

        Implementation note: It may be useful to stream the XOF output a block
        a time rather than pre-compute the entire bufferr.
        """
        xof_output = wr_joint_rand_xof.next(
            chunk_count(4, self.num_wr_checks * self.dimension))

        wr_check_results = [self.Field(0)] * self.num_wr_checks
        for i, rand_bits in enumerate(bit_chunks(xof_output, 2)):
            wr_check_index = i // self.dimension
            x = encoded_gradient[i % self.dimension]
            if rand_bits == 0b00:
                rand_field = -self.Field(1)
            elif rand_bits == 0b01 or rand_bits == 0b10:
                rand_field = self.Field(0)
            else:
                rand_field = self.Field(1)
            wr_check_results[wr_check_index] += rand_field * x
        return wr_check_results

    def encode_wr_checks(self, encoded_gradient, wr_joint_rand_xof):
        """
        Run the wraparound checks and return the dot product for each check
        (`wr_check_results`) and the range-checked result for each check
        (`wr_check_bits`).

        The string `encoded_gradient + wr_check_bits + wr_check_results` is
        passed to the circuit. The Aggregators are expected to re-compute the
        dot products on their own, with the encoded gradient and the wraparound
        joint randomness XOF.
        """
        wr_check_results = self.run_wr_checks(encoded_gradient,
                                              wr_joint_rand_xof)

        wr_check_bits = []

        # Number of successful checks. If the prover has passed more than
        # `self.num_wr_successes`, don't set the success bit to be 1 anymore,
        # because verifier will check that exactly `self.num_wr_successes`
        # checks passed.
        wr_success_count = 0

        for i in range(self.num_wr_checks):
            # This wraparound check passes if the current dot product is in
            # range `[-wr_check_bound, wr_check_bound+1]`. To prove this, the
            # prover sends the bit-encoding of `wr_check_v =
            # wr_check_results[i] + wr_check_bound` to the verifier. The
            # verifier re-computes this value and makes sure it matches the
            # reported value.
            (is_in_range, wr_check_v, _) = range_check(
                wr_check_results[i],
                -self.wr_check_bound,
                self.wr_check_bound + self.Field(1),
            )

            if is_in_range and wr_success_count < self.num_wr_successes:
                # If the result of the current wraparound check is
                # in range, and the number of passing checks hasn't
                # reached `self.num_wr_successes`. Set the success bit
                # to be 1.
                wr_success_count += 1
                wr_check_g = self.Field(1)
            else:
                # Otherwise set the success bit to be 0.
                wr_check_g = self.Field(0)

            # Send the bits of wraparound check result, and the success bit.
            wr_check_bits += self.Field.encode_into_bit_vector(
                wr_check_v.as_unsigned(), self.num_bits_for_wr_check,
            )
            wr_check_bits.append(wr_check_g)

        # Sanity check the Client has passed `self.num_wr_successes`
        # number of checks, otherwise Client SHOULD retry.
        if wr_success_count != self.num_wr_successes:
            raise Exception(
                "Client should retry wraparound check with "
                "different wraparound joint randomness."
            )
        return (wr_check_bits, wr_check_results)

    def truncate(self, meas: list[Field]):
        return meas[:self.dimension]

    def decode(self,
               output: list[Field],
               num_measurements: Unsigned) -> AggResult:
        return [self.decode_f64_from_field(x) for x in output]

    def encode_f64_into_field(self, x: float) -> Field:
        if (math.isnan(x) or not math.isfinite(x) or
            (x != 0.0 and abs(x) < sys.float_info.min)):
            # Reject NAN, infinity, and subnormal floats,
            # per {{fp-encoding}}.
            raise ValueError("f64 encoding doesn't accept NAN, "
                             "infinite, or subnormal floats.")
        x_encoded = math.floor(x * (2 ** self.num_frac_bits))
        return self.Field(x_encoded)

    def decode_f64_from_field(self, field_elem: Field) -> float:
        decoded = field_elem.as_unsigned()
        # If the field is larger than half of the field size, then
        # decoded result should be negative.
        if decoded > math.floor(field_elem.MODULUS / 2):
            # We need to take the difference between the result
            # and the field modulus, and return the result as negative.
            decoded = -(field_elem.MODULUS - decoded)
        # Divide by 2^num_frac_bits and we will get a float back.
        decoded_float = decoded / (2 ** self.num_frac_bits)
        return decoded_float

    def test_vec_set_type_param(self, test_vec):
        test_vec["l2_norm_bound"] = self.l2_norm_bound
        test_vec["num_frac_bits"] = self.num_frac_bits
        test_vec["dimension"] = self.dimension
        test_vec["chunk_length"] = self.chunk_length
        test_vec["field"] = self.Field.__name__
        test_vec["num_wr_checks"] = self.num_wr_checks
        test_vec["num_wr_successes"] = self.num_wr_successes
        test_vec["alpha"] = self.alpha
        return ["l2_norm_bound", "num_frac_bits", "dimension", "field",
                "num_wr_checks", "num_wr_successes", "alpha"]

def bit_chunks(buf: bytes, num_chunk_bits: int):
    """
    Output the bit chunks, at `num_chunk_bits` bits at a time, from `buf`.
    """
    assert(8 % num_chunk_bits == 0 and
           0 < num_chunk_bits and num_chunk_bits <= 8)
    # Mask to extract the least significant `num_chunk_bits` bits.
    mask = (1 << num_chunk_bits) - 1
    for byte in buf:
        for chunk_start in reversed(range(0, 8, num_chunk_bits)):
            yield (byte >> chunk_start) & mask

def range_check(dot_prod: Field,
                lower_bound: Field,
                upper_bound: Field) -> tuple[bool, Field, Field]:
    """
    Check if the dot product is in the range `[lower_bound, upper_bound]`,
    per Section 4.1 of PINE paper.
    """
    v = dot_prod - lower_bound
    u = upper_bound - dot_prod
    is_in_range = \
        v.as_unsigned() <= (upper_bound - lower_bound).as_unsigned()
    return (is_in_range, v, u)

def chunk_count(chunk_length, length):
    return (length + chunk_length - 1) // chunk_length
