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
                 chunk_length: Unsigned):
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
        # larger than or equal to `ALPHA * encoded_norm_bound_unsigned
        # + 1` is a power of 2. Using a power of 2 allows us to use
        # the optimization of Remark 3.2 without degrading
        # completeness.
        self.wr_check_bound = self.Field(
            next_power_of_2(ALPHA * encoded_norm_bound_unsigned + 1) - 1
        )

        # Number of bits to represent each wraparound check result, which
        # should be in range `[-wr_bound, wr_bound + 1]`. The number of
        # values in this range is `2 * (wr_bound + 1)`, so take the `log2`
        # of `wr_bound + 1` and add 1 to it.
        self.num_bits_for_wr_res = 1 + math.ceil(math.log2(
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
            (self.num_bits_for_wr_res + 1) * NUM_WR_CHECKS

        # Set `Valid` parameters.
        # The measurement length expected by `Flp.eval()` contains the encoded
        # gradient, the expected bits, and the dot products in wraparound
        # checks.
        self.MEAS_LEN = dimension + self.bit_checked_len + NUM_WR_CHECKS
        # 1 for bit check, 1 for wraparound check, 1 for final reduction.
        # Note we don't include the number of wraparound joint randomness field
        # elements in `JOINT_RAND_LEN`.
        self.JOINT_RAND_LEN = 1 + 1 + 1
        self.OUTPUT_LEN = dimension

        self.chunk_length = chunk_length
        self.GADGET_CALLS = [
            chunk_count(chunk_length, self.bit_checked_len) + \
            chunk_count(chunk_length, dimension) + \
            chunk_count(chunk_length, NUM_WR_CHECKS)
        ]
        self.GADGETS = [ParallelSum(Mul(), chunk_length)]

    def eval(self,
             meas: list[Field],
             joint_rand: list[Field],
             num_shares: Unsigned) -> Field:
        """Validity circuit for PINE. """
        self.check_valid_eval(meas, joint_rand)
        shares_inv = self.Field(num_shares).inv()

        # Unpack `meas = encoded_gradient || bit_checked || wr_dot_prods`
        (encoded_gradient, rest) = front(self.dimension, meas)
        (bit_checked, rest) = front(self.bit_checked_len, rest)
        (wr_dot_prods, rest) = front(NUM_WR_CHECKS, rest)
        assert len(rest) == 0

        # Unpack the joint randomness.
        [r_bit_check, r_wr_check, r_final] = joint_rand

        # 0/1 bit checks:
        bit_check_res = self.bit_check(
            r_bit_check, bit_checked, shares_inv,
        )

        # L2-norm check:
        (norm_bits, wr_check_bits) = \
            front(2 * self.num_bits_for_sq_norm, bit_checked)
        (norm_equality_check_res, norm_range_check_res) = \
            self.norm_check(encoded_gradient, norm_bits, shares_inv)

        # Wraparound checks:
        assert len(wr_check_bits) == \
               (self.num_bits_for_wr_res + 1) * NUM_WR_CHECKS
        assert len(wr_dot_prods) == NUM_WR_CHECKS
        (wr_mul_check_res, wr_success_count_check_res) = self.wr_check(
            wr_check_bits,
            wr_dot_prods,
            r_wr_check,
            shares_inv,
        )

        # Reduce over all circuits.
        return bit_check_res + \
            r_final * norm_equality_check_res + \
            r_final**2 * norm_range_check_res + \
            r_final**3 * wr_mul_check_res + \
            r_final**4 * wr_success_count_check_res

    def bit_check(self,
                  r_bit_check,
                  bit_checked,
                  shares_inv):
        """
        Compute the bit checks, consisting of a random linear combination of a
        range check for each element of `bit_checked`.
        """
        mul_inputs = []
        r_power = self.Field(1)
        for bit in bit_checked:
            mul_inputs += [r_power * bit, bit - shares_inv]
            r_power *= r_bit_check
        return self.parallel_sum(mul_inputs)

    def norm_check(self,
                   encoded_gradient: list[Field],
                   norm_bits: list[Field],
                   shares_inv: Field) -> tuple[Field, Field]:
        """
        Compute the squared L2-norm of the gradient and test that it is in range.
        """
        mul_inputs = []
        for val in encoded_gradient:
            mul_inputs += [val, val]
        computed_sq_norm = self.parallel_sum(mul_inputs)

        # The `v` bits (difference between squared L2-norm and lower bound)
        # and `u` bits (difference between squared L2-norm and upper bound)
        # claimed by the Client for the range check of the squared L2-norm.
        (norm_range_check_v_bits, norm_range_check_u_bits) = front(
            self.num_bits_for_sq_norm, norm_bits
        )
        norm_range_check_v = \
            self.Field.decode_from_bit_vector(norm_range_check_v_bits)
        norm_range_check_u = \
            self.Field.decode_from_bit_vector(norm_range_check_u_bits)

        return (
            # Check that the computed squared L2-norm result matches
            # the value claimed by the Client.
            norm_range_check_v - computed_sq_norm,
            # Check the squared L2-norm is in range
            # `[0, encoded_sq_norm_bound]`.
            (norm_range_check_v + norm_range_check_u -
             self.encoded_sq_norm_bound * shares_inv),
        )

    def wr_check(self,
                 wr_check_bits,
                 wr_dot_prods,
                 r_wr_check,
                 shares_inv):
        """
        Compute the wraparound checks, consisting of (i) checking that, for
        each wraparound test, either the Client indicated failure or the Client
        indicated success and the dot product is equal to the claimed value and
        is in range; and (ii) checking that the Client indicated the expected
        number of successes.
        """
        mul_inputs = []
        r_power = self.Field(1)
        wr_success_count_check_res = \
            -self.Field(NUM_WR_SUCCESSES) * shares_inv
        for check in range(NUM_WR_CHECKS):
            # Add the dot product result and the absolute value of the
            # wraparound check lower bound.
            computed_wr_res = wr_dot_prods[check] + self.wr_check_bound * shares_inv

            # Wraparound check result indicated by the Client:
            (wr_res_bits, wr_check_bits) = \
                front(self.num_bits_for_wr_res, wr_check_bits)
            wr_res = self.Field.decode_from_bit_vector(wr_res_bits)

            # Success bit, the Client's indication as to whether the current
            # check passed.
            ([success_bit], wr_check_bits) = front(1, wr_check_bits)
            wr_success_count_check_res += success_bit

            # The Client share is considered valid if the multiplication of
            # the difference between computed wraparound check result and
            # the result from the bits, and the success bit is equal to 0.
            # This means either:
            # - success bit is 0, the current check failed.
            #   The difference can be any arbitrary value.
            # - success bit is 1, the current check passed.
            #   Then the difference must be 0, i.e., the bits of `wr_res`
            #   must match `computed_wr_res`.
            mul_inputs += [r_power * (computed_wr_res - wr_res), success_bit]
            r_power *= r_wr_check
        assert len(wr_check_bits) == 0
        return (self.parallel_sum(mul_inputs), wr_success_count_check_res)

    def parallel_sum(self, inputs):
        """
        Split `inputs` into chunks, call the gadget on each chunk, and return
        the sum of the results. If there is a partial leftover, then it is
        padded with zeros.
        """
        s = self.Field(0)
        while len(inputs) >= 2*self.chunk_length:
            (chunk, inputs) = front(2*self.chunk_length, inputs)
            s += self.GADGETS[0].eval(self.Field, chunk)
        if len(inputs) > 0:
            chunk = self.Field.zeros(2*self.chunk_length)
            for i in range(len(inputs)):
                chunk[i] = inputs[i]
            s += self.GADGETS[0].eval(self.Field, chunk)
        return s

    def encode_gradient(self, measurement):
        """
        Encode everything except wraparound check results:
        - Encode each f64 value into field element.
        - Encode L2-norm check results.
        """
        if len(measurement) != self.dimension:
            raise ValueError("Unexpected gradient dimension.")
        encoded = [self.encode_f64_into_field(x) for x in measurement]

        # Encode results for range check of the squared L2-norm.
        sq_encoded = sum((x**2 for x in encoded), self.Field(0))
        (_, norm_range_check_v, norm_range_check_u) = range_check(
            sq_encoded, self.Field(0), self.encoded_sq_norm_bound,
        )
        encoded += self.Field.encode_into_bit_vector(
            norm_range_check_v.as_unsigned(),
            self.num_bits_for_sq_norm,
        )
        encoded += self.Field.encode_into_bit_vector(
            norm_range_check_u.as_unsigned(),
            self.num_bits_for_sq_norm,
        )
        return encoded

    def run_wr_checks(self, encoded_gradient, wr_joint_rand_xof):
        """
        Compute the dot products of `encoded_gradient` with the wraparound
        joint randomness, sampled from the XOF `wr_joint_rand_xof`. `eval()`
        function expects the circuit input to contain the dot products.
        """

        # Sample the wraparound joint randomness. This is a `{-1, 0, 1}`-vector
        # with `NUM_WR_CHECKS` chunks, where the length of each chunk is
        # `self.dimension`. Each element is independently distributed as
        # follows:
        # - -1 with probability 1/4;
        # - 0 with probability 1/2; and
        # - 1 with probability 1/4.
        #
        # To sample this distribution exactly, we will look at the bytes from
        # `Xof` two bits at a time, so the number of field elements we can
        # generate from each byte is 4.
        #
        # Implementation note: Note in a real implementation with large
        # dimension, in order to not consume a lot of memory when sampling, we
        # can sample from the XOF one block at a time.
        xof_output = wr_joint_rand_xof.next(
            chunk_count(4, NUM_WR_CHECKS * self.dimension))

        wr_dot_prods = [self.Field(0)] * NUM_WR_CHECKS
        for i, rand_bits in enumerate(bit_chunks(xof_output, 2)):
            wr_check_index = i // self.dimension
            x = encoded_gradient[i % self.dimension]
            if rand_bits == 0b00:
                rand_field = -self.Field(1)
            elif rand_bits == 0b01 or rand_bits == 0b10:
                rand_field = self.Field(0)
            else:
                rand_field = self.Field(1)
            wr_dot_prods[wr_check_index] += rand_field * x
        return wr_dot_prods

    def encode_wr_checks(self, encoded_gradient, wr_joint_rand_xof):
        """
        Run the wraparound checks and return the dot product for each check
        (`wr_dot_prods`) and the range-checked result for each check
        (`wr_check_bits`).

        The dot products are passed into the `PineValid.eval()` function, and
        the wraparound check results are sent to the Aggregators. The
        Aggregators are expected to re-compute the dot products on their own,
        with the encoded gradient and the wraparound joint randomness XOF.
        """
        # First compute the dot product between `encoded_gradient` and the
        # wraparound joint randomness field elements in each wraparound check,
        # sampled from the XOF `wr_joint_rand_xof`.
        wr_dot_prods = \
            self.run_wr_checks(encoded_gradient, wr_joint_rand_xof)

        # Stores wraparound check result bits, and success bits.
        wr_check_bits = []
        # Keep track of the number of passing checks in
        # `num_passed_wr_checks`. If the Client has passed more than
        # `NUM_WR_SUCCESSES`, don't set the success bit to be 1
        # anymore, because Aggregators will check that exactly
        # `NUM_WR_SUCCESSES` checks passed.
        num_passed_wr_checks = 0
        for check in range(NUM_WR_CHECKS):
            # This wraparound check passes if the current dot product is in
            # range `[-wr_bound, wr_bound+1]`. To prove this, the Client sends
            # the bit-encoding of `wr_res = wr_dot_prods[check] + wr_bound`
            # to the Aggregators. To check that the dot product is larger or
            # equal to the lower bound, the Aggregator then reconstructs
            # `wr_res` from the bits, and checks that it is equal to
            # `computed_wr_res = wr_dot_prods[check] + wr_bound`. The upper
            # bound is checked implicitly by virtue of the fact that
            # `wr_res` is encoded with `1 + ceil(log2(wr_bound + 1))` bits.
            (is_in_range, wr_res, _) = range_check(
                wr_dot_prods[check],
                -self.wr_check_bound,
                self.wr_check_bound + self.Field(1),
            )

            if is_in_range and num_passed_wr_checks < NUM_WR_SUCCESSES:
                # If the result of the current wraparound check is
                # in range, and the number of passing checks hasn't
                # reached `NUM_WR_SUCCESSES`, set the success bit
                # to be 1.
                num_passed_wr_checks += 1
                success_bit = self.Field(1)
            else:
                # Otherwise set the success bit to be 0, and set wraparound
                # check result to be 0.
                wr_res = self.Field(0)
                success_bit = self.Field(0)

            # Send the bits of wraparound check result, and the success bit.
            wr_check_bits += self.Field.encode_into_bit_vector(
                wr_res.as_unsigned(), self.num_bits_for_wr_res,
            )
            wr_check_bits.append(success_bit)

        # Sanity check the Client has passed `NUM_WR_SUCCESSES`
        # number of checks, otherwise Client SHOULD retry.
        if num_passed_wr_checks != NUM_WR_SUCCESSES:
            raise Exception(
                "Client should retry wraparound check with "
                "different wraparound joint randomness."
            )
        return (wr_check_bits, wr_dot_prods)

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
        return ["l2_norm_bound", "num_frac_bits", "dimension"]

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
