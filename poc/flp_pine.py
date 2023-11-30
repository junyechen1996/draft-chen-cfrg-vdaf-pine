"""Validity circuit for PINE. """

import functools
import math
import os
import sys

# Access poc folder in submoduled VDAF draft.
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "draft-irtf-cfrg-vdaf", "poc"))
from common import Unsigned, Vec, front, gen_rand, next_power_of_2, vec_add
from field import Field, Field128
from flp import additive_secret_share
from flp_generic import FlpGeneric, Mul, ParallelSum, PolyEval, Valid
from xof import Xof, XofShake128


class PineValid(Valid):
    # Operational parameters set by user.
    l2_norm_bound: float = None # Set by constructor
    num_frac_bits: Unsigned = None # Set by constructor
    dimension: Unsigned = None # Set by constructor

    # Internal operational parameters.
    # TODO(junyechen1996): Figure out how to fix them safely, so it doesn't
    # negatively impact soundness and completeness error. (#24)
    ALPHA: float = 7
    NUM_WR_CHECKS: Unsigned = 135
    TAU: float = 0.75
    NUM_PASS_WR_CHECKS: Unsigned = math.floor(TAU * NUM_WR_CHECKS)
    encoded_sq_norm_bound: Field = None # Set by constructor
    num_bits_for_norm: Unsigned = None # Set by constructor
    wr_bound: Field = None # Set by constructor
    num_bits_for_wr_res: Unsigned = None # Set by constructor
    wr_joint_rand_len = None # Set by constructor
    vf_joint_rand_len = None # Set by constructor

    # Associated types for `Valid`.
    Measurement = Vec[float]
    AggResult = Vec[float]
    Field = Field128

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
        self.num_bits_for_norm = \
            math.ceil(math.log2(self.encoded_sq_norm_bound.as_unsigned() + 1))
        # TODO(junyechen1996): check other field size requirement,
        # specifically the requirements from Figure 1 and Lemma 4.3
        # in the PINE paper.

        # Compute wraparound check bound.
        # Pick `alpha' >= alpha`, such that
        # `alpha' * encoded_norm_bound_unsigned + 1` is a
        # power of 2, so the bounds for wraparound check becomes
        # `[-alpha' * encoded_norm_bound_unsigned,
        #   alpha' * encoded_norm_bound_unsigned + 1]`,
        # which won't negatively impact completeness error,
        # and allows us to perform the optimization in Remark 4.11
        # of the PINE paper.
        self.wr_bound = self.Field(
            next_power_of_2(self.ALPHA * encoded_norm_bound_unsigned + 1) - 1
        )
        # Number of bits to represent each wraparound check result, which
        # should be in range `[-wr_bound, wr_bound + 1]`. The number of
        # values in this range is `2 * (wr_bound + 1)`, so take the `log2`
        # of `wr_bound + 1` and add 1 to it.
        self.num_bits_for_wr_res = 1 + math.ceil(math.log2(
            self.wr_bound.as_unsigned() + 1
        ))
        self.wr_joint_rand_len = self.NUM_WR_CHECKS * dimension
        # 1 for bit check, 1 for wraparound check, 1 for final reduction.
        self.vf_joint_rand_len = 1 + 1 + 1

        # Set `Valid` parameters.
        # Total number of bits is:
        # - The number of bits for L2-norm check.
        # - The number of bits for each wraparound check result, and the
        #   success bit for each wraparound check.
        total_num_bits = 2 * self.num_bits_for_norm + \
                         (self.num_bits_for_wr_res + 1) * self.NUM_WR_CHECKS
        self.MEAS_LEN = dimension + total_num_bits
        self.JOINT_RAND_LEN = self.wr_joint_rand_len + self.vf_joint_rand_len
        self.OUTPUT_LEN = dimension

        self.chunk_length = chunk_length
        self.GADGET_CALLS = [
            chunk_count(chunk_length, total_num_bits) + \
            chunk_count(chunk_length, dimension) + \
            chunk_count(chunk_length, self.NUM_WR_CHECKS)
        ]
        self.GADGETS = [ParallelSum(Mul(), chunk_length)]

    def eval(self,
             meas: Vec[Field],
             joint_rand: Vec[Field],
             num_shares: Unsigned) -> Field:
        self.check_valid_eval(meas, joint_rand)
        shares_inv = self.Field(num_shares).inv()

        wr_joint_rand, joint_rand = front(self.wr_joint_rand_len, joint_rand)
        [bit_check_red_joint_rand], joint_rand = front(1, joint_rand)
        [wr_check_red_joint_rand], joint_rand = front(1, joint_rand)
        [final_red_joint_rand], joint_rand = front(1, joint_rand)
        assert len(joint_rand) == 0 # sanity check

        # 0/1 bit checks:
        x, bit_checked = front(self.dimension, meas)
        bit_check_res = self.bit_check(
            bit_check_red_joint_rand,
            bit_checked,
            shares_inv,
        )

        # L2-norm check:
        norm_bits, bit_checked = front(
            2 * self.num_bits_for_norm, bit_checked
        )
        (norm_equality_check_res, norm_range_check_res) = \
            self.norm_check(x, norm_bits, shares_inv)

        # Wraparound checks:
        assert(len(bit_checked) ==
               (self.num_bits_for_wr_res + 1) * self.NUM_WR_CHECKS)
        (wr_test_check_res, wr_success_count_check_res) = self.wr_check(
            x,
            bit_checked,
            wr_check_red_joint_rand,
            wr_joint_rand,
            shares_inv,
        )

        # Reduce over all circuits.
        return bit_check_res + \
            final_red_joint_rand * norm_equality_check_res + \
            final_red_joint_rand**2 * norm_range_check_res + \
            final_red_joint_rand**3 * wr_test_check_res + \
            final_red_joint_rand**4 * wr_success_count_check_res

    def bit_check(self,
                  bit_check_red_joint_rand,
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
            r_power *= bit_check_red_joint_rand
        return self.parallel_sum(mul_inputs)

    def norm_check(self,
                   x: Vec[Field],
                   norm_bits: Vec[Field],
                   shares_inv: Field) -> tuple[Field, Field]:
        """
        Compute the squared L2-norm of the gradient and test that it is in range.
        """
        mul_inputs = []
        for val in x:
            mul_inputs += [val, val]
        computed_sq_norm = self.parallel_sum(mul_inputs)

        # The `v` bits (difference between squared L2-norm and lower bound)
        # and `u` bits (difference between squared L2-norm and upper bound)
        # claimed by the Client for the range check of the squared L2-norm.
        norm_range_check_v_bits, norm_range_check_u_bits = front(
            self.num_bits_for_norm, norm_bits
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
                 x,
                 bit_checked,
                 wr_check_red_joint_rand,
                 wr_joint_rand,
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
        wr_mul_check_res = self.Field(0)
        wr_success_count_check_res = \
            -self.Field(self.NUM_PASS_WR_CHECKS) * shares_inv
        for check in range(self.NUM_WR_CHECKS):
            z, wr_joint_rand = front(self.dimension, wr_joint_rand)
            # Compute the dot product of `x` and `z`, and add the
            # absolute value of the wraparound check lower bound.
            computed_wr_res = dot_prod(x, z) + self.wr_bound * shares_inv

            # Wraparound check result indicated by the Client:
            wr_res_bits, bit_checked = \
                front(self.num_bits_for_wr_res, bit_checked)
            wr_res = self.Field.decode_from_bit_vector(wr_res_bits)

            # Success bit, the Client's indication as to whether the current
            # check passed.
            [success_bit], bit_checked = front(1, bit_checked)
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
            r_power *= wr_check_red_joint_rand
        assert(len(wr_joint_rand) == 0)
        assert(len(bit_checked) == 0)
        return (self.parallel_sum(mul_inputs), wr_success_count_check_res)

    def parallel_sum(self, inputs):
        """
        Split `inputs` into chunks, call the gadget on each chunk, and return
        the sum of the results. If there is a partial leftover, then it is
        padded with zeros.
        """
        s = self.Field(0)
        while len(inputs) >= 2*self.chunk_length:
            chunk, inputs = front(2*self.chunk_length, inputs)
            s += self.GADGETS[0].eval(self.Field, chunk)
        if len(inputs) > 0:
            chunk = self.Field.zeros(2*self.chunk_length)
            for i in range(len(inputs)):
                chunk[i] = inputs[i]
            s += self.GADGETS[0].eval(self.Field, chunk)
        return s

    def encode(self, measurement: Measurement) -> Vec[Field]:
        """
        Encode everything except wraparound check results:
        - Encode each f64 value into field element.
        - Encode L2-norm check results.
        """
        if len(measurement) != self.dimension:
            raise ValueError("Unexpected gradient dimension.")
        encoded = [self.encode_f64_into_field(x) for x in measurement]

        # Encode results for range check of the squared L2-norm.
        (_, norm_range_check_v, norm_range_check_u) = range_check(
            encoded, encoded, self.Field(0), self.encoded_sq_norm_bound,
        )
        encoded += self.Field.encode_into_bit_vector(
            norm_range_check_v.as_unsigned(),
            self.num_bits_for_norm,
        )
        encoded += self.Field.encode_into_bit_vector(
            norm_range_check_u.as_unsigned(),
            self.num_bits_for_norm,
        )
        return encoded

    def sample_wr_joint_rand(self, xof: Xof):
        # Each element in the wraparound joint randomness is:
        # - -1 with probability 1/4,
        # - 0 with probability 1/2,
        # - 1 with probability 1/4.
        # To sample this distribution exactly, we will look at the
        # bytes from `Xof` two bits at a time, so the number of field
        # elements we can generate from each byte is 4.
        NUM_ELEMS_IN_ONE_BYTE = 4
        # Compute the number of bytes we will need from `Xof`
        # in order to sample `self.wr_joint_rand_len`
        # number of field elements.
        # Taking the ceiling of the division makes sure we can sample
        # one more byte if `self.wr_joint_rand_len` is not
        # divisible by `NUM_ELEMS_IN_ONE_BYTE`.
        xof_output_len = math.ceil(
            self.wr_joint_rand_len / NUM_ELEMS_IN_ONE_BYTE
        )
        rand_buf = xof.next(xof_output_len)

        wr_joint_rand = []
        for rand_bits in bit_chunks(rand_buf, 2):
            if rand_bits == 0b00:
                rand_field = self.Field(self.Field.MODULUS - 1)
            elif rand_bits == 0b01 or rand_bits == 0b10:
                rand_field = self.Field(0)
            else:
                rand_field = self.Field(1)
            wr_joint_rand.append(rand_field)
        return wr_joint_rand

    def run_wr_checks(self,
                      x: Vec[Field],
                      wr_joint_rand: Vec[Field]) -> Vec[Field]:
        full_wr_res = []
        # Keep track of the number of passing checks in
        # `num_passed_wr_checks`. If the Client has passed more than
        # `self.NUM_PASS_WR_CHECKS`, don't set the success bit to be 1
        # anymore, because Aggregators will check that exactly
        # `self.NUM_PASS_WR_CHECKS` checks passed.
        num_passed_wr_checks = 0
        for check in range(self.NUM_WR_CHECKS):
            z, wr_joint_rand = front(self.dimension, wr_joint_rand)
            # This wraparound check passes if `dot_prod(x, z)` is in range
            # `[-wr_bound, wr_bound+1]`. To prove this, the Client sends the
            # bit-encoding of `wr_res = dot_prod(x, z) + wr_bound` to the
            # Aggregators. To check that the dot product is larger or equal to
            # the lower bound, the Aggregator then reconstructs `wr_res` from
            # the bits, and checks that it is equal to
            # `computed_wr_res = dot_prod(x, z) + wr_bound`. The upper bound
            # is checked implicitly by virtue of the fact that `wr_res` is
            # encoded with `1 + ceil(log2(wr_bound + 1))` bits.
            (is_in_range, wr_res, _) = range_check(
                x, z, -self.wr_bound, self.wr_bound + self.Field(1),
            )

            if is_in_range and num_passed_wr_checks < self.NUM_PASS_WR_CHECKS:
                # If the result of the current wraparound check is
                # in range, and the number of passing checks hasn't
                # reached `self.NUM_PASS_WR_CHECKS`, set the success bit
                # to be 1.
                num_passed_wr_checks += 1
                success_bit = self.Field(1)
            else:
                # Otherwise set the success bit to be 0, and set wraparound
                # check result to be 0.
                wr_res = self.Field(0)
                success_bit = self.Field(0)

            # Send the bits of wraparound check result, and the success bit.
            full_wr_res += self.Field.encode_into_bit_vector(
                wr_res.as_unsigned(),
                self.num_bits_for_wr_res,
            )
            full_wr_res.append(success_bit)

        assert(len(wr_joint_rand) == 0)
        # Sanity check the Client has passed `self.NUM_PASS_WR_CHECKS`
        # number of checks, otherwise Client SHOULD retry.
        if num_passed_wr_checks != self.NUM_PASS_WR_CHECKS:
            raise Exception(
                "Client should retry wraparound check with "
                "different wraparound joint randomness."
            )
        return full_wr_res

    def truncate(self, meas: Vec[Field]):
        return meas[:self.dimension]

    def decode(self,
               output: Vec[Field],
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
        if x >= 0:
            return self.Field(x_encoded)
        return self.Field(self.Field.MODULUS + x_encoded)

    def decode_f64_from_field(self, field_elem: Field) -> float:
        decoded = field_elem.as_unsigned()
        # If the aggregated field is larger than half of the field
        # size, the decoded result should be negative.
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

def dot_prod(vec0: Vec[Field], vec1: Vec[Field]):
    """Compute the dot product of vectors `vec0` and `vec1`."""
    if len(vec0) != len(vec1):
        raise ValueError("Mismatched input vector length")
    res = map(lambda x: x[0] * x[1], zip(vec0, vec1))
    return functools.reduce(lambda x, y: x + y, res)

def range_check(vec0: Vec[Field],
                vec1: Vec[Field],
                lower_bound: Field,
                upper_bound: Field) -> tuple[bool, Field, Field]:
    """
    Compute the dot product of `vec0` and `vec1`, and check if it is in the
    range `[lower_bound, upper_bound]`, per Section 4.1 of PINE paper.
    """
    res = dot_prod(vec0, vec1)
    v = res - lower_bound
    u = upper_bound - res
    is_in_range = \
        v.as_unsigned() <= (upper_bound - lower_bound).as_unsigned()
    return (is_in_range, v, u)

def chunk_count(chunk_length, length):
    return (length + chunk_length - 1) // chunk_length

# Adapted from `FlpGeneric.test_flp_generic` to take joint randomness
# as part of a test case, because wraparound joint randomness
# is part of FLP encoding for PINE, and the same joint randomness
# needs to be passed to `run_wr_checks` and `eval` functions.
def test_flp_pine(flp, test_cases):
    for (i, (meas, joint_rand, expected_decision)) in enumerate(test_cases):
        assert len(meas) == flp.MEAS_LEN
        assert len(flp.truncate(meas)) == flp.OUTPUT_LEN

        # Evaluate validity circuit.
        v = flp.Valid.eval(meas, joint_rand, 1)
        if (v == flp.Field(0)) != expected_decision:
            print('validity test {} failed: validity circuit returned {}'.format(i, v))

        # Run the FLP.
        decision = run_flp_pine(flp, meas, joint_rand, 2)
        if decision != expected_decision:
            print('flp test {} failed: proof evaluation resulted in {}; '
                  'want {}'.format(
                i, decision, expected_decision
            ))


# Adapted from `Flp.run_flp` with joint randomness, because wraparound
# joint randomness is part of FLP encoding for PINE, and the same joint
# randomness needs to be passed to `run_wr_checks` and `run_flp`, instead
# of `run_flp` sampling joint randomness directly.
def run_flp_pine(flp,
                 meas: Vec[PineValid.Field],
                 joint_rand: Vec[PineValid.Field],
                 num_shares: Unsigned):
    """Run PINE FLP on an encoded measurement and joint randomness. """
    prove_rand = flp.Field.rand_vec(flp.PROVE_RAND_LEN)
    query_rand = flp.Field.rand_vec(flp.QUERY_RAND_LEN)

    # Prover generates the proof.
    proof = flp.prove(meas, prove_rand, joint_rand)

    # Shard the measurement and the proof.
    meas_shares = additive_secret_share(meas, num_shares, flp.Field)
    proof_shares = additive_secret_share(proof, num_shares, flp.Field)

    # Verifier queries the meas shares and proof shares.
    verifier_shares = [
        flp.query(
            meas_share,
            proof_share,
            query_rand,
            joint_rand,
            num_shares,
        )
        for meas_share, proof_share in zip(meas_shares, proof_shares)
    ]

    # Combine the verifier shares into the verifier.
    verifier = flp.Field.zeros(len(verifier_shares[0]))
    for verifier_share in verifier_shares:
        verifier = vec_add(verifier, verifier_share)

    # Verifier decides if the measurement is valid.
    return flp.decide(verifier)


def test_bit_chunks():
    buf = os.urandom(16)
    # Test that chunking the buffer with `num_chunk_bits` bits at a time and
    # joining them back together should output the original buffer.
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

def test_pine_valid_roundtrip():
    valid = PineValid(1.0, 15, 2, 1)
    f64_vals = [0.5, 0.5]
    assert f64_vals == valid.decode(valid.truncate(valid.encode(f64_vals)), 1)

def test():
    test_bit_chunks()
    test_pine_valid_roundtrip()

    l2_norm_bound = 1.0
    # 4 fractional bits should be enough to keep 1 decimal digit.
    num_frac_bits = 4
    dimension = 4
    # A gradient with a L2-norm of exactly 1.0.
    measurement = [l2_norm_bound / 2] * dimension
    pine_valid = PineValid(l2_norm_bound, num_frac_bits, dimension, 150)
    flp = FlpGeneric(pine_valid)

    # Test PINE FLP with verification.
    xof = XofShake128(gen_rand(16), b"", b"")
    wr_joint_rand = pine_valid.sample_wr_joint_rand(xof)
    partially_encoded = flp.encode(measurement)
    encoded = partially_encoded + \
        pine_valid.run_wr_checks(partially_encoded[:dimension],
                                 wr_joint_rand)

    vf_joint_rand = flp.Field.rand_vec(pine_valid.vf_joint_rand_len)
    joint_rand = wr_joint_rand + vf_joint_rand
    test_flp_pine(flp, [(encoded, joint_rand, True)])


if __name__ == '__main__':
    test()
