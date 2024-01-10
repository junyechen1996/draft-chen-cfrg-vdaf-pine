"""PINE VDAF. """

import os
import sys
from typing import Union

# Access poc folder in submoduled VDAF draft.
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "draft-irtf-cfrg-vdaf", "poc"))
import xof
from common import (Unsigned, byte, concat, front, gen_rand, to_be_bytes,
                    vec_add, vec_sub, zeros)
from field import Field, Field128, Field64
from flp_generic import FlpGeneric
from flp_pine import PineValid, NUM_WR_CHECKS, NUM_WR_SUCCESSES
from vdaf import Vdaf, test_vdaf
from vdaf_prio3 import (
    USAGE_MEAS_SHARE, USAGE_PROOF_SHARE, USAGE_JOINT_RANDOMNESS,
    USAGE_PROVE_RANDOMNESS, USAGE_QUERY_RANDOMNESS, USAGE_JOINT_RAND_SEED,
    USAGE_JOINT_RAND_PART
)

# Additional usage passed to domain separation tag for PINE VDAF, make sure
# they use distinct values from the ones in Prio3.
# Used to derive the wraparound joint randomness field elements from the Xof.
USAGE_WR_JOINT_RANDOMNESS = 8
# Used to derive the wraparound joint randomness seed from the parts.
USAGE_WR_JOINT_RAND_SEED = 9
# Used to derive each wraparound joint randomness seed part.
USAGE_WR_JOINT_RAND_PART = 10

# PINE draft version.
VERSION = 0


class Pine(Vdaf):
    """The Pine VDAF."""

    # Operational parameters set by user.
    Flp = FlpGeneric  # Set by `with_params()`. It is a `FlpGeneric` with a
                      # concrete `PineValid`.
    PROOFS = 1  # Set by `with_params()`, number of proofs to run with `Flp`.

    # Associated parameters for `Pine`.
    MEAS_LEN = None  # Set by `with_params()`, based on `Flp.MEAS_LEN`, minus
                     # the number of wraparound checks, because Clients
                     # don't send the dot products in wraparound checks.

    # Associated parameters required by `Vdaf`.
    ID = 0xFFFFFFFF  # Private codepoint that will be updated later.
    VERIFY_KEY_SIZE = PineValid.Xof.SEED_SIZE  # Set based on `Xof`.
    NONCE_SIZE = 16
    RAND_SIZE = None  # Computed from `Xof.SEED_SIZE` and `SHARES`
    ROUNDS = 1
    SHARES = None  # A number between `[2, 256)` set later

    # Associated types required by `Vdaf`.
    Measurement = PineValid.Measurement
    PublicShare = tuple[
        list[bytes],  # A list of wraparound joint randomness parts.
        list[bytes],  # A list of verification joint randomness parts.
    ]
    InputShare = tuple[
        Union[
            # Leader: expanded measurement share and proof share.
            tuple[list[Flp.Field], list[Flp.Field]],
            # Helper: seeds both measurement share and proof share.
            tuple[bytes, bytes]
        ],
        bytes,  # wraparound joint randomness blind
        bytes,  # verification joint randomness blind
    ]
    OutShare = list[Flp.Field]
    AggShare = list[Flp.Field]
    AggResult = PineValid.AggResult
    PrepShare = tuple[
        list[Flp.Field],  # verifier share
        bytes,            # wraparound joint randomness part
        bytes,            # verification joint randomness part
    ]
    PrepState = tuple[
        list[Flp.Field],  # output share
        bytes,            # corrected wraparound joint randomness seed
        bytes,            # corrected verification joint randomness seed
    ]
    # Joint randomness seed check for both wraparound joint randomness
    # and verification joint randomness.
    PrepMessage = tuple[bytes, bytes]

    @classmethod
    def with_params(Pine,
                    l2_norm_bound: float,
                    num_frac_bits: Unsigned,
                    dimension: Unsigned,
                    chunk_length: Unsigned,
                    num_shares: Unsigned,
                    field: Field,
                    num_proofs: Unsigned):
        class PineWithParams(Pine):
            # TODO(issue#39) Decide how many proofs to use and enforce
            # robustness.
            Flp = FlpGeneric(PineValid.with_field(field)(
                l2_norm_bound, num_frac_bits, dimension, chunk_length
            ))
            PROOFS = num_proofs
            MEAS_LEN = Flp.MEAS_LEN - NUM_WR_CHECKS
            # The size of randomness is the seed size times the sum of
            # the following:
            # - One prover randomness seed.
            # - One measurement share seed for each Helper.
            # - One proof share seed for each Helper.
            # - Two joint randomness seed blind for each Aggregator, one for
            #   wraparound check, one for verification.
            RAND_SIZE = (1 + 2 * (num_shares - 1) + 2 * num_shares) * \
                Flp.Valid.Xof.SEED_SIZE
            SHARES = num_shares
        return PineWithParams

    @classmethod
    def shard(Pine, measurement, nonce, rand):
        l = Pine.Flp.Valid.Xof.SEED_SIZE
        seeds = [rand[i:i+l] for i in range(0, Pine.RAND_SIZE, l)]

        meas = Pine.Flp.Valid.encode_gradient(measurement)
        assert len(meas) == Pine.Flp.Valid.encoded_gradient_len

        # Parse Helper seeds. Each Helper has 4 seeds:
        # - one for measurement share.
        # - one for proof share.
        # - one for wraparound joint randomness blind.
        # - one for verification joint randomness blind.
        # TODO(junyechen1996): We may be able to reuse the seed blind for all
        # shares, but needs security analysis. Related issue #185 in VDAF draft.
        num_helper_seeds = 4
        (k_helper_seeds, seeds) = front(
            (Pine.SHARES - 1) * num_helper_seeds, seeds
        )
        k_helper_meas_shares = [
            k_helper_seeds[i]
            for i in range(0, (Pine.SHARES - 1) * num_helper_seeds,
                           num_helper_seeds)
        ]
        k_helper_proofs_shares = [
            k_helper_seeds[i]
            for i in range(1, (Pine.SHARES - 1) * num_helper_seeds,
                           num_helper_seeds)
        ]
        k_helper_wr_joint_rand_blinds = [
            k_helper_seeds[i]
            for i in range(2, (Pine.SHARES - 1) * num_helper_seeds,
                           num_helper_seeds)
        ]
        k_helper_vf_joint_rand_blinds = [
            k_helper_seeds[i]
            for i in range(3, (Pine.SHARES - 1) * num_helper_seeds,
                           num_helper_seeds)
        ]
        # Parse leader seeds.
        (
            [k_leader_wr_joint_rand_blind, k_leader_vf_joint_rand_blind],
            seeds
        ) = front(2, seeds)
        ((k_prove,), seeds) = front(1, seeds)
        assert len(seeds) == 0  # sanity check

        # Compute wraparound joint randomness parts.
        (_, k_wr_joint_rand_parts) = Pine.leader_meas_share_and_joint_rand_parts(
            meas,
            k_helper_wr_joint_rand_blinds,
            k_helper_meas_shares,
            k_leader_wr_joint_rand_blind,
            nonce,
            USAGE_WR_JOINT_RAND_PART
        )
        # Initialize the `Xof` for wraparound joint randomness, with the seed
        # computed from the parts.
        wr_joint_rand_xof = Pine.wr_joint_rand_xof(
            Pine.joint_rand_seed(k_wr_joint_rand_parts,
                                 USAGE_WR_JOINT_RAND_SEED)
        )

        # Run wraparound checks with wraparound joint randomness XOF, and append
        # wraparound check results at the end of the encoded gradient.
        # Note Client doesn't send the dot products in wraparound checks,
        # because Aggregators are expected to derive the wrapround joint
        # randomness themselves, but the dot products are passed to
        # `Flp.Valid.eval()` later to avoid computing the dot products again.
        (wr_check_bits, wr_check_results) = \
            Pine.Flp.Valid.encode_wr_checks(meas, wr_joint_rand_xof)
        meas += wr_check_bits
        assert len(meas) == Pine.MEAS_LEN

        # Compute Leader's measurement share and verification joint randomness
        # parts.
        (leader_meas_share, k_vf_joint_rand_parts) = \
            Pine.leader_meas_share_and_joint_rand_parts(
                meas,
                k_helper_vf_joint_rand_blinds,
                k_helper_meas_shares,
                k_leader_vf_joint_rand_blind,
                nonce,
                USAGE_JOINT_RAND_PART
            )
        # Compute verification joint randomness field elements.
        vf_joint_rands = Pine.vf_joint_rands(Pine.joint_rand_seed(
            k_vf_joint_rand_parts, USAGE_JOINT_RAND_SEED,
        ))

        # Generate the proof and shard it into proof shares.
        meas += wr_check_results
        prove_rands = Pine.prove_rands(k_prove)
        leader_proofs_share = []
        for _ in range(Pine.PROOFS):
            (prove_rand, prove_rands) = \
                front(Pine.Flp.PROVE_RAND_LEN, prove_rands)
            (vf_joint_rand, vf_joint_rands) = \
                front(Pine.Flp.JOINT_RAND_LEN, vf_joint_rands)
            leader_proofs_share += Pine.Flp.prove(meas,
                                                  prove_rand,
                                                  vf_joint_rand)
        # Sanity check:
        assert len(prove_rands) == 0
        assert len(vf_joint_rands) == 0
        for j in range(Pine.SHARES-1):
            leader_proofs_share = vec_sub(
                leader_proofs_share,
                Pine.helper_proofs_share(j+1, k_helper_proofs_shares[j]),
            )

        # Each Aggregator's input share contains:
        # - its measurement share,
        # - its proof share,
        # - its wraparound joint randomness blind,
        # - its verification joint randomness blind.
        # The public share contains the joint randomness parts for both
        # wraparound joint randomness and verification joint randomness.
        input_shares = []
        input_shares.append((
            leader_meas_share,
            leader_proofs_share,
            k_leader_wr_joint_rand_blind,
            k_leader_vf_joint_rand_blind,
        ))
        for j in range(Pine.SHARES-1):
            input_shares.append((
                k_helper_meas_shares[j],
                k_helper_proofs_shares[j],
                k_helper_wr_joint_rand_blinds[j],
                k_helper_vf_joint_rand_blinds[j],
            ))
        return (
            (k_wr_joint_rand_parts, k_vf_joint_rand_parts), input_shares
        )

    @classmethod
    def is_valid(Pine, _agg_param, previous_agg_params):
        """
        Checks if `previous_agg_params` is empty, as input shares in Pine may
        only be used once.
        """
        return len(previous_agg_params) == 0

    @classmethod
    def prep_init(Pine,
                  verify_key,
                  agg_id,
                  _agg_param,
                  nonce,
                  public_share,
                  input_share):
        (k_wr_joint_rand_parts, k_vf_joint_rand_parts) = public_share
        (
            meas_share,
            proofs_share,
            k_wr_joint_rand_blind,
            k_vf_joint_rand_blind
        ) = Pine.expand_input_share(agg_id, input_share)
        out_share = Pine.Flp.truncate(meas_share)

        # Compute this Aggregator's wraparound joint randomness part and seed.
        # The part is exchanged with other Aggregators, and the seed is used to
        # expand into the wraparound joint randomness field elements.
        (k_wr_joint_rand_part, k_corrected_wr_joint_rand_seed) = \
            Pine.joint_rand_part_and_seed_for_aggregator(
                agg_id,
                k_wr_joint_rand_blind,
                meas_share[:Pine.Flp.Valid.encoded_gradient_len],
                nonce,
                k_wr_joint_rand_parts,
                USAGE_WR_JOINT_RAND_PART,
                USAGE_WR_JOINT_RAND_SEED,
            )
        # Now initialize the wraparound joint randomness XOF, in order to
        # compute the dot products in wraparound checks.
        wr_joint_rand_xof = \
            Pine.wr_joint_rand_xof(k_corrected_wr_joint_rand_seed)
        wr_dot_prods = Pine.Flp.Valid.run_wr_checks(meas_share,
                                                    wr_joint_rand_xof)

        # Compute this Aggregator's verification joint randomness part and seed.
        (k_vf_joint_rand_part, k_corrected_vf_joint_rand_seed) = \
            Pine.joint_rand_part_and_seed_for_aggregator(
                agg_id,
                k_vf_joint_rand_blind,
                meas_share,
                nonce,
                k_vf_joint_rand_parts,
                USAGE_JOINT_RAND_PART,
                USAGE_JOINT_RAND_SEED,
            )
        vf_joint_rands = Pine.vf_joint_rands(k_corrected_vf_joint_rand_seed)

        # Query the measurement and proof share.
        # `PineValid.eval()` expects the dot products for wraparound checks to be
        # appended at the end of Client's encoded measurement.
        flp_meas_share = meas_share + wr_dot_prods
        query_rands = Pine.query_rands(verify_key, nonce)
        verifiers_share = []
        for _ in range(Pine.PROOFS):
            (proof_share, proofs_share) = front(Pine.Flp.PROOF_LEN, proofs_share)
            (vf_joint_rand, vf_joint_rands) = \
                front(Pine.Flp.JOINT_RAND_LEN, vf_joint_rands)
            (query_rand, query_rands) = \
                front(Pine.Flp.QUERY_RAND_LEN, query_rands)
            verifiers_share += Pine.Flp.query(flp_meas_share,
                                              proof_share,
                                              query_rand,
                                              vf_joint_rand,
                                              Pine.SHARES)
        # Sanity check:
        assert len(proofs_share) == 0
        assert len(vf_joint_rands) == 0
        assert len(query_rands) == 0

        return (
            # Prepare state:
            (
                out_share,
                k_corrected_wr_joint_rand_seed,
                k_corrected_vf_joint_rand_seed
            ),
            # Prepare share that is exchanged with other Aggregators:
            (
                verifiers_share,
                k_wr_joint_rand_part,
                k_vf_joint_rand_part
            )
        )

    @classmethod
    def prep_shares_to_prep(Pine, _agg_param, prep_shares):
        # Unshard the verifier shares into the verifier message.
        verifiers = Pine.Flp.Field.zeros(Pine.Flp.VERIFIER_LEN * Pine.PROOFS)
        k_wr_joint_rand_parts = []
        k_vf_joint_rand_parts = []
        for (verifiers_share,
             k_wr_joint_rand_part,
             k_vf_joint_rand_part) in prep_shares:
            verifiers = vec_add(verifiers, verifiers_share)
            k_wr_joint_rand_parts.append(k_wr_joint_rand_part)
            k_vf_joint_rand_parts.append(k_vf_joint_rand_part)

        # Verify that each proof is well-formed and input is valid.
        for _ in range(Pine.PROOFS):
            (verifier, verifiers) = front(Pine.Flp.VERIFIER_LEN, verifiers)
            if not Pine.Flp.decide(verifier):
                # Proof verifier check failed.
                raise ValueError("Decision function failed after combining all "
                                 "verifier shares.")
        # Sanity check:
        assert len(verifiers) == 0

        # Combine the joint randomness parts computed by the Aggregators
        # into the true joint randomness seeds, which are checked by all
        # Aggregators.
        k_wr_joint_rand_seed = Pine.joint_rand_seed(
            k_wr_joint_rand_parts, USAGE_WR_JOINT_RAND_SEED,
        )
        k_vf_joint_rand_seed = Pine.joint_rand_seed(
            k_vf_joint_rand_parts, USAGE_JOINT_RAND_SEED,
        )
        return (k_wr_joint_rand_seed, k_vf_joint_rand_seed)

    @classmethod
    def prep_next(Pine, prep_state, prep_msg):
        # Joint randomness seeds sent by the Leader.
        (k_wr_joint_rand_seed, k_vf_joint_rand_seed) = prep_msg
        (
            out_share,
            k_corrected_wr_joint_rand_seed,
            k_corrected_vf_joint_rand_seed
        ) = prep_state

        # Make sure the seeds from the Leader are consistent with what the
        # current Aggregator sees.
        if (k_wr_joint_rand_seed != k_corrected_wr_joint_rand_seed or
            k_vf_joint_rand_seed != k_corrected_vf_joint_rand_seed):
            raise ValueError("Inconsistency between the seed from Leader and "
                             "the seed seen by the current Aggregator.")
        return out_share

    @classmethod
    def aggregate(Pine, _agg_param, out_shares):
        agg_share = Pine.Flp.Field.zeros(Pine.Flp.OUTPUT_LEN)
        for out_share in out_shares:
            agg_share = vec_add(agg_share, out_share)
        return agg_share

    @classmethod
    def unshard(Pine, _agg_param, agg_shares, num_measurements):
        agg = Pine.Flp.Field.zeros(Pine.Flp.OUTPUT_LEN)
        for agg_share in agg_shares:
            agg = vec_add(agg, agg_share)
        return Pine.Flp.decode(agg, num_measurements)

    @classmethod
    def domain_separation_tag(Pine, usage):
        return concat([
            to_be_bytes(VERSION, 1),
            to_be_bytes(Pine.ID, 4),
            to_be_bytes(usage, 2),
        ])

    # Helper functions:

    @classmethod
    def helper_meas_share(Pine,
                          agg_id: Unsigned,
                          k_share: bytes,
                          meas_len: Unsigned) -> list[Flp.Field]:
        """
        Generate the helper measurement share up to length `meas_len`,
        for aggregator ID `agg_id`, with measurement share seed `k_share`.
        """
        return Pine.Flp.Valid.Xof.expand_into_vec(
            Pine.Flp.Field,
            k_share,
            Pine.domain_separation_tag(USAGE_MEAS_SHARE),
            byte(agg_id),
            meas_len,
        )

    @classmethod
    def helper_proofs_share(Pine,
                            agg_id: Unsigned,
                            k_share: bytes) -> list[Flp.Field]:
        """
        Generate the helper proofs share for aggregator ID `agg_id`, with
        proof share seed `k_share`.
        """
        return Pine.Flp.Valid.Xof.expand_into_vec(
            Pine.Flp.Field,
            k_share,
            Pine.domain_separation_tag(USAGE_PROOF_SHARE),
            byte(Pine.PROOFS) + byte(agg_id),
            Pine.Flp.PROOF_LEN * Pine.PROOFS
        )

    @classmethod
    def expand_input_share(
        Pine,
        agg_id: Unsigned,
        input_share: InputShare,
    ) -> tuple[list[Flp.Field], list[Flp.Field], bytes, bytes]:
        """Expand Helper's seed into a vector of field elements. """
        (
            meas_share,
            proofs_share,
            k_wr_joint_rand_blind,
            k_vf_joint_rand_blind
        ) = input_share
        if agg_id > 0:
            meas_share = \
                Pine.helper_meas_share(agg_id, meas_share, Pine.MEAS_LEN)
            proofs_share = Pine.helper_proofs_share(agg_id, proofs_share)
        return (meas_share,
                proofs_share,
                k_wr_joint_rand_blind,
                k_vf_joint_rand_blind)

    @classmethod
    def prove_rands(Pine, k_prove: bytes) -> list[Flp.Field]:
        """Generate the prover randomness based on the seed blind `k_prove`."""
        return Pine.Flp.Valid.Xof.expand_into_vec(
            Pine.Flp.Field,
            k_prove,
            Pine.domain_separation_tag(USAGE_PROVE_RANDOMNESS),
            byte(Pine.PROOFS),
            Pine.Flp.PROVE_RAND_LEN * Pine.PROOFS
        )

    @classmethod
    def query_rands(Pine,
                    verify_key: bytes,
                    nonce: bytes) -> list[Flp.Field]:
        """
        Generate the query randomness based on the verification key and nonce.
        """
        return Pine.Flp.Valid.Xof.expand_into_vec(
            Pine.Flp.Field,
            verify_key,
            Pine.domain_separation_tag(USAGE_QUERY_RANDOMNESS),
            byte(Pine.PROOFS) + nonce,
            Pine.Flp.QUERY_RAND_LEN * Pine.PROOFS
        )

    @classmethod
    def joint_rand_part(Pine,
                        agg_id: Unsigned,
                        k_blind: bytes,
                        meas_share: list[Flp.Field],
                        nonce: bytes,
                        usage: Unsigned) -> bytes:
        """Derive joint randomness part for an Aggregator. """
        return Pine.Flp.Valid.Xof.derive_seed(
            k_blind,
            Pine.domain_separation_tag(usage),
            byte(agg_id) + nonce + Pine.Flp.Field.encode_vec(meas_share),
        )

    @classmethod
    def leader_meas_share_and_joint_rand_parts(
        Pine,
        encoded_measurement: list[Flp.Field],
        k_helper_joint_rand_blinds: list[bytes],
        k_helper_meas_shares: list[bytes],
        k_leader_joint_rand_blind: bytes,
        nonce: bytes,
        part_usage: Unsigned
    ) -> tuple[list[Flp.Field], list[bytes]]:
        """
        Return the leader measurement share and joint randomness parts with
        domain separation tag `part_usage`.
        This function shards the encoded measurement into shares and feed
        each sequence of secret shares into `Xof`, in order to compute all
        Aggregators' joint randomness parts.
        """
        leader_meas_share = encoded_measurement
        k_joint_rand_parts = []
        for j in range(Pine.SHARES-1):
            helper_meas_share = Pine.helper_meas_share(
                j+1, k_helper_meas_shares[j], len(encoded_measurement)
            )
            leader_meas_share = vec_sub(leader_meas_share, helper_meas_share)
            k_joint_rand_parts.append(Pine.joint_rand_part(
                j+1, k_helper_joint_rand_blinds[j], helper_meas_share, nonce,
                part_usage
            ))
        k_joint_rand_parts.insert(0, Pine.joint_rand_part(
            0, k_leader_joint_rand_blind, leader_meas_share, nonce, part_usage
        ))
        return (leader_meas_share, k_joint_rand_parts)

    @classmethod
    def joint_rand_seed(Pine,
                        k_joint_rand_parts: list[bytes],
                        usage: Unsigned) -> bytes:
        """
        Derive the joint randomness seed from its parts and based on the usage.
        """
        return Pine.Flp.Valid.Xof.derive_seed(
            zeros(Pine.Flp.Valid.Xof.SEED_SIZE),
            Pine.domain_separation_tag(usage),
            concat(k_joint_rand_parts),
        )

    @classmethod
    def joint_rand_part_and_seed_for_aggregator(
        Pine,
        agg_id: Unsigned,
        k_joint_rand_blind: bytes,
        meas_share: list[Flp.Field],
        nonce: bytes,
        k_joint_rand_parts: list[bytes],
        joint_rand_part_usage: Unsigned,
        joint_rand_seed_usage: Unsigned
    ) -> bytes:
        """
        The Aggregator `agg_id` computes the joint randomness seed. This is a
        common routine to compute both wraparound and verification joint
        randomness.
        The Aggregator first computes its joint randomness part, from the
        seed blind `k_joint_rand_blind`, and its secret share of the Client
        measurement `meas_share`.
        With the joint randomness parts for other Aggregators sent by the
        Client, the Aggregator computes the joint randomness seed.
        """
        k_joint_rand_part = Pine.joint_rand_part(
            agg_id, k_joint_rand_blind, meas_share, nonce, joint_rand_part_usage
        )
        k_joint_rand_parts[agg_id] = k_joint_rand_part
        k_corrected_joint_rand_seed = Pine.joint_rand_seed(
            k_joint_rand_parts, joint_rand_seed_usage
        )
        return (k_joint_rand_part, k_corrected_joint_rand_seed)

    @classmethod
    def wr_joint_rand_xof(Pine, k_wr_joint_rand_seed: bytes) -> PineValid.Xof:
        """Initialize the XOF to sample wraparound joint randomness. """
        return Pine.Flp.Valid.Xof(
            k_wr_joint_rand_seed,
            Pine.domain_separation_tag(USAGE_WR_JOINT_RANDOMNESS),
            b'',
        )

    @classmethod
    def vf_joint_rands(Pine,
                       k_joint_rand_seed: bytes) -> list[Flp.Field]:
        """
        Derive the verification joint randomness based on the initial seed.
        """
        return Pine.Flp.Valid.Xof.expand_into_vec(
            Pine.Flp.Field,
            k_joint_rand_seed,
            Pine.domain_separation_tag(USAGE_JOINT_RANDOMNESS),
            byte(Pine.PROOFS),
            Pine.Flp.JOINT_RAND_LEN * Pine.PROOFS
        )

    # Methods for generating test vectors:

    @classmethod
    def test_vec_encode_input_share(Pine, input_share):
        (
            meas_share,
            proofs_share,
            k_wr_joint_rand_blind,
            k_vf_joint_rand_blind
        ) = input_share
        encoded = bytes()
        if type(meas_share) == list and type(proofs_share) == list:  # leader
            encoded += Pine.Flp.Field.encode_vec(meas_share)
            encoded += Pine.Flp.Field.encode_vec(proofs_share)
        elif type(meas_share) == bytes and type(proofs_share) == bytes:  # helper
            encoded += meas_share
            encoded += proofs_share
        return encoded + k_wr_joint_rand_blind + k_vf_joint_rand_blind

    @classmethod
    def test_vec_encode_public_share(Pine, public_share):
        (k_wr_joint_rand_parts, k_vf_joint_rand_parts) = public_share
        return concat(k_wr_joint_rand_parts) + concat(k_vf_joint_rand_parts)

    @classmethod
    def test_vec_encode_agg_share(Pine, agg_share):
        return Pine.Flp.Field.encode_vec(agg_share)

    @classmethod
    def test_vec_encode_prep_share(Pine, prep_share):
        (verifier_share, k_wr_joint_rand_part, k_vf_joint_rand_part) = \
            prep_share
        return Pine.Flp.Field.encode_vec(verifier_share) + \
            k_wr_joint_rand_part + k_vf_joint_rand_part

    @classmethod
    def test_vec_encode_prep_msg(Pine, prep_message):
        (k_wr_joint_rand_seed, k_vf_joint_rand_seed) = prep_message
        return k_wr_joint_rand_seed + k_vf_joint_rand_seed


# Tests:
def test_shard_result_share_length(Vdaf: Pine):
    """Check the result shares of `shard()` have the expected lengths. """
    measurement = [0.0] * Vdaf.Flp.Valid.dimension
    nonce = gen_rand(Vdaf.NONCE_SIZE)
    rand = gen_rand(Vdaf.RAND_SIZE)
    (public_share, input_shares) = Vdaf.shard(measurement, nonce, rand)
    assert public_share is not None
    assert input_shares is not None and len(input_shares) == Vdaf.SHARES

    [wr_joint_rand_parts, vf_joint_rand_parts] = public_share
    assert len(wr_joint_rand_parts) == Vdaf.SHARES
    assert len(vf_joint_rand_parts) == Vdaf.SHARES
    assert(all(len(part) == Vdaf.Flp.Valid.Xof.SEED_SIZE
               for part in wr_joint_rand_parts))
    assert(all(len(part) == Vdaf.Flp.Valid.Xof.SEED_SIZE
               for part in vf_joint_rand_parts))

    # Check leader share length.
    (meas_share, proofs_share, wr_joint_rand_blind, vf_joint_rand_blind) = \
        input_shares[0]
    assert len(meas_share) == Vdaf.MEAS_LEN
    assert len(proofs_share) == Vdaf.Flp.PROOF_LEN * Vdaf.PROOFS

if __name__ == '__main__':
    usages = [USAGE_MEAS_SHARE, USAGE_PROOF_SHARE, USAGE_JOINT_RANDOMNESS,
              USAGE_PROVE_RANDOMNESS, USAGE_QUERY_RANDOMNESS,
              USAGE_JOINT_RAND_SEED, USAGE_JOINT_RAND_PART]
    if usages != list(range(1, len(usages) + 1)):
        raise ValueError("Expect Prio3's usage string in domain separation "
                         "tag to be unique from 1 to " + str(len(usages)) + ".")

    # Check `Pine.domain_separation_tag` output length: 1 byte for draft
    # version, 4 bytes for algorithm ID, 2 bytes for usage string.
    assert(len(Pine.domain_separation_tag(0)) == 7)

    # Instantiate `Pine` with different field sizes and number of proofs, but
    # with the same user parameters:
    # `l2_norm_bound = 1.0`, `num_frac_bits = 4`, `dimension = 4`,
    # `chunk_length = 150`, `num_shares = 2`.
    args = [1.0, 4, 4, 150, 2]

    # Test happy cases.
    for (field, num_proofs) in [(Field64, 2), (Field128, 1)]:
        concrete_pine = Pine.with_params(*args, field, num_proofs)
        assert concrete_pine.Flp.Field == field
        assert concrete_pine.PROOFS == num_proofs
        test_shard_result_share_length(concrete_pine)
        test_vdaf(
            concrete_pine,
            None,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            [1.0, 1.0, 0.0, 0.0],
        )
