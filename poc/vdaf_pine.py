"""PINE VDAF. """

import os
from typing import Generic, TypeAlias, TypeVar, Union, cast

from vdaf_poc.common import (byte, concat, front, to_be_bytes, vec_add,
                             vec_sub, zeros)
from vdaf_poc.field import Field64, Field128, NttField
from vdaf_poc.flp_bbcggi19 import FlpBBCGGI19
from vdaf_poc.vdaf import Vdaf
from vdaf_poc.vdaf_prio3 import (USAGE_JOINT_RAND_PART, USAGE_JOINT_RAND_SEED,
                                 USAGE_JOINT_RANDOMNESS, USAGE_MEAS_SHARE,
                                 USAGE_PROOF_SHARE, USAGE_PROVE_RANDOMNESS,
                                 USAGE_QUERY_RANDOMNESS)
from vdaf_poc.xof import Xof, XofTurboShake128

from field32 import Field32
from flp_pine import (ALPHA, NUM_WR_CHECKS, NUM_WR_SUCCESSES, PineValid,
                      construct_circuits)
from xof_hmac_sha256_aes128 import XofHmacSha256Aes128

F = TypeVar("F", bound=NttField)
X = TypeVar("X", bound=Xof)

# Additional usage passed to domain separation tag for PINE VDAF, make sure
# they use distinct values from the ones in Prio3.
# Used to derive the wraparound joint randomness field elements from the Xof.
USAGE_WR_JOINT_RANDOMNESS = 8
# Used to derive the wraparound joint randomness seed from the parts.
USAGE_WR_JOINT_RAND_SEED = 9
# Used to derive each wraparound joint randomness seed part.
USAGE_WR_JOINT_RAND_PART = 10

# PINE draft version.
dir_name = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dir_name, "VERSION")) as f:
    VERSION = int(f.read())
    # Sanity check it can be represented with one byte.
    assert VERSION >= 0 and VERSION <= 255

PinePublicShare: TypeAlias = tuple[
    list[bytes],  # wr joint randomness parts
    list[bytes],  # vf joint randomness parts
]
PineInputShare: TypeAlias = Union[
    tuple[  # leader input share
        list[F],  # measurement share
        list[F],  # proof share
        bytes,  # wr joint randomness blind
        bytes,  # vf joint randomness blind
    ],
    tuple[  # helper input share
        bytes,  # measurement share seed
        bytes,  # proof share seed
        bytes,  # wr joint randomness blind
        bytes,  # vf joint randomness blind
    ],
]
PinePrepState: TypeAlias = tuple[
    list[F],  # output share
    bytes,  # corrected wr joint randomness seed
    bytes,  # corrected vf joint randomness seed
]
PinePrepShare: TypeAlias = tuple[
    list[F],  # verifier share
    bytes,  # wr joint randomness part
    bytes,  # vf joint randomness part
]


class Pine(
        Generic[F, X],
        Vdaf[
            list[float],  # Measurement
            None,  # AggParam
            PinePublicShare,  # PublicShare
            PineInputShare[F],  # InputShare
            list[F],  # OutShare
            list[F],  # AggShare
            list[float],  # AggResult
            PinePrepState[F],  # PrepState
            PinePrepShare[F],  # PrepShare
            bytes,  # PrepMessage, joint randomness seed check
        ]):
    """The Pine VDAF."""
    test_vec_name = "Pine"

    NONCE_SIZE = 16
    ROUNDS = 1

    # Parameters set by constructor.
    valid: PineValid[F]
    flp_norm_equality: FlpBBCGGI19[list[float], list[float], F]
    flp: FlpBBCGGI19[list[float], list[float], F]
    xof: type[X]
    PROOFS: int
    PROOFS_NORM_EQUALITY: int
    MEAS_LEN: int
    RAND_SIZE: int
    SHARES: int

    # Operational parameters for generating test vectors.
    ID = 0xffff_ffff

    def __init__(self,
                 field: type[F],
                 xof: type[X],
                 l2_norm_bound: int,
                 num_frac_bits: int,
                 dimension: int,
                 chunk_length: int,
                 chunk_length_norm_equality: int,
                 num_shares: int,
                 num_proofs: int,
                 num_proofs_norm_equality: int,
                 alpha: float = ALPHA,
                 num_wr_checks: int = NUM_WR_CHECKS,
                 num_wr_successes: int = NUM_WR_SUCCESSES,
                 vdaf_id: int = ID):
        (valid_norm_equality, self.valid) = construct_circuits(
            field=field,
            l2_norm_bound=l2_norm_bound,
            num_frac_bits=num_frac_bits,
            dimension=dimension,
            chunk_length=chunk_length,
            chunk_length_norm_equality=chunk_length_norm_equality,
            alpha=alpha,
            num_wr_checks=num_wr_checks,
            num_wr_successes=num_wr_successes,
        )
        self.flp_norm_equality = FlpBBCGGI19(valid_norm_equality)
        self.flp = FlpBBCGGI19(self.valid)
        self.xof = xof
        self.PROOFS = num_proofs
        self.PROOFS_NORM_EQUALITY = num_proofs_norm_equality
        self.MEAS_LEN = self.flp.MEAS_LEN - num_wr_checks
        self.ID = vdaf_id
        self.VERIFY_KEY_SIZE = self.xof.SEED_SIZE
        # The size of randomness is the seed size times the sum of
        # the following:
        # - One prover randomness seed.
        # - One measurement share seed for each Helper.
        # - One proof share seed for each Helper.
        # - Two joint randomness seed blind for each Aggregator, one for
        #   wraparound check, one for verification.
        self.RAND_SIZE = (1 + 2 * (num_shares - 1) + 2 * num_shares) * \
            self.xof.SEED_SIZE
        self.SHARES = num_shares

    def shard(self,
              measurement: list[float],
              nonce: bytes,
              rand: bytes
              ) -> tuple[PinePublicShare, list[PineInputShare]]:
        l = self.xof.SEED_SIZE
        seeds = [rand[i:i + l] for i in range(0, self.RAND_SIZE, l)]

        meas = self.valid.encode_gradient_and_norm(measurement)
        assert len(meas) == self.valid.encoded_gradient_and_norm_len

        # Parse Helper seeds. Each Helper has 4 seeds:
        # - one for measurement share.
        # - one for proof share.
        # - one for wraparound joint randomness blind.
        # - one for verification joint randomness blind.
        # TODO(junyechen1996): We may be able to reuse the seed blind for all
        # shares, but needs security analysis. Related issue #185 in VDAF draft.
        num_helper_seeds = 4
        (k_helper_seeds, seeds) = front(
            (self.SHARES - 1) * num_helper_seeds, seeds
        )
        k_helper_meas_shares = [
            k_helper_seeds[i]
            for i in range(0, (self.SHARES - 1) * num_helper_seeds,
                           num_helper_seeds)
        ]
        k_helper_proofs_shares = [
            k_helper_seeds[i]
            for i in range(1, (self.SHARES - 1) * num_helper_seeds,
                           num_helper_seeds)
        ]
        k_helper_wr_joint_rand_blinds = [
            k_helper_seeds[i]
            for i in range(2, (self.SHARES - 1) * num_helper_seeds,
                           num_helper_seeds)
        ]
        k_helper_vf_joint_rand_blinds = [
            k_helper_seeds[i]
            for i in range(3, (self.SHARES - 1) * num_helper_seeds,
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
        (_, k_wr_joint_rand_parts) = self.leader_meas_share_and_joint_rand_parts(
            meas,
            k_helper_wr_joint_rand_blinds,
            k_helper_meas_shares,
            k_leader_wr_joint_rand_blind,
            nonce,
            USAGE_WR_JOINT_RAND_PART
        )
        # Initialize the `Xof` for wraparound joint randomness, with the seed
        # computed from the parts.
        wr_joint_rand_xof = self.wr_joint_rand_xof(
            self.joint_rand_seed(k_wr_joint_rand_parts,
                                 USAGE_WR_JOINT_RAND_SEED)
        )

        # Run wraparound checks with wraparound joint randomness XOF, and append
        # wraparound check results at the end of the encoded gradient and norm.
        # Note Client doesn't send the dot products in wraparound checks,
        # because Aggregators are expected to derive the wrapround joint
        # randomness themselves, but the dot products are passed to
        # `Flp.Valid.eval()` later to avoid computing the dot products again.
        (wr_check_bits, wr_check_results) = \
            self.valid.encode_wr_checks(meas[:self.valid.dimension],
                                        wr_joint_rand_xof)
        meas += wr_check_bits
        assert len(meas) == self.MEAS_LEN

        # Compute Leader's measurement share and verification joint randomness
        # parts.
        (leader_meas_share, k_vf_joint_rand_parts) = \
            self.leader_meas_share_and_joint_rand_parts(
                meas,
                k_helper_vf_joint_rand_blinds,
                k_helper_meas_shares,
                k_leader_vf_joint_rand_blind,
                nonce,
                USAGE_JOINT_RAND_PART
        )
        # Compute verification joint randomness field elements.
        vf_joint_rands = self.vf_joint_rands(self.joint_rand_seed(
            k_vf_joint_rand_parts, USAGE_JOINT_RAND_SEED,
        ))

        # Generate the proof and shard it into proof shares.
        meas += wr_check_results
        prove_rands = self.prove_rands(k_prove)
        leader_proofs_share = []

        # Proofs for norm equality check.
        for _ in range(self.PROOFS_NORM_EQUALITY):
            (prove_rand, prove_rands) = \
                front(self.flp_norm_equality.PROVE_RAND_LEN, prove_rands)
            leader_proofs_share += self.flp_norm_equality.prove(
                meas, prove_rand, [])
        # Proofs for all other checks.
        for _ in range(self.PROOFS):
            (prove_rand, prove_rands) = \
                front(self.flp.PROVE_RAND_LEN, prove_rands)
            (vf_joint_rand, vf_joint_rands) = \
                front(self.flp.JOINT_RAND_LEN, vf_joint_rands)
            leader_proofs_share += self.flp.prove(meas,
                                                  prove_rand,
                                                  vf_joint_rand)
        # Sanity check:
        assert len(prove_rands) == 0
        assert len(vf_joint_rands) == 0
        for j in range(self.SHARES - 1):
            leader_proofs_share = vec_sub(
                leader_proofs_share,
                self.helper_proofs_share(j + 1, k_helper_proofs_shares[j]),
            )

        # Each Aggregator's input share contains:
        # - its measurement share,
        # - its proof share,
        # - its wraparound joint randomness blind,
        # - its verification joint randomness blind.
        # The public share contains the joint randomness parts for both
        # wraparound joint randomness and verification joint randomness.
        input_shares: list[PineInputShare] = []
        input_shares.append((
            leader_meas_share,
            leader_proofs_share,
            k_leader_wr_joint_rand_blind,
            k_leader_vf_joint_rand_blind,
        ))
        for j in range(self.SHARES - 1):
            input_shares.append((
                k_helper_meas_shares[j],
                k_helper_proofs_shares[j],
                k_helper_wr_joint_rand_blinds[j],
                k_helper_vf_joint_rand_blinds[j],
            ))
        return (
            (k_wr_joint_rand_parts, k_vf_joint_rand_parts), input_shares
        )

    def is_valid(self, _agg_param, previous_agg_params):
        """
        Checks if `previous_agg_params` is empty, as input shares in Pine may
        only be used once.
        """
        return len(previous_agg_params) == 0

    def prep_init(self,
                  verify_key: bytes,
                  agg_id: int,
                  _agg_param: None,
                  nonce: bytes,
                  public_share: PinePublicShare,
                  input_share: PineInputShare):
        (k_wr_joint_rand_parts, k_vf_joint_rand_parts) = public_share
        (
            meas_share,
            proofs_share,
            k_wr_joint_rand_blind,
            k_vf_joint_rand_blind
        ) = self.expand_input_share(agg_id, input_share)
        out_share = self.flp.truncate(meas_share)

        # Compute this Aggregator's wraparound joint randomness part and seed.
        # The part is exchanged with other Aggregators, and the seed is used to
        # expand into the wraparound joint randomness field elements.
        (k_wr_joint_rand_part, k_corrected_wr_joint_rand_seed) = \
            self.joint_rand_part_and_seed_for_aggregator(
                agg_id,
                k_wr_joint_rand_blind,
                meas_share[:self.valid.encoded_gradient_and_norm_len],
                nonce,
                k_wr_joint_rand_parts,
                USAGE_WR_JOINT_RAND_PART,
                USAGE_WR_JOINT_RAND_SEED,
        )
        # Now initialize the wraparound joint randomness XOF, in order to
        # compute the dot products in wraparound checks.
        wr_joint_rand_xof = \
            self.wr_joint_rand_xof(k_corrected_wr_joint_rand_seed)
        wr_check_results = self.valid.run_wr_checks(
            meas_share[:self.valid.dimension], wr_joint_rand_xof
        )

        # Compute this Aggregator's verification joint randomness part and seed.
        (k_vf_joint_rand_part, k_corrected_vf_joint_rand_seed) = \
            self.joint_rand_part_and_seed_for_aggregator(
                agg_id,
                k_vf_joint_rand_blind,
                meas_share,
                nonce,
                k_vf_joint_rand_parts,
                USAGE_JOINT_RAND_PART,
                USAGE_JOINT_RAND_SEED,
        )
        vf_joint_rands = self.vf_joint_rands(k_corrected_vf_joint_rand_seed)

        # Query the measurement and proof share.
        # `PineValid.eval()` expects the wraparound check results (i.e., the dot
        # products) to be appended at the end of Client's encoded measurement.
        flp_meas_share = meas_share + wr_check_results
        query_rands = self.query_rands(verify_key, nonce)
        verifiers_share = []
        # Query the proofs for norm equality check.
        for _ in range(self.PROOFS_NORM_EQUALITY):
            (proof_share, proofs_share) = front(
                self.flp_norm_equality.PROOF_LEN, proofs_share)
            (query_rand, query_rands) = \
                front(self.flp_norm_equality.QUERY_RAND_LEN, query_rands)
            verifiers_share += self.flp_norm_equality.query(flp_meas_share,
                                                            proof_share,
                                                            query_rand,
                                                            [],
                                                            self.SHARES)
        # Query the proofs for all other checks.
        for _ in range(self.PROOFS):
            (proof_share, proofs_share) = front(
                self.flp.PROOF_LEN, proofs_share)
            (vf_joint_rand, vf_joint_rands) = \
                front(self.flp.JOINT_RAND_LEN, vf_joint_rands)
            (query_rand, query_rands) = \
                front(self.flp.QUERY_RAND_LEN, query_rands)
            verifiers_share += self.flp.query(flp_meas_share,
                                              proof_share,
                                              query_rand,
                                              vf_joint_rand,
                                              self.SHARES)
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

    def prep_shares_to_prep(self, _agg_param: None,
                            prep_shares: list[PinePrepShare[F]]):
        # Unshard the verifier shares into the verifier message.
        verifiers = self.flp.field.zeros(
            self.flp_norm_equality.VERIFIER_LEN * self.PROOFS_NORM_EQUALITY +
            self.flp.VERIFIER_LEN * self.PROOFS)
        k_wr_joint_rand_parts = []
        k_vf_joint_rand_parts = []
        for (verifiers_share,
             k_wr_joint_rand_part,
             k_vf_joint_rand_part) in prep_shares:
            verifiers = vec_add(verifiers, verifiers_share)
            k_wr_joint_rand_parts.append(k_wr_joint_rand_part)
            k_vf_joint_rand_parts.append(k_vf_joint_rand_part)

        # Verify that each proof is well-formed and input is valid.
        for _ in range(self.PROOFS_NORM_EQUALITY):
            (verifier, verifiers) = front(
                self.flp_norm_equality.VERIFIER_LEN, verifiers)
            if not self.flp_norm_equality.decide(verifier):
                # Proof verifier check failed.
                raise ValueError("Decision function for norm equality check "
                                 "failed after combining all verifier shares.")
        for _ in range(self.PROOFS):
            (verifier, verifiers) = front(self.flp.VERIFIER_LEN, verifiers)
            if not self.flp.decide(verifier):
                # Proof verifier check failed.
                raise ValueError("Decision function for other checks failed "
                                 "after combining all verifier shares.")
        # Sanity check:
        assert len(verifiers) == 0

        # Combine the joint randomness parts computed by the Aggregators
        # into the true joint randomness seeds, which are checked by all
        # Aggregators.
        k_wr_joint_rand_seed = self.joint_rand_seed(
            k_wr_joint_rand_parts, USAGE_WR_JOINT_RAND_SEED,
        )
        k_vf_joint_rand_seed = self.joint_rand_seed(
            k_vf_joint_rand_parts, USAGE_JOINT_RAND_SEED,
        )
        return (k_wr_joint_rand_seed, k_vf_joint_rand_seed)

    def prep_next(self, prep_state: PinePrepState[F], prep_msg: bytes):
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

    def aggregate(self, _agg_param, out_shares):
        agg_share = self.flp.field.zeros(
            self.flp.OUTPUT_LEN)
        for out_share in out_shares:
            agg_share = vec_add(agg_share, out_share)
        return agg_share

    def unshard(self, _agg_param, agg_shares, num_measurements):
        agg = self.flp.field.zeros(
            self.flp.OUTPUT_LEN)
        for agg_share in agg_shares:
            agg = vec_add(agg, agg_share)
        return self.flp.decode(agg, num_measurements)

    def domain_separation_tag(self, usage):
        return concat([
            to_be_bytes(VERSION, 1),
            to_be_bytes(self.ID, 4),
            to_be_bytes(usage, 2),
        ])

    # Helper functions:

    def helper_meas_share(self,
                          agg_id: int,
                          k_share: bytes,
                          meas_len: int) -> list[F]:
        """
        Generate the helper measurement share up to length `meas_len`,
        for aggregator ID `agg_id`, with measurement share seed `k_share`.
        """
        return self.xof.expand_into_vec(
            self.flp.field,
            k_share,
            self.domain_separation_tag(USAGE_MEAS_SHARE),
            byte(agg_id),
            meas_len,
        )

    def helper_proofs_share(self,
                            agg_id: int,
                            k_share: bytes) -> list[F]:
        """
        Generate the helper proofs share for aggregator ID `agg_id`, with
        proof share seed `k_share`.
        """
        return self.xof.expand_into_vec(
            self.flp.field,
            k_share,
            self.domain_separation_tag(USAGE_PROOF_SHARE),
            byte(self.PROOFS_NORM_EQUALITY) + byte(self.PROOFS) + byte(agg_id),
            self.flp_norm_equality.PROOF_LEN * self.PROOFS_NORM_EQUALITY +
            self.flp.PROOF_LEN * self.PROOFS
        )

    def expand_input_share(
        self,
        agg_id: int,
        input_share: PineInputShare[F],
    ) -> tuple[list[F], list[F], bytes, bytes]:
        """Expand Helper's seed into a vector of field elements. """
        if agg_id > 0:
            (
                k_meas_share,
                k_proofs_share,
                k_wr_joint_rand_blind,
                k_vf_joint_rand_blind,
            ) = input_share
            return (
                self.helper_meas_share(agg_id, cast(
                    bytes, k_meas_share), self.MEAS_LEN),
                self.helper_proofs_share(agg_id, cast(bytes, k_proofs_share)),
                k_wr_joint_rand_blind,
                k_vf_joint_rand_blind,
            )
        return cast(tuple[list[F], list[F], bytes, bytes], input_share)

    def prove_rands(self, k_prove: bytes) -> list[F]:
        """Generate the prover randomness based on the seed blind `k_prove`."""
        return self.xof.expand_into_vec(
            self.flp.field,
            k_prove,
            self.domain_separation_tag(USAGE_PROVE_RANDOMNESS),
            byte(self.PROOFS_NORM_EQUALITY) + byte(self.PROOFS),
            self.flp_norm_equality.PROVE_RAND_LEN * self.PROOFS_NORM_EQUALITY +
            self.flp.PROVE_RAND_LEN * self.PROOFS
        )

    def query_rands(self,
                    verify_key: bytes,
                    nonce: bytes) -> list[F]:
        """
        Generate the query randomness based on the verification key and nonce.
        """
        return self.xof.expand_into_vec(
            self.flp.field,
            verify_key,
            self.domain_separation_tag(USAGE_QUERY_RANDOMNESS),
            byte(self.PROOFS_NORM_EQUALITY) + byte(self.PROOFS) + nonce,
            self.flp_norm_equality.QUERY_RAND_LEN * self.PROOFS_NORM_EQUALITY +
            self.flp.QUERY_RAND_LEN * self.PROOFS
        )

    def joint_rand_part(self,
                        agg_id: int,
                        k_blind: bytes,
                        meas_share: list[F],
                        nonce: bytes,
                        usage: int) -> bytes:
        """Derive joint randomness part for an Aggregator. """
        return self.xof.derive_seed(
            k_blind,
            self.domain_separation_tag(usage),
            byte(agg_id) + nonce + self.flp.field.encode_vec(meas_share),
        )

    def leader_meas_share_and_joint_rand_parts(
        self,
        encoded_measurement: list[F],
        k_helper_joint_rand_blinds: list[bytes],
        k_helper_meas_shares: list[bytes],
        k_leader_joint_rand_blind: bytes,
        nonce: bytes,
        part_usage: int,
    ) -> tuple[list[F], list[bytes]]:
        """
        Return the leader measurement share and joint randomness parts with
        domain separation tag `part_usage`.
        This function shards the encoded measurement into shares and feed
        each sequence of secret shares into `Xof`, in order to compute all
        Aggregators' joint randomness parts.
        """
        leader_meas_share = encoded_measurement
        k_joint_rand_parts = []
        for j in range(self.SHARES - 1):
            helper_meas_share = self.helper_meas_share(
                j + 1, k_helper_meas_shares[j], len(encoded_measurement)
            )
            leader_meas_share = vec_sub(leader_meas_share, helper_meas_share)
            k_joint_rand_parts.append(self.joint_rand_part(
                j + 1, k_helper_joint_rand_blinds[j], helper_meas_share, nonce,
                part_usage
            ))
        k_joint_rand_parts.insert(0, self.joint_rand_part(
            0, k_leader_joint_rand_blind, leader_meas_share, nonce, part_usage
        ))
        return (leader_meas_share, k_joint_rand_parts)

    def joint_rand_seed(self,
                        k_joint_rand_parts: list[bytes],
                        usage: int) -> bytes:
        """
        Derive the joint randomness seed from its parts and based on the usage.
        """
        return self.xof.derive_seed(
            zeros(self.xof.SEED_SIZE),
            self.domain_separation_tag(usage),
            concat(k_joint_rand_parts),
        )

    def joint_rand_part_and_seed_for_aggregator(
        self,
        agg_id: int,
        k_joint_rand_blind: bytes,
        meas_share: list[F],
        nonce: bytes,
        k_joint_rand_parts: list[bytes],
        joint_rand_part_usage: int,
        joint_rand_seed_usage: int,
    ) -> tuple[bytes, bytes]:
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
        k_joint_rand_part = self.joint_rand_part(
            agg_id, k_joint_rand_blind, meas_share, nonce, joint_rand_part_usage
        )
        k_joint_rand_parts[agg_id] = k_joint_rand_part
        k_corrected_joint_rand_seed = self.joint_rand_seed(
            k_joint_rand_parts, joint_rand_seed_usage
        )
        return (k_joint_rand_part, k_corrected_joint_rand_seed)

    def wr_joint_rand_xof(self, k_wr_joint_rand_seed: bytes) -> X:
        """Initialize the XOF to sample wraparound joint randomness. """
        return self.xof(
            k_wr_joint_rand_seed,
            self.domain_separation_tag(USAGE_WR_JOINT_RANDOMNESS),
            b'',
        )

    def vf_joint_rands(self,
                       k_joint_rand_seed: bytes) -> list[F]:
        """
        Derive the verification joint randomness based on the initial seed.
        """
        return self.xof.expand_into_vec(
            self.flp.field,
            k_joint_rand_seed,
            self.domain_separation_tag(USAGE_JOINT_RANDOMNESS),
            byte(self.PROOFS),
            self.flp.JOINT_RAND_LEN * self.PROOFS
        )

    # Methods for generating test vectors:

    def test_vec_set_type_param(self, test_vec):
        params = self.flp_norm_equality.test_vec_set_type_param(
            test_vec)
        params += self.flp.test_vec_set_type_param(test_vec)
        test_vec["proofs"] = self.PROOFS
        test_vec["proofs_norm_equality"] = self.PROOFS_NORM_EQUALITY
        return params + ["proofs", "proofs_norm_equality"]

    def test_vec_encode_input_share(self, input_share):
        (
            meas_share,
            proofs_share,
            k_wr_joint_rand_blind,
            k_vf_joint_rand_blind
        ) = input_share
        encoded = bytes()

        if type(meas_share) == list and type(proofs_share) == list:  # leader
            encoded += self.flp.field.encode_vec(meas_share)
            encoded += self.flp.field.encode_vec(proofs_share)
        elif type(meas_share) == bytes and type(proofs_share) == bytes:  # helper
            encoded += meas_share
            encoded += proofs_share
        return encoded + k_wr_joint_rand_blind + k_vf_joint_rand_blind

    def test_vec_encode_public_share(self, public_share):
        (k_wr_joint_rand_parts, k_vf_joint_rand_parts) = public_share
        return concat(k_wr_joint_rand_parts) + concat(k_vf_joint_rand_parts)

    def test_vec_encode_agg_share(self, agg_share):
        return self.flp.field.encode_vec(agg_share)

    def test_vec_encode_prep_share(self, prep_share):
        (verifier_share, k_wr_joint_rand_part, k_vf_joint_rand_part) = \
            prep_share
        return self.flp.field.encode_vec(verifier_share) + \
            k_wr_joint_rand_part + k_vf_joint_rand_part

    def test_vec_encode_prep_msg(self, prep_message):
        (k_wr_joint_rand_seed, k_vf_joint_rand_seed) = prep_message
        return k_wr_joint_rand_seed + k_vf_joint_rand_seed


class Pine128(Pine[Field128, XofTurboShake128]):
    test_vec_name = "Pine128"

    def __init__(self,
                 l2_norm_bound: int,
                 num_frac_bits: int,
                 dimension: int,
                 chunk_length: int,
                 chunk_length_norm_equality,
                 num_shares: int,
                 alpha: float = ALPHA,
                 num_wr_checks: int = NUM_WR_CHECKS,
                 num_wr_successes: int = NUM_WR_SUCCESSES):
        return super().__init__(
            Field128,
            XofTurboShake128,
            l2_norm_bound=l2_norm_bound,
            num_frac_bits=num_frac_bits,
            dimension=dimension,
            chunk_length=chunk_length,
            chunk_length_norm_equality=chunk_length_norm_equality,
            num_shares=num_shares,
            num_proofs=1,
            num_proofs_norm_equality=1,
            alpha=alpha,
            num_wr_checks=num_wr_checks,
            num_wr_successes=num_wr_successes
        )


class Pine64(Pine[Field64, XofTurboShake128]):
    test_vec_name = "Pine64"

    def __init__(self,
                 l2_norm_bound: int,
                 num_frac_bits: int,
                 dimension: int,
                 chunk_length: int,
                 chunk_length_norm_equality: int,
                 num_shares: int,
                 alpha: float = ALPHA,
                 num_wr_checks: int = NUM_WR_CHECKS,
                 num_wr_successes: int = NUM_WR_SUCCESSES):
        return super().__init__(
            Field64,
            XofTurboShake128,
            l2_norm_bound=l2_norm_bound,
            num_frac_bits=num_frac_bits,
            dimension=dimension,
            chunk_length=chunk_length,
            chunk_length_norm_equality=chunk_length_norm_equality,
            num_shares=num_shares,
            num_proofs=2,
            num_proofs_norm_equality=1,
            alpha=alpha,
            num_wr_checks=num_wr_checks,
            num_wr_successes=num_wr_successes
        )


class Pine64HmacSha256Aes128(Pine[Field64, XofHmacSha256Aes128]):
    test_vec_name = "Pine64HmacSha256Aes128"
    ID = 0xFFFF1004

    def __init__(self,
                 l2_norm_bound: int,
                 num_frac_bits: int,
                 dimension: int,
                 chunk_length: int,
                 chunk_length_norm_equality: int,
                 num_shares: int,
                 num_proofs: int = 2,
                 num_proofs_norm_equality: int = 1,
                 num_wr_checks: int = NUM_WR_CHECKS,
                 num_wr_successes: int = NUM_WR_SUCCESSES):
        return super().__init__(
            Field64,
            XofHmacSha256Aes128,
            l2_norm_bound=l2_norm_bound,
            num_frac_bits=num_frac_bits,
            dimension=dimension,
            chunk_length=chunk_length,
            chunk_length_norm_equality=chunk_length_norm_equality,
            num_shares=num_shares,
            num_proofs=num_proofs,
            num_proofs_norm_equality=num_proofs_norm_equality,
            alpha=ALPHA,
            num_wr_checks=num_wr_checks,
            num_wr_successes=num_wr_successes,
            vdaf_id=self.ID
        )


class Pine32HmacSha256Aes128(Pine[Field32, XofHmacSha256Aes128]):
    test_vec_name = "Pine32HmacSha256Aes128"
    ID = 0xFFFF1005

    def __init__(self,
                 l2_norm_bound: int,
                 num_frac_bits: int,
                 dimension: int,
                 chunk_length: int,
                 chunk_length_norm_equality: int,
                 num_shares: int,
                 num_proofs: int = 5,
                 num_proofs_norm_equality: int = 1,
                 num_wr_checks: int = NUM_WR_CHECKS,
                 num_wr_successes: int = NUM_WR_SUCCESSES):
        return super().__init__(
            Field32,
            XofHmacSha256Aes128,
            l2_norm_bound=l2_norm_bound,
            num_frac_bits=num_frac_bits,
            dimension=dimension,
            chunk_length=chunk_length,
            chunk_length_norm_equality=chunk_length_norm_equality,
            num_shares=num_shares,
            num_proofs=num_proofs,
            num_proofs_norm_equality=num_proofs_norm_equality,
            alpha=ALPHA,
            num_wr_checks=num_wr_checks,
            num_wr_successes=num_wr_successes,
            vdaf_id=self.ID
        )
