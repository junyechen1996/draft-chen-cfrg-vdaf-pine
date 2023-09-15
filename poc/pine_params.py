"""Helper functions to search for parameters for PINE. """

from scipy.stats import binom

import math
import numpy as np
import os
import sys

from vdaf_pine import Pine128, Pine64


def log2_wr_zk_error(alpha: float,
                     num_wr_checks: int,
                     num_wr_successes: int) -> float:
    # Compute the zero-knowledge (ZK) error of running wraparound checks for
    # `num_wr_checks` number of times, and requiring the Client to pass
    # `num_wr_successes` of them.
    # Note `alpha` determines the ZK error for one wraparound check:
    # `2.0 / (e ** (alpha ** 2))` (See "The Protocol" in Figure 2).
    # To summarize the relationship between ZK error and each of the parameters,
    # assuming other parameters are fixed:
    # - `alpha` increases, ZK error decreases.
    # - `num_wr_checks` increases, ZK error increases.
    # - `num_wr_successes` increases, ZK error increases.
    zk_error_one_check = 2.0 / math.exp(alpha ** 2)
    # Compute the overall ZK error:
    # Assume the probability of an honest Client failing one wraparound check is
    # `zk_error_one_check`, compute the probability that the Client fails more
    # than `num_wr_checks - num_wr_successes` checks, out of `num_wr_checks`.
    log_zk_error = binom.logsf(num_wr_checks - num_wr_successes,
                               num_wr_checks,
                               zk_error_one_check)
    return log_zk_error / math.log(2)

def log2_wr_soundness_error(num_wr_checks: int, num_wr_successes: int) -> float:
    # The soundness error of one wraparound check is always 1/2.
    # Compute the overall soundness error, i.e., the probability of the
    # malicious Client passing at least `num_wr_successes` checks, out of
    # `num_wr_checks`.
    return binom.logsf(num_wr_successes - 1, num_wr_checks, 0.5) / math.log(2)

def log2_gadget_soundness_error(pine):
    valid = pine.Flp.Valid
    # Compute the soundness error of the degree-2 checks in PINE,
    # based on Lemma 3.12.
    # `sqrt(n)` is what we referred to as the "gadget calls" in flp_pine.py.
    gadget_calls = sum(valid.GADGET_CALLS)
    # The number of constraints are:
    # - All the bit entries pass the bit check.
    # - The L2-norm check.
    # - The degree-2 check in wraparound check.
    num_constraints = valid.bit_checked_len + 1 + valid.num_wr_checks
    # With parallel repetitions of proofs, we can reduce soundness error to
    # the power of `pine.PROOFS`, based on Lemma 3.13.
    soundness_error = \
        (gadget_calls * 2 / (valid.Field.MODULUS - gadget_calls) +
         num_constraints / valid.Field.MODULUS) ** pine.PROOFS
    return math.log2(soundness_error)

def log2_overall_error(pine):
    alpha = pine.Flp.Valid.alpha
    num_wr_checks = pine.Flp.Valid.num_wr_checks
    num_wr_successes = pine.Flp.Valid.num_wr_successes
    zk_error = log2_wr_zk_error(alpha, num_wr_checks, num_wr_successes)
    wr_soundness_error = log2_wr_soundness_error(num_wr_checks,
                                                 num_wr_successes)
    gadget_soundness_error = log2_gadget_soundness_error(pine)
    print("wraparound check soundness error: 2^(%f), "
          "gadget soundness error: 2^(%f)" % (
            wr_soundness_error, gadget_soundness_error))
    overall_soundness_error = np.logaddexp2(wr_soundness_error,
                                            gadget_soundness_error)
    return overall_soundness_error, zk_error

def main():
    alpha = 7.99996948
    l2_norm_bound = 1.0
    num_frac_bits = 15
    dimension = 100_000
    chunk_length = 400
    num_wr_checks = 101
    num_wr_successes = 100
    for pine_cls in [Pine128, Pine64]:
        pine = pine_cls.with_params(l2_norm_bound = l2_norm_bound,
                                    num_frac_bits = num_frac_bits,
                                    dimension = dimension,
                                    chunk_length = chunk_length,
                                    num_shares = 2,
                                    alpha = alpha,
                                    num_wr_checks = num_wr_checks,
                                    num_wr_successes = num_wr_successes)
        log2_soundness, log2_zk = log2_overall_error(pine)
        print("%s: soundness error: 2^(%f), ZK error: 2^(%f)" % (
            pine_cls.__name__, log2_soundness, log2_zk))


if __name__ == "__main__":
    main()
