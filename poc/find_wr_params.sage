# Script for helping us choose parameters for the wraparound tests (number of
# tests, minimum number of successes, and alpha).
#
# Run this script with sage directly to ensure that rational numbers don't get
# converted to floats:
#
# ```
# $ sage find_wr_params.sage
# ```
#
# Methodology:
# To find the optimal set of wraparound check parameters:
# Given the user parameters, we search over a range of `num_wr_checks`. For each
# `num_wr_checks`, find the smallest `num_wr_successes` that produces soundness
# error close to the target (usually `2**-100`). Then for each pair of
# `(num_wr_checks, num_wr_successes)`, find the `alpha` that produces a ZK error
# close to the target (usually `2**-100`), and also produces the lowest
# communication overhead, i.e., the number of field elements that we need to
# encode the wraparound check results in the FLP input.
#
# Then given each set of wraparound check parameters and user parameters,
# compute the number of proofs needed to achieve the target FLP soundness error.

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "draft-irtf-cfrg-vdaf", "poc"))
from field import Field, Field128, Field64
from flp_pine import PineValid
from sage.all import GF

def Bin(succ, total, prob):
    '''
    As defined in arxiv.org/abs/2311.10237, the probability of at least
    `succ` trials passing out of `total`. Each trial has success probability
    `prob`.
    '''
    b = 0
    for k in range(succ, total+1):
        b += binomial(total, k) * prob**k * (1 - prob)**(total - k)
    return b

def compute_alpha(eta):
    return sqrt(log(2/eta))

def compute_min_alpha():
    # Compute the minimum `alpha`. Using `alpha` below this value will make
    # `eta` greater than 1, which doesn't make sense because `eta` is a
    # probability and should not be greater than 1.
    return compute_alpha(1.0)

def compute_eta(alpha):
    # Sage doesn't like returning `2 / exp(alpha ** 2)` directly (could be
    # because of the `exp`), so take the `bits_of_security`, which makes the
    # ZK error larger.
    return 2**(-bits_of_security(2 / exp(alpha ** 2)))

def zk(r, r_succ, eta):
    '''$rho_C$ as defined in arxiv.org/abs/2311.10237, Lemma 3.3.'''
    return 1 - Bin(r_succ, r, 1 - eta)

def wr_sound(r, r_succ):
    '''$rho_S$ as defined in arxiv.org/abs/2311.10237, Lemma 3.3.'''
    return Bin(r_succ, r, 1/2)

def flp_sound(valid: PineValid):
    # Compute the soundness error of the degree-2 checks in PINE FLP, based on
    # Lemma 3.12.
    # `sqrt(n)` in the lemma is what we referred to as the "gadget calls"
    # in flp_pine.py.
    gadget_calls = sum(valid.GADGET_CALLS)
    # The number of constraints are:
    # - All the bit entries pass the bit check.
    # - The degree-2 check in wraparound check, i.e., g_k * S_k = 0
    #   in bullet point 3 in Figure 2.
    # - The quantities we are asserting in the final random linear combination,
    #   which includes the L2-norm range check, L2-norm equality check, the
    #   wraparound success count check, and the reduced result of degree-2 check
    #   in wraparound check:
    #   https://github.com/junyechen1996/draft-chen-cfrg-vdaf-pine/blob/21c43447f9b3ed283cc44500001ab4e9411a72c7/poc/flp_pine.py#L212-L215
    num_constraints = valid.bit_checked_len + valid.num_wr_checks + 4
    return gadget_calls * 2 / (valid.Field.MODULUS - gadget_calls) + \
           num_constraints / valid.Field.MODULUS

def overhead(pine_valid):
    '''
    Number of field elements to encode range-checked, wraparound check results.
    '''
    return pine_valid.num_wr_checks * (pine_valid.num_bits_for_wr_check + 1)

def bits_of_security(x):
    return floor(-log(x)/log(2))

def search_num_wr_successes(target_soundness_bits, num_wr_checks):
    '''
    Search for the smallest number of passing wraparound checks that satisfies
    the soundness error requirement `target_soundness_bits`, given the Client
    runs the wraparound check `num_wr_checks` times. The smallest number of
    passing checks is the most optimal for ZK error, because increasing it
    negatively impacts ZK error.
    '''
    for num_wr_successes in range(num_wr_checks, -1, -1):
        prob = wr_sound(num_wr_checks, num_wr_successes)
        if bits_of_security(prob) < target_soundness_bits - 1:
            if num_wr_successes == num_wr_checks:
                raise ValueError(
                    "num_wr_checks {} is too small to achieve target soundness "
                    "bits {}.".format(num_wr_checks, target_soundness_bits))
            return num_wr_successes + 1
    raise ValueError("Target soundness bits cannot be 0.")

def search_wr_params(target_soundness_bits,
                     target_zk_bits,
                     user_params,
                     max_num_wr_checks,
                     num_wr_checks_step):
    '''
    Search for the optimal wraparound check parameters for each set of user
    parameters in `user_params`. The search logic works as follows:
    - Find all pairs of `num_wr_checks` and `num_wr_successes` that can satisfy
      soundness error. We do so by finding the `num_wr_successes`, for each
      `num_wr_checks` from `target_soundness_bits` up to `max_num_wr_checks`,
      with a step of `num_wr_checks_step`. For each `num_wr_checks`, the search
      logic is handled in `search_num_wr_successes`.
    - Then for each set of parameters in `user_params`, go through the pairs of
      `num_wr_checks` and `num_wr_successes`, and find a viable `alpha`. For
      each possible combination of `num_wr_checks`, `num_wr_successes`, and
      `alpha`, compute the overhead in terms of the number of field elements
      needed to encode the wraparound check results. We pick the combination
      that gives the lowest overhead.
    '''
    # The search range for `alpha`.
    # We pick this maximum alpha, because it ensures a sufficiently low ZK error
    # for all user parameters.
    max_alpha = 10
    min_alpha = compute_min_alpha()
    assert min_alpha <= max_alpha

    # Find all pairs of `num_wr_checks` and `num_wr_successes`.
    num_wr_checks_and_successes = []
    # Search `num_wr_checks` in increasing order, because one more wraparound
    # check gives an additional bit of soundness (more robust).
    for num_wr_checks in range(target_soundness_bits,
                               max_num_wr_checks + 1,
                               num_wr_checks_step):
        num_wr_successes = search_num_wr_successes(target_soundness_bits,
                                                   num_wr_checks)
        num_wr_checks_and_successes.append((num_wr_checks, num_wr_successes))

    # Collect derived wraparound check parameters and user parameters that
    # satisfy the target ZK error.
    wr_and_user_params = []
    for (l2_norm_bound, num_frac_bits, dimension, field) in user_params:
        # Record the combination of `num_wr_checks`, `num_wr_successes`, and
        # `alpha` that gives the lowest overhead in terms of the number of
        # field elements to encode range-checked, wraparound check results.
        best_num_wr_checks = None
        best_num_wr_successes = None
        best_alpha = None
        best_overhead = None
        for (num_wr_checks, num_wr_successes) in num_wr_checks_and_successes:
            # Search for `alpha` in `[min_alpha, max_alpha]`, given
            # `num_wr_checks` and `num_wr_successes`.
            curr_min_alpha = min_alpha
            curr_max_alpha = max_alpha
            alpha = None
            # Do a binary search between `curr_min_alpha` and `curr_max_alpha`,
            # and stop when `curr_min_alpha` is at least `curr_max_alpha`, or
            # the difference between `curr_min_alpha` and `curr_max_alpha` is
            # < 0.1.
            while (curr_min_alpha < curr_max_alpha and
                   abs(curr_min_alpha - curr_max_alpha) >= 0.1):
                mid_alpha = curr_min_alpha + \
                            (curr_max_alpha - curr_min_alpha) / 2
                eta = compute_eta(mid_alpha)
                if (bits_of_security(zk(num_wr_checks, num_wr_successes, eta)) <
                    target_zk_bits - 1):
                    # `mid_alpha` doesn't satisfy ZK requirement, increase lower
                    # bound.
                    curr_min_alpha = mid_alpha
                else:
                    # `mid_alpha` satisfies ZK requirement. Check if it can be
                    # used to instantiate `PineValid`. If so, record this
                    # `mid_alpha`.
                    # We will decrease the search upper bound in the else case,
                    # because either (1) `mid_alpha` works for this field size,
                    # we can try using a even smaller alpha, (2) `mid_alpha` is
                    # too large for this field size, so we need to reduce alpha.
                    try:
                        pine_valid = PineValid.with_field(field)(
                            l2_norm_bound,
                            num_frac_bits,
                            dimension,
                            num_wr_checks = num_wr_checks,
                            num_wr_successes = num_wr_successes,
                            alpha = mid_alpha
                        )
                        # `mid_alpha` works for this field size, record it.
                        alpha = mid_alpha
                    except ValueError as e:
                        # `mid_alpha` is too large for field size.
                        pass
                    curr_max_alpha = mid_alpha
            if alpha is None:
                # Couldn't find `alpha` for the current `num_wr_checks` and
                # `num_wr_successes`.
                continue

            pine_valid = PineValid.with_field(field)(
                l2_norm_bound = l2_norm_bound,
                num_frac_bits = num_frac_bits,
                # Doesn't change parameters in wraparound checks.
                dimension = dimension,
                alpha = alpha,
                num_wr_checks = num_wr_checks,
                num_wr_successes = num_wr_successes
            )
            curr_overhead = overhead(pine_valid)
            if best_overhead is None or curr_overhead < best_overhead:
                best_num_wr_checks = num_wr_checks
                best_num_wr_successes = num_wr_successes
                best_alpha = alpha
                best_overhead = curr_overhead

        if best_overhead is None:
            print("Failed to find alpha in [{}, {}] with any combination "
                  "of num_wr_checks and num_wr_successes, for user parameters "
                  "l2_norm_bound = {}, num_frac_bits = {}, field = {}.".format(
                    float(min_alpha),
                    float(max_alpha),
                    l2_norm_bound,
                    num_frac_bits,
                    field.__name__))
            continue

        # Record the best set of wraparound check parameters for the current set
        # of user parameters.
        best_eta = compute_eta(best_alpha)
        wr_and_user_params.append((best_num_wr_checks,
                                   best_num_wr_successes,
                                   best_alpha,
                                   best_eta,
                                   zk(best_num_wr_checks,
                                      best_num_wr_successes,
                                      best_eta),
                                   wr_sound(best_num_wr_checks,
                                            best_num_wr_successes),
                                   best_overhead,
                                   l2_norm_bound,
                                   num_frac_bits,
                                   dimension,
                                   field))
    return wr_and_user_params

def display_wr_params(wr_and_user_params):
    # Print outputs that can be displayed in markdown.
    col_widths = [10, 10, 10, 15, 20, 20, 15, 15, 15, 10]
    col_names = ["l2_norm",
                 "frac_bits",
                 "field",
                 "num_wr_checks",
                 "num_wr_successes",
                 "alpha",
                 "-log2(eta)",
                 "-log2(zk)",
                 "-log2(sound)",
                 "overhead"]
    headers = "|"
    header_separator = "|"
    for (col_name, col_width) in zip(col_names, col_widths):
        headers += col_name.ljust(col_width) + "|"
        header_separator += ":" + "-" * (col_width - 1) + "|"
    printed_output = headers + "\n" + header_separator + "\n"

    for (num_wr_checks,
         num_wr_successes,
         alpha,
         eta,
         zk_error,
         wr_sound_error,
         overhead,
         l2_norm_bound,
         num_frac_bits,
         _dimension,
         field) in wr_and_user_params:
        row = [str(l2_norm_bound),
               str(num_frac_bits),
               field.__name__,
               str(num_wr_checks),
               str(num_wr_successes),
               str(float(alpha)),
               str(bits_of_security(eta)),
               str(bits_of_security(zk_error)),
               str(bits_of_security(wr_sound_error)),
               str(overhead)]
        printed_output += "|"
        for (col, col_width) in zip(row, col_widths):
            printed_output += col.ljust(col_width) + "|"
        printed_output += "\n"
    print("Displaying feasible set of wraparound check and user parameters:")
    print(printed_output)

def search_vdaf_params(target_soundness_bits, wr_and_user_params):
    '''
    For each set of wraparound check and user parameters that satisfy the ZK
    error and soundness error (produced by `search_wr_params`), compute the
    number of proofs needed to satisfy the FLP soundness error. This gives us
    the full VDAF parameters.
    '''
    vdaf_params = []
    for (num_wr_checks,
         num_wr_successes,
         alpha,
         eta,
         zk_error,
         wr_sound_error,
         overhead,
         l2_norm_bound,
         num_frac_bits,
         dimension,
         field) in wr_and_user_params:
        valid = PineValid.with_field(field)(
            l2_norm_bound = l2_norm_bound,
            num_frac_bits = num_frac_bits,
            dimension = dimension,
            alpha = alpha,
            num_wr_checks = num_wr_checks,
            num_wr_successes = num_wr_successes
        )
        flp_sound_one_proof = flp_sound(valid)
        num_proofs = ceil(
            target_soundness_bits / bits_of_security(flp_sound_one_proof)
        )
        # Compute overall soundness error per Theorem 3.14, i.e. the sum of
        # FLP soundness error and wraparound check soundness error.
        overall_sound = flp_sound_one_proof**num_proofs + wr_sound_error
        vdaf_params.append((l2_norm_bound,
                            num_frac_bits,
                            dimension,
                            valid.chunk_length,
                            field,
                            num_proofs,
                            num_wr_checks,
                            num_wr_successes,
                            zk_error,
                            overall_sound))
    return vdaf_params

def display_vdaf_params(vdaf_params):
    # Print outputs that can be displayed in markdown.
    col_widths = [10, 10, 10, 10, 10, 10, 15, 20, 10, 20]
    col_names = ["l2_norm",
                 "frac_bits",
                 "dimension",
                 "chunk_len",
                 "field",
                 "proofs",
                 "num_wr_checks",
                 "num_wr_successes",
                 "-log2(zk)",
                 "-log2(sound)"]
    headers = "|"
    header_separator = "|"
    for (col_name, col_width) in zip(col_names, col_widths):
        headers += col_name.ljust(col_width) + "|"
        header_separator += ":" + "-" * (col_width - 1) + "|"
    printed_output = headers + "\n" + header_separator + "\n"

    for (l2_norm_bound,
         num_frac_bits,
         dimension,
         chunk_length,
         field,
         num_proofs,
         num_wr_checks,
         num_wr_successes,
         zk_error,
         overall_sound) in vdaf_params:
        row = [str(l2_norm_bound),
               str(num_frac_bits),
               str(dimension),
               str(chunk_length),
               field.__name__,
               str(num_proofs),
               str(num_wr_checks),
               str(num_wr_successes),
               str(bits_of_security(zk_error)),
               str(bits_of_security(overall_sound))]
        printed_output += "|"
        for (col, col_width) in zip(row, col_widths):
            printed_output += col.ljust(col_width) + "|"
        printed_output += "\n"
    print(printed_output)


class Field32(Field):
    """A fake 32-bit finite field. """

    # Taken from https://github.com/divviup/libprio-rs/blob/bc8c3ec5feed9c6f68f113d148eff2788354d346/src/fp.rs#L346
    MODULUS = 4293918721
    ENCODED_SIZE = 4

    # Operational parameters
    gf = GF(MODULUS)

class Field40(Field):
    """A fake 40-bit finite field. """

    MODULUS = (2^40).previous_prime()  # May not be FFT-friendly.
    ENCODED_SIZE = 5

    # Operational parameters
    gf = GF(MODULUS)

class Field48(Field):
    """A fake 48-bit finite field. """

    MODULUS = (2^48).previous_prime()  # May not be FFT-friendly.
    ENCODED_SIZE = 6

    # Operational parameters
    gf = GF(MODULUS)

class Field56(Field):
    """A fake 56-bit finite field. """

    MODULUS = (2^56).previous_prime()  # May not be FFT-friendly.
    ENCODED_SIZE = 7

    # Operational parameters
    gf = GF(MODULUS)


# Populate a list of user parameters that we will compute operational
# parameters for. The user parameters need to pass the basic check
# in `PineValid.__init__`, e.g., the field size is large enough to represent
# the range of possible floating point values, and aggregate the gradients
# from `max_num_clients` so that the sum of the floating point values don't
# overflow the field size.
user_params = []
max_num_clients = 100_000
# Use fixed `l2_norm_bound` and `dimension` when we search for wraparound
# check parameters.
l2_norm_bound = 1
dimension = 200_000
for field in [Field32, Field40, Field48, Field56, Field64, Field128]:
    for num_frac_bits in [14, 20, 24]:
        try:
            # Check if this combination of user parameters can be used to
            # initialize `PineValid`, with the smallest `alpha` possible
            # (`alpha` will be checked later when we search for wraparound
            # check parameters).
            valid = PineValid.with_field(field)(l2_norm_bound,
                                                num_frac_bits,
                                                dimension,
                                                alpha = compute_min_alpha())
        except Exception as e:
            print("Failed to initialize PineValid with l2_norm_bound = {}, "
                  "num_frac_bits = {}, dimension = {}, field = {}, "
                  "due to: {}".format(
                  l2_norm_bound,
                  num_frac_bits,
                  dimension,
                  field.__name__,
                  str(e)))
            continue

        encoded_norm_bound = \
            valid.encode_f64_into_field(l2_norm_bound).as_unsigned()
        # We reserve the first half of the field size for positive
        # floating point values, so require the field size to be 2 times
        # the number of clients, times the encoded norm bound.
        if field.MODULUS / 2 < encoded_norm_bound * max_num_clients:
            print("{} is not large enough to handle aggregation of {} clients "
                  "clients for l2_norm_bound = {}, num_frac_bits = {}.".format(
                  field.__name__,
                  max_num_clients,
                  l2_norm_bound,
                  num_frac_bits))
            continue
        user_params.append((l2_norm_bound, num_frac_bits, dimension, field))

# In order to achieve the target soundness and ZK error, compute the operational
# parameters (wraparound check parameters, number of proofs) for the
# `user_params`. It works as follows:
# We first compute the wraparound check parameters for each set of
# `user_params`. Then we will compute the number of proofs for each set of
# parameters to achieve the desired FLP soundness error.
for (target_bits,
     max_num_wr_checks,
     num_wr_checks_step) in [(100, 1000, 50),
                             (32, 160, 5),
                             (64, 500, 20),
                             (80, 1000, 30)]:
    print("Parameters to satisfy 2^-{} soundness and 2^-{} ZK error".format(
        target_bits, target_bits))
    # First find the wraparound check parameters. Output each possible
    # combination of wraparound check parameters and user parameters.
    wr_and_user_params = search_wr_params(target_bits,
                                          target_bits,
                                          user_params,
                                          max_num_wr_checks,
                                          num_wr_checks_step)
    display_wr_params(wr_and_user_params)

    # Then compute the number of proofs to achieve the desired FLP soundness
    # error, for each set of parameters in `wr_and_user_params`.
    # This gives us the full parameters for PINE VDAF.
    vdaf_params = search_vdaf_params(target_bits,
                                     wr_and_user_params)
    display_vdaf_params(vdaf_params)
