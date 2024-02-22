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
# Methodology: Find pairs of `(r, r_succ)` that produce soundness error close to
# the target (usually `2**-64`), then find the smallest `eta` for which the ZK
# error is close to `2**-128`. The optimal set of parameters is the one that
# minimizes communication overhead, i.e., the number of field elements that we
# need to encode the wraparound check results in the FLP input.

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "draft-irtf-cfrg-vdaf", "poc"))
from field import Field128, Field64
from flp_pine import PineValid

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

def overhead(r, r_succ, eta, l2_norm_bound, num_frac_bits):
    '''Number of field elements to encode wrap around tests.'''
    alpha = compute_alpha(eta)
    pine_valid = PineValid.with_field(Field64)(
        l2_norm_bound = l2_norm_bound,
        num_frac_bits = num_frac_bits,
        dimension = 100000,  # Doesn't change overhead from wraparound checks.
        chunk_length = 100,  # Doesn't change overhead from wraparound checks.
        alpha = alpha,
        num_wr_checks = r,
        num_wr_successes = r_succ
    )
    return (r * (pine_valid.num_bits_for_wr_check + 1), alpha)

def bits_of_security(x):
    return floor(-log(x)/log(2))

def search(target_soundness_bits):
    '''
    Find values of `(r, r_succ)` with soundness error close to the target
    soundness (in bits of security). For example, `target_soundness_bits = 64`
    produces choices with soundness error close to `2**-64`.
    '''
    params = []
    failures = 0
    total = target_soundness_bits
    while failures < 16:
        while bits_of_security(wr_sound(total, total - failures)) \
                not in range(target_soundness_bits-1,target_soundness_bits+1):
            total += 1
        params.append((total, total - failures, None))
        failures += 1
    return params

def display_wr_params(target_soundness_bits, params):
    '''
    For reach result for which `eta` is set, print soundness error, ZK error,
    and overhead for the given L2 norm bound and number of fractional bits. To
    complete the parameters, we run this on the output of `search()`, make an
    initial guess of `eta`, and manually tune until the desired ZK is reached.
    '''
    # Print outputs that can be displayed in markdown.
    col_widths = [10, 10, 15, 15, 15, 10, 20]
    col_names = ["r",
                 "r_succ",
                 "-log2(eta)",
                 "-log2(zk)",
                 "-log2(sound)",
                 "overhead",
                 "alpha"]
    headers = "|"
    header_separator = "|"
    for (col_name, col_width) in zip(col_names, col_widths):
        headers += col_name.ljust(col_width) + "|"
        header_separator += ":" + "-" * (col_width - 1) + "|"
    print(headers)
    print(header_separator)

    # Also write outputs as a CSV file.
    with open("wr_params_{}.csv".format(target_soundness_bits), "w+") as f:
        f.write(",".join(col_names) + "\n")
        for (r, r_succ, eta) in params:
            if eta != None:
                (num_elems, alpha) = overhead(r, r_succ, eta, 1.0, 15)
                row = [str(r),
                       str(r_succ),
                       str(bits_of_security(eta)),
                       str(bits_of_security(zk(r, r_succ, eta))),
                       str(bits_of_security(wr_sound(r, r_succ))),
                       str(num_elems),
                       str(float(alpha))]
                csv_row = ",".join(row) + "\n"
                f.write(csv_row)
                printed_row = "|"
                for (col, col_width) in zip(row, col_widths):
                    printed_row += col.ljust(col_width) + "|"
                print(printed_row)
    print()

def display_user_params(target_soundness_bits,
                        user_params,
                        r,
                        r_succ,
                        eta):
    '''
    Given a chosen set of wraparound check parameters, display the soundness
    error from the gadgets, based on a list of user parameters.
    '''
    # Print outputs that can be displayed in markdown.
    col_widths = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 20]
    col_names = ["l2_norm",
                 "frac_bits",
                 "dimension",
                 "chunk_len",
                 "field",
                 "proofs",
                 "r",
                 "r_succ",
                 "-log2(eta)",
                 "-log2(zk)",
                 "-log2(sound)"]
    headers = "|"
    header_separator = "|"
    for (col_name, col_width) in zip(col_names, col_widths):
        headers += col_name.ljust(col_width) + "|"
        header_separator += ":" + "-" * (col_width - 1) + "|"
    print(headers)
    print(header_separator)

    # Also write outputs as a CSV file.
    with open("user_params_{}.csv".format(target_soundness_bits), "w+") as f:
        f.write(",".join(col_names) + "\n")
        for (l2_norm_bound,
             num_frac_bits,
             dimension,
             chunk_length,
             field) in user_params:
            alpha = compute_alpha(eta)
            valid = PineValid.with_field(field)(
                l2_norm_bound = l2_norm_bound,
                num_frac_bits = num_frac_bits,
                dimension = dimension,
                chunk_length = chunk_length,
                alpha = alpha,
                num_wr_checks = r,
                num_wr_successes = r_succ
            )
            flp_sound_one_proof = flp_sound(valid)
            num_proofs = ceil(
                target_soundness_bits / bits_of_security(flp_sound_one_proof)
            )
            # Compute overall soundness error per Theorem 3.14, i.e. the sum of
            # FLP soundness error and wraparound check soundness error.
            overall_sound = \
                flp_sound_one_proof**num_proofs + wr_sound(r, r_succ)
            row = [str(l2_norm_bound),
                   str(num_frac_bits),
                   str(dimension),
                   str(valid.chunk_length),
                   field.__name__,
                   str(num_proofs),
                   str(r),
                   str(r_succ),
                   str(bits_of_security(eta)),
                   str(bits_of_security(zk(r, r_succ, eta))),
                   str(bits_of_security(overall_sound))]
            csv_row = ",".join(row) + "\n"
            f.write(csv_row)
            printed_row = "|"
            for (col, col_width) in zip(row, col_widths):
                printed_row += col.ljust(col_width) + "|"
            print(printed_row)
        print()


user_params = [
    (1, 15, 1000, None, Field64),
    (1, 15, 1000, None, Field128),
    (1, 15, 10000, None, Field64),
    (1, 15, 10000, None, Field128),
    (1, 15, 100000, None, Field64),
    (1, 15, 100000, None, Field128)
]

# print(search(32))
params_32 = [
    (32, 32, 2**-134),
    (37, 36, 2**-69),
    (41, 39, 2**-48),
    (45, 42, None),
    (49, 45, None),
    (53, 48, None),
    (57, 51, None),
    (60, 53, None),
    (64, 56, None),
    (67, 58, None),
    (70, 60, None),
    (73, 62, None),
    (77, 65, None),
    (80, 67, None),
    (83, 69, None),
    (86, 71, 2**-12),
]
print('------- Target soundness error: 2^(-32)')
print('Wraparound check parameters:')
display_wr_params(32, params_32)
print('User parameters:')
display_user_params(32, user_params, *params_32[0])

# print(search(64))
params_64 = [
    (64, 64, 2**-134),
    (70, 69, 2**-69),
    (75, 73, 2**-48),
    (80, 77, None),
    (84, 80, None),
    (89, 84, None),
    (93, 87, None),
    (97, 90, None),
    (101, 93, 2**-19),
    (105, 96, None),
    (109, 99, None),
    (113, 102, None),
    (116, 104, None),
    (120, 107, None),
    (123, 109, None),
    (127, 112, 2**-12),
]
print('------- Target soundness error: 2^(-64)')
print('Wraparound check parameters:')
display_wr_params(64, params_64)
print('User parameters:')
display_user_params(64, user_params, *params_64[0])

# print(search(80))
params_80 = [
    (80, 80, 2**-134),
    (86, 85, 2**-69),
    (92, 90, 2**-48),
    (97, 94, None),
    (102, 98, None),
    (106, 101, None),
    (111, 105, None),
    (115, 108, None),
    (119, 111, None),
    (123, 114, None),
    (127, 117, None),
    (131, 120, None),
    (135, 123, None),
    (139, 126, None),
    (142, 128, None),
    (146, 131, 2**-12),
]
print('------- Target soundness error: 2^(-80)')
print('Wraparound check parameters:')
display_wr_params(80, params_80)
print('User parameters:')
display_user_params(80, user_params, *params_80[0])
