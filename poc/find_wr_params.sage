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
from common import next_power_of_2
from field import Field64
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

def zk(r, r_succ, eta):
    '''$rho_C$ as defined in arxiv.org/abs/2311.10237, Lemma 3.3.'''
    return 1 - Bin(r_succ, r, 1 - eta)

def sound(r, r_succ):
    '''$rho_S$ as defined in arxiv.org/abs/2311.10237, Lemma 3.3.'''
    return Bin(r_succ, r, 1/2)

def overhead(r, r_succ, eta, l2_norm_bound, num_frac_bits):
    '''Number of field elements to encode wrap around tests.'''
    alpha = sqrt(log(2/eta))
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
        while bits_of_security(sound(total, total - failures)) \
                not in range(target_soundness_bits-1,target_soundness_bits+1):
            total += 1
        params.append((total, total - failures, None))
        failures += 1
    return params

def display_params(params):
    '''
    For reach result for which `eta` is set, print soundness error, ZK error,
    and overhead for the given L2 norm bound and number of fractional bits. To
    complete the parameters, we run this on the output of `search()`, make an
    initial guess of `eta`, and manually tune until the desired ZK is reached.
    '''
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
    for (r, r_succ, eta) in params:
        if eta != None:
            (num_elems, alpha) = overhead(r, r_succ, eta, 1.0, 15)
            print("|{}|{}|{}|{}|{}|{}|{}|".format(
                str(r).ljust(col_widths[0]),
                str(r_succ).ljust(col_widths[1]),
                str(bits_of_security(eta)).ljust(col_widths[2]),
                str(bits_of_security(zk(r, r_succ, eta))).ljust(col_widths[3]),
                str(bits_of_security(sound(r, r_succ))).ljust(col_widths[4]),
                str(num_elems).ljust(col_widths[5]),
                str(float(alpha)).ljust(col_widths[6]),
            ))
    print()


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
display_params(params_32)

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
display_params(params_64)

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
display_params(params_80)
