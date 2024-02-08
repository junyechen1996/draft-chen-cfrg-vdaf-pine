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
# Methodology: Find pairs of `(r, tau)` that produce soundness error close to
# the target (usually `2**64`), then find the smallest `eta` for which the ZK
# error is close to `2**128`. The optimal set of parameters is the one that
# minimizes communication overhead, i.e., the number of field elements that we
# need to encode the wraparound check results in the FLP input.

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, "draft-irtf-cfrg-vdaf", "poc"))
from common import next_power_of_2

def Bin(succ, total, prob):
    '''
    As defined in arxiv.org/abs/2311.10237, the probability of at least
    `succ` trials passing out of `total`. Each trial has success probability
    `prob`.
    '''
    b = 0
    for k in range(succ, total+1):
        b += binomial(total, k) * prob**k * (1-prob)**(total - k)
    return b

def zk(r, tau, eta):
    '''$rho_C$ as defined in arxiv.org/abs/2311.10237, Lemma 3.3.'''
    return 1 - Bin(r * tau, r, 1 - eta)

def sound(r, tau):
    '''$rho_S$ as defined in arxiv.org/abs/2311.10237, Lemma 3.3.'''
    return Bin(r * tau, r, 1/2)

def overhead(r, eta, l2_norm_bound, num_frac_bits):
    '''Number of field elements to encode wrap around tests.'''
    alpha = ceil(sqrt(log(2/eta)))
    l2_norm_bound_encoded = floor(l2_norm_bound * 2**num_frac_bits)
    wr_check_bound = next_power_of_2(alpha * l2_norm_bound_encoded + 1) - 1
    num_bits_for_wr_check = 1 + ceil(log(wr_check_bound + 1)/log(2))
    return (r * (num_bits_for_wr_check + 1), alpha)

def bits_of_security(x):
    return floor(-log(x)/log(2))

def search(target_soundness_bits):
    '''
    Find values of `(r, tau)` with soundness error close to the target
    soundness (in bits of security). For example, `target_soundness_bits = 64`
    produces choices with soundness error close to `2**-64`.
    '''
    params = []
    failures = 0
    total = target_soundness_bits
    while failures < 16:
        while bits_of_security(sound(total, (total - failures) / total)) \
                not in range(target_soundness_bits-1,target_soundness_bits+1):
            total += 1
        params.append((total, (total - failures) / total, None))
        failures += 1
    return params

def display_params(params):
    '''
    For reach result for which `eta` is set, print soundness error, ZK error,
    and overhead for the given L2 norm bound and number of fractional bits. To
    complete the parameters, we run this on the output of `search()`, make an
    initial guess of `eta`, and manually tune until the desired ZK is reached.
    '''
    for (r, tau, eta) in params:
        if eta != None:
            print(r, tau, eta)
            print('zk      ', bits_of_security(zk(r, tau, eta)))
            print('sound   ', bits_of_security(sound(r, tau)))
            print('overhead {} (alpha={})'.format(*overhead(r, eta, 1.0, 15)))
    print()


# print(search(32))
params_32 = [
    (32, 1, 2**-134),
    (37, 36/37, 2**-69),
    (41, 39/41, 2**-48),
    (45, 14/15, None),
    (49, 45/49, None),
    (53, 48/53, None),
    (57, 17/19, None),
    (60, 53/60, None),
    (64, 7/8, None),
    (67, 58/67, None),
    (70, 6/7, None),
    (73, 62/73, None),
    (77, 65/77, None),
    (80, 67/80, None),
    (83, 69/83, None),
    (86, 71/86, 2**-12),
]
print('------- 32')
display_params(params_32)

# print(search(64))
params_64 = [
    (64, 1, 2**-134),
    (70, 69/70, 2**-69),
    (75, 73/75, 2**-48),
    (80, 77/80, None),
    (84, 20/21, None),
    (89, 84/89, None),
    (93, 29/31, None),
    (97, 90/97, None),
    (101, 93/101, 2**-19),
    (105, 32/35, None),
    (109, 99/109, None),
    (113, 102/113, None),
    (116, 26/29, None),
    (120, 107/120, None),
    (123, 109/123, None),
    (127, 112/127, 2**-12),
]
print('------- 64')
display_params(params_64)

# print(search(80))
params_80 = [
    (80, 1, 2**-134),
    (86, 85/86, 2**-69),
    (92, 45/46, 2**-48),
    (97, 94/97, None),
    (102, 49/51, None),
    (106, 101/106, None),
    (111, 35/37, None),
    (115, 108/115, None),
    (119, 111/119, None),
    (123, 38/41, None),
    (127, 117/127, None),
    (131, 120/131, None),
    (135, 41/45, None),
    (139, 126/139, None),
    (142, 64/71, None),
    (146, 131/146, 2**-12),
]
print('------- 80')
display_params(params_80)
