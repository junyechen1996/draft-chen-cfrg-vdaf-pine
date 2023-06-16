---
title: "Private Inexpensive Norm Enforcement (PINE) VDAF"
category: info

docname: draft-chen-cfrg-vdaf-pine-latest
submissiontype: IRTF  # also: "independent", "IAB", or "IRTF"
number:
date:
consensus: true
v: 3
area: "IRTF"
workgroup: "Crypto Forum"
keyword:
 - next generation
 - unicorn
 - sparkling distributed ledger
venue:
  group: "Crypto Forum"
  type: "Research Group"
  mail: "cfrg@ietf.org"
  arch: "https://mailarchive.ietf.org/arch/search/?email_list=cfrg"
  github: "junyechen1996/draft-chen-cfrg-vdaf-pine"
  latest: "https://junyechen1996.github.io/draft-chen-cfrg-vdaf-pine/draft-chen-cfrg-vdaf-pine.html"

author:
 -
    fullname: Junye Chen
    organization: Apple Inc.
    email: "junyec@apple.com"

normative:

informative:

  BBCGGI19:
    title: "Zero-Knowledge Proofs on Secret-Shared Data via Fully Linear PCPs"
    author:
      - ins: D. Boneh
      - ins: E. Boyle
      - ins: H. Corrigan-Gibbs
      - ins: N. Gilboa
      - ins: Y. Ishai
    date: 2019
    seriesinfo: CRYPTO 2019
    target: https://ia.cr/2019/188

--- abstract

A new Verifiable Distributed Aggregation Function (VDAF) named
Private Inexpensive Norm Enforcement (PINE) that supports aggregating
high-dimensional real number vectors bounded by a configurable L2-norm bound,
which is a fundamental primitive to support private federated learning.


--- middle

# Introduction

Aggregating high-dimensional real number vectors is a fundamental primitive
to support federated learning. There have been various approaches that attempt
to support such use cases with strong security and privacy guarantee, such as
differential privacy (DP), secure aggregation, etc. In this document, we propose
a protocol that attempts to achieve the following properties:

* Malicious Clients should be prevented from sending vectors with invalid
  L2-norm and poisoning the final aggregated result with high probability.

* Honest Clients should be accepted with high probability.

* An attacker that controls the Collector, a subset of Clients, and all but one
  Aggregators, learns statistically close to nothing about the measurements of
  honest Clients.

Previous approaches that attempt to achieve these properties either only
implement an approximate verification of Client measurements, or incur a high
communication overhead between Client and Aggregators. This document outlines
a proposal of a new VDAF that enables more efficient and accurate secure
aggregation that conforms to the VDAF
protocol {{!VDAF=I-D.draft-irtf-cfrg-vdaf-05}}.

The VDAF protocol explicitly introduces a Prio3 scheme {{Section 7 of !VDAF}},
which can support various types of aggregated statistics and can verify certain
properties of Client measurements. This document aims to extend the Prio3 scheme
to support aggregating real number vectors with bounded L2-norm, specifically
from the following aspects:

* A new type of joint randomness agreed between Client and Aggregators, that
  supports the computation of "wraparound protocol" in PINE VDAF. We denote it
  as "wraparound joint randomness".

* A new "Fully Linear Proof (FLP)" system that incorporates the
  "wraparound joint randomness" and supports PINE as a new VDAF.


# Conventions and Definitions

{::boilerplate bcp14-tagged}

This document uses the same parameters and conventions specified for:

* Definitions for VDAF in {{Section 5 of !VDAF}}.

* Field elements in {{Section 6.1 of !VDAF}}.

* Pseudorandom Generators in {{Section 6.2 of !VDAF}}.

* Prio3 FLP system in {{Section 7 of !VDAF}}, with some extensions.

The computation parameters used in the protocols of PINE are listed in
{{pine-comp-param}}:

| Parameter | Description    |
|:----------|:---------------|
| `b`       | L2-norm bound. This is an inclusive upper bound. Users of PINE can use this parameter to control the L2-norm bound of its Client vectors. |
| `d`       | Client vector dimension. |
| `f`       | Number of fractional bits to keep in Client vector values. Users of PINE can use this parameter to control the precision. |
| `B`       | Squared L2-norm bound in field integer. It is equal to `(floor(b * 2^f))^2`. This is an inclusive upper bound. |
| `xs`      | Original Client vector. |
| `X`       | Client vector after the original real number vector is encoded into a vector of field integers. |
| `V0`      | Computed squared L2-norm of X, i.e. `SUM_i X_i^2`. |
| `U0`      | Difference between `V0` and `B`. |
| `b0`      | Number of bits to represent `B`, `V0`, and `U0`. It should be equal to `ceil(log_2(B + 1))`. The `+ 1` is to support the inclusive squared L2-norm upper bound `B`. |
| `r`       | Number of wraparound protocol repetitions. |
| `TAU`     | Minimum proportion of `r` wraparound protocol repetitions that Clients are required to pass. |
| `Z`       | Random vector in wraparound protocol, each element of which is -1, 0, or 1. |
| `ETA`     | Completeness error in one repetition of wraparound protocol. |
| `ALPHA`   | Parameter in wraparound protocol, equal to `sqrt(ln(2 / ETA))`. |
| `Y1`      | Computed wraparound protocol result. |
| `V1`      | Shifted wraparound protocol result, which is always non-negative. It is computed as `Y1 + ALPHA * sqrt(B)`. |
| `U1`      | Difference between shifted wraparound protocol result and the shifted upper bound (`2 * ALPHA * sqrt(B)`). |
| `g`       | Success bit, one bit for each wraparound protocol repetition. |
| `S`       | Difference between computed shifted wraparound protocol result, and client-encoded shifted result. This is a single field element for each repetition of wraparound protocol, and is only non-zero when the Client has failed that repetition. |
| `b1`      | Number of bits to represent `V1` and `U1` in one repetition of wraparound protocol. It should be equal to `ceil(log_2(2 * ALPHA * sqrt(B) + 1))`. The `+ 1` is to support the inclusive bound in wraparound protocol. |
| `F`       | Field element type, could be `Field64` (8 bytes), `Field128` (16 bytes). |
| `q`       | Field modulus. |
| `t`       | Repetitions of verifying degree-2 polynomial checks with the techniques in Corollary 4.7 and Remark 4.8 of {{BBCGGI19}}, in order to reduce soundness error of verification exponentially with `t`. |
{: #pine-comp-param title="Computation Parameters Used in PINE."}


# Private Inexpensive Norm Enforcement (PINE) VDAF {#pine}

PINE supports aggregating real number vectors bounded by a configured L2-norm.
It asks each Client to compute results needed by protocols in PINE to verify
the L2-norm, and then send the results in the submitted vector. In this draft,
we are specifically interested in the implementation of the statistical
zero-knowledge protocol in section 4 of the PINE paper, in the VDAF setting.

> TODO Add citations for PINE paper once the public link is available.

The Clients are expected to first encode their real number vectors into a
vector of field integers, because all the protocols in PINE are computed and
verified with field integers. The vectors of field integers are then aggregated
and decoded into a vector of IEEE-754 compatible float64 values.

## Encoding Real Numbers into Field Integers {#fp-encoding}

To keep the necessary number of precision bits `f` configured in PINE VDAF,
the high-level idea is to multiply the real numbers by `2^f`.

For L2-norm bound `b`, the field integer value is `floor(b * 2^f)`. To compute
the parameter `B`, we square the encoded field integer directly,
i.e. `(floor(b * 2^f))^2`.

For each value `x` in Client vector `xs`:

* If `x >= 0`, encode the value as `floor(x * 2^f)`, i.e. positive real numbers
  are encoded into the front end of the field integer range,
  i.e. `[0, floor(b * 2^f)]`, or `[0, sqrt(B)]`.

* If `x < 0`, encode the value as `q - floor(|x| * 2^f)`, i.e. negative real
  numbers are encoded into the tail end of the field integer range,
  i.e. `[q - floor(b * 2^f), q)`, or `[q - sqrt(B), q)`.

This effectively maps each possible value `x` in the range `[-b, b]` into
the field integer range `[-sqrt(B), sqrt(B)]`.

## Aggregation {#agg}

The aggregation of the real number vectors is therefore done in field integers
as well, after each original Client vector is encoded into a vector of field
integers.

## Decoding Field Integers into IEEE-754 Compatible Float64 {#fi-decoding}

To decode a (aggregated) field integer `agg` back to float, assuming there are
`c` clients, the aggregated field integer should fall in the range of
`[-c * sqrt(B), c * sqrt(B)]`. That is:

* If the aggregated value is negative, we expect the field integer to be in
the range `[q - c * sqrt(B), q)`.

* If the aggregated value is positive, we expect the field integer to be in
the range `[0, c * sqrt(B)]`.

We will first check that the two ranges don't collapse, which happens when
the number of clients `c` is too big. We verify that
`2 * c * sqrt(B) < q`.

Then in order to decode, the basic idea is to divide the aggregated field
integer by `2^f`, specifically:

* When `0 <= agg <= c * sqrt(B)`, we will decode it as `agg / 2^f`.

* Otherwise, decode it as `-((q - agg) / 2^f)`.

## Protocols in PINE

Now we will assume Clients and Aggregators can execute the protocols in PINE
with field integers, after Clients has encoded their real number vectors
into field integer vectors as specified in {{fp-encoding}}. There are three
main protocols in PINE:

* 0/1 bit check protocol

* L2-norm sum-check protocol

* Wraparound protocol

### 0/1 Bit Check Protocol {#bit-check}

The Client will encode the bit representation of results computed in
L2-norm sum-check protocol and wraparound protocol, secret-share the bits,
and submit them to the Aggregators.

The Aggregators MUST verify the indices that are expected to be bits are
indeed 0 or 1.

The Client and Aggregators can follow the proof system constructed in
{{Section 7.3.1.1 of !VDAF}} that computes a random linear combination of
the `Range2` polynomial (`Range2(x) = x^2 - x`) evaluated at all indices
that are expected to be bits, with the verification joint randomness derived
in TODO.

Assume we have a total of `n_b` number of indices to verify, the security and
privacy guarantee of this protocol is characterized by the following:

* Soundness: With Corollary 4.7 of {{BBCGGI19}}, the number of variables in
  this degree-2 polynomial is `n_b`, so the soundness error is
  `2 * sqrt(n_b) / (q - sqrt(n_b))`. And because we compute a random linear
  combination of the results over all indices, the soundness error increases
  by `n_b / q`, based on Remark 4.8 of {{BBCGGI19}}. The overall soundness
  error is therefore `2 * sqrt(n_b) / (q - sqrt(n_b)) + n_b / q`. If we repeat
  this protocol for `t` times, the soundness error reduces exponentially
  with respect to `t`, i.e. `(2 * sqrt(n_b) / (q - sqrt(n_b)) + n_b / q)^t`.

* Completeness: Valid Clients under this protocol will always be accepted.

* Zero-Knowledge: This protocol is perfect honest-verifier zero-knowledge.

### L2-Norm Sum-Check Protocol {#l2-sum-check}

The goal of this protocol is to verify the sum of square (i.e. squared
L2-norm) of values in `X` modulo field size `q` is at most `B`, which is the
squared L2-norm bound in field integer.

To verify this inequality, we will first turn this inequality check into an
equality check, per Section 4.1 of the PINE paper, with `\beta_1` equal to 0,
and `\beta_2` equal to `B`. Additionally, the minimum field size requirement for
this protocol MUST be at least `3 * B + 2`.

Clients are expected to compute `V0 = \SUM_i X[i]^2`, and also the difference
`U0 = B - V0`, and send both `V0` and `U0` as bits in its vector. The number of
bits to represent both MUST be `b0`, which is equal to `log_2(B + 1)`.

The Client vector after running this protocol looks like the following:

~~~
|--- X ---|--- bits of V0 ---|--- bits of U0 ---|
~~~

Aggregators then collectively verify the following:

1. The entries that encode `V0` and `U0` are indeed bits, which can be
   achieved by 0/1 bit check protocol {{bit-check}}.

1. The computed squared L2-norm from the secret shares of `X` matches the
   secret-shared bit representation of `V0`, as per Section 4.3 of the
   PINE paper.

1. `V0 + U0 = B`, based on the secret-shared bit representation of `V0`
   and `U0`, as per Section 4.1 of the PINE paper.

It's important to recognize that verifying property 2 involves a
non-affine operation, that is a degree-2 polynomial to square each value in `X`,
`Sq(x) = x^2`. Then we verify the summation of the polynomials is equal to
`V0`. We will again use the same proof system in {{Section 7.3.1 of !VDAF}} and
{{Section 7.3.1.1 of !VDAF}}, by constructing a degree-2 polynomial gadget
at each value of `X` and computing the summations of the gadgets. Specifically,
assume `inp` is the final vector from the Client, and the vector has the first
`d` entries for `X`, and next `b0` entries for `V0`, and next `b0` entries
for `U0`. The exact circuit `SC_0` computed for the second property is:

~~~
rec_V0(inp) = 2^0 * inp[d] + 2^1 * inp[d + 1] + ... + \
              2^(b0 - 1) * inp[d + b0 - 1]
SC_0(inp) = Sq(inp[0]) + Sq(inp[1]) + ... + Sq(inp[d - 1]) - rec_V0(inp)
~~~

Property 3 contains all affine operations, so each Aggregator can
directly compute its circuit output with its secret shares locally, and
exchange the outputs. The exact circuit `SC_1` computed for the third property
is:

~~~
rec_U0(inp) = 2^0 * inp[d + b0] + 2^1 * inp[d + b0 + 1] + ... + \
              2^(b0 - 1) * inp[d + 2 * b0 - 1]
SC_1(inp) = rec_V0(inp) + rec_U0(inp) - B / SHARES
~~~

The Client vector is deemed valid if and only if both `SC_0` and `SC_1`
evaluate to zero. Note we divide `B` by `SHARES` to account for the case when
this circuit is computed in the secret shares, when the number of Aggregators
is `SHARES`.

The security and privacy guarantee of this protocol is defined as the
following:

* Soundness: We already include the soundness error of verifying property 1 in
  {{bit-check}}. Now the soundness error of this protocol only comes from
  verifying property 2. We again use Corollary 4.7 of {{BBCGGI19}}, so the
  soundness error is `2 * sqrt(d) / (q - sqrt(d))`. By repeating the degree-2
  check over `t` repetitions, the soundness error becomes
  `(2 * sqrt(d) / (q - sqrt(d)))^t`.

* Completeness: Valid Clients under this protocol will always be accepted.

* Zero-Knowledge: This protocol is perfect honest-verifier zero-knowledge.

Note if the Client manages to overflow the field size `q` with its L2-norm, but
keeps the L2-norm at most `B` modulo `q`, the Client will be accepted under
this protocol, but will be rejected by wraparound protocol {{wraparound}}
with high probability.

### Wraparound Protocol {#wraparound}

The goal of this protocol is to verify the sum of square of values in `X`
doesn't overflow field size `q`, i.e. whether there is a "wraparound".
This protocol requires field size `q` to be at least
`max(81 * B * ln(2 / ETA), 10000, 2 * r)`, where `ETA` is the completeness error
of one repetition of the protocol, `r` is the number of repetitions.

Each repetition needs a random vector `Z` of length `d` provided by the
Aggregators. Each value in `Z` is sampled independently to be -1 with
probability 1/4, 0 with probability 1/2, and 1 with probability 1/4.
Since we don't have any interaction from Aggregators to Clients in VDAF,
we will generate the pseudorandom generator (PRG) seed for `Z`
(i.e. "wraparound joint randomness") based on the derivation in TODO.
We will sample each value in `Z`, by looking at the generated bytes from the
PRG seed two bits at a time:

* If the bits are `00`, set the value to be `-1`, or `q-1` in field integer.

* If the bits are `01` or `10`, we set the value to be `0`.

* If the bits are `11`, set the value to be `1`.

We will define the lower bound `L` and upper bound `H` for the wraparound
protocol result. In the general case, `L` is `-ALPHA * sqrt(B)`, and `H` is
`ALPHA * sqrt(B)`, where `ALPHA = sqrt(ln(2 / ETA))`. In a particular
repetition `k` of the protocol, the Client computes `Y1_k`, the dot product of
`Z_k` and `X` modulo `q`, where `Z_k` is the random vector. If its result
modulo `q` falls in `[q - abs(L), q)` or `[0, H]`, the Client passes that
repetition, and fails otherwise.

We ask Clients to repeat the procedure for `r` number of times, to reduce the
overall soundness and completeness error of this protocol, with motivations
described in Section 4.2 of the PINE paper. We define a threshold `TAU`, so that
the Clients are required to pass at least `TAU * r` repetitions. Otherwise,
they SHOULD abort, and Aggregators MUST reject.

To be more specific about each repetition `k` of wraparound protocol, assuming
the Client has computed `Y1_k`, if it passes that repetition, it then does the
following:

1. The Client computes `V1_k = Y1_k + abs(L)`, i.e. the shifted result, and
   encodes its bit representation. Now `V1_k` MUST be in `[0, H + abs(L)]`.
   In order for Aggregators to verify `V1_k` doesn't exceed `H + abs(L)`, the
   Client also computes `U1_k = H + abs(L) - V1_k`, so Aggregators can run the
   protocol in Section 4.1 of the PINE paper to turn this inequality check into
   equality check.
1. It records a success bit `g_k` of 1, indicating it has passed.
1. It sets `S_k = 0`. This is a single field element being secret-shared.

If the Client fails a repetition `k`, it does the following:

1. It sets the bits of `V1_k` to represent the value `H + abs(L)`,
   and the bits of `U1_k` to be all zeros.
1. It records a success bit `g_k` of 0.
1. It sets `S_k` to be `Y1_k + abs(L) - V1_k`.
   This should be non-zero, since we assume the Client fails.

Aggregators then collectively verify the following properties:

1. The entries that encode all `V1`, `U1`, `g` are indeed bits, which can be
   achieved by {{bit-check}}.

1. For each repetition `k`, verify the linear equality
   `V1_k + U1_k = H + abs(L)`, per Section 4.1 of the PINE paper.
   This verifies `V1_k` doesn't exceed `H + abs(L)`.

1. For each repetition `k`, verify the linear equality
   `S_k = (SUM_i Z_(k,i) * X_i) + abs(L) - V1_k`. This ensures the
   Aggregator-computed shifted wraparound protocol result matches what the
   Client encoded in `V1_k` and `S_k`.

1. For each repetition `k`, verify degree-2 polynomial `g_k * S_k = 0`, which
   ensures at least one of `g_k` and `S_k` is 0, i.e. if success bit `g_k` is 1,
   `S_k` MUST be 0.

1. `\SUM_k g_k = TAU * r`, the Client has passed at least `TAU * r` repetitions.

It is important to notice an optimization proposed in Remark 4.12 of the
PINE paper, that if we can come up with wraparound protocol bounds
`L >= -ALPHA * sqrt(B)` and `H <= ALPHA * sqrt(B)`, such that `H + L + 1` is a
power of 2, we can avoid asking Clients to send `U1_k` for each repetition, and
Aggregators don't have to verify property 2 above, because the maximum number of
bits to encode `V1_k` naturally limits the upper bound.

One way to achieve this is to come up with an `ALPHA' < ALPHA`, such that it
fulfills the following conditions:

* `2 * ALPHA' * sqrt(B) + 2` is a power of 2.

* `L = -ALPHA' * sqrt(B)` and `H = ALPHA' * sqrt(B) + 1`.

For the rest of this draft, we will assume these bounds for wraparound protocol,
because this is a non-trivial saving in communication cost between Clients and
Aggregators to avoid sending `U1_k`, as we repeat this protocol for `r` times.

We ask the Client to submit its vector with wraparound protocol results in the
following order:

~~~
|--- X ---|--- bits of V0 ---|--- bits of U0 ---|
|--- bits of V1_0 ---|--- g_0 ---|--- bits of V1_1 ---|--- g_1 ---|...
|--- bits of V1_(r-1) ---|--- g_(r-1) ---|
|--- S_0 ---|--- S_1 ---|...|--- S_(r-1) ---|
~~~

Therefore, for each repetition `k`, we know the number of bit entries needed by
`V1_k`, `g_k`, and also the starting index offsets of `V1_k`, `g_k`, and `S_k`.
The index offsets are computed as the following:

~~~
num_bits_per_rep = b1 + 1
offset_V1(k) = d + 2 * b0 + k * num_bits_per_rep
offset_g(k) = d + 2 * b0 + k * num_bits_per_rep + b1
offset_S(k) = d + 2 * b0 + r * num_bits_per_rep + k
~~~

Verifiers can then reconstruct `V1_k` from their bits, and compute `Y1_k`,
which is the dot product of `Z_k` and `X` modulo `q`. They are computed
as the following:

~~~
rec_V1(inp, k) = 2^0 * inp[offset_V1(k)] + 2^1 * inp[offset_V1(k) + 1] + \
                 2^(b1 - 1) * inp[offset_V1(k) + b1 - 1]
comp_Y1(Z, X, k) = Z[k * d] * X[0] + Z[k * d + 1] * X[1] + ... + \
                   Z[(k + 1) * d - 1] * X[d - 1]
~~~

And therefore, for each repetition `k`, Aggregators collectively verify
the following circuits are all zero, for properties 3 and 4 above:

~~~
WC_0(inp, k) = comp_Y1(Z, X, k) + ALPHA' * sqrt(B) / SHARES - rec_V1(inp, k) - \
               inp[offset_S(k)]
WC_1(inp, k) = inp[offset_g(k)] * inp[offset_S(k)]
~~~

And eventually, Aggregators verify property 5:

~~~
r_pass = floor(TAU * r)
WC_2(inp) = inp[offset_g(0)] + inp[offset_g(1)] + ... + inp[offset_g(k - 1)] - \
            r_pass / SHARES
~~~

The Client vector is deemed valid if and only if `WC_0` and `WC_1` are both zero
for all `k`, and `WC_2` is zero. Note we divide the fixed constants by `SHARES`
to account for the case when these circuits are computed in the secret shares,
when the number of Aggregators is `SHARES`.

`WC_0` and `WC_2` only contain affine operations, so each Aggregator
can compute its local outputs and exchange them with other Aggregators, but
`WC_1` contains a non-affine operation, on two variables that are both
secret-shared. We will again use the proof system in {{{Section 7.3.1 of !VDAF}}
and {{Section 7.3.1.1 of !VDAF}} to construct a `Mul` gadget between `g_k` and
`S_k`, and computes a random linear combination of all `Mul` gadget results.
That is, with verification joint randomness `r_v` derived in TODO,
the following circuit is expected to be 0:

~~~
WC_1(inp) = r_v * WC_1(inp, 0) + r_v^2 * WC_1(inp, 1) + ... +
            r_v^r * WC_1(inp, r - 1)
~~~

The privacy and security guarantee of the entire wraparound protocol is
characterized from the following aspects:

* Soundness: The soundness error of one repetition of wraparound protocol is
  1/2. Over `r` repetitions, the soundness error is therefore
  `Bin((TAU * r); r, 1/2)`, see Lemma 4.3 and also Claim 4.9 in PINE paper.
  There is also a soundness error when we verify the degree-2 polynomial in
  `WC_1` over all `r` repetitions. The soundness error is computed based on
  Corollary 4.7 and Remark 4.8 of {{BBCGGI19}},
  `2 * sqrt(r) / (q - sqrt(r)) + r / q`. If we repeat this degree-2 check
  over `t` repetitions, the overall soundness error is
  `Bin((TAU * r); r, 1/2) + (2 * sqrt(r) / (q - sqrt(r)) + r / q)^t`.

* Completeness: The completeness error of one repetition of wraparound protocol
  is `ETA`, or `2 / e^(ALPHA' ^ 2)`, where `ALPHA'` is the actual parameter used
  after applying optimization in Remark 4.12 in the PINE paper.
  Therefore, the completeness error over all repetitions is
  `1 - Bin((TAU * r); r, 1 - ETA)`, see Lemma 4.3 and also Claim 4.9 in
  PINE paper. This is the probability that a Client fails more than
  `(1 - TAU) * r` repetitions.

* `rho`-Statistical Zero-Knowledge: Aggregators learn close to nothing about
  an honest Client. The zero knowledge leakage is quantified by the
  completeness error `rho` from above, which is negligible. See discussion in
  Lemma 4.3 of the PINE paper.

### Summary of PINE Protocols {#summary-protocols}

Aggregators MUST verify the circuits among all of the above protocols evaluate
to zero. We just described the final Client vector after running all the
protocols, we can specify exactly the bits to verify in {{bit-check}}.
These include:

* indices in `[d, d + b0)` that encode `V0`,

* indices in `[d + b0, d + 2 * b0)` that encode `U0`,

* indices in `[d + 2 * b0, d + 2 * b0 + r * (b1 + 1))` that encode
  `V1`, and `g` over all `r` repetitions of wraparound protocol.

Assume we compute the verification joint randomness `r_v` based on TODO.
The circuit to compute for {{bit-check}} is:

~~~
RC_0(inp) = r_v * Range2(inp[d]) + r_v^2 * Range2(inp[d + 1]) + ... +
            r_v^(b0) * Range2(inp[d + b0 - 1]) +
            r_v^(b0 + 1) * Range2(inp[d + b0]) +
            r_v^(b0 + 2) * Range2(inp[d + b0 + 1]) + ... +
            r_v^(2 * b0) * Range2(inp[d + 2 * b0 - 1]) +
            r_v^(2 * b0 + 1) * Range2(inp[d + 2 * b0]) +
            r_v^(2 * b0 + 2) * Range2(inp[d + 2 * b0 + 1]) + ... +
            r_v^(2 * b0 + r * (b1 + 1) + 1) *
            Range2(inp[d + 2 * b0 + r * (b1 + 1)])
~~~

Finally, we only consider a Client has passed the protocol if all the circuits
evaluate to 0:

* `RC_0`, the reduced output of 0/1 bit check protocol {{bit-check}},

* `SC_0` and `SC_1`, the outputs of L2-norm sum-check protocol {{l2-sum-check}},

* `WC_0` for all repetitions of wraparound protocol {{wraparound}},

* `WC_1` and `WC_2` in wraparound protocol {{wraparound}}.


# PINE Fully Linear Proof (FLP) in VDAF {#pine-flp}

TODO


# Security Considerations {#security}

PINE achieves the following privacy and security guarantees:

* Soundness: Malicious Clients should be prevented from sending vectors with
  invalid L2-norm and poisoning the final aggregated result with
  high probability.

* Completeness: Honest Clients should be accepted with high probability.

* Privacy: An attacker that controls the Collector, a subset of Clients, and
  all but one Aggregators, learns statistically close to nothing about the
  measurements of honest Clients.


# IANA Considerations

This document has no IANA actions.


--- back

# Acknowledgments
{:numbered="false"}

TODO acknowledge.
