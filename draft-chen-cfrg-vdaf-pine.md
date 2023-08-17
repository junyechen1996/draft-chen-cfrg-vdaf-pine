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

  MR17:
    title: "Federated Learning: Collaborative Machine Learning without Centralized Training Data"
    author:
      - ins: B. McMahan
      - ins: D. Ramage
    date: 2017
    target: https://ai.googleblog.com/2017/04/federated-learning-collaborative.html

  Lem12:
    title: "Cauchy and the gradient method"
    author:
      - ins: C. Lemaréchal
    date: 2012
    target: https://www.elibm.org/article/10011456

  PINE:
    title: "[TODO: Add arxiv link when it's ready]"

  Tal22:
    title: "Differential Secrecy for Distributed Data and Applications to Robust Differentially Secure Vector Summation"
    author:
      - ins: K. Talwar
    date: 2022
    target: https://arxiv.org/abs/2202.10618

  DivviUpVDAF:
    title: "DivviUp LibPrio Rust"
    target: https://github.com/divviup/libprio-rs

  IEEE754-2019:
    title: "IEEE Standard for Floating-Point Arithmetic"
    date: 2019
    target: https://ieeexplore.ieee.org/document/8766229

--- abstract

A Verifiable Distributed Aggregation Function (VDAF) named
Private Inexpensive Norm Enforcement (PINE) that supports aggregating
high-dimensional real number vectors bounded by a configurable L2-norm bound,
which is a fundamental primitive to support private federated learning.


--- middle

# Introduction


> CP: Here is a proposed rewrite. (This is a skeleton, some bits still need to
> be filled out.)

The goal of federated learning {{MR17}} is to enable training of machine
learning models from data stored on user' devices. The bulk of the computation
is carried out on-device: each user trains the model on its data locally, then
sends a model update to a central server. These model updates are commonly
referred to as a "gradients" {{Lem12}}. The server then aggregates the
gradients, applies them to the central model, and sends the updated model to
the users to repeat the process.

> CP: A diagram showing how this works would be helpful here.

Federated learning improves user privacy by ensuring the training data never
leaves the device. However, the gradients themselves reveal a significant
amount of information about each user's input. [CP: 1-2 sentences describing
the risk here would be useful.] One way to mitigate this risk is to distribute
the aggregation step across multiple servers such that no server sees any
gradient in the clear.

In a Verifiable Distributed Aggregation Function
{{!VDAF=I-D.draft-irtf-cfrg-vdaf-06}}, this is achieved by having each user
shard their gradient into a number of secret shares, one for each aggregation
server. Each server aggregates their shares locally, then combines their share
of the aggregate with the other servers to get the aggregate result.

Along with keeping the gradients privacy, it is also desirable to ensure
robustness of the overall computation. In particular, to prevent clients from
"posioning" the model, federated learning systems typically reject gradients
whose L2-norm exceed a certain threeshold. [CP: Say that the gradients are
vectors of real numbers, define L2-norm, and say why this bound is significant.
Is it just that this is what federated learning people have decided is
sufficient to get reasonable performance?]

This dcoument describes Pine {{PINE}} ("P"rivate "I"nexpensive "N"orm
"E"forcement"), a VDAF for secure aggregation of gradients for federated
learning. Pine shares many of the same techniques as Prio3 {{Section 7 of
!VDAF}}, including the use of Fully Linear Proofs {{Section 7.1 of !VDAF}} for
validating the gradients. However, Pine introduces a new technique and
supporting analysis that, for high-dimensonal data, significantly improves
communication cost compared to what appears to be possible for Prio3.

We given an overview of this technique in {{wraparound-overview}}. In {{flp}}
we describe our FLP for bounded L2 norm. In {{vdaf}} we describe the complete
Pine VDAF.

> CP: BELOW HERE IS THE ORIGINAL INTRO, WITH SOME INLINE COMMENTS.

Aggregating high-dimensional real number vectors is a fundamental primitive
to support federated learning {{MR17}}, that allows data scientists to train
machine learning models with data from many users' devices. Each user's device
will train the model with its local data and send the model updates to the
servers. The model updates are typically referred to as "gradients" {{Lem12}},
and are typically expressed as a vector of real numbers. [CP: The data type of
the gradient is not relevant (yet!)] The servers will obtain the aggregated
model updates, and apply them to the central model. This process repeats as the
servers send the new model to the devices.

There have been various approaches discussed in the Introduction section of
{{PINE}} to support such use cases, [CP: Don't assume the reader has read the
paper. You need to sell it here, too.] but they either only implement an
approximate verification of Client measurements ({{Tal22}}), or incur a high
communication overhead between Client and Aggregators, e.g. VDAF
{{!VDAF=I-D.draft-irtf-cfrg-vdaf-06}} proposes a Prio3 scheme as a multi-party
computation (MPC) protocol to verify certain property of each Client
measurement, and the proposed implementation in {{DivviUpVDAF}} has to
secret-share each bit of each vector dimension as a finite field element in the
Client measurement.

In this document, we propose a VDAF that enables more efficient and accurate
secure aggregation that conforms to the VDAF interface {{!VDAF}}. We want to
achieve the following properties in our VDAF:

* Malicious Clients should be prevented from sending vectors with invalid
  L2-norm with high probability. The L2-norm of a vector is defined as the
  square root of the sum of squares of values at all vector dimensions.
  Rejecting vectors with high L2-norm helps prevent Clients from poisoning the
  final aggregate result obtained by the Aggregators and minimizes the risk of
  training a bad machine learning model.

  > CP: Minimizes the risk or eliminates it? What do you mean by "bad"? The
  > abstract of the paper mentions "poisoning attacks"; I'd suggest defining
  > this here in the intro.

* Honest Clients should be accepted with high probability.

  > CP: Do you mean the measurements generated by honest Clients?

  > CP: Prio3 always accepts honest measurements. It's worth nothing here that
  > PINE trades non-zero completeness error for reduced proof size.

* An attacker that controls the Collector, a subset of Clients, and all but one
  Aggregators, learns statistically close to nothing about the measurements of
  honest Clients.

This document aims to extend the Prio3 scheme {{Section 7 of !VDAF}} to support
aggregating real number vectors with bounded L2-norm, specifically
from the following aspects:

* A new type of joint randomness agreed between Client and Aggregators, that
  supports the computation of "wraparound protocol" in PINE VDAF. We denote it
  as "wraparound joint randomness".

* A "Fully Linear Proof (FLP)" system that incorporates the
  "wraparound joint randomness" and supports PINE as a VDAF.




# Conventions and Definitions

{::boilerplate bcp14-tagged}

This document uses the same parameters and conventions specified for:

* Definitions for VDAF in {{Section 5 of !VDAF}}.

* Field elements in {{Section 6.1 of !VDAF}}.

* Pseudorandom Generators in {{Section 6.2 of !VDAF}}.

* Prio3 FLP system in {{Section 7 of !VDAF}}, with some extensions.

The computation parameters used in the protocols of PINE are listed in
{{pine-comp-param}}:

| Parameter | Type | Description    |
|:----------|:-----|----------------|
| `l2_norm_bound` | float | L2-norm bound. This is an inclusive upper bound. Users of PINE can use this parameter to control the L2-norm bound of its Client vectors. |
| `dimension`     | Unsigned | Client vector dimension. |
| `num_frac_bits` | Unsigned | Number of binary fractional bits to keep in Client vector values. Users of PINE can use this parameter to control the precision. We require this parameter to be less than 128 as specified in {{fp-encoding}}. |
{: #pine-user-param title="User parameters for PINE."}

> TODO: Figure out which of these are needed to describe the algorithm. For
> those that are not needed, remove them. For those that are needed, describe
> them as fucntions of `l2_norm_bound`, etc.

| Parameter | Description |
|:----------|:------------|
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

# Overview of the "Wraparound Check" in Pine {#wraparound-overview}

> CP: Describe a circuit for verifying that the L2-norm (modulo q) is in the
> desired range.

This simple computation is sufficient as long as the sum of squares does not
"wrap around" the field modulus. But when evaluated over secret shares of data
represented in a finite field, there is no way for the Aggregators to determine
if this has happened.

One way to mitigate this issue is to encode the gradient so that wrap around is
impossible. [CP: Summarize what's going on in the SumVecBoundedL2Norm type in
llibprio.] However this results in a signficant amount of communication
overhead, since [CP. why?].

The key technical idea underlying Pine is a stastical test carried out by the
verifier (with assistance by the prover). [CP: Describe how this works at a
high level.]

> CP: End this section with the headlines from the analysis. Namely:
>
> * Pine admits a small completeness error, resulting from false negative
>   wraparound tests (is this the right way to describe this?)
> * The FLP is statistical zero-knowledge rather than perfect zero-knowledge
>   (this is not a big deal but is worth mentioning)

# An FLP for Pine {#flp}

> CP: Here is where the most of the technical meat goes. The goal here is to
> describe the end-to-end FLP, including the wraparound check, the bit checks,
> and the norm check.
>
> Say up front that we're describing an instance of FlpGeneric
> {{Section 7.3 of !VDAF}}, i.e., we're describing an arithmetic circuit.
>
> Assume you get the wraparound randomness and joint randomness from the sky.
> (We'll describe how it's derived in the next section.) Then put them together
> into an implementation of the Flp interface.

# The Pine VDAF {#vdaf}

> CP: Here's where you define the end-to-end protocol as an implementation of
> the Vdaf interface. Here you'll use the Flp described in the previous section
> and derive the wraparound and joint randomness.

# BELOW HERE IS THE TEXT THAT NEEDS TO BE MASSAGED INTO THE SECTIONS ABOVE.

# Private Inexpensive Norm Enforcement (PINE) VDAF {#pine}

PINE supports aggregating real number vectors bounded by a configured L2-norm.
It asks each Client to compute results needed by protocols in PINE to verify
the L2-norm, and then send the results in the submitted vector. In this draft,
we are specifically interested in the implementation of the statistical
zero-knowledge protocol in section 4 of {{PINE}}, in the VDAF setting.

The Clients are expected to first encode their real number vectors into a
vector of field integers, because all the protocols in PINE are computed and
verified with field integers. The vectors of field integers are then aggregated
and decoded into a vector of IEEE-754 compatible float64 values.

## Encoding Real Numbers into Field Integers {#fp-encoding}

To keep the necessary number of precision bits `f` configured in PINE VDAF,
the high-level idea is to multiply a real number by `2^f`, and provide a 1-1
mapping between the multiplied value and a field integer. In order to represent
real numbers with sufficient precision and also allow implementations to
represent `2^f` in common integer representations, we require `f < 128`, which
is precise enough to represent 127 binary fractional bits, or equivalently,
approximately 38 decimal bits.

We don't recommend a specific real number representation during encoding,
but a common choice is to use some fixed-point precision representation or
IEEE-754 compatible float64 representation {{IEEE754-2019}}. In this draft,
we will use IEEE-754 float64 as an example, because it's commonly available
in most hardwares.

We consider the following float64 values as invalid inputs during encoding:

* NaN, which can be obtained by dividing 0.0 by 0.0.
* Positive and negative infinity, which can be obtained by dividing any non-zero
  value by 0.0.
* Subnormal numbers, whose absolute values are less than `2.225e-308`. Since
  we require `f < 128`, it's not possible to use unique field integers to
  represent subnormal numbers.

Otherwise, for a float64 value `x`, we will encode the result of
`floor(x * 2^f)` into a field integer. We will reserve the first half of the
field size `[1, floor(q/2)]` to encode positive values, and the second half of
the field size `[ceil(q/2), q)` to encode negative values. Specifically:

* A positive or negative 0.0 in float64 will be encoded into a field integer
  of 0. IEEE-754 float64 has the notion of both positive and negative 0, but
  their values are equal for our purposes.
* If `x` is positive, encode it as `floor(x * 2^f)`.
* If `x` is negative, encode it as `q - floor(|x| * 2^f)`.

Since field size `q` is a prime number, the number of unique values in both
ranges should be equal. This encoding will effectively map the negative values
in `[-b, 0)` to `[q-sqrt(B), q)`, and map the positive values in `[0, b]` to
`[0, sqrt(B)]`, where `b` is the L2-norm bound and also the maximum value of
any vector element, `B` is the squared L2-norm bound in field integer,
computed via `(floor(b * 2^f))^2`. We must make sure the two ranges don't
overlap with each other, i.e. the chosen values of `b` and `f` must satisfy
`floor(b * 2^f) <= floor(q/2)`.

## Aggregation {#agg}

The aggregation of the real number vectors is therefore done in field integers
as well, after each original Client vector is encoded into a vector of field
integers.

## Decoding Field Integers into IEEE-754 Compatible Float64 {#fi-decoding}

To decode a (aggregated) field integer `agg` back to float, assuming there are
`c` clients, the aggregated field integer should fall in the range of
`[-c * sqrt(B), c * sqrt(B)]`. Based on the encoding mechanisms in
{{fp-encoding}}, we decode the aggregated field integer into float64 as the
following:

* If the field integer is 0, the aggregated float64 value should be 0.0.
* If the field integer `agg` is in `[q - c * sqrt(B), q)`, the aggregated
  float64 value should be negative. We will decode `agg` as
  `-((q - agg) / 2^f)`.
* If the field integer `agg` is in `(0, c * sqrt(B)]`, the aggregated
  float64 value should be positive. We will decode `agg` as `agg / 2^f`.

Similar to how we require field integer ranges for positive and negative
float64 values to not overlap, we also need to make sure the number of Clients
`c` is not so big that it causes the two ranges after aggregation to overlap.
Therefore, we must check that `c * sqrt(B) <= floor(q/2)`.

## Validity Checks for PINE

Similar to Prio3, each Client's measurement (a vector of floating point
numbers) is encoded as a vector over a finite field and secret-shared among the
Aggregators. During preparation, the Aggregators verify the following
properties of the encoded vector:

1. L2-norm sum-check: TODO(junye) Describe this at a high level.
1. 0/1 bit check: TODO(junye)
1. Wraparound protocol: TODO(junye)

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
in {{pine-verification-joint-randomness}}.

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
equality check, per Section 4.1 of {{PINE}}, with `\beta_1` equal to 0,
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
   secret-shared bit representation of `V0`, as per Section 4.3 of {{PINE}}.

1. `V0 + U0 = B`, based on the secret-shared bit representation of `V0`
   and `U0`, as per Section 4.1 of {{PINE}}.

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
(i.e. "wraparound joint randomness") based on the derivation in
{{pine-wraparound-joint-randomness}}.
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
described in Section 4.2 of {{PINE}}. We define a threshold `TAU`, so that
the Clients are required to pass at least `TAU * r` repetitions. Otherwise,
Clients SHOULD retry, but Aggregators MUST reject.

To be more specific about each repetition `k` of wraparound protocol, assuming
the Client has computed `Y1_k`, if it passes that repetition, it then does the
following:

1. The Client computes `V1_k = Y1_k + abs(L)`, i.e. the shifted result, and
   encodes its bit representation. Now `V1_k` MUST be in `[0, H + abs(L)]`.
   In order for Aggregators to verify `V1_k` doesn't exceed `H + abs(L)`, the
   Client also computes `U1_k = H + abs(L) - V1_k`, so Aggregators can run the
   protocol in Section 4.1 of {{PINE}} to turn this inequality check into
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
   `V1_k + U1_k = H + abs(L)`, per Section 4.1 of {{PINE}}.
   This verifies `V1_k` doesn't exceed `H + abs(L)`.

1. For each repetition `k`, verify the linear equality
   `S_k = (SUM_i Z_(k,i) * X_i) + abs(L) - V1_k`. This ensures the
   Aggregator-computed shifted wraparound protocol result matches what the
   Client encoded in `V1_k` and `S_k`.

1. For each repetition `k`, verify degree-2 polynomial `g_k * S_k = 0`, which
   ensures at least one of `g_k` and `S_k` is 0, i.e. if success bit `g_k` is 1,
   `S_k` MUST be 0.

1. `\SUM_k g_k = TAU * r`, the Client has passed at least `TAU * r` repetitions.

It is important to notice an optimization proposed in Remark 4.12 of {{PINE}},
that if we can come up with wraparound protocol bounds
`L >= -ALPHA * sqrt(B)` and `H <= ALPHA * sqrt(B)`, such that `H + L + 1` is a
power of 2, we can avoid asking Clients to send `U1_k` for each repetition, and
Aggregators don't have to verify property 2 above, because the maximum number of
bits to encode `V1_k` naturally limits the upper bound.

One way to achieve this is to come up with an `ALPHA' < ALPHA`, such that it
fulfills the following conditions:

* `ALPHA' * sqrt(B) + 1` is a power of 2.

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
secret-shared. We will again use the proof system in {{Section 7.3.1 of !VDAF}}
and {{Section 7.3.1.1 of !VDAF}} to construct a `Mul` gadget between `g_k` and
`S_k`, and computes a random linear combination of all `Mul` gadget results.
That is, with verification joint randomness `r_v` derived in
{{pine-verification-joint-randomness}}, the following circuit is expected
to be 0:

~~~
WC_1(inp) = r_v * WC_1(inp, 0) + r_v^2 * WC_1(inp, 1) + ... +
            r_v^r * WC_1(inp, r - 1)
~~~

The privacy and security guarantee of the entire wraparound protocol is
characterized from the following aspects:

* Soundness: The soundness error of one repetition of wraparound protocol is
  1/2. Over `r` repetitions, the soundness error is therefore
  `Bin((TAU * r); r, 1/2)`, see Lemma 4.3 and also Claim 4.9 in {{PINE}}.
  There is also a soundness error when we verify the degree-2 polynomial in
  `WC_1` over all `r` repetitions. The soundness error is computed based on
  Corollary 4.7 and Remark 4.8 of {{BBCGGI19}},
  `2 * sqrt(r) / (q - sqrt(r)) + r / q`. If we repeat this degree-2 check
  over `t` repetitions, the overall soundness error is
  `Bin((TAU * r); r, 1/2) + (2 * sqrt(r) / (q - sqrt(r)) + r / q)^t`.

* Completeness: The completeness error of one repetition of wraparound protocol
  is `ETA`, or `2 / e^(ALPHA' ^ 2)`, where `ALPHA'` is the actual parameter used
  after applying optimization in Remark 4.12 in {{PINE}}.
  Therefore, the completeness error over all repetitions is
  `1 - Bin((TAU * r); r, 1 - ETA)`, see Lemma 4.3 and also Claim 4.9 in
  {{PINE}}. This is the probability that a Client fails more than
  `(1 - TAU) * r` repetitions.

* `rho`-Statistical Zero-Knowledge: Aggregators learn close to nothing about
  an honest Client. The zero knowledge leakage is quantified by the
  completeness error `rho` from above, which is negligible. See discussion in
  Lemma 4.3 of {{PINE}}.

### Summary of PINE Protocols {#summary-protocols}

Aggregators MUST verify the circuits among all of the above protocols evaluate
to zero. We just described the final Client vector after running all the
protocols, we can specify exactly the bits to verify in {{bit-check}}.
These include:

* indices in `[d, d + b0)` that encode `V0`,

* indices in `[d + b0, d + 2 * b0)` that encode `U0`,

* indices in `[d + 2 * b0, d + 2 * b0 + r * (b1 + 1))` that encode
  `V1`, and `g` over all `r` repetitions of wraparound protocol.

Assume we compute the verification joint randomness `r_v` based on
{{pine-verification-joint-randomness}}. The circuit to compute for {{bit-check}}
is:

~~~
RC_0(inp) = r_v * Range2(inp[d]) + r_v^2 * Range2(inp[d + 1]) + ... +
            r_v^(b0) * Range2(inp[d + b0 - 1]) +
            r_v^(b0 + 1) * Range2(inp[d + b0]) +
            r_v^(b0 + 2) * Range2(inp[d + b0 + 1]) + ... +
            r_v^(2 * b0) * Range2(inp[d + 2 * b0 - 1]) +
            r_v^(2 * b0 + 1) * Range2(inp[d + 2 * b0]) +
            r_v^(2 * b0 + 2) * Range2(inp[d + 2 * b0 + 1]) + ... +
            r_v^(2 * b0 + r * (b1 + 1)) *
            Range2(inp[d + 2 * b0 + r * (b1 + 1) - 1])
~~~

Finally, we only consider a Client has passed the protocol if all the circuits
evaluate to 0:

* `RC_0`, the reduced output of 0/1 bit check protocol {{bit-check}},

* `SC_0` and `SC_1`, the outputs of L2-norm sum-check protocol {{l2-sum-check}},

* `WC_0` for all repetitions of wraparound protocol {{wraparound}},

* `WC_1` and `WC_2` in wraparound protocol {{wraparound}}.


# PINE Fully Linear Proof (FLP) in VDAF {#pine-flp}

We aim to define a FLP for PINE that reuses most of Prio3 FLP components in
{{Section 7.1 of !VDAF}}, with a new type of joint randomness named
"wraparound joint randomness" to support wraparound protocol {{wraparound}}
in PINE.

We reuse all the FLP parameters in {{Section 7.1 of !VDAF}}, and add the
parameters used for wraparound joint randomness. All the parameters are listed
in {{pine-flp-param}}:

| Parameter                     | Description               |
|:------------------------------|:--------------------------|
| `PROVE_RAND_LEN`              | Length of the prover randomness, the number of random field elements consumed by the prover when generating a proof |
| `QUERY_RAND_LEN`              | Length of the query randomness, the number of random field elements consumed by the verifier |
| `WRAPAROUND_JOINT_RAND_LEN`   | Length of the joint randomness needed during computation of wraparound protocol {{wraparound}} in PINE. This length should be equal to `d * r`, with `d` elements for each run of wraparound protocol. |
| `VERIFICATION_JOINT_RAND_LEN` | Length of joint randomness used to reduce over multiple circuit outputs in {{Section 7.3.1.1 of !VDAF}}, to trade a small soundness error for a shorter proof, as described in Remark 4.8 of {{BBCGGI19}}. |
| `JOINT_RAND_LEN`              | Length of the total joint randomness of `WRAPAROUND_JOINT_RAND_LEN` and `VERIFICATION_JOINT_RAND_LEN` |
| `INPUT_LEN`                   | Length of the encoded measurement ({{pine-flp-encoding}}). The encoded measurement is treated as the input into the FLP. |
| `OUTPUT_LEN`                  | Length of the aggregatable output ({{pine-flp-encoding}}) necessary to perform aggregation |
| `PROOF_LEN`                   | Length of the proof     |
| `VERIFIER_LEN`                | Length of the verifier message generated by querying the input and proof |
| `Measurement`                 | Type of the measurement. For PINE, this is a vector of floating point numbers.  |
| `AggResult`                   | Type of the aggregate result. For PINE, this is a vector of IEEE-754 compatible floating point numbers. |
| `Field`                       | As defined in {{Section 6.1 of !VDAF}} |
{: #pine-flp-param title="Constants and types used in PINE FLP."}


We will introduce PINE FLP in the following steps:

* The prover encodes its measurement into a vector of field integers as
  specified in {{pine-flp-encoding}}. The encoded measurement should
  contain the outputs of running the PINE protocols {{pine}}.

* The prover secret shares the encoded measurement into multiple shares.
  Each share is sent to a different verifier. The secret shares are also
  used to derive the wraparound joint randomness and verification
  joint randomness, specified in {{pine-joint-randomness}}.

* The prover and verifiers execute the FLP as specified in
  {{pine-flp-execution}} once the encoded measurement and joint randomness
  are available.

* Eventually we put all of the above together in the VDAF context.

## Encoding {#pine-flp-encoding}

PINE FLP encoding is divided into three stages:

1. The Client first encodes its real number vector into a vector of field
   integers as described in {{fp-encoding}}.

1. In order to fully execute the protocols in PINE, Client and Aggregators
   need to agree on the "wraparound joint randomness", which is necessary
   for wraparound protocol {{wraparound}}. The derivation of this randomness
   depends on the secret shares from the previous step, as specified
   in {{pine-wraparound-joint-randomness}}.

1. After wraparound joint randomness is derived, the Client encodes the results
   of L2-norm sum-check protocol {{l2-sum-check}} and wraparound protocol
   {{wraparound}}, in its vector.

Therefore, we have the following interfaces in `Pine.Flp`:

* `Pine.Flp.encode(measurement: Measurement) -> Vec[Field]` encodes a raw
  measurement of a vector of floating point numbers into a vector of field
  elements. The output length of this function MUST be equal to the length of
  the input measurement, i.e. `d`.

* `Pine.Flp.finalize_encoding_with_wraparound_joint_rand(
  partially_encoded: Vec[Field], wraparound_joint_rand: Vec[Field])
  -> Vec[Field]` executes the checks in PINE, and appends the results at the
  end of `partially_encoded`. `partially_encoded` is the output of
  `Pine.Flp.encode` function, and `wraparound_joint_rand` is the wraparound
  joint randomness necessary to execute wraparound check {{wraparound}}.
  The output length of this function MUST be `INPUT_LEN`.

Wraparound protocol {{wraparound}} describes a process of sampling
`wraparound_joint_rand` (the `Z` vector) from a seed stream directly, so we will
have another interface that allows the FLP to sample the field elements from
a `Prg` seed stream:

* `Pine.Flp.sample_wraparound_joint_rand(prg: Prg) -> Vec[Field]` allows PINE
  FLP to sample `Z` vector. The input `Prg` is the wraparound joint randomness
  as derived in {{pine-wraparound-joint-randomness}}. The output length of this
  function MUST be `WRAPAROUND_JOINT_RAND_LEN`.

> TODO We introduce this interface because it allows us to sample each value
in `Z` accurately, by looking at two bits at a time from the `Prg` seed stream,
as we are dealing with probabilities 1/4 and 1/2. We can also accept field
elements uniformly distributed in the field size, but field size `q` is always
a prime, and not divisible by 2 or 4.

We would like to directly reuse these two functions from Prio3 FLP in
{{Section 7.1.1 of !VDAF}}, specifically:

* `Flp.truncate(input: Vec[Field]) -> Vec[Field]` takes the first `OUTPUT_LEN`
  from `input` to perform aggregation. The input length to this function MUST
  be `INPUT_LEN`, and the output of this function MUST be `OUTPUT_LEN`.

* `Flp.decode(output: Vec[Field], num_measurements: Unsigned) -> AggResult`
  decodes an aggregated vector of field integers into a vector of IEEE-754
  compatible float64 values, as specified in {{fi-decoding}}.

## Joint Randomness {#pine-joint-randomness}

Joint randomness is used both by prover and verifiers to make sure they use
the same randomness to compute the protocols. Since VDAF doesn't allow interaction
between prover and verifiers, we will use Fiat-Shamir Heuristic to generate
the joint randomness, same as Prio3, but for two types of joint randomness.

### Wraparound Joint Randomness {#pine-wraparound-joint-randomness}

Wraparound protocol {{wraparound}} computation needs the `Z` vector. The Client
and Aggregators MUST both be able to derive this randomness, in order for them
to produce consistent outputs. The procedure for the Client is described as the
following:

1. The Client encodes its raw measurement into a vector of field integers as
   described in {{fp-encoding}}, which is also the first step of
   {{pine-flp-encoding}}, and produces the secret shares from the vector,
   with number of shares equal to the number of Aggregators.
1. Derive a wraparound joint randomness "part" for each Aggregator, based
   on the Client nonce, an initial seed "blind" for that Aggregator, and
   the secret share for that Aggregator.
1. Then derive the wraparound joint randomness seed, based on all wraparound
   joint randomness parts.

> TODO Kunal mentioned it would be good to add a VDAF TaskId to both joint
randomness part derivation, so the salt for joint randomness is fresh for each
VDAF initialization. Client "nonce" may be sufficient if Client is not allowed
to replay its share (which is the case in DAP), because we don't want a Client
that figures out an exploit in Fiat-Shamir to continue to attack the VDAF.

The Client sends its nonce, and derived seed parts to all Aggregators. The
Client also sends the respective seed blind, and secret share to each
Aggregator.

Each Aggregator MUST also know how to derive the wraparound joint randomness,
with the following steps:

1. Each Aggregator receives its seed blind and its secret shares, and uses the
   first `d` elements of the secret shares to derive the seed part.
1. Then it computes the wraparound joint randomness based on its computed seed
   part, and the seed parts for other Aggregators sent by the Client. Then the
   Aggregator can use the joint randomness seed to generate the `Z` vector.

### Verification Joint Randomness {#pine-verification-joint-randomness}

In 0/1 bit check protocol and wraparound protocol, we use the extensions in
{{Section 7.3.1.1 of !VDAF}} to compute a random linear combination of multiple
repeated degree-2 polynomials, which allows us to trade a small soundness error
for a shorter proof, as described in Remark 4.8 of {{BBCGGI19}}. We call the
coefficient of the random linear combination as "verification joint randomness",
which is same as the joint randomness in Prio3 FLP. The derivation is similar
to {{pine-wraparound-joint-randomness}}, but it uses the secret shares of
the Client outputs after it has completely finished {{pine-flp-encoding}}.

Specifically, the Client does the following:

1. The Client produces a vector of field integers from {{pine-flp-encoding}},
   and produces the secret shares from the vector. It should use the same
   helper seeds as those used in {{pine-wraparound-joint-randomness}}.
1. Derive a verification joint randomness "part" for each Aggregator, based
   on the Client nonce, an initial seed "blind" for that Aggregator, and
   the secret share for that Aggregator. It should use a different seed blind
   from the seed blind for wraparound joint randomness.
1. Then derive the verification joint randomness seed, based on all seed parts.

Each Aggregator follows the same procedure to derive the verification joint
randomness as the procedure described in {{pine-wraparound-joint-randomness}},
except each Aggregator needs to use the full `INPUT_LEN` of its secret share
to derive its verification joint randomness seed part.

The full derivation of joint randomness can also be found in
{{pine-joint-randomness-derivation}}.

## PINE FLP Instantiation {#pine-flp-instantiation}

This section instantiates `FlpGeneric` ({{Section 7.3.2 of !VDAF}}) for PINE
by specifying the validity circuit and parameters.

The encoding and decoding interface for PINE are defined as the following
based on {{pine-flp-encoding}}:

~~~
def encode(Pine, measurement: list[float]):
    encoded_measurement = []
    for x in measurement:
        x_encoded = Pine.encode_f64_into_field(x)
        encoded_measurement.append(x_encoded)
    return encoded_measurement

def sample_wraparound_joint_rand(Pine, prg: Prg):
    # Since we generate field element by looking at the random byte
    # array two bits at a time, the number of field elements we can
    # generate from each byte is 4.
    NUM_ELEMS_IN_ONE_BYTE = 4
    # Number of wraparound protocol repetitions * dimension.
    output_len = Pine.r * Pine.d
    # We look at bytes from the PRG two bits at a time, so the
    # number of bytes fed into PRG is `ceil(output_len / 4)`.
    # `4` is the number of field elements we can sample from one byte.
    prg_output_len = math.ceil(output_len / NUM_ELEMS_IN_ONE_BYTE)
    rand_buf = prg.next(prg_output_len)

    wraparound_joint_rand = []
    for i in range(output_len):
        rand_buf_index = i // NUM_ELEMS_IN_ONE_BYTE
        offset = i % NUM_ELEMS_IN_ONE_BYTE
        rand_bits = (rand_buf[rand_buf_index] >> offset) & 0b11
        if rand_bits == 0b00:
            rand_field = Pine.Field(Pine.Field.MODULUS - 1)
        elif rand_bits == 0b01 or rand_bits == 0b10:
            rand_field = Pine.Field(0)
        else:
            rand_field = Pine.Field(1)
        wraparound_joint_rand.append(rand_field)
    return wraparound_joint_rand

def finalize_encoding_with_wraparound_joint_rand(
    Pine,
    partially_encoded: list[Pine.Field],
    wraparound_joint_rand: list[Pine.Field],
):
    b0 = Pine.num_bits_for_sq_l2_norm()
    encoded_measurement = partially_encoded[:]

    # Compute L2-norm sum-check results.
    sq_l2_norm = Pine.Field(0)
    for val in partially_encoded:
        sq_l2_norm += val * val
    sq_l2_norm_diff = Pine.Field(Pine.B) - sq_l2_norm
    # Append the bit representation of `sq_l2_norm` and
    # `sq_l2_norm_diff` to `encoded_measurement`.
    encoded_measurement.extend(
        Pine.Field.encode_into_bit_vector(sq_l2_norm, b0)
    )
    encoded_measurement.extend(
        Pine.Field.encode_into_bit_vector(sq_l2_norm_diff, b0)
    )

    # Compute wraparound check results.
    # Infer the bounds based on the PINE parameters.
    (abs_wr_lower_bound, abs_wr_upper_bound, wr_shifted_bound) = \
        Pine.wr_bounds()
    # Number of bits to represent shifted wraparound check result.
    b1 = math.ceil(math.log2(wr_shifted_bound.as_unsigned() + 1))

    # Keep track of the "difference" field elements for each
    # repetition, i.e. the difference between the shifted wraparound
    # check result and the shifted upper bound.
    # This is also the `s` parameter in the parameter table.
    diff_field_elems = []
    # Compute the number of repetitions `r_pass` that the Client
    # is expected to pass. Also keep track of the number of passing
    # repetitions in `r_passed`. If the Client has passed more than
    # `r_pass`, don't set success bit to be 1 anymore,
    # because Aggregators are doing an equality check between the
    # `r_passed` and `r_pass`. See guidance in {{wraparound}}.
    r_pass = Pine.r_pass()
    r_passed = 0
    for rep in range(Pine.r):
        # Compute shifted wraparound check result in the current
        # wraparound check repetition.
        Z = wraparound_joint_rand[(rep*d):((rep+1)*d)]
        shifted_wr_res = Pine.compute_dot_prod(
            partially_encoded, Z
        )
        shifted_wr_res += abs_wr_lower_bound

        if shifted_wr_res <= wr_shifted_bound:
            # Successful repetition:
            # - set success bit to be 1, if `r_passed` hasn't
            #   exceeded `r_pass`, otherwise set success bit to
            #   to be 0.
            # - set difference field element to be 0, because
            #   Client has passed this repetition.
            if r_passed >= r_pass:
                success_bit = 0
            else:
                success_bit = 1
                r_passed += 1
            diff_field_elem = 0
            adj_shifted_wr_res = shifted_wr_res
        else:
            # Failing repetition:
            # - set success bit to be 0.
            # - set the shifted wraparound check result and the
            #   difference field element according to {{wraparound}}.
            success_bit = 0
            diff_field_elem = shifted_wr_res - wr_shifted_bound
            adj_shifted_wr_res = wr_shifted_bound

        # Append bit-encoded `adj_shifted_wr_res` into
        # `encoded_measurement`.
        encoded_measurement.extend(
            Field.encode_into_bit_vector(adj_shifted_wr_res, b1)
        )
        # Append success bit right after the bits for shifted
        # wraparound check result.
        encoded_measurement.append(success_bit)
        # Append difference field element.
        diff_field_elems.append(diff_field_elem)

    # Sanity check the Client has passed `r_pass` number of
    # repetitions, otherwise Client SHOULD retry as recommended
    # in {{wraparound}}.
    if r_passed != r_pass:
        raise Exception(
            "Client should retry encoding PINE check results with "
            "a different wraparound joint randomness."
        )
    encoded_measurement.extend(diff_field_elems)
    return encoded_measurement

def truncate(Pine, inp: Vec[Pine.Field]):
    return inp[:Pine.d]

def decode(Pine,
           output: Vec[Pine.Field],
           num_measurements: Unsigned):
    decoded_output = []
    for val in output:
        decoded_output.append(
            Pine.decode_f64_from_field(val, num_measurements)
        )
    return decoded_output
~~~

The gadgets we will use to verify the degree-2 polynomials in PINE are:

* `Range2(x) = x^2 - x` polynomial to verify the expected entries are bits
  {{bit-check}}.

* `Sq(x) = x^2` polynomial to compute the squared L2-norm sum in
  {{l2-sum-check}}.

* `Mul(g, s) = g * s` to verify the multiplication of success bit `g` and
  difference field element `s` in each wraparound protocol repetition is 0,
  as specified in {{wraparound}}.

~~~
def Range2(x):
    return x ** 2 - x

def Sq(x):
    return x ** 2

def Mul(g, s):
    return g * s
~~~

Therefore, the validity circuits to verify are defined as the following:

~~~
def bit_check(Pine,
              inp: list[Pine.Flp.Field],
              verification_joint_rand_0: Pine.Field):
    # Compute a random linear combination of the `Range2` polynomial
    # evaluated at all entries that are expected to be bits.
    res = Pine.Field(0)
    b0 = Pine.num_bits_for_sq_l2_norm()
    b1 = Pine.num_bits_for_shifted_wr_res()
    # Vector values between `[d, d + 2 * b0 + r * (b1 + 1))`
    # are all expected to be bits.
    for i in range(2 * b0 + Pine.r * (b1 + 1)):
        res += (verification_joint_rand_0 ** (i+1)) * \
               Range2(inp[Pine.d + i])
    return res

def l2_norm_sum_check(Pine,
                      inp: list[Pine.Field],
                      num_shares: Unsigned):
    computed_sq_l2_norm = Pine.Field(0)
    for i in range(Pine.d):
        computed_sq_l2_norm += Sq(inp[i])
    b0 = Pine.num_bits_for_sq_l2_norm()
    encoded_sq_l2_norm = Field.decode_from_bit_vector(
        inp[Pine.d : (Pine.d + b0)]
    )
    encoded_sq_l2_norm_diff = Field.decode_from_bit_vector(
        inp[(Pine.d + b0) : (Pine.d + 2 * b0)]
    )
    B_shares = Pine.B * Pine.Field(num_shares).inv()
    return (
        computed_sq_l2_norm - encoded_sq_l2_norm,
        encoded_sq_l2_norm + encoded_sq_l2_norm_diff - B_shares
    )

# Index offset that contains bit-encoded shifted wraparound check
# result for the `k`-th wraparound protocol repetition.
def offset_shifted_wr_res(Pine, k):
    return Pine.d + 2 * Pine.num_bits_for_sq_l2_norm() + \
        k * (Pine.num_bits_for_shifted_wr_res() + 1)

# Index offset that contains the success bit for the `k`-th
# wraparound check repetition.
def offset_success_bit(Pine, k):
    # Skipping over the L2-norm sum-check results,
    # and the bits for the shifted wraparound check results
    # for the first `k + 1` repetitions (+1 to include the current
    # repetition), and the success bits of the first `k` repetitions.
    return Pine.d + 2 * Pine.num_bits_for_sq_l2_norm() + \
        (k + 1) * Pine.num_bits_for_shifted_wr_res() + k

# Index offset that contains the "difference" field element between
# the shifted wraparound check result and shifted upper bound for
# the `k`-th wraparound protocol repetition.
def offset_diff_field_elem(Pine, k):
    # Skipping over all bits, and the difference field elements
    # for the first `k` repetitions.
    return Pine.d + 2 * Pine.num_bits_for_sq_l2_norm() + \
        Pine.r * (Pine.num_bits_for_shifted_wr_res() + 1) + k

def wraparound_check(Pine,
                     inp: list[Pine.Field],
                     wraparound_joint_rand: list[Pine.Field],
                     verification_joint_rand_1: Pine.Field,
                     num_shares: Unsigned):
    checks = []
    (abs_wr_lower_bound, abs_wr_upper_bound, wr_shifted_bound) = \
        Pine.wr_bounds()
    abs_wr_lower_bound_shares = \
        abs_wr_lower_bound * Pine.Field(num_shares).inv()
    # Number of bits to represent shifted wraparound check result.
    b1 = Pine.num_bits_for_shifted_wr_res()

    # Check for sum of success bits.
    success_bit_check = -(
        Pine.Field(Pine.r_pass()) * \
        Pine.Field(num_shares).inv()
    )
    # Degree-2 check of `g_k * s_k = 0` for all repetitions `k`.
    degree_2_check = Pine.Field(0)

    for k in range(Pine.r):
        # Compute wraparound protcol result for the `k`-th repetition,
        # i.e. the dot product of `X` and `Z` vector
        # (`wraparound_joint_rand`).
        dot_prod_res = Pine.compute_dot_prod(
            inp,
            wraparound_joint_rand[(k * d) : ((k + 1) * d)],
        )
        # Recover bit-encoded wraparound protocol result for
        # the `k`-th repetition.
        shifted_wr_res_start = Pine.offset_shifted_wr_res(k)
        shifted_wr_res = Pine.Field.decode_from_bit_vector(
            inp[shifted_wr_res_start : (shifted_wr_res_start + b1)]
        )
        # Difference field element between the shifted wraparound
        # check result and the shifted upper bound for the `k`-th
        # repetition.
        diff_field_elem = inp[Pine.offset_diff_field_elem(k)]
        # This equation should be equal to 0.
        check = dot_prod_res + abs_wr_lower_bound_shares - \
            shifted_wr_res - diff_field_elem
        checks.append(check)

        # Success bit for the `k`-th repetition.
        success_bit = inp[Pine.offset_success_bit(k)]
        success_bit_check += success_bit
        # Degree-2 check accumulation.
        degree_2_check += \
            (verification_joint_rand_1 ** (k + 1)) * \
            Mul(success_bit, diff_field_elem)
    checks.append(success_bit_check)
    checks.append(degree_2_check)
    return checks

def pine_valid(Pine,
               inp: Vec[Pine.Field],
               joint_rand: Vec[Pine.Field],
               num_shares: Unsigned):
    wraparound_joint_rand = joint_rand[:(Pine.d * Pine.r)]
    verification_joint_rand = joint_rand[(Pine.d * Pine.r):]

    bit_check_res = Pine.bit_check(inp, verification_joint_rand[0])
    (sum_check_res_0, sum_check_res_1) = \
        Pine.l2_norm_sum_check(inp, num_shares)
    wraparound_checks = \
        Pine.wraparound_check(inp,
                              wraparound_joint_rand,
                              verification_joint_rand[1])

    final_reduction_joint_rand = verification_joint_rand[2]
    res = \
        final_reduction_joint_rand * bit_check_res + \
        final_reduction_joint_rand ** 2 * sum_check_res_0 + \
        final_reduction_joint_rand ** 3 * sum_check_res_1
    for i in range(len(wraparound_checks)):
        res += final_reduction_joint_rand ** (4+i) * \
               wraparound_checks[i]
    return res
~~~

The parameters provided to the general-purpose FLP in Prio3 are therefore:

| Parameter                     | Value                                                                                   |
|:------------------------------|:----------------------------------------------------------------------------------------|
| `GADGETS`                     | `[Range2, Sq, Mul]`                                                                     |
| `GADGET_CALLS`                | `[ceil(sqrt(2 * b0 + r * (b1 + 1))), ceil(sqrt(d)), ceil(sqrt(r))]`                     |
| `INPUT_LEN`                   | `d + 2 * b0 + r * (b1 + 1) + r`                                                         |
| `OUTPUT_LEN`                  | `d`                                                                                     |
| `WRAPAROUND_JOINT_RAND_LEN`   | `d * r`                                                                                 |
| `VERIFICATION_JOINT_RAND_LEN` | 3                                                                                       |
| `JOINT_RAND_LEN`              | total length of `WRAPAROUND_JOINT_RAND_LEN` and `VERIFICATION_JOINT_RAND_LEN`           |
| `Measurement`                 | `Vec[float]`                                                                              |
| `AggResult`                   | `Vec[float]`                                                                              |
| `Field`                       | Defined during instantiation of PINE, based on the parameters in {{Section 5 of !VDAF}} |
{: title="Parameters of validity circuit in PINE FLP."}


## FLP Execution {#pine-flp-execution}

After the Client has produced its encoded measurement in {{pine-flp-encoding}},
and has derived the joint randomness based on {{pine-joint-randomness}}, the
FLP execution is very similar to that of Prio3. We reuse the following functions
from Prio3 FLP:

* `Flp.prove(input: Vec[Field], prove_rand: Vec[Field], joint_rand: Vec[Field])
  -> Vec[Field]` is the deterministic proof-generation algorithm run by the
  prover. Prover produces the proof that is necessary for verifiers to linearly
  query the degree-2 polynomials, along with `input`. Note `joint_rand` here
  includes both wraparound joint randomness and verification joint randomness.

* `Flp.query(input: Vec[Field], proof: Vec[Field], query_rand: Vec[Field],
  joint_rand: Vec[Field], num_shares: Unsigned) -> Vec[Field]` is the
  query-generation algorithm run by the verifier, as described in
  {{Section 7.3.3.2 of !VDAF}}. This allows verifiers to verify all the protocol
  outputs in PINE, and also verify each degree-2 polynomial is well-formed.

* `Flp.decide(verifier: Vec[Field]) -> Bool` is the deterministic decision
  algorithm run by the verifier.

The PINE FLP is executed by the prover and verifier as follows. Note we assume
`inp` is the encoded Client vector after {{pine-flp-encoding}}, and
`wraparound_joint_rand` is the vector of wraparound joint randomness derived
based on {{pine-wraparound-joint-randomness}}.

~~~
def run_pine_flp(Pine.Flp,
                 inp: Vec[Pine.Flp.Field],
                 wraparound_joint_rand: Vec[Pine.Flp.Field],
                 num_shares: Unsigned):
    verification_joint_rand = Pine.Flp.Field.rand_vec(
        Flp.VERIFICATION_JOINT_RAND_LEN,
    )
    joint_rand = wraparound_joint_rand + verification_joint_rand
    prove_rand = Pine.Flp.Field.rand_vec(Pine.Flp.PROVE_RAND_LEN)
    query_rand = Pine.Flp.Field.rand_vec(Pine.Flp.QUERY_RAND_LEN)

    # Prover generates the proof.
    proof = Pine.Flp.prove(inp, prove_rand, joint_rand)

    # Verifier queries the input and proof.
    verifier = Pine.Flp.query(inp, proof, query_rand, joint_rand, num_shares)

    # Verifier decides if the input is valid.
    return Pine.Flp.decide(verifier)
~~~
{: #run-flp title="Execution of PINE FLP."}

## FLP Construction in VDAF {#pine-flp-construction}

Putting everything together, we will describe how PINE fits into the `Vdaf`
interface {{Section 5 of !VDAF}}. It reuses the same VDAF parameters and
constants in {{Section 7.2 of !VDAF}} as Prio3.

### Sharding {#pine-flp-sharding}

Client sharding is done in the following steps:

1. Encode the Client's raw measurement into a vector of field integers,
   as described in {{fp-encoding}}, and the first step in {{pine-flp-encoding}}.
1. Derive the wraparound joint randomness parts and seed, based on the output
   from the previous step, and procedures described in
   {{pine-wraparound-joint-randomness}}.
1. Finish rest of the encoding in {{pine-flp-encoding}}, which includes
   computing results for protocols in PINE.
1. Shard the encoded measurement into input shares for the Aggregators from the
   previous step, and derive the verification joint randomness parts and seed as
   described in {{pine-verification-joint-randomness}}.
1. Run the FLP proof-generation algorithm using both wraparound joint randomness
   and verification joint randomness.
1. Shard the proof into a sequence of proof shares.
1. The Client sends the sequence of input shares, proof shares to the
   Aggregators, and also the seed blinds and seed parts for both wraparound
   joint randomness and verification joint randomness.

The algorithm is specified below. Notice that only one set of input and proof
shares (called the "leader" shares below) are vectors of field elements. The
other shares (called the "helper" shares) are represented instead by PRG seeds,
which are expanded into vectors of field elements.

We reuse the `Prg` interface in {{Section 6.2 of !VDAF}}, the parameters and
constants in {{Section 7.2 of !VDAF}} for Prio3.
We also define some auxiliary functions in {{pine-auxiliary}} for joint
randomness derivation and message serialization.

~~~
def measurement_to_input_shares(Pine, measurement, nonce, rand):
    l = Pine.Prg.SEED_SIZE
    d = len(measurement) # Client measurement length.

    # Split the coins into the various seeds we'll need.
    if len(rand) != Pine.RAND_SIZE:
        raise ERR_INPUT # unexpected length for random coins

    # Seed blinds. We use them for the following:
    # - input share
    # - proof share
    # - wraparound joint randomness
    # - verification joint randomness
    num_seed_blinds = 4
    seeds = [rand[i:i+l] for i in range(0,Pine.RAND_SIZE,l)]

    # Split the Helper seeds from Leader seeds.
    k_helper_seeds, seeds = front((Pine.SHARES-1) * num_seed_blinds, seeds)
    k_helper_meas_shares = [
        k_helper_seeds[i]
        for i in range(0, (Pine.SHARES-1) * num_seed_blinds, num_seed_blinds)
    ]
    k_helper_proof_shares = [
        k_helper_seeds[i]
        for i in range(1, (Pine.SHARES-1) * num_seed_blinds, num_seed_blinds)
    ]
    k_helper_wraparound_joint_rand_blinds = [
        k_helper_seeds[i]
        for i in range(2, (Pine.SHARES-1) * num_seed_blinds, num_seed_blinds)
    ]
    (k_leader_wraparound_joint_rand_blind,), seeds = front(1, seeds)
    k_helper_verification_joint_rand_blinds = [
        k_helper_seeds[i]
        for i in range(3, (Pine.SHARES-1) * num_seed_blinds, num_seed_blinds)
    ]
    (k_leader_verification_joint_rand_blind,), seeds = front(1, seeds)

    # Extract prover randomness seed.
    (k_prove,), seeds = front(1, seeds)

    inp = Pine.Flp.encode(measurement)

    # Compute wraparound joint randomness parts, based on Client nonce,
    # secret shares of the first `d` field elements (encoded field vector from
    # Client measurement), and all Aggregators' seed blinds.
    k_wraparound_joint_rand_parts = \
        Pine.derive_joint_rand_parts_from_encoded_vec(
            nonce,
            inp,
            d,
            k_helper_meas_shares,
            k_leader_wraparound_joint_rand_blind,
            k_helper_wraparound_joint_rand_blinds,
        )
    # Derive the wraparound joint randomness from the parts.
    k_wraparound_joint_rand_prg = Pine.joint_rand(k_wraparound_joint_rand_parts)
    wraparound_joint_rand = Pine.Flp.sample_wraparound_joint_rand(
        k_wraparound_joint_rand_prg,
    )

    # Now compute wraparound protocol.
    inp = Pine.Flp.finalize_encoding_with_wraparound_joint_rand(
        inp, wraparound_joint_rand
    )

    # Compute verification joint randomness parts, based on Client nonce,
    # secret shares of all field elements from the encoded vector,
    # all Aggregators' seed blinds.
    k_verification_joint_rand_parts = \
        Pine.derive_joint_rand_parts_from_encoded_vec(
            nonce,
            inp,
            Pine.Flp.INPUT_LEN,
            k_helper_meas_shares,
            k_leader_verification_joint_rand_blind,
            k_helper_verification_joint_rand_blinds,
        )
    # Compute verification joint randomness based on the parts.
    verification_joint_rand = Pine.Prg.expand_into_vec(
        Pine.Flp.Field,
        Pine.joint_rand(k_verification_joint_rand_parts),
        Pine.domain_separation_tag(USAGE_JOINT_RANDOMNESS),
        b'',
        Pine.Flp.VERIFICATION_JOINT_RAND_LEN,
    )

    joint_rand = wraparound_joint_rand + verification_joint_rand

    # Finish the proof shares.
    prove_rand = Pine.Prg.expand_into_vec(
        Pine.Flp.Field,
        k_prove,
        Pine.domain_separation_tag(USAGE_PROVE_RANDOMNESS),
        b'',
        Pine.Flp.PROVE_RAND_LEN,
    )
    proof = Pine.Flp.prove(inp, prove_rand, joint_rand)
    leader_proof_share = proof
    for j in range(Pine.SHARES-1):
        helper_proof_share = Pine.Prg.expand_into_vec(
            Pine.Flp.Field,
            k_helper_proof_shares[j],
            Pine.domain_separation_tag(USAGE_PROOF_SHARE),
            byte(j+1),
            Pine.Flp.PROOF_LEN,
        )
        leader_proof_share = vec_sub(leader_proof_share,
                                     helper_proof_share)

    # Each Aggregator's input share contains its measurement share,
    # proof share, and wraparound and verification joint randomness blinds.
    # The public share contains the Aggregators' joint randomness parts.
    input_shares = []
    input_shares.append(Pine.encode_leader_share(
        leader_meas_share,
        leader_proof_share,
        k_leader_wraparound_joint_rand_blind,
        k_leader_verification_joint_rand_blind,
    ))
    for j in range(Pine.SHARES-1):
        input_shares.append(Pine.encode_helper_share(
            k_helper_meas_shares[j],
            k_helper_proof_shares[j],
            k_helper_wraparound_joint_rand_blinds[j],
            k_helper_verification_joint_rand_blinds[j],
        ))
    public_share = Pine.encode_public_share(
        k_wraparound_joint_rand_parts,
        k_verification_joint_rand_parts,
    )
    return (public_share, input_shares)
~~~
{: #pine-eval-input title="Input-distribution algorithm for PINE."}

### Preparation {#pine-flp-preparation}

This section is similar to the preparation step specified in
{{Section 7.2.2 of !VDAF}}, except Aggregators need to derive the same
wraparound joint randomness and the verification joint randomness as the Client,
specified in {{pine-joint-randomness}}.

It is important that Aggregators compute the same joint randomness seed.
The Client sends each Aggregator the joint randomness seed parts from all other
Aggregators, so each Aggregator can compute its verifier share using the joint
randomness seed derived with its locally computed seed part and the seed parts
of other Aggregators, according to procedure described in
{{pine-joint-randomness}}.
Aggregators later exchange their locally computed joint randomness seed parts
along with their verifier shares, and all of them can verify that hashing
together the exchanged joint randomness seed parts gives the same joint
randomness seed they were using to compute verifier shares.

The definitions of constants and a few auxiliary functions are defined in
{{pine-auxiliary}}.

~~~
def prep_init(Pine, verify_key, agg_id, _agg_param,
              nonce, public_share, input_share):
    d = Pine.d # Client measurement length, `d` is a parameter for PINE.

    k_wraparound_joint_rand_parts, k_verification_joint_rand_parts = \
        Pine.decode_public_share(public_share)
    (meas_share,
     proof_share,
     k_wraparound_joint_rand_blind,
     k_verification_joint_rand_blind) = \
        Pine.decode_leader_share(input_share) if agg_id == 0 else \
        Pine.decode_helper_share(agg_id, input_share)
    out_share = Pine.Flp.truncate(meas_share)

    # Compute wraparound joint randomness.
    # Use the first `d` field elements to compute wraparound joint randomness
    # "part" and replace the wraparound joint randomness part in
    # `public_share` with the part computed by the current Aggregator.
    encoded = Pine.Flp.Field.encode_vec(
        meas_share[:Pine.Flp.d],
    )
    k_wraparound_joint_rand_part = Pine.Prg.derive_seed(
        k_wraparound_joint_rand_blind,
        Pine.domain_separation_tag(USAGE_JOINT_RAND_PART),
        byte(agg_id) + nonce + encoded,
    )
    k_wraparound_joint_rand_parts[agg_id] = k_wraparound_joint_rand_part
    k_corrected_wraparound_joint_rand_prg = Pine.joint_rand(
        k_wraparound_joint_rand_parts,
    )
    wraparound_joint_rand = Pine.Flp.sample_wraparound_joint_rand(
        k_corrected_wraparound_joint_rand_prg,
    )

    # Compute verification joint randomness.
    # Replace the verification joint randomness "part" in `public_share` with
    # the part computed by the current Aggregator.
    encoded = Pine.Flp.Field.encode_vec(meas_share)
    k_verification_joint_rand_part = Pine.Prg.derive_seed(
        k_verification_joint_rand_blind,
        Pine.domain_separation_tag(USAGE_JOINT_RAND_PART),
        byte(agg_id) + nonce + encoded,
    )
    k_verification_joint_rand_parts[agg_id] = k_verification_joint_rand_part
    k_corrected_verification_joint_rand_prg = Pine.joint_rand(
        k_verification_joint_rand_parts,
    )
    verification_joint_rand = Pine.Prg.expand_into_vec(
        Pine.Flp.Field,
        k_corrected_verification_joint_rand_prg,
        Pine.domain_separation_tag(USAGE_JOINT_RANDOMNESS),
        b'',
        Pine.Flp.VERIFICATION_JOINT_RAND_LEN,
    )

    # Concatenate both joint randomness and use it in query algorithm.
    # This should match what the Client uses in `Pine.Flp.prove()`.
    joint_rand = wraparound_joint_rand + verification_joint_rand

    # Query the measurement and proof share.
    query_rand = Pine.Prg.expand_into_vec(
        Pine.Flp.Field,
        verify_key,
        Pine.domain_separation_tag(USAGE_QUERY_RANDOMNESS),
        nonce,
        Pine.Flp.QUERY_RAND_LEN,
    )
    verifier_share = Pine.Flp.query(meas_share,
                                    proof_share,
                                    query_rand,
                                    joint_rand,
                                    Pine.SHARES)

    prep_msg = Pine.encode_prep_share(verifier_share,
                                      k_wraparound_joint_rand_part,
                                      k_verification_joint_rand_part)
    return (out_share,
            k_corrected_wraparound_joint_rand_prg,
            k_corrected_verification_joint_rand_prg,
            prep_msg)

def prep_next(Pine, prep, inbound):
    (out_share,
     k_corrected_wraparound_joint_rand_prg,
     k_corrected_verification_joint_rand_prg,
     prep_msg) = prep

    if inbound is None:
        return (prep, prep_msg)

    k_wraparound_joint_rand_prg_check, k_verification_joint_rand_prg_check = \
        Pine.decode_prep_msg(inbound)
    if ((k_wraparound_joint_rand_prg_check !=
         k_corrected_wraparound_joint_rand_prg) or
        (k_verification_joint_rand_prg_check !=
         k_corrected_verification_joint_rand_prg)):
        # Hashing together the exchanged joint randomness parts should be
        # equal to the joint randomness seed used by the verifier.
        raise ERR_VERIFY

    return out_share

def prep_shares_to_prep(Pine, _agg_param, prep_shares):
    verifier = Pine.Flp.Field.zeros(Pine.Flp.VERIFIER_LEN)
    # Accumulate the joint randomness seed parts sent by all Aggregators.
    k_wraparound_joint_rand_parts, k_verification_joint_rand_parts = [], []
    for encoded in prep_shares:
        (verifier_share,
         k_wraparound_joint_rand_part,
         k_verification_joint_rand_part) = Pine.decode_prep_share(encoded)

        verifier = vec_add(verifier, verifier_share)

        k_wraparound_joint_rand_parts.append(k_wraparound_joint_rand_part)
        k_verification_joint_rand_parts.append(k_verification_joint_rand_part)

    if not Pine.Flp.decide(verifier):
        raise ERR_VERIFY # proof verifier check failed

    k_wraparound_joint_rand_prg_check = Pine.joint_rand(
        k_wraparound_joint_rand_parts,
    )
    k_verification_joint_rand_prg_check = Pine.joint_rand(
        k_verification_joint_rand_parts,
    )
    return Pine.encode_prep_msg(
        k_wraparound_joint_rand_prg_check,
        k_verification_joint_rand_prg_check
    )
~~~
{: #pine-prep-state title="Preparation state for PINE."}

### Aggregation

Similar to Prio3 aggregation in {{Section 7.2.4 of !VDAF}}, aggregating a set of
output shares is simply a matter of adding up the vectors element-wise.

~~~
def out_shares_to_agg_share(Pine, _agg_param, out_shares):
    agg_share = Pine.Flp.Field.zeros(Pine.Flp.OUTPUT_LEN)
    for out_share in out_shares:
        agg_share = vec_add(agg_share, out_share)
    return Pine.Flp.Field.encode_vec(agg_share)
~~~
{: #pine-out2agg title="Aggregation algorithm for PINE."}

### Unsharding

To unshard a set of aggregate shares, the Collector first adds up the vectors
element-wise. It then decodes the field integer vector based on
`Pine.Flp.decode()`. Each field integer decoding is done based on
{{fi-decoding}}.

~~~
def agg_shares_to_result(Pine, _agg_param,
                         agg_shares, num_measurements):
    agg = Pine.Flp.Field.zeros(Pine.Flp.OUTPUT_LEN)
    for agg_share in agg_shares:
        agg = vec_add(agg, Pine.Flp.Field.decode_vec(agg_share))
    return Pine.Flp.decode(agg, num_measurements)
~~~
{: #pine-agg-output title="Computation of the aggregate result for PINE."}


## Auxiliary Functions {#pine-auxiliary}

This section defines a number of auxiliary functions referenced by the main
algorithms for PINE in the preceding sections.

### Conversion Between Float64 and Field Element {#f64-field-util}

~~~
def encode_f64_into_field(Pine, x: float) -> Pine.Field:
    if math.isnan(x) or math.isfinite(x) or abs(x) < sys.float_info.min:
        # Reject NAN, infinity, and subnormal floats.
        raise ERR_ENCODE
    x_encoded = int(x * (2 ** Pine.f))
    if x >= 0:
        return Pine.Field(x_encoded)
    return Pine.Flp.Field(Pine.Field.MODULUS + x_encoded)

def decode_f64_from_field(
    Pine,
    field_elem: Pine.Field,
    num_measurements: Unsigned,
) -> float:
    # The upper bound in the field, below which we will think the
    # aggregated field element indicates a positive value in
    # floating point world, otherwise the result should be negative.
    positive_upper_bound = \
        num_measurements * math.sqrt(Pine.B.as_unsigned())
    decoded = field_elem.as_unsigned()
    if field_elem_unsigned > positive_upper_bound:
        # We need to take the difference between the result
        # and the field modulus, and return the result as negative.
        decoded = -(Pine.Field.MODULUS - decoded)
    # Divide by 2^f and we will get a float back.
    decoded_float = decoded / (2 ** Pine.f)
    return decoded_float
~~~

### Conversion Between Field Element And Its Bit Representation {#field-repr}

A common subroutine in PINE is to encode a field element into its bit
representation, as an array of field elements, and vice versa.
So we define two new methods on `Field`.

~~~
def is_valid_bit_length(
    Field,
    bits: Unsigned,
) -> bool:
    if bits >= 8 * Field.ENCODED_SIZE:
        # If number of bits is already more than the maximum number of
        # bits to represent this field type.
        return False
    # Check if 2^{bits} - 1 <= Field.MODULUS.
    return Field.MODULUS >> bits != 0

def encode_into_bit_vector(
    Field,
    field_elem: Field,
    bits: Unsigned,
) -> list[Field]:
    result = []
    for i in range(bits):
        # Take the least significant bit and right shift.
        result.append(Field(field_elem & 1))
        field_elem >>= 1
    if field_elem != 0:
        # The bit vector representation should be longer than `bits`.
        raise ERR_ENCODE
    return result

def decode_from_bit_vector(
    Field,
    vec: list[Field],
) -> Field:
    if not Field.is_valid_bit_length(len(vec)):
        # Field modulus cannot represent this bit vector
        # representation.
        raise ERR_DECODE
    result = Field(0)
    for (l, bit) in enumerate(vec):
        result += Field(1 << l) * bit
    return result
~~~

### Wraparound Check Utility Functions {#wraparound-utility}

There are some common subroutines that wraparound checks use, such as computing
the parameters for wraparound check, and computing dot product of field element
vectors `X` and `Z`.

~~~
def wr_bounds(Pine) -> tuple[
    Pine.Field, Pine.Field, Pine.Field
]:
    # Compute the wraparound check bounds to optimize for
    # communication cost mentioned in {{wraparound}}.
    alpha: float = Pine.ALPHA
    b: float = Pine.b
    f: Unsigned = Pine.f

    # {{wraparound}} mentions talking about optimizing communication
    # cost by finding an `alpha' <= alpha`, such that
    # `alpha' * sqrt(B) + 1` is a power of 2.
    # Note `sqrt(B)` is the L2-norm bound in field integer, by
    # encoding float64 `b` into a field integer, with number of
    # fractional bits `f`.
    alpha_times_b = alpha * b
    # This is the absolute value of the wraparound check lower bound
    # in field integer.
    abs_wr_lower_bound = Pine.encode_f64_into_field(alpha_times_b)
    # The upper bound is always one more than the absolute
    # lower bound.
    abs_wr_upper_bound = abs_wr_lower_bound + 1
    # Our goal is to round down `abs_wr_upper_bound`, so it is a
    # power of 2.
    optim_abs_wr_upper_bound = Pine.Field(
        1 << (int(math.log2(abs_wr_upper_bound)))
    )
    optim_abs_wr_lower_bound = optim_abs_wr_upper_bound - 1
    # Client sends the shifted wraparound check result, by shifting
    # the negative results to positive, so the shifted bound is simply
    # the sum of the absolute lower bound and upper bound.
    # Note this is an inclusive bound, thus the maximum shifted result
    # that allows the Client to pass a particular repetition.
    shifted_bound = \
        optim_abs_wr_lower_bound + optim_abs_wr_upper_bound
    return (
        optim_abs_wr_lower_bound, optim_abs_wr_upper_bound,
        shifted_bound,
    )

def num_bits_for_sq_l2_norm(Pine):
    # Compute the number of bits to represent squared L2-norm.
    # This is an inclusive bound, so +1 before taking the log2.
    return math.ceil(math.log2(Pine.B.as_unsigned() + 1))

def num_bits_for_shifted_wr_res(Pine):
    # Compute the number of bits to represent the shifted wraparound
    # check result sent by the Clients.
    (_, _, wr_shifted_bound) = Pine.wr_bounds()
    # This is an inclusive bound, so +1 before taking the log2.
    return math.ceil(math.log2(wr_shifted_bound.unsigned() + 1))

def r_pass(Pine):
    # Compute the number of wraparound check repetitions that the
    # Client is expected to pass.
    return math.floor(Pine.r * Pine.TAU)

def compute_dot_prod(
    Pine,
    vec1: list[Pine.Field],
    vec2: list[Pine.Field],
) -> Pine.Field:
    # Compute the dot product of two field element vectors.
    result = Pine.Field(0)
    for val1, val2 in zip(vec1, vec2):
        result += val1 * val2
    return result
~~~


### Joint Randomness Derivation {#pine-joint-randomness-derivation}

The following methods are called by the sharding and preparation algorithms to
derive the joint randomness parts and joint randomness seed.

~~~
def derive_joint_rand_parts_from_encoded_vec(
    Pine,
    nonce,
    inp,
    inp_len,
    k_helper_meas_shares,
    k_leader_joint_rand_blind,
    k_helper_joint_rand_blinds,
):
    """
    Compute joint randomness part for each Aggregator, based on Client nonce,
    secret shares of inp[:inp_len], and initial seed blind for each Aggregator.
    """
    leader_meas_share = inp
    k_joint_rand_parts = []
    for j in range(Pine.SHARES-1):
        # Secret share the first `inp_len` field elements to compute the
        # joint randomness part.
        helper_meas_share = Pine.Prg.expand_into_vec(
            Pine.Flp.Field,
            k_helper_meas_shares[j],
            Pine.domain_separation_tag(USAGE_MEASUREMENT_SHARE),
            byte(j+1),
            inp_len,
        )
        leader_meas_share = vec_sub(leader_meas_share,
                                    helper_meas_share)

        # Derive the joint randomness part.
        encoded = Pine.Flp.Field.encode_vec(helper_meas_share)
        k_joint_rand_part = Pine.Prg.derive_seed(
            k_helper_joint_rand_blinds[j],
            Pine.domain_separation_tag(USAGE_JOINT_RAND_PART),
            byte(j+1) + nonce + encoded,
        )
        k_joint_rand_parts.append(k_joint_rand_part)

    # Finish computing joint randomness.
    encoded = Pine.Flp.Field.encode_vec(leader_meas_share)
    k_leader_joint_rand_part = Pine.Prg.derive_seed(
        k_leader_joint_rand_blind,
        Pine.domain_separation_tag(USAGE_JOINT_RAND_PART),
        byte(0) + nonce + encoded,
    )
    k_joint_rand_parts.insert(0, k_leader_joint_rand_part)
    return k_joint_rand_parts

def joint_rand(Pine, k_joint_rand_parts):
    """Compute joint randomness seed from the parts. """
    return Pine.Prg.derive_seed(
        zeros(Pine.Prg.SEED_SIZE),
        Pine.domain_separation_tag(USAGE_JOINT_RAND_SEED),
        concat(k_joint_rand_parts),
    )
~~~

### Message Serialization

~~~
def encode_leader_share(Pine,
                        meas_share,
                        proof_share,
                        k_wraparound_joint_rand_blind,
                        k_verification_joint_rand_blind):
    encoded = Bytes()
    encoded += Pine.Flp.Field.encode_vec(meas_share)
    encoded += Pine.Flp.Field.encode_vec(proof_share)
    encoded += k_wraparound_joint_rand_blind
    encoded += k_verification_joint_rand_blind
    return encoded

def decode_leader_share(Pine, encoded):
    l = Pine.Flp.Field.ENCODED_SIZE * Pine.Flp.INPUT_LEN
    encoded_meas_share, encoded = encoded[:l], encoded[l:]
    meas_share = Pine.Flp.Field.decode_vec(encoded_meas_share)
    l = Pine.Flp.Field.ENCODED_SIZE * Pine.Flp.PROOF_LEN
    encoded_proof_share, encoded = encoded[:l], encoded[l:]
    proof_share = Pine.Flp.Field.decode_vec(encoded_proof_share)

    # Parse joint randomness blinds.
    l = Pine.Prg.SEED_SIZE
    if len(encoded) != 2 * l:
        raise ERR_DECODE
    wraparound_joint_rand_blind, verification_joint_rand_blind = \
        encoded[:l], encoded[l:]

    return (meas_share,
            proof_share,
            wraparound_joint_rand_blind,
            verification_joint_rand_blind)

def encode_helper_share(Pine,
                        k_meas_share,
                        k_proof_share,
                        k_wraparound_joint_rand_blind,
                        k_verification_joint_rand_blind):
    encoded = Bytes()
    encoded += k_meas_share
    encoded += k_proof_share
    encoded += k_wraparound_joint_rand_blind
    encoded += k_verification_joint_rand_blind
    return encoded

def decode_helper_share(Pine, agg_id, encoded):
    c_meas_share = Pine.domain_separation_tag(USAGE_MEASUREMENT_SHARE)
    c_proof_share = Pine.domain_separation_tag(USAGE_PROOF_SHARE)
    l = Pine.Prg.SEED_SIZE
    k_meas_share, encoded = encoded[:l], encoded[l:]
    meas_share = Pine.Prg.expand_into_vec(Pine.Flp.Field,
                                          k_meas_share,
                                          c_meas_share,
                                          byte(agg_id),
                                          Pine.Flp.INPUT_LEN)
    k_proof_share, encoded = encoded[:l], encoded[l:]
    proof_share = Pine.Prg.expand_into_vec(Pine.Flp.Field,
                                           k_proof_share,
                                           c_proof_share,
                                           byte(agg_id),
                                           Pine.Flp.PROOF_LEN)

    # Parse the seed blinds for both joint randomness.
    if len(encoded) != 2 * l:
        raise ERR_DECODE
    wraparound_joint_rand_blind, verification_joint_rand_blind = \
        encoded[:l], encoded[l:]

    return (meas_share,
            proof_share,
            wraparound_joint_rand_blind,
            verification_joint_rand_blind)

def encode_public_share(Pine,
                        k_wraparound_joint_rand_parts,
                        k_verification_joint_rand_parts):
    encoded = Bytes()
    encoded += concat(k_wraparound_joint_rand_parts)
    encoded += concat(k_verification_joint_rand_parts)
    return encoded

def decode_public_share(Pine, encoded):
    l = Pine.Prg.SEED_SIZE
    k_wraparound_joint_rand_parts, k_verification_joint_rand_parts = [], []
    for i in range(Pine.SHARES):
        k_wraparound_joint_rand_part, encoded = encoded[:l], encoded[l:]
        k_wraparound_joint_rand_parts.append(k_wraparound_joint_rand_part)
    for i in range(Pine.SHARES):
        k_verification_joint_rand_part, encoded = encoded[:l], encoded[l:]
        k_verification_joint_rand_parts.append(k_verification_joint_rand_part)
    if len(encoded) != 0:
        raise ERR_DECODE
    return k_wraparound_joint_rand_parts, k_verification_joint_rand_parts

def encode_prep_share(Pine,
                      verifier,
                      k_wraparound_joint_rand_prg,
                      k_verification_joint_rand_prg):
    encoded = Bytes()
    encoded += Pine.Flp.Field.encode_vec(verifier)
    encoded += k_wraparound_joint_rand_prg
    encoded += k_verification_joint_rand_prg
    return encoded

def decode_prep_share(Pine, encoded):
    l = Pine.Flp.Field.ENCODED_SIZE * Pine.Flp.VERIFIER_LEN
    encoded_verifier, encoded = encoded[:l], encoded[l:]
    verifier = Pine.Flp.Field.decode_vec(encoded_verifier)

    # Parse the seeds for both joint randomness.
    l = Pine.Prg.SEED_SIZE
    if len(encoded) != 2 * l:
        raise ERR_DECODE
    wraparound_joint_rand_prg, verification_joint_rand_prg = \
        encoded[:l], encoded[l:]

    return (verifier, wraparound_joint_rand_prg, verification_joint_rand_prg)

def encode_prep_msg(Pine,
                    k_wraparound_joint_rand_prg_check,
                    k_verification_joint_rand_prg_check):
    encoded = Bytes()
    encoded += k_wraparound_joint_rand_prg_check
    encoded += k_verification_joint_rand_prg_check
    return encoded

def decode_prep_msg(Pine, encoded):
    l = Pine.Prg.SEED_SIZE
    if len(encoded) != 2 * l:
        raise ERR_DECODE
    wraparound_joint_rand_prg_check, verification_joint_rand_prg_check = \
        encoded[:l], encoded[l:]
    return wraparound_joint_rand_prg_check, verification_joint_rand_prg_check
~~~

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
