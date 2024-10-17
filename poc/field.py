from typing import Self

from sage.all import GF
from vdaf_poc.field import NttField


class Field32(NttField):
    """The finite field GF(4293918721)."""

    MODULUS = 4293918721
    GEN_ORDER = 2**20
    ENCODED_SIZE = 4

    # Sage finite field object.
    gf = GF(MODULUS)

    @classmethod
    def gen(cls):
        return cls(3925978153)


class Field40(NttField):
    """
    The finite field GF(2^20 * 1048555 + 1).
    Modulus is selected using https://github.com/cfrg/draft-irtf-cfrg-vdaf/blob/main/misc/prime-hunt.md
    first one by running `search(20, 40, 200)`
    q = 5 * 43 * 4877 = 1048555
    p = 2^20 * q + 1
    generator = pow(7, q, p)
    """

    MODULUS = 2**20 * 1048555 + 1
    GEN_ORDER = 2**20
    ENCODED_SIZE = 5

    # Sage finite field object.
    gf = GF(MODULUS)

    @classmethod
    def gen(cls) -> Self:
        return cls(7) ** 1048555
