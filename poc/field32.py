from sage.all import GF
from vdaf_poc.field import FftField


class Field32(FftField):
    """The finite field GF(4293918721)."""

    MODULUS = 4293918721
    GEN_ORDER = 2**20
    ENCODED_SIZE = 4

    # Sage finite field object.
    gf = GF(MODULUS)

    @classmethod
    def gen(cls):
        return cls(3925978153)
