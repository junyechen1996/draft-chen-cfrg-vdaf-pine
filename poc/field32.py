from vdaf_poc.field import NttField


class Field32(NttField):
    """The finite field GF(4293918721)."""

    MODULUS = 4293918721
    GEN_ORDER = 2**20
    ENCODED_SIZE = 4

    @classmethod
    def gen(cls):
        return cls(3925978153)
