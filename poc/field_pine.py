from sage.all import GF
from field import FftField

class Field48(FftField):
    """The finite field GF(2^21 * 3^2 * 14913079 + 1)."""

    MODULUS = 2**21 * 3**2 * 14913079 + 1
    GEN_ORDER = 2**21
    ENCODED_SIZE = 6

    # Operational parameters
    gf = GF(MODULUS)

    @classmethod
    def gen(cls):
        return cls(cls.gf.primitive_element()**(3**2 * 14913079))
