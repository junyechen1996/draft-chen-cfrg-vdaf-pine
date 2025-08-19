from Cryptodome.Cipher import AES
from Cryptodome.Hash import HMAC, SHA256
from vdaf_poc.common import to_le_bytes, zeros
from vdaf_poc.xof import Xof


class XofHmacSha256Aes128(Xof):
    """
    XOF based on HMAC-SHA256 and AES128.
    """

    # Associated parameters
    SEED_SIZE = 32

    def __init__(self, seed, dst, binder):
        hmac = HMAC.new(seed, digestmod=SHA256)
        dst_length = to_le_bytes(len(dst), 1)
        hmac.update(dst_length)
        hmac.update(dst)
        hmac.update(binder)
        hmac_output = hmac.digest()
        self.key = hmac_output[:16]
        self.nonce = hmac_output[16:24]
        self.initial_value = hmac_output[24:]
        self.length_consumed = 0

    def next(self, length):
        self.length_consumed += length

        # CTR-mode encryption of the all-zero string of the desired
        # length and a random IV.
        cipher = AES.new(self.key,
                         AES.MODE_CTR,
                         nonce=self.nonce,
                         initial_value=self.initial_value)
        stream = cipher.encrypt(zeros(self.length_consumed))
        return stream[-length:]
