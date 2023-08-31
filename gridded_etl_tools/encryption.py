"""Allow Zarrs to be saved using XChaCha20-Poly1305 encryption.

In order to use encryption with this library, one need only generate an encryption key,
and then pass the generated key into the constructor of a
:class:`.dataset_manager.DatasetManager` subclass. The user only needs to keep track of
their encryption key. Everything else is managed from there.

Underneath the hood, encryption keys at run time are hashed using a cryptographically
secure hashing algorithm (SHA3-256) and then registered. Because the
:class:`EncryptionFilter` gets serialized in place with the Zarr, its configuration is
saved as part of the Zarr. The hash of its encryption key is stored in the Zarr, and
then turned into the actual key by looking it up by hash in the registry. In this way,
we can know at run time which key was used to encrypt the Zarr, as long as it is present
in he current run time registry, but attackers who have access to the Zarr but not the
encryption key cannot decrypt the Zarr.
"""
import functools
import hashlib
import io

from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes

from numcodecs import register_codec
from numcodecs.abc import Codec

_key_registry = {}


def generate_encryption_key() -> str:
    """Generate a random encryption key.

    Returns
    -------

    key: str
        A hex string representing a 32 byte (256 bit) randomly generated encryption key.
    """
    return get_random_bytes(32).hex()


def register_encryption_key(key: str) -> str:
    """Register an encryption key in a runtime registry.

    Generates a cryptographically secure hash of the given key, and then stores the key
    in runtime registry using the hash as a lookup key. To use encryption with this
    library, you must register your key before reading or writing any Zarr that uses
    encryption. Zarrs that are written to storage will contain the hashed version of
    their encryption key which is used to look up the real key at runtime.

    Parameters
    ----------

    key: str
        A hex string representing a 32 byte (256 bit) randomly generated encryption key,
        as returned by :func:`generate_encryption_key`.

    Returns
    -------

    key_hash, str:
        The cryptographically secure hash of the registered key, used to later look up
        the key in the registry.
    """
    key = bytes.fromhex(key)
    hashed = _hash(key)
    _key_registry[hashed] = key

    return hashed


def _hash(key: bytes) -> str:
    """Generate cryptographically secure hash for key."""
    hashed = hashlib.sha3_256(key)

    return hashed.hexdigest()


class EncryptionFilter(Codec):
    """An encryption filter for ZARR data.

    This class is serialized and stored along with the Zarr it is used with, so instead
    of storing the encryption key, we store the hash of the encryption key, so it can be
    looked up in the key registry at run time as needed.

    Parameters
    ----------

    key_hash: str
        The hex digest of an encryption key. A key may be generated using
        :func:`generate_encryption_key`. The hex digest is obtained by registering the
        key using :func:`register_encryption_key`.
    """

    codec_id = "xchacha20poly1305"
    header = b"dClimate-Zarr"

    def __init__(self, key_hash: str):
        self.key_hash = key_hash

    @functools.cached_property
    def _key(self):
        # names beginning with underscore are not included in the serialized configuration
        if self.key_hash not in _key_registry:
            raise MissingKeyError(self.key_hash)

        return _key_registry[self.key_hash]

    def encode(self, buf):
        raw = io.BytesIO()
        raw.write(buf)
        nonce = get_random_bytes(24)  # XChaCha uses 24 byte (192 bit) nonce
        cipher = ChaCha20_Poly1305.new(key=self._key, nonce=nonce)
        cipher.update(self.header)
        ciphertext, tag = cipher.encrypt_and_digest(raw.getbuffer())

        return nonce + tag + ciphertext

    def decode(self, buf, out=None):
        nonce, tag, ciphertext = buf[:24], buf[24:40], buf[40:]
        cipher = ChaCha20_Poly1305.new(key=self._key, nonce=nonce)
        cipher.update(self.header)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)

        if out is not None:
            outbuf = io.BytesIO(plaintext)
            outbuf.readinto(out)
            return out

        return plaintext


register_codec(EncryptionFilter)


class MissingKeyError(Exception):
    def __init__(self, key):
        super().__init__(f"Cannot find encryption key with hash: {key}")
