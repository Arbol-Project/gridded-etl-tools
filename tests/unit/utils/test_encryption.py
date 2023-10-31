import io
import unittest.mock

import numpy as np
import pytest

from gridded_etl_tools.utils.encryption import (
    generate_encryption_key,
    EncryptionFilter,
    MissingKeyError,
    register_encryption_key,
)


def test_gen_encryption_key():
    key = generate_encryption_key()
    key = bytes.fromhex(key)
    assert len(key) == 32
    assert isinstance(key, bytes)


class TestEncryptionFilter:
    key = b"abcdefghijklmnopqrstuvwxyz123456".hex()
    key_hash = register_encryption_key(key)

    def _make_one(self):
        return EncryptionFilter(self.key_hash)

    def test_constructor(self):
        filter = self._make_one()
        assert filter.key_hash == self.key_hash
        assert filter._key == bytes.fromhex(self.key)
        assert filter.codec_id == "xchacha20poly1305"

    def test_round_trip(self):
        # Integration test
        filter = self._make_one()
        cleartext = b"Hello, Dad. I'm in jail."
        encrypted = filter.encode(cleartext)
        assert encrypted != cleartext
        decrypted = filter.decode(encrypted)
        assert decrypted == cleartext

    def test_round_trip_ndarray(self):
        # Integration test
        filter = self._make_one()
        data = np.arange(27).reshape(
            (
                3,
                3,
                3,
            )
        )

        encrypted = filter.encode(data)
        decrypted = filter.decode(encrypted)

        reconstituted = np.zeros((3, 3, 3), dtype=data.dtype)
        io.BytesIO(decrypted).readinto(reconstituted)

        assert np.array_equal(reconstituted, data)

    def test_round_trip_ndarray_with_out_argument(self):
        # Integration test
        filter = self._make_one()
        data = np.arange(27).reshape(
            (
                3,
                3,
                3,
            )
        )

        encrypted = filter.encode(data)
        reconstituted = np.zeros((3, 3, 3), dtype=data.dtype)
        filter.decode(encrypted, out=reconstituted)

        assert np.array_equal(reconstituted, data)

    def test_hmac(self):
        # Integration test
        filter = self._make_one()
        cleartext = b"Hello, Dad. I'm in jail."
        encrypted = filter.encode(cleartext)

        # Change 'header' used to compute tag out from under filter to force an HMAC
        # failure, to make sure it's really working the way we think
        filter.header = b"hahaha i'm sneaky"
        with pytest.raises(ValueError):
            filter.decode(encrypted)

    @unittest.mock.patch("gridded_etl_tools.utils.encryption.ChaCha20_Poly1305")
    @unittest.mock.patch("gridded_etl_tools.utils.encryption.get_random_bytes")
    def test_encode(self, get_random_bytes, ChaCha20_Poly1305):
        # Unit test
        get_random_bytes.return_value = b"|nonce-----------------|"
        cipher = ChaCha20_Poly1305.new.return_value
        cipher.encrypt_and_digest.return_value = (
            b"encrypted text",
            b"|tag-----------|",
        )
        filter = self._make_one()

        encrypted = filter.encode(b"Hello Dad. I'm in jail.")

        assert encrypted == b"|nonce-----------------||tag-----------|encrypted text"
        get_random_bytes.assert_called_once_with(24)
        ChaCha20_Poly1305.new.assert_called_once_with(key=bytes.fromhex(self.key), nonce=b"|nonce-----------------|")
        cipher.encrypt_and_digest.assert_called_once_with(b"Hello Dad. I'm in jail.")
        cipher.update.assert_called_once_with(filter.header)

    @unittest.mock.patch("gridded_etl_tools.utils.encryption.ChaCha20_Poly1305")
    def test_decode(self, ChaCha20_Poly1305):
        # Unit test
        cipher = ChaCha20_Poly1305.new.return_value
        cipher.decrypt_and_verify.return_value = b"Hello Dad. I'm in jail."
        filter = self._make_one()

        decrypted = filter.decode(b"|nonce-----------------||tag-----------|encrypted text")

        assert decrypted == b"Hello Dad. I'm in jail."
        ChaCha20_Poly1305.new.assert_called_once_with(key=bytes.fromhex(self.key), nonce=b"|nonce-----------------|")
        cipher.decrypt_and_verify.assert_called_once_with(b"encrypted text", b"|tag-----------|")
        cipher.update.assert_called_once_with(filter.header)

    def test_decode_missing_key(self):
        # Unit test
        missing_key = "-I-don't-exist-"
        filter = EncryptionFilter(missing_key)

        with pytest.raises(MissingKeyError):
            filter.decode(b"anything at all")
