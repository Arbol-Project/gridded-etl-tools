from unittest.mock import Mock

import dag_cbor
from requests.exceptions import Timeout as TimeoutError
from requests.exceptions import HTTPError


class TestIPFS:
    @staticmethod
    def test_constructor_defaults(manager_class):
        dm = manager_class()
        assert dm._host == "http://127.0.0.1:5001"
        assert dm._default_hash == "sha2-256"
        assert dm._default_base == "base32"
        assert dm._default_timeout == 600

    @staticmethod
    def test_ipfs_get(manager_class):
        dm = manager_class()
        dm.ipfs_session = Mock()
        response = Mock(content=dag_cbor.encode({"hi": "mom!"}), spec=("content", "raise_for_status"))
        cid = "identifier"
        dm.ipfs_session.post.return_value = response
        assert dm.ipfs_get(cid) == {"hi": "mom!"}
        dm.ipfs_session.post.assert_called_once_with(
            "http://127.0.0.1:5001/api/v0/block/get", timeout=600, params={"arg": "identifier"}
        )
        response.raise_for_status.assert_called_once_with()

    @staticmethod
    def test_ipfs_put(manager_class):
        dm = manager_class()
        dm.ipfs_session = Mock()
        response = Mock(spec=("json", "raise_for_status"), json=Mock(return_value={"Cid": {"/": "the_new_cid"}}))
        dm.ipfs_session.post.return_value = response
        assert dm.ipfs_put(b"hi mom!") == "the_new_cid"
        dm.ipfs_session.post.assert_called_once_with(
            "http://127.0.0.1:5001/api/v0/dag/put",
            params={"store-codec": "dag-cbor", "input-codec": "dag-json", "pin": True, "hash": "sha2-256"},
            files={"dummy": b"hi mom!"},
        )
        response.raise_for_status.assert_called_once_with()

    @staticmethod
    def test_ipfs_put_should_not_pin(manager_class):
        dm = manager_class()
        dm.ipfs_session = Mock()
        response = Mock(spec=("json", "raise_for_status"), json=Mock(return_value={"Cid": {"/": "the_new_cid"}}))
        dm.ipfs_session.post.return_value = response
        assert dm.ipfs_put(b"hi mom!", should_pin=False) == "the_new_cid"
        dm.ipfs_session.post.assert_called_once_with(
            "http://127.0.0.1:5001/api/v0/dag/put",
            params={"store-codec": "dag-cbor", "input-codec": "dag-json", "pin": False, "hash": "sha2-256"},
            files={"dummy": b"hi mom!"},
        )
        response.raise_for_status.assert_called_once_with()

    @staticmethod
    def test_ipns_resolve(manager_class):
        dm = manager_class()
        dm.ipfs_session = Mock()
        response = Mock(spec=("json", "raise_for_status"), json=Mock(return_value={"Path": "/ipfs/identifier"}))
        dm.ipfs_session.post.return_value = response
        dm.ipns_key_list = Mock(return_value={"thename": "theipnskey"})
        assert dm.ipns_resolve("thename") == "identifier"
        dm.ipfs_session.post.assert_called_once_with(
            "http://127.0.0.1:5001/api/v0/name/resolve", timeout=600, params={"arg": "theipnskey"}
        )
        response.raise_for_status.assert_called_once_with()
        dm.ipns_key_list.assert_called_once_with()

    @staticmethod
    def test_ipns_publish(manager_class):
        dm = manager_class()
        dm.ipfs_session = Mock()
        response = Mock(spec=("json", "raise_for_status"), json=Mock(return_value={"Name": "the_new_name_hash"}))
        dm.ipfs_session.post.return_value = response
        assert dm.ipns_publish("thekey", "thecid") == "the_new_name_hash"
        dm.ipfs_session.post.assert_called_once_with(
            "http://127.0.0.1:5001/api/v0/name/publish",
            timeout=600,
            params={"arg": "thecid", "key": "thekey", "allow-offline": False, "offline": False},
        )
        response.raise_for_status.assert_called_once_with()

    @staticmethod
    def test_ipns_publish_offline(manager_class):
        dm = manager_class()
        dm.ipfs_session = Mock()
        response = Mock(spec=("json", "raise_for_status"), json=Mock(return_value={"Name": "the_new_name_hash"}))
        dm.ipfs_session.post.return_value = response
        assert dm.ipns_publish("thekey", "thecid", offline=True) == "the_new_name_hash"
        dm.ipfs_session.post.assert_called_once_with(
            "http://127.0.0.1:5001/api/v0/name/publish",
            timeout=600,
            params={"arg": "thecid", "key": "thekey", "allow-offline": True, "offline": True},
        )
        response.raise_for_status.assert_called_once_with()

    @staticmethod
    def test_ipns_key_list(manager_class):
        dm = manager_class()
        dm.ipfs_session = Mock()
        response = Mock(
            spec=("json", "raise_for_status"),
            json=Mock(
                return_value={
                    "Keys": [{"Id": "some key", "Name": "self"}, {"Id": "some other key", "Name": "catalog"}]
                }
            ),
        )
        dm.ipfs_session.post.return_value = response
        assert dm.ipns_key_list() == {"self": "some key", "catalog": "some other key"}
        dm.ipfs_session.post.assert_called_once_with("http://127.0.0.1:5001/api/v0/key/list", timeout=600)
        response.raise_for_status.assert_called_once_with()

    @staticmethod
    def test_ipns_generate_name(manager_class):
        dm = manager_class()
        dm.ipfs_session = Mock()
        response = Mock(spec=("json", "raise_for_status"), json=Mock(return_value={"Id": "new_name_hash"}))
        dm.ipfs_session.post.return_value = response
        dm.ipns_key_list = Mock(return_value={})
        assert dm.ipns_generate_name() == "new_name_hash"
        dm.ipfs_session.post.assert_called_once_with(
            "http://127.0.0.1:5001/api/v0/key/gen", timeout=600, params={"arg": "DummyManager-daily", "type": "rsa"}
        )
        response.raise_for_status.assert_called_once_with()
        dm.ipns_key_list.assert_called_once_with()

    @staticmethod
    def test_ipns_generate_name_pass_in_key(manager_class):
        dm = manager_class()
        dm.ipfs_session = Mock()
        response = Mock(spec=("json", "raise_for_status"), json=Mock(return_value={"Id": "new_name_hash"}))
        dm.ipfs_session.post.return_value = response
        dm.ipns_key_list = Mock(return_value={})
        assert dm.ipns_generate_name("hotdogs-hourly") == "new_name_hash"
        dm.ipfs_session.post.assert_called_once_with(
            "http://127.0.0.1:5001/api/v0/key/gen", timeout=600, params={"arg": "hotdogs-hourly", "type": "rsa"}
        )
        response.raise_for_status.assert_called_once_with()
        dm.ipns_key_list.assert_called_once_with()

    @staticmethod
    def test_ipns_generate_name_name_already_exists(manager_class):
        dm = manager_class()
        dm.ipfs_session = Mock()
        response = Mock(spec=("json", "raise_for_status"), json=Mock(return_value={"Id": "new_name_hash"}))
        dm.ipfs_session.post.return_value = response
        dm.ipns_key_list = Mock(return_value={"hotdogs-hourly": "hotdog-name-hash"})
        assert dm.ipns_generate_name("hotdogs-hourly") == "hotdog-name-hash"
        dm.ipfs_session.post.assert_not_called()
        response.raise_for_status.assert_not_called()
        dm.ipns_key_list.assert_called_once_with()

    @staticmethod
    def test_ipns_retrieve_object(manager_class):
        dm = manager_class()
        dm.ipns_key_list = Mock(return_value={"hotdogs-hourly": "ipns-hash"})
        dm.ipns_resolve = Mock(return_value="ipfs-hash")
        dm.ipfs_get = Mock(return_value={"hi": "mom!"})
        assert dm.ipns_retrieve_object("hotdogs-hourly") == ({"hi": "mom!"}, "ipns-hash")
        dm.ipns_key_list.assert_called_once_with()
        dm.ipns_resolve.assert_called_once_with("hotdogs-hourly")
        dm.ipfs_get.assert_called_once_with("ipfs-hash")

    @staticmethod
    def test_latest_hash(manager_class):
        dm = manager_class()
        dm.check_stac_on_ipns = Mock(return_value=True)
        cid = Mock(spec=("set",))
        cid.set.return_value = "thecid"
        dm.load_stac_metadata = Mock(return_value={"assets": {"zmetadata": {"href": cid}}})
        assert dm.latest_hash() == "thecid"

        dm.check_stac_on_ipns.assert_called_once_with("DummyManager-daily")
        dm.load_stac_metadata.assert_called_once_with("DummyManager-daily")
        cid.set.assert_called_once_with(base="base32")

    @staticmethod
    def test_latest_hash_customized(manager_class):
        dm = manager_class()
        dm.custom_latest_hash = "this is the latest hash now"
        dm.check_stac_on_ipns = Mock(return_value=True)
        cid = Mock(spec=("set",))
        cid.set.return_value = "thecid"
        dm.load_stac_metadata = Mock(return_value={"assets": {"zmetadata": {"href": cid}}})
        assert dm.latest_hash() == "this is the latest hash now"

        dm.check_stac_on_ipns.assert_not_called()
        dm.load_stac_metadata.assert_not_called()
        cid.set.assert_not_called()

    @staticmethod
    def test_latest_hash_pass_in_key(manager_class):
        dm = manager_class()
        dm.check_stac_on_ipns = Mock(return_value=True)
        cid = Mock(spec=("set",))
        cid.set.return_value = "thecid"
        dm.load_stac_metadata = Mock(return_value={"assets": {"zmetadata": {"href": cid}}})
        assert dm.latest_hash("hotdogs-hourly") == "thecid"

        dm.check_stac_on_ipns.assert_called_once_with("hotdogs-hourly")
        dm.load_stac_metadata.assert_called_once_with("hotdogs-hourly")
        cid.set.assert_called_once_with(base="base32")

    @staticmethod
    def test_latest_hash_already_set(manager_class):
        dm = manager_class()
        dm.dataset_hash = "this is the hash now"
        dm.check_stac_on_ipns = Mock(return_value=True)
        cid = Mock(spec=("set",))
        cid.set.return_value = "thecid"
        dm.load_stac_metadata = Mock(return_value={"assets": {"zmetadata": {"href": cid}}})
        assert dm.latest_hash() == "this is the hash now"

        dm.check_stac_on_ipns.assert_not_called()
        dm.load_stac_metadata.assert_not_called()
        cid.set.assert_not_called()

    @staticmethod
    def test_latest_hash_no_stac_metadata(manager_class):
        dm = manager_class()
        dm.check_stac_on_ipns = Mock(return_value=False)
        cid = Mock(spec=("set",))
        cid.set.return_value = "thecid"
        dm.load_stac_metadata = Mock(return_value={"assets": {"zmetadata": {"href": cid}}})
        assert dm.latest_hash() is None

        dm.check_stac_on_ipns.assert_called_once_with("DummyManager-daily")
        dm.load_stac_metadata.assert_not_called()
        cid.set.assert_not_called()

    @staticmethod
    def test_check_stac_on_ipns(manager_class):
        dm = manager_class()
        dm.ipns_retrieve_object = Mock(return_value=({"stac_version": 666}, "foo"))
        assert dm.check_stac_on_ipns("hotdogs-daily") is True

        dm.ipns_retrieve_object.assert_called_once_with("hotdogs-daily")

    @staticmethod
    def test_check_stac_on_ipns_not_stac_compliant(manager_class):
        dm = manager_class()
        dm.ipns_retrieve_object = Mock(return_value=({"nonstac_version": 666}, "foo"))
        assert dm.check_stac_on_ipns("hotdogs-daily") is False

        dm.ipns_retrieve_object.assert_called_once_with("hotdogs-daily")

    @staticmethod
    def test_check_stac_on_ipns_key_error(manager_class):
        dm = manager_class()
        dm.ipns_retrieve_object = Mock(side_effect=KeyError("hotdogs-daily"))
        assert dm.check_stac_on_ipns("hotdogs-daily") is False

        dm.ipns_retrieve_object.assert_called_once_with("hotdogs-daily")

    @staticmethod
    def test_check_stac_on_ipns_value_error(manager_class):
        dm = manager_class()
        dm.ipns_retrieve_object = Mock(side_effect=ValueError("hotdogs-daily"))
        assert dm.check_stac_on_ipns("hotdogs-daily") is False

        dm.ipns_retrieve_object.assert_called_once_with("hotdogs-daily")

    @staticmethod
    def test_check_stac_on_ipns_timeout(manager_class):
        dm = manager_class()
        dm.ipns_retrieve_object = Mock(side_effect=TimeoutError())
        assert dm.check_stac_on_ipns("hotdogs-daily") is False

        dm.ipns_retrieve_object.assert_called_once_with("hotdogs-daily")

    @staticmethod
    def test_check_stac_on_ipns_http_error(manager_class):
        dm = manager_class()
        dm.ipns_retrieve_object = Mock(side_effect=HTTPError())
        assert dm.check_stac_on_ipns("hotdogs-daily") is False

        dm.ipns_retrieve_object.assert_called_once_with("hotdogs-daily")
