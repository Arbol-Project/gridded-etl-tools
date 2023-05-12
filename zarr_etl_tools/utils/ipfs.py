import dag_cbor
import requests


from multiformats import multicodec, multihash

from requests.adapters import HTTPAdapter, Retry
from requests.exceptions import Timeout as TimeoutError
from requests.exceptions import HTTPError

# Base methods


def get_retry_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(connect=5, total=5, backoff_factor=4)
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


class IPFS:
    """
    Methods to be inherited by a DatasetManager that needs to instantiate and interact with an IPFS client
    """

    def __init__(
        self,
        host: str,
        default_hash: str
        | int
        | multicodec.Multicodec
        | multihash.Multihash = "sha2-256",
        default_base: str = "base32",
        default_timeout: int = 600,
    ):
        self._host = host
        self._default_base = default_base
        self._default_timeout = default_timeout
        self._default_hash = default_hash

        self.ipfs_session = get_retry_session()

    # FUNDAMENTAL METHODS

    def ipfs_get(self, cid: str) -> dict:
        """
        Fetch a DAG CBOR object by its IPFS hash and return it as a JSON

        Parameters
        ----------
        cid : str
            The IPFS hash corresponding to a given object (implicitly DAG CBOR)

        Returns
        -------
        dict
            The referenced DAG CBOR object decoded as a JSON
        """
        res = self.ipfs_session.post(
            self._host + "/api/v0/block/get",
            timeout=self._default_timeout,
            params={"arg": str(cid)},
        )
        res.raise_for_status()
        return dag_cbor.decode(res.content)

    def ipfs_put(self, bytes_obj: bytes, should_pin: bool = True) -> str:
        """
        Turn a bytes object (file type object) into a DAG CBOR object compatible with IPFS and return its corresponding multihash

        Parameters
        ----------
        bytes_obj : bytes
            A file type (io.BytesIO) object to be converted into a DAG object and put on IPFS

        should_pin : bool, optional
            Whether to automatically pin this object when converting it to a DAG. Defauls to True.

        Returns
        -------
        str
            The IPFS hash (base32 encoded) corresponding to the newly created DAG object
        """
        res = self.ipfs_session.post(
            self._host + "/api/v0/dag/put",
            params={
                "store-codec": "dag-cbor",
                "input-codec": "dag-json",
                "pin": should_pin,
                "hash": self._default_hash,
            },
            files={"dummy": bytes_obj},
        )
        res.raise_for_status()
        return res.json()["Cid"]["/"]  # returns hash of DAG object created

    # IPNS METHODS

    def ipns_resolve(self, key: str) -> str:
        """
        Resolve the IPFS hash corresponding to a given key

        Parameters
        ----------
        key : str
            The IPNS key (human readable name) referencing a given dataset

        Returns
        -------
        str
            The IPFS hash corresponding to a given IPNS name hash
        """
        ipns_key =  self.ipns_key_list()[key]
        res = self.ipfs_session.post(
            self._host + "/api/v0/name/resolve",
            timeout=self._default_timeout,
            params={"arg": ipns_key},
        )
        res.raise_for_status()
        return res.json()["Path"][6:]  # 6: shaves off leading '/ipfs/'

    def ipns_publish(self, key: str, cid: str, offline: bool = False) -> str:
        """
        Publish an IPNS key string and return the corresponding name hash

        Parameters
        ----------
        key : str
            The human readable key part of the IPNS key pair referencing an object
        cid : str
            The hash the key pair will resolve to
        offline : bool, optional
            An optional trigger to disable the publication of this IPNS key and name hash over the IPFS network.
            Offline mode will be much faster but will not publish the key pair to peers' Distributed Hash Tables on the global network.

        Returns
        -------
        str
            The IPNS "name" hash corresponding to the published key
        """
        # Pin the IPFS CID and publish your key, linking they key to the desired CID
        res = self.ipfs_session.post(
            self._host + "/api/v0/name/publish",
            timeout=self._default_timeout,
            params={
                "arg": cid,
                "key": key,
                "allow-offline": offline,
                "offline": offline,
            },
        )
        res.raise_for_status()
        ipns_name_hash = res.json()["Name"]
        self.info(
            f"Published CID {cid} for key {key} to name hash {ipns_name_hash} and pinned it in the process"
        )
        return ipns_name_hash

    def ipns_key_list(self) -> dict:
        """
        Return IPFS's Key List as a dict corresponding of key strings and associated ipns name hashes

        Returns
        -------
        ipns_key_hash_dict : dict
            All the IPNS name hashes and keys in the local IPFS repository
        """
        ipns_key_hash_dict = {}
        for name_hash_pair in self.ipfs_session.post(
            self._host + "/api/v0/key/list", timeout=self._default_timeout
        ).json()["Keys"]:
            key, val = tuple(name_hash_pair.values())
            ipns_key_hash_dict[key] = val
        return ipns_key_hash_dict

    def ipns_generate_name(self, key: str = None) -> str:
        """
        Generate a stable IPNS name hash to populate the `href` field of any STAC Object.
        If a name hash already exists, return it.

        Parameters
        ----------
        key : str
            The IPNS key string to be used to reference a given object

        Returns
        -------
        ipns_name_hash : str
            The IPNS name hash (str) resulting from the publication of an empty dict
        """
        if key is None:
            key = self.json_key()
        # Only generate the key in IPFS's registry if it doesn't already exist
        if key not in self.ipns_key_list():
            res = self.ipfs_session.post(
                self._host + "/api/v0/key/gen",
                timeout=self._default_timeout,
                params={"arg": key, "type": "rsa"},
            )
            self.info(f"Key '{key}' generated in key list")
            res.raise_for_status()
            return res.json()["Id"]
        else:
            return self.ipns_key_list()[key]

    def ipns_retrieve_object(self, ipns_name: str) -> tuple[dict, str]:
        """
        Retrieve a JSON object using its human readable IPNS name key
        (e.g. 'prism-precip-hourly).

        Parameters
        ----------
        key : str
            The IPNS key string referencing a given object
        timeout : int, optional
            Time in seconds to wait for a response from `ipfs.name.resolve` and `ipfs.dag.get` before failing. Defaults to 30.

        Returns
        -------
        tuple[dict, str, str] | None
            A tuple of the JSON and the hash part of the IPNS key pair
        """
        ipns_key_hash = self.ipns_key_list()[ipns_name]
        ipfs_hash = self.ipns_resolve(ipns_name)
        json_obj = self.ipfs_get(ipfs_hash)
        return json_obj, ipns_key_hash

    # RETRIEVE LATEST OBJECT

    def latest_hash(self, key: str = None) -> str | None:
        """
        Get the latest hash of the climate data for this dataset. This hash can be loaded into xarray through xarray.open_zarr if
        this is a Zarr compatible dataset. This will be the hash contained within the STAC metadata if this is STAC compatible dataset.

        Parameters
        ----------
        key : str, optional
            The name of the dataset in the format it is stored in the IPNS namespace. If `None`, the value of `self.json_key()`
            will be used.

        Returns
        -------
        str | None
            The IPFS/IPLD hash corresponding to the climate data, or `None` if no data was found
        """
        if self.custom_latest_hash is not None:
            return self.custom_latest_hash
        else:
            if key is None:
                key = self.json_key()
            if hasattr(self, "dataset_hash") and self.dataset_hash:
                return self.dataset_hash
            elif self.check_stac_on_ipns(key):
                # the dag_cbor.decode call in `self.ipfs_get` will auto-convert the `{'\' : <CID>}``
                # it finds to a CID object. Convert it back to a hash of type `str``
                return str(
                    self.load_stac_metadata(key)["assets"]["zmetadata"]["href"].set(
                        base=self._default_base
                    )
                )
            else:
                return None

    def check_stac_on_ipns(self, key: str) -> bool:
        """
        Convenience function to check whether a STAC object is registered with IPNS under the assigned key

        Parameters
        ----------
        key : str
            The IPNS key (human readable name) referencing a given dataset

        Returns
        -------
        exists : bool
            Whether the IPNS key exists in the IPFS key registry or not
        """
        exists = True
        try:
            obj_json = self.ipns_retrieve_object(key)[0]
            if "stac_version" not in obj_json:
                raise TypeError
        except TypeError:
            self.info(f"Non-STAC compliant object found for {key}")
            exists = False
        except (KeyError, ValueError):
            self.info(
                f"No existing STAC-compliant object found for {key}."
            )
            exists = False
        except (HTTPError, TimeoutError):
            self.info(f"No object found at {key}")
            exists = False
        return exists
