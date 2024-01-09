import json

from unittest import mock

import pytest


@pytest.fixture
def input_files(tmp_path, mocker):
    folder = tmp_path / "datasets" / "input_files"
    folder.mkdir(parents=True)

    files = {
        folder / "one.nc": {"one": "fun"},
        folder / "two.nc4": {"two": "shoe"},
        folder / "three.grib": {"three": "free"},
        folder / "four.grib1": {"four": "soar"},
        folder / "five.grib2": {"five": "alive"},
        folder / "six.grb1": {"six": "styx"},
        folder / "seven.grb2": {"seven": "eleven"},
    }

    for file in files:
        with open(file, "w") as f:
            json.dump(files[file], f)

    def kerchunkify(self, path):
        return json.load(open(path))

    mocker.patch("gridded_etl_tools.utils.zarr_methods.Transform.kerchunkify", kerchunkify)
    mocker.patch("gridded_etl_tools.utils.zarr_methods.Transform.input_files", mock.Mock(return_value=files))

    return files


class TestTransform:
    @staticmethod
    def test_create_zarr_json(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path

        md.create_zarr_json()

        mzz.assert_called_once_with(
            path=list(map(str, input_files.keys())),
            indicts=list(input_files.values()),
            remote_protocol="handshake",
            remote_options={"anon": True},
            identical_dims=["x", "y"],
            concat_dims=["z", "zz"],
            preprocess=md.preprocess_kerchunk,
        )

        outfile = tmp_path / "datasets" / "merged_zarr_jsons" / "DummyManager_zarr.json"
        mzz.return_value.translate.assert_called_once_with(filename=outfile)

    @staticmethod
    def test_create_zarr_json_w_file_filter(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path

        md.create_zarr_json(file_filters=["three", "five"])

        mzz.assert_called_once_with(
            path=[str(list(input_files.keys())[i]) for i in [2, 4]],
            indicts=[list(input_files.values())[i] for i in [2, 4]],
            remote_protocol="handshake",
            remote_options={"anon": True},
            identical_dims=["x", "y"],
            concat_dims=["z", "zz"],
            preprocess=md.preprocess_kerchunk,
        )

        outfile = tmp_path / "datasets" / "merged_zarr_jsons" / "DummyManager_zarr.json"
        mzz.return_value.translate.assert_called_once_with(filename=outfile)

    @staticmethod
    def test_create_zarr_json_w_zarr_jsons_attribute(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path
        md.zarr_jsons = ["how's", "my", "driving?"]

        md.create_zarr_json()

        mzz.assert_called_once_with(
            path=["how's", "my", "driving?"],
            remote_protocol="handshake",
            remote_options={"anon": True},
            identical_dims=["x", "y"],
            concat_dims=["z", "zz"],
            preprocess=md.preprocess_kerchunk,
        )

        outfile = tmp_path / "datasets" / "merged_zarr_jsons" / "DummyManager_zarr.json"
        mzz.return_value.translate.assert_called_once_with(filename=outfile)

    @staticmethod
    def test_create_zarr_json_w_outfile_path(manager_class, tmp_path, mocker, input_files):
        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path

        md.create_zarr_json(outfile_path="put/it/here")

        mzz.assert_called_once_with(
            path=list(map(str, input_files.keys())),
            indicts=list(input_files.values()),
            remote_protocol="handshake",
            remote_options={"anon": True},
            identical_dims=["x", "y"],
            concat_dims=["z", "zz"],
            preprocess=md.preprocess_kerchunk,
        )

        mzz.return_value.translate.assert_called_once_with(filename="put/it/here")

    @staticmethod
    def test_create_zarr_json_already_done(manager_class, tmp_path, mocker):
        folder = tmp_path / "datasets" / "merged_zarr_jsons"
        folder.mkdir(parents=True)
        outfile = folder / "DummyManager_zarr.json"
        open(outfile, "w").write("already done")

        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path

        md.create_zarr_json(force_overwrite=False)
        mzz.assert_not_called()

    @staticmethod
    def test_create_zarr_json_force_overwrite(manager_class, tmp_path, mocker, input_files):
        folder = tmp_path / "datasets" / "merged_zarr_jsons"
        folder.mkdir(parents=True)
        outfile = folder / "DummyManager_zarr.json"
        open(outfile, "w").write("already done")

        mzz = mocker.patch("gridded_etl_tools.utils.zarr_methods.MultiZarrToZarr")
        md = manager_class()
        md._root_directory = tmp_path

        md.create_zarr_json()

        mzz.assert_called_once_with(
            path=list(map(str, input_files.keys())),
            indicts=list(input_files.values()),
            remote_protocol="handshake",
            remote_options={"anon": True},
            identical_dims=["x", "y"],
            concat_dims=["z", "zz"],
            preprocess=md.preprocess_kerchunk,
        )

        outfile = tmp_path / "datasets" / "merged_zarr_jsons" / "DummyManager_zarr.json"
        mzz.return_value.translate.assert_called_once_with(filename=outfile)

    @staticmethod
    def test_kerchunkify_local(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock()
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        assert md.use_local_zarr_jsons is False

        assert md.kerchunkify("/home/body/over/here.json") is md.local_kerchunk.return_value
        md.local_kerchunk.assert_called_once_with("/home/body/over/here.json", 0)
        md.remote_kerchunk.assert_not_called()
        md.zarr_json_in_memory_to_file.assert_not_called()

    @staticmethod
    def test_kerchunkify_remote(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock()
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        assert md.use_local_zarr_jsons is False

        assert md.kerchunkify("s3://remote/over/here.json", 42) is md.remote_kerchunk.return_value
        md.local_kerchunk.assert_not_called()
        md.remote_kerchunk.assert_called_once_with("s3://remote/over/here.json", 42)
        md.zarr_json_in_memory_to_file.assert_not_called()

    @staticmethod
    def test_kerchunkify_local_use_local_zarr_jsons(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock()
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        md.use_local_zarr_jsons = True
        md.protocol = "file"

        assert md.kerchunkify("/home/body/over/here.json") is md.local_kerchunk.return_value
        md.local_kerchunk.assert_called_once_with("/home/body/over/here.json", 0)
        md.remote_kerchunk.assert_not_called()
        md.zarr_json_in_memory_to_file.assert_called_once_with(
            md.local_kerchunk.return_value, "/home/body/over/here.json"
        )

    @staticmethod
    def test_kerchunkify_remote_use_local_zarr_jsons(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock()
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        md.use_local_zarr_jsons = True
        md.protocol = "s3"

        assert (
            md.kerchunkify("s3://remote/over/here.json", local_file_path="/local/here.json")
            is md.remote_kerchunk.return_value
        )
        md.local_kerchunk.assert_not_called()
        md.remote_kerchunk.assert_called_once_with("s3://remote/over/here.json", 0)
        md.zarr_json_in_memory_to_file.assert_called_once_with(md.remote_kerchunk.return_value, "/local/here.json")

    @staticmethod
    def test_kerchunkify_remote_use_local_zarr_jsons_missing_local_file_path(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock()
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        md.use_local_zarr_jsons = True
        md.protocol = "s3"

        with pytest.raises(ValueError):
            md.kerchunkify("s3://remote/over/here.json")

    @staticmethod
    def test_kerchunkify_local_use_local_zarr_jsons_kerchunk_returns_list(manager_class):
        md = manager_class()
        md.local_kerchunk = mock.Mock(return_value=["abacus", "bertrand", "chloroform"])
        md.remote_kerchunk = mock.Mock()
        md.zarr_json_in_memory_to_file = mock.Mock()
        md.use_local_zarr_jsons = True
        md.protocol = "file"

        assert md.kerchunkify("/home/body/over/here.json") is md.local_kerchunk.return_value
        md.local_kerchunk.assert_called_once_with("/home/body/over/here.json", 0)
        md.remote_kerchunk.assert_not_called()
        md.zarr_json_in_memory_to_file.assert_has_calls(
            [
                mock.call("abacus", "/home/body/over/here.json"),
                mock.call("bertrand", "/home/body/over/here.json"),
                mock.call("chloroform", "/home/body/over/here.json"),
            ]
        )

    @staticmethod
    def test_local_kerchunk_netcdf(manager_class, mocker):
        fsspec = mocker.patch("gridded_etl_tools.utils.zarr_methods.fsspec")
        fs = fsspec.filesystem.return_value
        infile = fs.open.return_value.__enter__.return_value

        SingleHdf5ToZarr = mocker.patch("gridded_etl_tools.utils.zarr_methods.SingleHdf5ToZarr")
        scanned_zarr_json = SingleHdf5ToZarr.return_value.translate.return_value

        md = manager_class()
        md.file_type = "NetCDF"
        assert md.local_kerchunk("/read/from/here") is scanned_zarr_json

        fsspec.filesystem.assert_called_once_with("file")
        SingleHdf5ToZarr.assert_called_once_with(h5f=infile, url="/read/from/here", inline_threshold=5000)
        SingleHdf5ToZarr.return_value.translate.assert_called_once_with()

    @staticmethod
    def test_local_kerchunk_grib(manager_class, mocker):
        scan_grib = mocker.patch("gridded_etl_tools.utils.zarr_methods.scan_grib")
        scanned_zarr_json = mock.Mock()
        scan_grib.return_value = [scanned_zarr_json]

        md = manager_class()
        md.file_type = "GRIB"
        md.grib_filter = "iamafilter"  # TODO: Validate data manager attributes to make sure grib_filter exists
        assert md.local_kerchunk("/read/from/here") is scanned_zarr_json

        scan_grib.assert_called_once_with(url="/read/from/here", filter="iamafilter", inline_threshold=20)

    @staticmethod
    def test_local_kerchunk_invalid_file_type(manager_class):
        md = manager_class()
        with pytest.raises(ValueError):
            md.local_kerchunk("/read/from/here")

    @staticmethod
    def test_local_kerchunk_netcdf_os_error(manager_class, mocker):
        fsspec = mocker.patch("gridded_etl_tools.utils.zarr_methods.fsspec")
        fs = fsspec.filesystem.return_value
        fs.open.return_value.__enter__.return_value

        SingleHdf5ToZarr = mocker.patch("gridded_etl_tools.utils.zarr_methods.SingleHdf5ToZarr")
        SingleHdf5ToZarr.side_effect = OSError

        md = manager_class()
        md.file_type = "NetCDF"
        with pytest.raises(ValueError):
            md.local_kerchunk("/read/from/here")

    @staticmethod
    def test_remote_kerchunk_netcdf(manager_class, mocker):
        s3fs = mocker.patch("gridded_etl_tools.utils.zarr_methods.s3fs")
        s3 = s3fs.S3FileSystem.return_value
        infile = s3.open.return_value.__enter__.return_value

        SingleHdf5ToZarr = mocker.patch("gridded_etl_tools.utils.zarr_methods.SingleHdf5ToZarr")
        SingleHdf5ToZarr.return_value.translate.return_value = scanned_zarr_json = {"hi": "mom!"}

        md = manager_class()
        md.zarr_jsons = []  # TODO: See note about code smell in remote_kerchunk method
        md.file_type = "NetCDF"

        assert md.remote_kerchunk("over/here") is scanned_zarr_json
        assert md.zarr_jsons == [{"hi": "mom!"}]

        s3fs.S3FileSystem.assert_called_once_with()
        s3.open.assert_called_once_with("over/here", anon=True, default_cache_type="readahead")

        SingleHdf5ToZarr.assert_called_once_with(h5f=infile, url="over/here")
        SingleHdf5ToZarr.return_value.translate.assert_called_once_with()

    @staticmethod
    def test_remote_kerchunk_grib(manager_class, mocker):
        scan_grib = mocker.patch("gridded_etl_tools.utils.zarr_methods.scan_grib")
        scanned_zarr_json = {"hi": "mom!"}
        scan_grib.return_value = [scanned_zarr_json]

        md = manager_class()
        md.zarr_jsons = []  # TODO: See note about code smell in remote_kerchunk method
        md.file_type = "GRIB"
        md.grib_filter = "iamafilter"  # TODO: Validate data manager attributes to make sure grib_filter exists

        assert md.remote_kerchunk("over/here") is scanned_zarr_json
        assert md.zarr_jsons == [{"hi": "mom!"}]

        scan_grib.assert_called_once_with(
            url="over/here",
            storage_options={"anon": True, "default_cache_type": "readahead"},
            filter="iamafilter",
            inline_threshold=20,
        )

    @staticmethod
    def test_remote_kerchunk_grib_non_zero_scan_indices(manager_class, mocker):
        scan_grib = mocker.patch("gridded_etl_tools.utils.zarr_methods.scan_grib")
        scanned_zarr_json = {"hi": "mom!"}
        scan_grib.return_value = {42: scanned_zarr_json}

        md = manager_class()
        md.zarr_jsons = []  # TODO: See note about code smell in remote_kerchunk method
        md.file_type = "GRIB"
        md.grib_filter = "iamafilter"  # TODO: Validate data manager attributes to make sure grib_filter exists

        assert md.remote_kerchunk("over/here", scan_indices=42) is scanned_zarr_json
        assert md.zarr_jsons == [{"hi": "mom!"}]

        scan_grib.assert_called_once_with(
            url="over/here",
            storage_options={"anon": True, "default_cache_type": "readahead"},
            filter="iamafilter",
            inline_threshold=20,
        )

    @staticmethod
    def test_remote_kerchunk_unexpected_file_type(manager_class, mocker):
        md = manager_class()
        md.file_type = "Some other weird format"

        with pytest.raises(ValueError):
            md.remote_kerchunk("over/here")

    @staticmethod
    def test_remote_kerchunk_grib_remote_scan_returns_list(manager_class, mocker):
        scan_grib = mocker.patch("gridded_etl_tools.utils.zarr_methods.scan_grib")
        scanned_zarr_json = [{"hi": "mom!"}]
        scan_grib.return_value = [scanned_zarr_json]

        md = manager_class()
        md.zarr_jsons = []  # TODO: See note about code smell in remote_kerchunk method
        md.file_type = "GRIB"
        md.grib_filter = "iamafilter"  # TODO: Validate data manager attributes to make sure grib_filter exists

        assert md.remote_kerchunk("over/here") is scanned_zarr_json
        assert md.zarr_jsons == [{"hi": "mom!"}]

        scan_grib.assert_called_once_with(
            url="over/here",
            storage_options={"anon": True, "default_cache_type": "readahead"},
            filter="iamafilter",
            inline_threshold=20,
        )

    @staticmethod
    def test_remote_kerchunk_grib_remote_scan_returns_unexpected_structure(manager_class, mocker):
        scan_grib = mocker.patch("gridded_etl_tools.utils.zarr_methods.scan_grib")
        scanned_zarr_json = {"hi", "mom!"}
        scan_grib.return_value = [scanned_zarr_json]

        md = manager_class()
        md.zarr_jsons = []  # TODO: See note about code smell in remote_kerchunk method
        md.file_type = "GRIB"
        md.grib_filter = "iamafilter"  # TODO: Validate data manager attributes to make sure grib_filter exists

        with pytest.raises(ValueError):
            md.remote_kerchunk("over/here")

        scan_grib.assert_called_once_with(
            url="over/here",
            storage_options={"anon": True, "default_cache_type": "readahead"},
            filter="iamafilter",
            inline_threshold=20,
        )
