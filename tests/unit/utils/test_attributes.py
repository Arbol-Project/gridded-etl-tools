import pytest

from gridded_etl_tools.utils import attributes, store


class TestAttributes:
    @staticmethod
    def test_abstract_class_attribute_access_missing_attribute(manager_class):
        class Base(manager_class):
            foo = attributes.abstract_class_property()

        class Subclass(Base): ...

        with pytest.raises(TypeError):
            Subclass.foo

    @staticmethod
    def test_abstract_class_attribute_construct_instance_missing_attribute(manager_class):
        class Base(manager_class):
            foo = attributes.abstract_class_property()

        class Subclass(Base): ...

        with pytest.raises(TypeError):
            Subclass()

    @staticmethod
    def test_abstract_class_attribute_with_fallback(manager_class):
        class Base(manager_class):
            foo = attributes.abstract_class_property(fallback="bar")

        class Subclass(Base):
            @classmethod
            def bar(cls):
                return "hi mom!"

        with pytest.deprecated_call():
            assert Subclass.foo == "hi mom!"

    @staticmethod
    def test_abstract_class_attribute_instance_with_fallback(manager_class):
        class Base(manager_class):
            foo = attributes.abstract_class_property(fallback="bar")

        class Subclass(Base):
            @classmethod
            def bar(cls):
                return "hi mom!"

        with pytest.deprecated_call():
            assert Subclass().foo == "hi mom!"

    @staticmethod
    def test_abstract_class_attribute_with_missing_fallback(manager_class):
        class Base(manager_class):
            foo = attributes.abstract_class_property(fallback="bar")

        class Subclass(Base): ...

        with pytest.raises(TypeError):
            Subclass.foo

    @staticmethod
    def test_backwards_compatible_no_fallback_or_override(manager_class):
        class MyClass(manager_class):
            foo = attributes._backwards_compatible("bar", "get_foo")

        assert MyClass.foo == "bar"

    @staticmethod
    def test_backwards_compatible_w_attribute_override(manager_class):
        class Base(manager_class):
            foo = attributes._backwards_compatible("bar", "get_foo")

        class Subclass(Base):
            foo = "baz"

        assert Subclass.foo == "baz"

    @staticmethod
    def test_backwards_compatible_w_class_fallback_override(manager_class):
        class Base(manager_class):
            foo = attributes._backwards_compatible("bar", "get_foo")

        class Subclass(Base):
            @classmethod
            def get_foo(self):
                return "boo"

        with pytest.deprecated_call():
            assert Subclass.foo == "boo"

    @staticmethod
    def test_host_organization(manager_class):
        with pytest.deprecated_call():
            assert manager_class.host_organization() == ""

    @staticmethod
    def test_organization_read_only(manager_class):
        dm = manager_class()
        with pytest.raises(AttributeError):
            dm.organization = "Hydra"

    @staticmethod
    def test_dataset_name_fallback_to_name(base_class):
        class Subclass(base_class):
            @classmethod
            def name(cls):
                return "SubclassName"

        with pytest.deprecated_call():
            assert Subclass.dataset_name == "SubclassName"

    @staticmethod
    def test_name(manager_class):
        with pytest.deprecated_call():
            assert manager_class.name() == "DummyManager"

    @staticmethod
    def test_name_no_fallback(base_class):
        with pytest.deprecated_call():
            with pytest.raises(TypeError):
                base_class.name()

    @staticmethod
    def test_collection(manager_class):
        with pytest.deprecated_call():
            assert manager_class.collection() == "Vintage Guitars"

    @staticmethod
    def test_remote_protocol(manager_class):
        with pytest.deprecated_call():
            assert manager_class.remote_protocol() == "handshake"

    @staticmethod
    def test_identical_dims(manager_class):
        with pytest.deprecated_call():
            assert manager_class.identical_dims() == ["x", "y"]

    @staticmethod
    def test_concat_dims(manager_class):
        with pytest.deprecated_call():
            assert manager_class.concat_dims() == ["z", "zz"]

    @staticmethod
    def test_temporal_resolution(manager_class):
        with pytest.deprecated_call():
            assert str(manager_class.temporal_resolution()) == "daily"

    @staticmethod
    def test_missing_value_indicator(manager_class):
        with pytest.deprecated_call():
            assert manager_class.missing_value_indicator() == 42

    @staticmethod
    def test_irregular_update_cadence(manager_class):
        with pytest.deprecated_call():
            assert manager_class.irregular_update_cadence() is None

    @staticmethod
    def test_final_lag_in_days(manager_class):
        assert manager_class.final_lag_in_days == 3

    @staticmethod
    def test_preliminary_lag_in_days(manager_class):
        assert not manager_class.preliminary_lag_in_days

    @staticmethod
    def test_expected_nan_frequency(manager_class):
        assert manager_class.expected_nan_frequency == 0.2

    @staticmethod
    def test_kerchunk_s3_options(manager_class):
        assert manager_class.kerchunk_s3_options == {}

    @staticmethod
    def test_open_dataset_kwargs(manager_class):
        assert manager_class.open_dataset_kwargs == {}

    @staticmethod
    def test_release_status(manager_class):
        assert manager_class.release_status == "final"

    @staticmethod
    def test_version(manager_class):
        assert manager_class.version is None

    @staticmethod
    def test_region(manager_class):
        assert manager_class.region == "global"

    @staticmethod
    def test_coordinate_reference_system(manager_class):
        # DummyManagerBase sets this to "EPSG:4326"
        assert manager_class.coordinate_reference_system == "EPSG:4326"

    @staticmethod
    def test_provider_url(manager_class):
        # DummyManagerBase sets this to "http://example.com/provider"
        assert manager_class.provider_url == "http://example.com/provider"

    @staticmethod
    def test_data_download_url(manager_class):
        # DummyManagerBase sets this to "http://example.com/download"
        assert manager_class.data_download_url == "http://example.com/download"

    @staticmethod
    def test_publisher(manager_class):
        # DummyManagerBase sets this to "Test Publisher"
        assert manager_class.publisher == "Test Publisher"

    @staticmethod
    def test_title(manager_class):
        # DummyManagerBase sets this to "Test Dataset"
        assert manager_class.title == "Test Dataset"

    @staticmethod
    def test_provider_description(manager_class):
        # DummyManagerBase sets this to "Test provider description"
        assert manager_class.provider_description == "Test provider description"

    @staticmethod
    def test_dataset_description(manager_class):
        # DummyManagerBase sets this to "Test dataset description"
        assert manager_class.dataset_description == "Test dataset description"

    @staticmethod
    def test_license(manager_class):
        # DummyManagerBase sets this to "Test License"
        assert manager_class.license == "Test License"

    @staticmethod
    def test_terms_of_service(manager_class):
        # DummyManagerBase sets this to "Test ToS"
        assert manager_class.terms_of_service == "Test ToS"

    @staticmethod
    def test_standard_name(manager_class):
        # DummyManagerBase sets this to "test_var"
        assert manager_class.standard_name == "test_var"

    @staticmethod
    def test_long_name(manager_class):
        # DummyManagerBase sets this to "Test Variable"
        assert manager_class.long_name == "Test Variable"

    @staticmethod
    def test_unit_of_measurement(manager_class):
        # DummyManagerBase sets this to "parsecs"
        assert manager_class.unit_of_measurement == "parsecs"

    @staticmethod
    def test_store_with_correct_type(manager_class):
        dm = manager_class()
        local_store = store.Local(dm)
        dm.store = local_store
        assert dm.store is local_store
        assert dm.store.dm is dm

    @staticmethod
    def test_store_with_incorrect_type(manager_class):
        dm = manager_class()
        with pytest.raises(TypeError):
            dm.store = "not a store"
