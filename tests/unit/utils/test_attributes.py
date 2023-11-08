import pytest

from gridded_etl_tools.utils import attributes


class TestAttributes:
    @staticmethod
    def test_abstract_class_attribute_access_missing_attribute(manager_class):
        class Base(manager_class):
            foo = attributes.abstract_class_property()

        class Subclass(Base):
            ...

        with pytest.raises(TypeError):
            Subclass.foo

    @staticmethod
    def test_abstract_class_attribute_construct_instance_missing_attribute(manager_class):
        class Base(manager_class):
            foo = attributes.abstract_class_property()

        class Subclass(Base):
            ...

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

        class Subclass(Base):
            ...

        with pytest.raises(TypeError):
            Subclass.foo

    @staticmethod
    def test_host_organization(manager_class):
        assert manager_class.host_organization() == ""

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
