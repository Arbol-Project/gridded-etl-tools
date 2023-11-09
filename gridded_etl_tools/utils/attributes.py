from abc import ABC, abstractmethod
import typing
import warnings

import deprecation
import numpy as np

from .store import StoreInterface, Local

_NO_FALLBACK = object()


class abstract_class_property(property):
    def __init__(self, fallback=None):
        self.fallback = fallback

    def __get__(self, obj, cls):
        if self.fallback is not None:
            fallback = getattr(cls, self.fallback, None)
            if fallback is not None:
                warnings.warn(
                    f"{cls.__name__}.{self.fallback}() is deprecated. Use {cls.__name__}.{self.name}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return fallback()

        raise TypeError(f"No value in {cls.__name__} for abstract class attribute {self.name}")


class Attributes(ABC):
    """
    Abstract base class containing default attributes of Zarr ETLs
    These can be overriden in the ETL managers for a given ETL as needed
    """

    def __init_subclass__(cls, **kwargs):
        """Setup for abstract class properties."""
        super().__init_subclass__(**kwargs)
        for name, obj in list(cls.__dict__.items()):
            if isinstance(obj, abstract_class_property):
                # Tell property its own name
                obj.name = name

    @classmethod
    def _check_abstract_class_properties(cls):
        """Check that a subclass has provided concrete values for all required abstract class properties."""
        for superclass in cls.mro():
            for name, obj in superclass.__dict__.items():
                if isinstance(obj, abstract_class_property):
                    getattr(cls, name)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._check_abstract_class_properties()

    @classmethod
    def _find_fallback(cls, attr):
        # There would have to be a class with an empty mro for branch coverage to be happy
        for superclass in cls.mro():  # pragma NO BRANCH
            value = superclass.__dict__.get(attr, _NO_FALLBACK)
            if value is not _NO_FALLBACK:
                if isinstance(value, abstract_class_property):
                    break

                return value

        raise TypeError(f"No value in {cls.__name__} for abstract class attribute {attr}")

    @classmethod
    def host_organization(self) -> str:
        """
        Name of the organization (your organization) hosting the data being published. Used in STAC metadata.
        """
        return ""  # e.g. "Arbol"

    dataset_name = abstract_class_property(fallback="name")
    """
    The name of each ETL is built recursively by appending each child class name to the inherited name
    """

    @classmethod
    @deprecation.deprecated("Use the dataset_name attribute")
    def name(cls) -> str:
        return cls._find_fallback("dataset_name")

    collection_name = abstract_class_property(fallback="collection")
    """
    Name of the collection
    """

    @classmethod
    @deprecation.deprecated("Use the collection_name attribute")
    def collection(cls):
        return cls._find_fallback("collection_name")

    file_type = None
    """
    The file type of each child class (and edition if relevant), e.g. GRIB1 for ERA5 data, GRIB2 for RTMA, or NetCDF
    for Copernicus Marine Service

    Used to trigger file format-appropriate functions and methods for Kerchunking and Xarray operations.
    """

    protocol: str = abstract_class_property(fallback="remote_protocol")
    """
    Remote protocol string for MultiZarrToZarr and Xarray to use when opening input files. 'File' for local, 's3'
    for S3, etc. See fsspec docs for more details.
    """

    @classmethod
    @deprecation.deprecated("Use the protocol attribute")
    def remote_protocol(cls):
        return cls._find_fallback("protocol")

    identical_dimensions = abstract_class_property(fallback="identical_dims")
    """
    List of dimension(s) whose values are identical in all input datasets. This saves Kerchunk time by having it
    read these dimensions only one time, from the first input file
    """

    @classmethod
    @deprecation.deprecated("Use the identical_dimensions attribute")
    def identical_dims(cls):
        return cls._find_fallback("identical_dimensions")

    concat_dimensions = abstract_class_property(fallback="concat_dims")
    """
    List of dimension(s) by which to concatenate input files' data variable(s) -- usually time, possibly with some
    other relevant dimension
    """

    @classmethod
    @deprecation.deprecated("Use the concat_dimensions attribute")
    def concat_dims(cls):
        return cls._find_fallback("concat_dimensions")

    data_var_dtype: float = "<f4"
    """
    The data type of the data variable
    """

    spatial_resolution: typing.Optional[float] = None
    """
    The spatial resolution of a dataset in decimal degrees
    """

    spatial_precision: typing.Optional[float] = None
    """
    The spatial resolution of a dataset in decimal degrees
    """

    time_resolution: str = abstract_class_property(fallback="temporal_resolution")
    """
    The time resolution of the dataset as a string (e.g. "hourly", "daily", "monthly", etc.)
    """

    @classmethod
    @deprecation.deprecated("Use the time_resolution attribute")
    def temporal_resolution(cls) -> str:
        return cls._find_fallback("time_resolution")

    update_cadence: typing.Optional[str] = None
    """
    The frequency with which a dataset is updated.
    """

    missing_value: str = ""
    """
    Indicator of a missing value in a dataset
    """

    @classmethod
    @deprecation.deprecated("Use the missing_value attribute")
    def missing_value_indicator(cls) -> str:
        return cls.missing_value

    tags: list[str] == [""]
    """
    Tags for dataset.
    """

    forecast: bool = False
    """
    ``True`` if the dataset provides forecast data.
    """

    ensemble: bool = False
    """
    ``True`` if the dataset provides ensemble data.
    """

    hindcast: bool = False
    """
    ``True`` if the dataset privides hindcast data.
    """

    forecast_hours: list[int] = []
    """"
    Hours provided by the forecast, if any.
    """

    ensemble_numbers: list[int] = []
    """
    Numbers uses for ensemble, if any.
    """

    hindcast_steps: list[int] = []
    """
    Steps used for hindcast, if any.
    """

    update_cadence_bounds: typing.Optional[tuple[np.timedelta64, np.timedelta64]] = None
    """G
    If a dataset doesn't update on a monotonic schedule return a tuple noting the lower and upper bounds of acceptable
    updates. Intended to prevent time contiguity checks from short-circuiting valid updates for datasets with
    non-monotic update schedules.
    """

    @classmethod
    @deprecation.deprecated("Use the update_cadence_bounds attribute")
    def irregular_update_cadence(cls) -> None | tuple[np.timedelta64, np.timedelta64]:
        return cls.update_cadence_bounds

    bbox_rounding_value: int = 5
    """
    The number of decimal places to round bounding box values to.
    """


# Won't get called automatically, because Attributes isn't a subclass of itself
Attributes.__init_subclass__()
