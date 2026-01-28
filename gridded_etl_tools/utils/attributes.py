from abc import ABC
import warnings

import deprecation
import numpy as np
import typing

from gridded_etl_tools.utils.store import StoreInterface

_NO_FALLBACK = object()


class abstract_class_property(property):
    def __init__(self, fallback=None):
        self.fallback = fallback

    def __get__(self, obj, cls):
        if self.fallback is not None:
            fallback = getattr(cls, self.fallback, None)
            if fallback is not None:
                warnings.warn(
                    f"{cls.__name__} is using deprecated fallback, {self.fallback}, for {self.name}. "
                    f"{cls.__name__} should define {self.name}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return fallback()

        raise TypeError(f"No value in {cls.__name__} for abstract class attribute {self.name}")


class _backwards_compatible(property):
    """For backwards compatibility, until all deprecated fallbacks can be removed, we need to honor the deprecated
    fallback if it is overridden in a subclass. So, to get a class property in a backwards compatible way we have to
    traverse the mro of the class to see if we find the deprecated callback, and use that.

    Note that if a class attribute has been overridden as a class attribute then this descriptor won't be reached, so
    we don't check for the overridden class attribute here, just any possible class method fallbacks.
    """

    def __init__(self, value, fallback):
        self.value = value
        self.fallback = fallback

    def __get__(self, obj, cls):
        # There would have to be a class with an empty mro for branch coverage to be happy
        for superclass in cls.mro():  # pragma NO BRANCH
            if superclass is Attributes:
                return self.value

            if self.fallback in superclass.__dict__:
                fallback = getattr(superclass, self.fallback)
                warnings.warn(
                    f"{cls.__name__} is using deprecated fallback, {self.fallback}, for {self.name}. "
                    f"{cls.__name__} should define {self.name}.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return fallback()


class Attributes(ABC):
    """
    Abstract base class containing default attributes of Zarr ETLs
    These can be overriden in the ETL managers for a given ETL as needed
    """

    def __init_subclass__(cls, **kwargs):
        """Setup for abstract class properties."""
        super().__init_subclass__(**kwargs)
        for name, obj in list(cls.__dict__.items()):
            if isinstance(obj, (abstract_class_property, _backwards_compatible)):
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

    organization: str = _backwards_compatible("", "host_organization")  # e.g. "Arbol"
    """
    Name of the organization (your organization) hosting the data being published. Used in STAC metadata.
    """

    @classmethod
    @deprecation.deprecated("Use the organization attribute")
    def host_organization(cls) -> str:
        return cls.organization

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

    protocol: typing.Literal["s3", "file"] = abstract_class_property(fallback="remote_protocol")
    """
    Remote protocol string for MultiZarrToZarr and Xarray to use when opening input files. 'file' for local, 's3'
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

    spatial_resolution: float | None = None
    """
    The spatial resolution of a dataset in decimal degrees
    """

    spatial_precision: float | None = None
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

    update_cadence: str | None = None
    """
    The frequency with which a dataset is updated.
    """

    missing_value: str = _backwards_compatible("", "missing_value_indicator")
    """
    Indicator of a missing value in a dataset
    """

    spatial_dims: list[str] = ["latitude", "longitude"]
    """
    Strings representing the spatial dimensions of the dataset in ["y", "x"] format.
    """

    @classmethod
    @deprecation.deprecated("Use the missing_value attribute")
    def missing_value_indicator(cls) -> str:
        return cls.missing_value

    tags: list[str] = [""]
    """
    Tags for dataset.
    """

    dataset_category: typing.Literal["observation", "forecast", "ensemble", "hindcast"] = "observation"
    """
    The type of climate data provided in a given dataset. Used to control various processes.
    Valid options include "observation", "forecast", "ensemble", and "hindcast".

    Defaults to "observation".

    Ensembles and hindcasts are necessarily forecasts and semantically should be understood
    to provide (more elaborated) forecast data with 5 and 6 dimensions. Accordingly, "forecast"
    should be understood to specify 4 dimensional forecast data w/out ensembles or hindcasts.
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

    update_cadence_bounds: tuple[np.timedelta64, np.timedelta64] | None = _backwards_compatible(
        None, "irregular_update_cadence"
    )
    """
    If a dataset doesn't update on a monotonic schedule return a tuple noting the lower and upper bounds of acceptable
    updates. Intended to prevent time contiguity checks from short-circuiting valid updates for datasets with
    non-monotic update schedules.
    """

    has_nans: bool = False
    """
    If True, disable quality checks for NaN values to prevent wrongful flags
    Default value set as False
    """

    kerchunk_s3_options: dict[str, str] = {}
    """
    Options to pass to S3FileSystem when scanning remote files with Kerchunk.
    """

    open_dataset_kwargs: dict[str, str] = {}
    """Some dataset types (e.g. HDF5) need special kwargs to open in Xarray. This will pass them automatically
    during post-parse QC so these datasets can be checked automatically without issue"""

    @classmethod
    @deprecation.deprecated("Use the update_cadence_bounds attribute")
    def irregular_update_cadence(cls) -> None | tuple[np.timedelta64, np.timedelta64]:
        return cls.update_cadence_bounds

    bbox_rounding_value: int = 5
    """
    The number of decimal places to round bounding box values to.
    """

    final_lag_in_days: int = abstract_class_property()
    """
    The number of days betweenm an observation taking place and its publication in finalized form
    w/in the dataset. For example, if today is April 6th and the latest data in a dataset is for April 1st,
    it has a 5 day lag.

    May stretch into the hundreds or thousands for some datasets that publish finalized data on a an extreme lag.
    """

    preliminary_lag_in_days: int = None
    """
    The number of days betweenm an observation taking place and its publication in preliminary form
    w/in the dataset. For example, if today is April 6th and the latest data in a dataset is for April 1st,
    it has a 5 day lag.

    Only applicable to datasets that publish preliminary data, for example CHIRPS Preliminary
    """

    expected_nan_frequency: float = 0.0
    """
    Datasets contain NaN values in varying proportions depending on how the source provider encodes data.

    Updates with unusual proportions of NaN values possibly represent possibly corrupted data
    from the source provider and should be investigated manually.

    This property encodes the anticipated proportion of NaNs in a daily dataset, based on empirical study
    of the dataaset in question
    """

    @property
    def store(self) -> StoreInterface:
        """
        The store where output is written to.
        """
        # The constructor has called the setter, so we don't need to check for the presence of the attribute.
        return self._store

    @store.setter
    def store(self, new_store):
        if not isinstance(new_store, StoreInterface):
            raise TypeError("Expected instance of StoreInterface, got {type(new_store)}")

        self._store = new_store

    EXTREME_VALUES_BY_UNIT = {"deg_C": (-90, 60), "K": (183.15, 333.15), "deg_F": (-129, 140)}
    """
    minimum and maximum permissible values for common units
    """

    check_tolerance: float = 0.0001
    """
    Tolerance for matching selection coordinates within the post-parse QC `check_written_value` method.
    In the vast majority of cases this will be sufficient, only for non-standard datasets is it necessary
    to adjust this value
    """

    release_status: typing.Literal["final", "preliminary", "mixed final and preliminary", "historical"] = "final"
    """
    The release_status of the dataset.

    Valid options include "final", "preliminary", "mixed final and preliminary", and "historical".

    Defaults to "final".

    "final" is the default and indicates the latest version of the dataset.
    "preliminary" indicates a preliminary version of the dataset.
    "historical" indicates a historical version of the dataset.
    "mixed final and preliminary" indicates a dataset that contains both final and preliminary versions of the data.
    """

    version: str | None = None
    """
    The version of the dataset, i.e. version 2, version 3, satellite-based, model-based, etc.
    Flexible to accommodate the quirks of individual providers' publication practices.
    """

    region: str | None = "global"
    """
    The region of the dataset.

    Valid options include "global" and "regional".

    Defaults to "global".

    "global" indicates the dataset covers the entire globe.
    "regional" indicates the dataset covers a specific region.
    """

    coordinate_reference_system: str | None = None
    """
    The coordinate reference system of the dataset, typically as an EPSG code (e.g., "EPSG:4326").
    """

    provider_url: str | None = None
    """
    URL for the data provider's website.
    """

    data_download_url: str | None = None
    """
    URL where the source data can be downloaded from.
    """

    publisher: str | None = None
    """
    Name of the organization that publishes the data.
    """

    title: str | None = None
    """
    Human-readable title for the dataset.
    """

    provider_description: str | None = None
    """
    Description of the data provider organization.
    """

    dataset_description: str | None = None
    """
    Description of the dataset itself, including its characteristics and use cases.
    """

    license: str | None = None
    """
    License under which the data is distributed (e.g., "CC BY 4.0", "Public Domain").
    """

    terms_of_service: str | None = None
    """
    Terms of service or usage restrictions for the data.
    """

    standard_name: str | None = None
    """
    CF convention standard name for the data variable.
    """

    long_name: str | None = None
    """
    Human-readable name for the data variable.
    """

    unit_of_measurement: str | None = None
    """
    Unit of measurement for the data variable (e.g., "K", "m", "mm").
    """


# Won't get called automatically, because Attributes isn't a subclass of itself
Attributes.__init_subclass__()
