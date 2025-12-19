from .protocols import DataLoader

class DataLoaderRegistry:
    """
    A registry for managing different data loaders.
    """
    _loaders = {}

    @classmethod
    def register_loader(cls, name: str, loader_func):
        """
        Registers a data loader function.

        Args:
            name: The name to associate with the loader (e.g., "text_file").
            loader_func: An object that conforms to the DataLoader protocol.
        """
        if not callable(loader_func):
            raise TypeError("Loader must be a callable function.")
        if not isinstance(loader_func, DataLoader):
            raise TypeError("Loader does not conform to the DataLoader protocol.")
        cls._loaders[name] = loader_func

    @classmethod
    def get_loader(cls, name: str):
        """
        Retrieves a registered data loader function.

        Args:
            name: The name of the loader to retrieve.

        Returns:
            The registered loader function.

        Raises:
            ValueError: If no loader is registered with the given name.
        """
        loader = cls._loaders.get(name)
        if loader is None:
            raise ValueError(f"No data loader registered with name: {name}")
        return loader