
from typing import Protocol, runtime_checkable, List, Callable, Optional
from langchain_core.vectorstores import VectorStore
from dataclasses import dataclass

class VectorStoreManager(Protocol):
    """
    Protocol for a vector store manager.
    """
    def get_vectorstore(self, collection_name: str) -> VectorStore:
        """
        Returns the vector store instance.

        Returns:
            The vector store instance.
        """
        ...
    def name(self) -> str:
        ...    

@dataclass
class LoadedDocument:
    """A standard representation for a document loaded from any source."""
    content: Optional[str]
    path: str  # Unique identifier for the source (e.g., file path, URL, GDrive ID)
    last_modified: int # Unix timestamp

@runtime_checkable
class DataLoader(Protocol):
    """
    Protocol for a data loader.
    """
    def load_data(self, path: str) -> List[LoadedDocument]:
        """
        Loads all data from a given path (e.g., directory, folder ID).

        Args:
            path: The path or identifier for the data source.

        Returns:
            A list of loaded documents.
        """
        ...

    def watcher(self, path: str, callback: Callable[[List[LoadedDocument]], None]):
        """
        Watches a source for changes and calls a callback function.

        Args:
            path: The path or identifier for the data source to watch.
            callback: The function to call when changes are detected.
        """
        ...

        