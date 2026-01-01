import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from storage.protocols import DataLoader, LoadedDocument
from storage.registry import DataLoaderRegistry

# NOTE: To handle PDF files, you would need to install a library like PyPDF2.
# You can do this by running: pip install PyPDF2
# Then, you can uncomment the following line:
# import PyPDF2

logger = logging.getLogger(__name__)

class FileSystemLoader(DataLoader):
    """
    Loads documents from a local directory and watches for updates.

    This loader scans a specified directory for .txt and .pdf files,
    loads their content, and can watch for changes (creations,
    modifications, deletions) to keep the data up-to-date.
    """

    def __init__(self, path: str, poll_interval: int = 10, **kwargs: Any):
        self.path = path
        self.poll_interval = poll_interval
        self._known_files: Dict[str, float] = {}
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._supported_extensions = [".txt", ".pdf"]

    def _read_file(self, file_path: Path) -> Optional[LoadedDocument]:
        """Reads content from a file based on its extension."""
        try:
            content = ""
            last_modified = int(file_path.stat().st_mtime)

            if file_path.suffix == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            elif file_path.suffix == ".pdf":
                # This is a placeholder for PDF reading logic.
                # A real implementation would use a library like PyPDF2.
                # For example:
                # with open(file_path, "rb") as f:
                #     reader = PyPDF2.PdfReader(f)
                #     content = "\n".join(page.extract_text() for page in reader.pages)
                logger.info(
                    "PDF reading not fully implemented. Returning placeholder "
                    "for %s.", file_path.name
                )
                content = f"Content of PDF file: {file_path.name}"
            else:
                return None

            return LoadedDocument(
                content=content,
                path=str(file_path),
                last_modified=last_modified
            )
        except OSError as e:
            logger.error(f"Error reading or getting stats for file {file_path}: {e}")
            return None

    def _scan_directory(self) -> Dict[str, float]:
        """Scans the directory for supported files and their modification times."""
        path = Path(self.path)
        current_files: Dict[str, float] = {}

        if path.is_file():
            if path.suffix in self._supported_extensions:
                try:
                    current_files[str(path)] = path.stat().st_mtime
                except OSError as e:
                    logger.warning(f"Could not stat file {path}, skipping: {e}")
        elif path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.suffix in self._supported_extensions:
                    try:
                        current_files[str(file_path)] = file_path.stat().st_mtime
                    except OSError as e:
                        logger.warning(f"Could not stat file {file_path}, skipping: {e}")

        return current_files

    def _fetch_documents(self) -> List[LoadedDocument]:
        """
        Fetches documents that have been added or modified since the last scan.
        """
        logger.info("Scanning directory: %s", self.path)
        current_files = self._scan_directory()
        
        if not self._known_files:
            # First run, load all documents
            logger.info("Initial scan, loading all documents.")
            self._known_files = current_files
            return [
                doc for file_path in self._known_files
                if (doc := self._read_file(Path(file_path)))
            ]

        old_paths = set(self._known_files.keys())
        current_paths = set(current_files.keys())

        added_paths = current_paths - old_paths
        removed_paths = old_paths - current_paths
        # Check for modification time changes in existing files
        modified_paths = {
            path for path in old_paths.intersection(current_paths)
            if self._known_files[path] != current_files[path]
        }

        if not (added_paths or removed_paths or modified_paths):
            return []

        logger.info(
            "Changes detected: %d added, %d removed, %d modified.",
            len(added_paths),
            len(removed_paths),
            len(modified_paths)
        )

        self._known_files = current_files
        return [
            doc for path in added_paths.union(modified_paths)
            if (doc := self._read_file(Path(path)))
        ]

    def load_data(self) -> List[LoadedDocument]:
        """
        Loads all documents from the specified directory.
        """
        self._known_files = {}  # Reset to force load all
        return self._fetch_documents()

    def _watch_loop(self, callback: Callable[[List[LoadedDocument]], None]) -> None:
        """The loop that polls for changes."""
        logger.info("Starting watcher for directory %s...", self.path)
        while not self._stop_event.is_set():
            updated_docs = self._fetch_documents()
            if updated_docs:
                logger.info("Found %d updated documents. Triggering callback.", len(updated_docs))
                callback(updated_docs)
            self._stop_event.wait(self.poll_interval)
        logger.info("Filesystem watcher stopped.")

    def watcher(self, callback: Callable[[List[LoadedDocument]], None]) -> None:
        """Starts a background thread to watch for file changes."""

        if self._watcher_thread and self._watcher_thread.is_alive():
            logger.info("Watcher is already running.")
            return

        if not Path(self.path).is_dir():
            logger.error("Path '%s' is not a valid directory.", self.path)
            return

        self._stop_event.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_loop, args=(callback,), daemon=True
        )
        self._watcher_thread.start()

    def stop_watcher(self):
        """Stops the background watcher thread."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            logger.info("Stopping filesystem watcher...")
            self._stop_event.set()
            self._watcher_thread.join()

    def close(self):
        pass

# Register the loader instance with the existing registry
DataLoaderRegistry.register_loader("filesystem", FileSystemLoader)