"""
Notion Loader for syncing pages and databases.
"""
import concurrent.futures
import logging
import os
import threading
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document
from config.integrations import integration_settings
from storage.protocols import DataLoader

logger = logging.getLogger(__name__)

try:
    from notion_client import Client
except ImportError:
    Client = None


class NotionLoader(DataLoader):
    """
    Loader for Notion pages and databases.
    
    Attributes:
        path (str): The Notion Page ID or Database ID to sync.
        loader_config (dict): Configuration containing 'api_token'.
    """

    def __init__(self, loader_config: Dict[str, Any]):
        super().__init__(loader_config)
        if Client is None:
            raise ImportError(
                "The 'notion-client' library is required. "
                "Please install it via `pip install notion-client`."
            )

        self.name = loader_config.get("name", "notion")
        self.path = loader_config.get("path")
        self.loader_config = loader_config.get("loader_config", {})
        self.poll_interval = loader_config.get("sync_interval", 60)
        self._known_files: Dict[str, str] = {}  # page_id -> last_edited_time
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Resolve API Token
        self.api_token = None

        profile_name = self.loader_config.get("profile")
        if profile_name:
            profile = integration_settings.notion.get(profile_name)
            if profile:
                self.api_token = profile.token

        if not self.api_token:
            self.api_token = self.loader_config.get("api_token")

        if not self.api_token:
            self.api_token = os.getenv("NOTION_API_TOKEN")

        if not self.api_token:
            raise ValueError(
                "Notion API token is missing. "
                "Provide it in 'loader_config.api_token' or 'NOTION_API_TOKEN' env var."
            )

        self.client = Client(auth=self.api_token)

    def load_data(self) -> List[Document]:
        """
        Loads all documents from the specified Notion page or database.
        Resets the state to force a full load.
        """
        self._known_files = {}
        return self._fetch_documents()

    def _fetch_documents(self) -> List[Document]:
        """Fetches documents that have been added or modified."""
        current_pages = self._get_all_pages_metadata()

        if not self._known_files:
            logger.info("Initial scan: loading %d documents from Notion.", len(current_pages))
            self._known_files = {pid: p["last_edited_time"] for pid, p in current_pages.items()}
            return self._fetch_pages_content(list(current_pages.values()))

        # Detect changes
        changed_pages = []
        current_ids = set(current_pages.keys())
        known_ids = set(self._known_files.keys())

        added_ids = current_ids - known_ids
        for pid in added_ids:
            changed_pages.append(current_pages[pid])

        modified_ids = []
        for pid in current_ids.intersection(known_ids):
            if current_pages[pid]["last_edited_time"] != self._known_files[pid]:
                changed_pages.append(current_pages[pid])
                modified_ids.append(pid)

        removed_ids = known_ids - current_ids

        if not (added_ids or modified_ids or removed_ids):
            return []

        logger.info(
            "Changes detected in Notion: %d added, %d modified, %d removed.",
            len(added_ids),
            len(modified_ids),
            len(removed_ids),
        )

        # Update state
        self._known_files = {pid: p["last_edited_time"] for pid, p in current_pages.items()}

        return self._fetch_pages_content(changed_pages)

    def _get_all_pages_metadata(self) -> Dict[str, Dict]:
        """Retrieves metadata for all pages in the configured path (Page or Database)."""
        pages = {}
        try:
            obj = self.client.objects.retrieve(block_id=self.path)
            object_type = obj.get("object")

            if object_type == "database":
                query_params = {"database_id": self.path}
                has_more = True
                start_cursor = None
                while has_more:
                    if start_cursor:
                        query_params["start_cursor"] = start_cursor
                    response = self.client.databases.query(**query_params)
                    for page in response.get("results", []):
                        pages[page["id"]] = page
                    has_more = response.get("has_more", False)
                    start_cursor = response.get("next_cursor")
            elif object_type == "page":
                pages[obj["id"]] = obj
            else:
                logger.warning("Unsupported Notion object type: %s", object_type)
        except Exception as e:
            logger.error("Failed to fetch Notion metadata: %s", e)
        return pages

    def _fetch_pages_content(self, pages: List[Dict]) -> List[Document]:
        """Fetches the content for a list of pages in parallel."""
        docs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_page = {
                executor.submit(self._load_page, page["id"], page): page
                for page in pages
            }
            for future in concurrent.futures.as_completed(future_to_page):
                page = future_to_page[future]
                try:
                    docs.append(future.result())
                except Exception as e:
                    logger.error("Error loading page %s: %s", page.get("id"), e)
        return docs

    def _load_page(self, page_id: str, page_metadata: Optional[Dict] = None) -> Document:
        if not page_metadata:
            page_metadata = self.client.pages.retrieve(page_id=page_id)

        title = self._extract_title(page_metadata)
        url = page_metadata.get("url", "")
        last_edited_time = page_metadata.get("last_edited_time")

        content = self._fetch_page_blocks(page_id)

        metadata = {
            "source": url or f"notion://{page_id}",
            "title": title,
            "id": page_id,
            "last_edited_time": last_edited_time,
            "loader": "notion",
        }

        return Document(page_content=content, metadata=metadata)

    def _fetch_page_blocks(self, block_id: str) -> str:
        content_parts = []
        has_more = True
        start_cursor = None

        while has_more:
            response = self.client.blocks.children.list(
                block_id=block_id, start_cursor=start_cursor
            )
            blocks = response.get("results", [])

            for block in blocks:
                b_type = block.get("type")
                content_data = block.get(b_type, {})
                rich_text = content_data.get("rich_text", [])
                plain_text = "".join([t.get("plain_text", "") for t in rich_text])

                if plain_text:
                    content_parts.append(plain_text)

            has_more = response.get("has_more", False)
            start_cursor = response.get("next_cursor")

        return "\n".join(content_parts)

    def _extract_title(self, page: Dict) -> str:
        properties = page.get("properties", {})
        for prop in properties.values():
            if prop.get("type") == "title":
                return "".join([t.get("plain_text", "") for t in prop.get("title", [])])
        return "Untitled"

    def _watch_loop(self, callback: Callable[[List[Document]], None]) -> None:
        """The loop that polls for changes."""
        logger.info("Starting watcher for Notion path %s...", self.path)
        while not self._stop_event.is_set():
            updated_docs = self._fetch_documents()
            if updated_docs:
                logger.info("Found %d updated documents in Notion. Triggering callback.", len(updated_docs))
                callback(updated_docs)
            self._stop_event.wait(self.poll_interval)
        logger.info("Notion watcher stopped.")

    def watcher(self, callback: Callable[[List[Document]], None]) -> None:
        """Starts a background thread to watch for changes."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            logger.info("Notion watcher is already running.")
            return

        self._stop_event.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_loop, args=(callback,), daemon=True
        )
        self._watcher_thread.start()

    def stop_watcher(self):
        """Stops the background watcher thread."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            logger.info("Stopping Notion watcher...")
            self._stop_event.set()
            self._watcher_thread.join()