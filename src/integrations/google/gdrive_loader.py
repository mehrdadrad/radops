import io
import logging
import os.path
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from storage.protocols import DataLoader, LoadedDocument
from storage.registry import DataLoaderRegistry

logger = logging.getLogger(__name__)

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class GoogleDriveLoader(DataLoader):
    """
    Loads documents from a Google Drive folder and watches for updates.

    This is a simplified example. A real implementation would use the
    Google Drive API client library for authentication and file operations.
    
    This loader is designed to work with a folder ID rather than a file path,
    so the method signatures from the protocol are adapted for this use case.
    """

    def __init__(self, folder_ids: List[str], poll_interval: int = 60, **kwargs: Any):
        self.folder_ids = folder_ids
        self.poll_interval = poll_interval
        self.credentials_path = kwargs.get("credentials_path", "credentials.json")
        self.token_path = kwargs.get("token_path", "token.json")
        self._gdrive_service: Optional[Any] = None  # Placeholder for the real service object
        self._known_files: Dict[str, Dict[str, Any]] = {}  # file_id -> {name, modified_time, content}
        self._watcher_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def _authenticate(self) -> Any:
        """Authenticates with the Google Drive API using OAuth 2.0."""
        if self._gdrive_service:
            return self._gdrive_service
        
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing expired Google Drive token...")
                creds.refresh(Request())
            else:
                logger.info("No valid Google Drive token found, starting authentication flow...")
                if not os.path.exists(self.credentials_path):
                    logger.error(
                        f"Google Drive credentials file not found at '{self.credentials_path}'."
                    )
                    logger.error(
                        "Please download it from your Google Cloud project and place it in the root directory."
                    )
                    raise FileNotFoundError(f"'{self.credentials_path}' not found.")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_path, "w") as token:
                token.write(creds.to_json())
            logger.info(f"Google Drive token saved to '{self.token_path}'.")

        try:
            self._gdrive_service = build("drive", "v3", credentials=creds)
            logger.info("Google Drive authentication successful.")
        except HttpError as error:
            logger.error(f"An error occurred during Google Drive authentication: {error}")
            raise
        return self._gdrive_service
    
    def _get_current_files_in_folder(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetches the current state of files in the Drive folder using the API.
        """
        service = self._authenticate()
        logger.debug(
            f"Scanning Google Drive folders: {self.folder_ids} with service: {service}"
        )
        try:
            # Build a query to search in all specified parent folders
            query_parents = " or ".join(
                [f"'{folder_id}' in parents" for folder_id in self.folder_ids]
            )
            results = service.files().list(
                q=f"({query_parents}) and trashed=false",
                fields="files(id, name, modifiedTime, mimeType)"
            ).execute()
            files = results.get('files', [])
            # Filter out folders
            return {
                f['id']: {
                    'name': f['name'],
                    'modified_time': f['modifiedTime'],
                    'mimeType': f['mimeType']
                } for f in files
                if f['mimeType'] != 'application/vnd.google-apps.folder'
            }
        except HttpError as error:
            logger.error(f"An error occurred while listing files: {error}")
            return {}

    def _fetch_documents(self) -> List[LoadedDocument]:
        """Fetches documents that have been added, modified, or deleted."""
        is_initial_scan = not self._known_files
        if not is_initial_scan:
            logger.info("Scanning Google Drive folders for changes: %s", self.folder_ids)
        current_files = self._get_current_files_in_folder()

        if is_initial_scan and current_files:
            logger.info("Initial scan: loading all %d documents from Google Drive.", len(current_files))
            self._known_files = current_files
            # On first run, we only care about adding, not what was "removed" from an empty state
            return self._read_files(self._known_files.keys())

        old_ids = set(self._known_files.keys())
        current_ids = set(current_files.keys())

        added_ids = current_ids - old_ids
        removed_ids = old_ids - current_ids
        modified_ids = {
            id_ for id_ in old_ids.intersection(current_ids)
            if self._known_files[id_]['modified_time'] != current_files[id_]['modified_time']
        }

        if not (added_ids or removed_ids or modified_ids):
            return []

        logger.info(
            "Changes detected in GDrive: %d added, %d removed, %d modified.",
            len(added_ids), len(removed_ids), len(modified_ids)
        )

        docs_to_upsert = self._read_files(added_ids.union(modified_ids))
        docs_to_delete = [
            LoadedDocument(
                content=None,
                path=self._known_files[id_]['name'],
                last_modified=0
            )
            for id_ in removed_ids
        ]

        self._known_files = current_files
        return docs_to_upsert + docs_to_delete
    
    def _read_files(self, file_ids: List[str]) -> List[LoadedDocument]:
        """Reads file content from Google Drive using the API."""
        service = self._authenticate()
        docs = []
        for file_id in file_ids:
            # We need the *new* metadata for the file
            file_meta = self._known_files.get(file_id)
            if not file_meta:
                continue
            
            try:
                fh = io.BytesIO()
                
                # Use export_media for Google Docs, get_media for everything else
                if file_meta.get('mimeType') == 'application/vnd.google-apps.document':
                    logger.info(f"Exporting Google Doc: {file_meta['name']} ({file_id})")
                    request = service.files().export_media(fileId=file_id, mimeType='text/plain')
                else:
                    logger.info(f"Downloading file: {file_meta['name']} ({file_id})")
                    request = service.files().get_media(fileId=file_id)

                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    logger.debug(f"Download {int(status.progress() * 100)}%.")
                
                content = fh.getvalue().decode('utf-8')
                last_modified = int(time.mktime(time.strptime(
                    file_meta['modified_time'], "%Y-%m-%dT%H:%M:%S.%fZ"
                )))
                docs.append(LoadedDocument(
                    content=content,
                    path=file_meta['name'],
                    last_modified=last_modified
                ))
            except HttpError as error:
                logger.error(f"An error occurred downloading file {file_id}: {error}")
            
        return docs

    def load_data(self) -> List[LoadedDocument]:
        """
        Loads all documents from the specified folder.
        """
        self._known_files = {}  # Reset to force load all
        return self._fetch_documents()
    
    def _watch_loop(self, callback: Callable[[List[LoadedDocument]], None]) -> None:
        """The loop that polls for changes."""
        logger.info("Starting watcher for GDrive folders %s...", self.folder_ids)
        while not self._stop_event.is_set():
            updated_docs = self._fetch_documents()
            if updated_docs:
                logger.info("Found %d updated documents in GDrive. Triggering callback.", len(updated_docs))
                callback(updated_docs)
            self._stop_event.wait(self.poll_interval)
        logger.info("GDrive watcher stopped.")
    
    def watcher(self, callback: Callable[[List[LoadedDocument]], None]) -> None:
        """Starts a background thread to watch for file changes."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            logger.info("GDrive watcher is already running.")
            return

        self._authenticate()  # Authenticate once before starting the thread

        self._stop_event.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_loop, args=(callback,), daemon=True
        )
        self._watcher_thread.start()

    def stop_watcher(self):
        """Stops the background watcher thread."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            logger.info("Stopping GDrive watcher...")
            self._stop_event.set()
            self._watcher_thread.join()

    def close(self):
        """Stops the watcher and closes the connection to Google Drive."""
        if self._gdrive_service:
            self._gdrive_service.close()
        logger.info("GDrive loader closed.")


# Register the loader instance with the existing registry
DataLoaderRegistry.register_loader("google_drive", GoogleDriveLoader)            
