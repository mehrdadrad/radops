import base64
import fnmatch
import logging
import threading
from datetime import datetime, timezone
from typing import Callable, List, Optional, Dict, Any

from github import Github, Auth
from github.GithubException import GithubException
from config.integrations import integration_settings 

from storage.protocols import LoadedDocument

logger = logging.getLogger(__name__)


class GithubLoader:
    """
    Loader for GitHub repositories.
    Supports loading files from one or more repositories and syncing changes.
    """

    def __init__(
        self,
        repo_names: List[str],
        loader_config: Dict[str, Any],
        poll_interval: int = 60,
    ):
        """
        Initialize the GithubLoader.

        Args:
            repo_names: List of repository names (e.g., ["owner/repo"]).
            loader_config: Configuration dictionary for the loader.
            poll_interval: Interval in seconds to check for changes.
        """
        self.repo_names = [name.strip() for name in repo_names]
        self.poll_interval = poll_interval
        self.loader_config = loader_config

        self.branch = loader_config.get("branch")
        self.file_extensions = loader_config.get("file_extensions") or []
        self.exclude_patterns = loader_config.get("exclude_patterns", [])

        integration_profile = loader_config.get("integration_profile")
        self.access_token = self._get_access_token(integration_profile)

        self._stop_event = threading.Event()
        self._thread = None
        self._last_sync_time = {}  # repo_name -> datetime (UTC)

        if self.access_token:
            auth = Auth.Token(self.access_token)
            self.client = Github(auth=auth)
        else:
            logger.warning("No GitHub access token provided. Using unauthenticated client (rate limited).")
            self.client = Github()

    def _get_access_token(self, profile_name: str) -> Optional[str]:
        if not profile_name:
            return None
        try:
            profile = integration_settings.github.get(profile_name)
            if profile:
                return profile.token
            return None
        except Exception as e:
            logger.error(f"Error loading integration token: {e}")
            return None

    def _is_valid_extension(self, filename: str) -> bool:
        return any(filename.endswith(ext) for ext in self.file_extensions)

    def _is_excluded(self, filename: str) -> bool:
        return any(fnmatch.fnmatch(filename, pattern) for pattern in self.exclude_patterns)

    def load_data(self) -> List[LoadedDocument]:
        """Loads data from the configured repositories."""
        documents = []
        for repo_name in self.repo_names:
            logger.info(f"Loading data from GitHub repo: {repo_name}")
            try:
                repo = self.client.get_repo(repo_name)
                target_branch = self.branch or repo.default_branch
                
                try:
                    branch_obj = repo.get_branch(target_branch)
                    sha = branch_obj.commit.sha
                    last_modified_fallback = branch_obj.commit.commit.author.date.replace(tzinfo=timezone.utc).timestamp()
                    tree = repo.get_git_tree(sha=sha, recursive=True)
                except GithubException as e:
                    logger.error(f"Failed to get tree for {repo_name}: {e}")
                    continue

                for element in tree.tree:
                    if element.type == "blob" and self._is_valid_extension(element.path) and not self._is_excluded(element.path):
                        try:
                            blob = repo.get_git_blob(element.sha)
                            if blob.encoding == "base64":
                                content = base64.b64decode(blob.content).decode("utf-8")
                            else:
                                content = blob.content

                            documents.append(
                                LoadedDocument(
                                    path=f"{repo_name}/{element.path}",
                                    content=content,
                                    last_modified=last_modified_fallback,
                                )
                            )
                        except Exception as e:
                            logger.warning(f"Failed to process file {element.path} in {repo_name}: {e}")

                self._last_sync_time[repo_name] = datetime.now(timezone.utc)

            except Exception as e:
                logger.error(f"Error accessing repo {repo_name}: {e}")
        
        return documents

    def watcher(self, callback: Callable[[List[LoadedDocument]], None]):
        """Starts a background thread to poll for changes."""
        def _poll():
            while not self._stop_event.is_set():
                if self._stop_event.wait(self.poll_interval):
                    break
                for repo_name in self.repo_names:
                    try:
                        logger.info(f"Checking for updates in GitHub repo: {repo_name}")
                        repo = self.client.get_repo(repo_name)
                        last_sync = self._last_sync_time.get(repo_name)
                        
                        if not last_sync:
                            continue

                        target_branch = self.branch or repo.default_branch
                        commits = repo.get_commits(sha=target_branch, since=last_sync)
                        if commits.totalCount == 0:
                            continue

                        logger.info(f"Detected changes in {repo_name}, fetching updates...")
                        
                        changed_docs = []
                        processed_files = set()

                        for commit in commits:
                            for file in commit.files:
                                if file.filename in processed_files:
                                    continue
                                
                                if not self._is_valid_extension(file.filename) or self._is_excluded(file.filename):
                                    continue

                                processed_files.add(file.filename)
                                full_path = f"{repo_name}/{file.filename}"

                                if file.status == "removed":
                                    changed_docs.append(
                                        LoadedDocument(
                                            path=full_path,
                                            content=None,
                                            last_modified=commit.commit.author.date.replace(tzinfo=timezone.utc).timestamp()
                                        )
                                    )
                                else:
                                    try:
                                        content_file = repo.get_contents(file.filename, ref=commit.sha)
                                        content = content_file.decoded_content.decode("utf-8")
                                        changed_docs.append(
                                            LoadedDocument(
                                                path=full_path,
                                                content=content,
                                                last_modified=commit.commit.author.date.replace(tzinfo=timezone.utc).timestamp()
                                            )
                                        )
                                    except Exception as e:
                                        logger.warning(f"Failed to fetch updated file {file.filename}: {e}")

                        if changed_docs:
                            callback(changed_docs)
                        
                        self._last_sync_time[repo_name] = datetime.now(timezone.utc)

                    except Exception as e:
                        logger.error(f"Error in GitHub watcher for {repo_name}: {e}")

        self._thread = threading.Thread(target=_poll, daemon=True)
        self._thread.start()

    def stop_watcher(self):
        """Stops the watcher thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def close(self):
        """Closes resources."""
        self.stop_watcher()
        if hasattr(self.client, "close"):
            self.client.close()