"""
GitHub loader module for loading and syncing files from GitHub repositories.
"""
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
    # pylint: disable=too-many-instance-attributes

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
            logger.warning(
                "No GitHub access token provided. "
                "Using unauthenticated client (rate limited)."
            )
            self.client = Github()

    def _get_access_token(self, profile_name: str) -> Optional[str]:
        if not profile_name:
            return None
        try:
            # pylint: disable=no-member
            profile = integration_settings.github.get(profile_name)
            if profile:
                return profile.token
            return None
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error loading integration token: %s", e)
            return None

    def _is_valid_extension(self, filename: str) -> bool:
        return any(filename.endswith(ext) for ext in self.file_extensions)

    def _is_excluded(self, filename: str) -> bool:
        return any(
            fnmatch.fnmatch(filename, pattern)
            for pattern in self.exclude_patterns
        )

    def _load_repo(self, repo_name: str) -> List[LoadedDocument]:
        """Loads data from a single repository."""
        documents = []
        logger.info("Loading data from GitHub repo: %s", repo_name)
        try:
            repo = self.client.get_repo(repo_name)
            target_branch = self.branch or repo.default_branch

            try:
                branch_obj = repo.get_branch(target_branch)
                sha = branch_obj.commit.sha
                commit_date = branch_obj.commit.commit.author.date
                last_modified_fallback = commit_date.replace(
                    tzinfo=timezone.utc
                ).timestamp()
                tree = repo.get_git_tree(sha=sha, recursive=True)
            except GithubException as e:
                logger.error("Failed to get tree for %s: %s", repo_name, e)
                return []

            for element in tree.tree:
                if (element.type == "blob" and
                        self._is_valid_extension(element.path) and
                        not self._is_excluded(element.path)):
                    try:
                        blob = repo.get_git_blob(element.sha)
                        if blob.encoding == "base64":
                            content = base64.b64decode(
                                blob.content
                            ).decode("utf-8")
                        else:
                            content = blob.content

                        documents.append(
                            LoadedDocument(
                                path=f"{repo_name}/{element.path}",
                                content=content,
                                last_modified=last_modified_fallback,
                            )
                        )
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.warning(
                            "Failed to process file %s in %s: %s",
                            element.path, repo_name, e
                        )

            self._last_sync_time[repo_name] = datetime.now(timezone.utc)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error accessing repo %s: %s", repo_name, e)

        return documents

    def load_data(self) -> List[LoadedDocument]:
        """Loads data from the configured repositories."""
        documents = []
        for repo_name in self.repo_names:
            documents.extend(self._load_repo(repo_name))
        return documents

    def _check_repo_updates(self, repo_name: str) -> List[LoadedDocument]:
        """Checks for updates in a single repository."""
        changed_docs = []
        try:
            logger.info("Checking for updates in GitHub repo: %s", repo_name)
            repo = self.client.get_repo(repo_name)
            last_sync = self._last_sync_time.get(repo_name)

            if not last_sync:
                return []

            target_branch = self.branch or repo.default_branch
            commits = repo.get_commits(sha=target_branch, since=last_sync)
            if commits.totalCount == 0:
                return []

            logger.info("Detected changes in %s, fetching updates...", repo_name)

            processed_files = set()

            for commit in commits:
                for file in commit.files:
                    if file.filename in processed_files:
                        continue

                    if (not self._is_valid_extension(file.filename) or
                            self._is_excluded(file.filename)):
                        continue

                    processed_files.add(file.filename)
                    full_path = f"{repo_name}/{file.filename}"
                    commit_date = commit.commit.author.date
                    timestamp = commit_date.replace(
                        tzinfo=timezone.utc
                    ).timestamp()

                    if file.status == "removed":
                        changed_docs.append(
                            LoadedDocument(
                                path=full_path,
                                content=None,
                                last_modified=timestamp
                            )
                        )
                    else:
                        try:
                            content_file = repo.get_contents(
                                file.filename, ref=commit.sha
                            )
                            content = content_file.decoded_content.decode("utf-8")
                            changed_docs.append(
                                LoadedDocument(
                                    path=full_path,
                                    content=content,
                                    last_modified=timestamp
                                )
                            )
                        except Exception as e:  # pylint: disable=broad-exception-caught
                            logger.warning(
                                "Failed to fetch updated file %s: %s",
                                file.filename, e
                            )

            self._last_sync_time[repo_name] = datetime.now(timezone.utc)

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error in GitHub watcher for %s: %s", repo_name, e)

        return changed_docs

    def _watch_loop(self, callback: Callable[[List[LoadedDocument]], None]):
        """The loop that polls for changes."""
        logger.info("Starting watcher for GitHub repos...")
        while not self._stop_event.is_set():
            if self._stop_event.wait(self.poll_interval):
                break
            for repo_name in self.repo_names:
                changed_docs = self._check_repo_updates(repo_name)
                if changed_docs:
                    callback(changed_docs)
        logger.info("GitHub watcher stopped.")

    def watcher(self, callback: Callable[[List[LoadedDocument]], None]):
        """Starts a background thread to poll for changes."""
        if self._thread and self._thread.is_alive():
            logger.info("GitHub watcher is already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, args=(callback,), daemon=True)
        self._thread.start()

    def stop_watcher(self):
        """Stops the watcher thread."""
        if self._thread and self._thread.is_alive():
            logger.info("Stopping Github watcher...")
            self._stop_event.set()
            self._thread.join()

    def close(self):
        """Closes resources."""
        self.stop_watcher()
        if hasattr(self.client, "close"):
            logger.info("GitHub loader closed.")
            self.client.close()
