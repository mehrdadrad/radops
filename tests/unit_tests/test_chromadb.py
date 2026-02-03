import unittest
from unittest.mock import MagicMock, patch
from src.storage.chromadb import ChromaVectorStoreManager
from src.config.config import SyncLocationSettings
from src.storage.protocols import LoadedDocument

class TestChromaVectorStoreManager(unittest.TestCase):
    def setUp(self):
        self.mock_settings_patcher = patch('src.storage.chromadb.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.vector_store.providers = {"chroma": {"path": ".chromadb"}}
        
        self.mock_chroma_patcher = patch('src.storage.chromadb.chromadb')
        self.mock_chroma = self.mock_chroma_patcher.start()
        self.mock_chroma.__version__ = "0.0.0"
        
        self.mock_client = MagicMock()
        self.mock_chroma.PersistentClient.return_value = self.mock_client
        
        self.mock_fs_loader_patcher = patch('src.storage.chromadb.FileSystemLoader')
        self.mock_fs_loader = self.mock_fs_loader_patcher.start()
        
        self.mock_lc_chroma_patcher = patch('src.storage.chromadb.Chroma')
        self.mock_lc_chroma = self.mock_lc_chroma_patcher.start()

    def tearDown(self):
        self.mock_settings_patcher.stop()
        self.mock_chroma_patcher.stop()
        self.mock_fs_loader_patcher.stop()
        self.mock_lc_chroma_patcher.stop()

    def test_initialization(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        self.mock_chroma.PersistentClient.assert_called_once()
        self.assertEqual(manager.name(), "test_manager")

    def test_update_vector_store(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        # Mock vectorstore instance
        mock_vectorstore = self.mock_lc_chroma.return_value
        manager._vectorstores["test_coll"] = mock_vectorstore
        
        # Mock existing docs check (return empty to force update)
        mock_vectorstore.get.return_value = {'metadatas': []}
        
        docs = [LoadedDocument(content="test content", path="file.txt", last_modified=100)]
        manager._update_vector_store(docs, "test_coll")
        
        mock_vectorstore.add_documents.assert_called()

    def test_update_vector_store_deletion(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        mock_vectorstore = self.mock_lc_chroma.return_value
        manager._vectorstores["test_coll"] = mock_vectorstore
        
        docs = [LoadedDocument(content=None, path="file.txt", last_modified=100)]
        manager._update_vector_store(docs, "test_coll")
        
        mock_vectorstore.delete.assert_called()

    @patch('src.storage.chromadb.GoogleDriveLoader')
    def test_initialization_gdrive(self, mock_gdrive_loader):
        loc = SyncLocationSettings(
            name="test", type="gdrive", path="folder_id", collection="test_coll", sync_interval=0
        )
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        mock_gdrive_loader.assert_called_once()
        self.assertIn("test_coll", manager._vectorstores)

    @patch('src.storage.chromadb.GithubLoader')
    def test_initialization_github(self, mock_github_loader):
        loc = SyncLocationSettings(
            name="test", type="github", path="owner/repo", collection="test_coll", sync_interval=0
        )
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        mock_github_loader.assert_called_once()
        self.assertIn("test_coll", manager._vectorstores)

    def test_update_vector_store_skip_unchanged(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        mock_vectorstore = self.mock_lc_chroma.return_value
        manager._vectorstores["test_coll"] = mock_vectorstore
        
        # Mock existing docs with same timestamp
        mock_vectorstore.get.return_value = {
            'metadatas': [{'source': 'file.txt', 'last_modification': 100}]
        }
        
        docs = [LoadedDocument(content="test content", path="/tmp/file.txt", last_modified=100)]
        manager._update_vector_store(docs, "test_coll")
        
        # Should not add documents if unchanged
        mock_vectorstore.add_documents.assert_not_called()
        # Should not delete for upsert if unchanged
        mock_vectorstore.delete.assert_not_called()

    def test_update_vector_store_metadata_extraction(self):
        # Setup metadata config
        metadata_structure = MagicMock()
        metadata_structure.delimiter = "_"
        prop = MagicMock()
        prop.name = "category"
        metadata_structure.structure = [prop]
        
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        loc.metadata = metadata_structure
        
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        mock_vectorstore = self.mock_lc_chroma.return_value
        manager._vectorstores["test_coll"] = mock_vectorstore
        
        mock_vectorstore.get.return_value = {'metadatas': []}
        
        # Filename matches structure: category_rest.txt
        docs = [LoadedDocument(content="content", path="/tmp/report_2023.txt", last_modified=100)]
        
        manager._update_vector_store(docs, "test_coll")
        
        args, _ = mock_vectorstore.add_documents.call_args
        added_docs = args[0]
        self.assertEqual(added_docs[0].metadata["category"], "report")

    def test_periodic_sync(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=60
        )
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=60)
        
        # Check if watcher was called on loader
        mock_loader_instance = self.mock_fs_loader.return_value
        mock_loader_instance.watcher.assert_called()
        
        manager.stop_periodic_sync()
        mock_loader_instance.stop_watcher.assert_called()
        mock_loader_instance.close.assert_called()

    def test_close(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        with patch.object(manager, 'stop_periodic_sync') as mock_stop:
            manager.close()
            mock_stop.assert_called_once()