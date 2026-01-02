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