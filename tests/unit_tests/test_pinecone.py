import unittest
from unittest.mock import MagicMock, patch
from src.storage.pinecone import PineconeVectorStoreManager
from src.config.config import SyncLocationSettings
from src.storage.protocols import LoadedDocument

class TestPineconeVectorStoreManager(unittest.TestCase):
    def setUp(self):
        self.mock_settings_patcher = patch('src.storage.pinecone.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.vector_store.providers = {
            "pinecone": {"api_key": "test-key", "index_name": "test-index"}
        }
        
        self.mock_pinecone_patcher = patch('src.storage.pinecone.Pinecone')
        self.mock_pinecone = self.mock_pinecone_patcher.start()
        
        self.mock_pc_instance = MagicMock()
        self.mock_pinecone.return_value = self.mock_pc_instance
        self.mock_index = MagicMock()
        self.mock_pc_instance.Index.return_value = self.mock_index
        
        self.mock_fs_loader_patcher = patch('src.storage.pinecone.FileSystemLoader')
        self.mock_fs_loader = self.mock_fs_loader_patcher.start()
        
        self.mock_lc_pinecone_patcher = patch('src.storage.pinecone.PineconeVectorStore')
        self.mock_lc_pinecone = self.mock_lc_pinecone_patcher.start()

    def tearDown(self):
        self.mock_settings_patcher.stop()
        self.mock_pinecone_patcher.stop()
        self.mock_fs_loader_patcher.stop()
        self.mock_lc_pinecone_patcher.stop()

    def test_initialization(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_ns", sync_interval=0
        )
        manager = PineconeVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        self.mock_pinecone.assert_called_with(api_key="test-key")
        self.mock_pc_instance.Index.assert_called_with("test-index")
        self.assertEqual(manager.name(), "test_manager")

    def test_update_vector_store(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_ns", sync_interval=0
        )
        manager = PineconeVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        mock_vectorstore = self.mock_lc_pinecone.return_value
        manager._vectorstores["test_ns"] = mock_vectorstore
        
        # Mock index query to return no matches (forcing update)
        self.mock_index.query.return_value.matches = []
        self.mock_index.describe_index_stats.return_value = {"dimension": 1536}
        
        docs = [LoadedDocument(content="test content", path="file.txt", last_modified=100)]
        manager._update_vector_store(docs, "test_ns")
        
        mock_vectorstore.add_documents.assert_called()

    def test_update_vector_store_skip_existing(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_ns", sync_interval=0
        )
        manager = PineconeVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        mock_vectorstore = self.mock_lc_pinecone.return_value
        manager._vectorstores["test_ns"] = mock_vectorstore

        # Mock index query to return match with same timestamp
        match = MagicMock()
        match.metadata = {"last_modification": 100}
        self.mock_index.query.return_value.matches = [match]
        
        docs = [LoadedDocument(content="test content", path="file.txt", last_modified=100)]
        manager._update_vector_store(docs, "test_ns")
        
        # Should NOT add documents
        mock_vectorstore.add_documents.assert_not_called()