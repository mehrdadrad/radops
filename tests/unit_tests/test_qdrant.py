import unittest
from unittest.mock import MagicMock, patch
from src.storage.qdrant import QdrantVectorStoreManager
from src.config.config import SyncLocationSettings
from src.storage.protocols import LoadedDocument

class TestQdrantVectorStoreManager(unittest.TestCase):
    def setUp(self):
        self.mock_settings_patcher = patch('src.storage.qdrant.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.vector_store.providers = {
            "qdrant": {"url": "http://localhost:6333"}
        }
        
        self.mock_qdrant_client_patcher = patch('src.storage.qdrant.QdrantClient')
        self.mock_qdrant_client = self.mock_qdrant_client_patcher.start()
        self.mock_client_instance = MagicMock()
        self.mock_qdrant_client.return_value = self.mock_client_instance
        
        self.mock_async_client_patcher = patch('src.storage.qdrant.AsyncQdrantClient')
        self.mock_async_client = self.mock_async_client_patcher.start()
        
        self.mock_fs_loader_patcher = patch('src.storage.qdrant.FileSystemLoader')
        self.mock_fs_loader = self.mock_fs_loader_patcher.start()
        
        self.mock_lc_qdrant_patcher = patch('src.storage.qdrant.QdrantVectorStore')
        self.mock_lc_qdrant = self.mock_lc_qdrant_patcher.start()

    def tearDown(self):
        self.mock_settings_patcher.stop()
        self.mock_qdrant_client_patcher.stop()
        self.mock_async_client_patcher.stop()
        self.mock_fs_loader_patcher.stop()
        self.mock_lc_qdrant_patcher.stop()

    def test_initialization(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        # Mock collection check
        self.mock_client_instance.collection_exists.return_value = True
        
        manager = QdrantVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        self.mock_qdrant_client.assert_called()
        self.assertEqual(manager.name(), "test_manager")

    def test_create_collection_if_missing(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        self.mock_client_instance.collection_exists.return_value = False
        
        # Mock embedding to determine size
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 10
        
        QdrantVectorStoreManager("test_manager", [loc], mock_embeddings, sync_interval=0)
        
        self.mock_client_instance.create_collection.assert_called()

    def test_update_vector_store(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        self.mock_client_instance.collection_exists.return_value = True
        manager = QdrantVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        mock_vectorstore = self.mock_lc_qdrant.return_value
        manager._vectorstores["test_coll"] = mock_vectorstore
        
        # Mock filter unchanged docs to return the doc (simulating it is new/changed)
        manager._filter_unchanged_docs = MagicMock(side_effect=lambda docs, col: docs)
        
        docs = [LoadedDocument(content="test content", path="file.txt", last_modified=100)]
        manager._update_vector_store(docs, "test_coll")
        
        mock_vectorstore.add_documents.assert_called()