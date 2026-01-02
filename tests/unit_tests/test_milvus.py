import unittest
from unittest.mock import MagicMock, patch
from src.storage.milvus import MilvusVectorStoreManager
from src.config.config import SyncLocationSettings
from src.storage.protocols import LoadedDocument

class TestMilvusVectorStoreManager(unittest.TestCase):
    def setUp(self):
        self.mock_settings_patcher = patch('src.storage.milvus.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.vector_store.providers = {
            "milvus": {"uri": "http://localhost:19530"}
        }
        
        self.mock_milvus_client_patcher = patch('src.storage.milvus.MilvusClient')
        self.mock_milvus_client = self.mock_milvus_client_patcher.start()
        self.mock_client_instance = MagicMock()
        self.mock_milvus_client.return_value = self.mock_client_instance
        
        self.mock_fs_loader_patcher = patch('src.storage.milvus.FileSystemLoader')
        self.mock_fs_loader = self.mock_fs_loader_patcher.start()
        
        self.mock_lc_milvus_patcher = patch('src.storage.milvus.Milvus')
        self.mock_lc_milvus = self.mock_lc_milvus_patcher.start()

    def tearDown(self):
        self.mock_settings_patcher.stop()
        self.mock_milvus_client_patcher.stop()
        self.mock_fs_loader_patcher.stop()
        self.mock_lc_milvus_patcher.stop()

    def test_initialization(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = MilvusVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        self.mock_milvus_client.assert_called()
        self.assertEqual(manager.name(), "test_manager")

    def test_initialization_skip_sync(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = MilvusVectorStoreManager(
            "test_manager", [loc], MagicMock(), sync_interval=0, skip_initial_sync=True
        )
        
        self.mock_fs_loader.return_value.load_data.assert_not_called()

    def test_update_vector_store(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = MilvusVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        mock_vectorstore = self.mock_lc_milvus.return_value
        manager._vectorstores["test_coll"] = mock_vectorstore
        
        # Mock collection check
        self.mock_client_instance.has_collection.return_value = True
        # Mock query to return empty (forcing update)
        self.mock_client_instance.query.return_value = []
        
        docs = [LoadedDocument(content="test content", path="file.txt", last_modified=100)]
        manager._update_vector_store(docs, "test_coll")
        
        mock_vectorstore.add_documents.assert_called()

    def test_close(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = MilvusVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        manager.close()
        self.mock_client_instance.close.assert_called_once()