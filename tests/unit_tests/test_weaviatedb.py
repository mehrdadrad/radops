import unittest
from unittest.mock import MagicMock, patch, ANY
from src.storage.weaviatedb import WeaviateVectorStoreManager
from src.config.config import SyncLocationSettings
from src.storage.protocols import LoadedDocument

class TestWeaviateVectorStoreManager(unittest.TestCase):
    def setUp(self):
        self.mock_settings_patcher = patch('src.storage.weaviatedb.settings')
        self.mock_settings = self.mock_settings_patcher.start()
        self.mock_settings.vector_store.providers = {"weaviate": {}}
        
        self.mock_weaviate_patcher = patch('src.storage.weaviatedb.weaviate')
        self.mock_weaviate = self.mock_weaviate_patcher.start()
        self.mock_weaviate.__version__ = "0.0.0"
        
        self.mock_client = MagicMock()
        self.mock_weaviate.connect_to_custom.return_value = self.mock_client
        self.mock_client.is_ready.return_value = True
        
        self.mock_fs_loader_patcher = patch('src.storage.weaviatedb.FileSystemLoader')
        self.mock_fs_loader = self.mock_fs_loader_patcher.start()
        
        self.mock_lc_weaviate_patcher = patch('src.storage.weaviatedb.Weaviate')
        self.mock_lc_weaviate = self.mock_lc_weaviate_patcher.start()

    def tearDown(self):
        self.mock_settings_patcher.stop()
        self.mock_weaviate_patcher.stop()
        self.mock_fs_loader_patcher.stop()
        self.mock_lc_weaviate_patcher.stop()

    def test_initialization(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        # Mock collection existence check
        self.mock_client.collections.exists.return_value = True
        
        manager = WeaviateVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        self.mock_weaviate.connect_to_custom.assert_called_once()
        self.mock_client.collections.exists.assert_called_with("test_coll")
        self.assertEqual(manager.name(), "test_manager")

    def test_create_collection_if_missing(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        self.mock_client.collections.exists.return_value = False
        
        WeaviateVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        self.mock_client.collections.create.assert_called_with(
            name="test_coll", properties=ANY
        )

    def test_update_vector_store(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        self.mock_client.collections.exists.return_value = True
        manager = WeaviateVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        # Mock collection object and schema validation
        mock_collection = MagicMock()
        self.mock_client.collections.get.return_value = mock_collection
        # Mock property check to include 'source'
        mock_prop = MagicMock()
        mock_prop.name = "source"
        mock_collection.config.get.return_value.properties = [mock_prop]
        
        # Mock query response for existing docs (empty means all are new)
        mock_collection.query.fetch_objects.return_value.objects = []
        
        docs = [LoadedDocument(content="test content", path="file.txt", last_modified=100)]
        manager._update_vector_store(docs, "test_coll")
        
        # Verify add_documents was called on the vectorstore
        manager.get_vectorstore("test_coll").add_documents.assert_called()

    def test_close(self):
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = WeaviateVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        manager.close()
        self.mock_client.close.assert_called_once()