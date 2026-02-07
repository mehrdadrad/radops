import unittest
from unittest.mock import MagicMock, patch, PropertyMock
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

    def test_connection_failure(self):
        """Test behavior when ChromaDB connection fails."""
        self.mock_chroma.PersistentClient.side_effect = Exception("Connection failed")
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        with self.assertRaises(Exception):
            ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)

    def test_unsupported_location_type(self):
        """Test behavior with unsupported location type."""
        loc = SyncLocationSettings(
            name="test", type="ftp", path="/tmp", collection="test_coll", sync_interval=0
        )
        with self.assertLogs('src.storage.chromadb', level='WARNING') as cm:
            manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
            self.assertTrue(any("Unsupported sync location type" in o for o in cm.output))
            self.assertEqual(len(manager._loaders), 0)

    def test_update_vector_store_missing_collection(self):
        """Test updating a non-existent collection."""
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        
        with self.assertLogs('src.storage.chromadb', level='ERROR') as cm:
            manager._update_vector_store([], "non_existent_coll")
            self.assertTrue(any("No vector store found" in o for o in cm.output))

    @patch('src.storage.chromadb.RecursiveCharacterTextSplitter')
    def test_chunking_configuration(self, mock_splitter):
        """Test that loader config is passed to text splitter."""
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        loc.loader_config = {"chunk_size": 1000, "chunk_overlap": 200}
        
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        mock_vectorstore = self.mock_lc_chroma.return_value
        manager._vectorstores["test_coll"] = mock_vectorstore
        mock_vectorstore.get.return_value = {'metadatas': []}
        
        docs = [LoadedDocument(content="content", path="file.txt", last_modified=100)]
        manager._update_vector_store(docs, "test_coll")
        
        mock_splitter.assert_called_with(chunk_size=1000, chunk_overlap=200)

    def test_context_manager(self):
        """Test usage as a context manager."""
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        with patch.object(ChromaVectorStoreManager, 'close') as mock_close:
            with ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0) as manager:
                self.assertIsInstance(manager, ChromaVectorStoreManager)
            mock_close.assert_called_once()

    def test_get_vectorstore(self):
        """Test retrieving vector store instance."""
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        mock_vs = self.mock_lc_chroma.return_value
        manager._vectorstores["test_coll"] = mock_vs
        
        self.assertEqual(manager.get_vectorstore("test_coll"), mock_vs)
        self.assertIsNone(manager.get_vectorstore("non_existent"))

    def test_initialization_skip_sync(self):
        """Test initialization with skip_initial_sync=True."""
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )
        manager = ChromaVectorStoreManager(
            "test_manager", [loc], MagicMock(), sync_interval=0, skip_initial_sync=True
        )
        self.mock_fs_loader.return_value.load_data.assert_not_called()

    def test_metadata_extraction_error(self):
        """Test metadata extraction failure handles gracefully."""
        loc = SyncLocationSettings(
            name="test", type="fs", path="/tmp", collection="test_coll", sync_interval=0
        )

        class BrokenMetadata:
            @property
            def structure(self):
                raise AttributeError("Boom")

        loc.metadata = BrokenMetadata()

        manager = ChromaVectorStoreManager("test_manager", [loc], MagicMock(), sync_interval=0)
        mock_vectorstore = self.mock_lc_chroma.return_value
        manager._vectorstores["test_coll"] = mock_vectorstore
        mock_vectorstore.get.return_value = {'metadatas': []}
        
        docs = [LoadedDocument(content="content", path="file.txt", last_modified=100)]
        
        with self.assertLogs('src.storage.chromadb', level='WARNING') as cm:
            manager._update_vector_store(docs, "test_coll")
            self.assertTrue(any("Could not extract metadata" in o for o in cm.output))
        
        mock_vectorstore.add_documents.assert_called()

    def test_periodic_sync_mixed_intervals(self):
        """Test periodic sync with mixed intervals (some 0, some > 0)."""
        loc1 = SyncLocationSettings(
            name="test1", type="fs", path="/tmp/1", collection="coll1", sync_interval=60
        )
        loc2 = SyncLocationSettings(
            name="test2", type="fs", path="/tmp/2", collection="coll2", sync_interval=0
        )
        
        mock_loader1 = MagicMock()
        mock_loader2 = MagicMock()
        self.mock_fs_loader.side_effect = [mock_loader1, mock_loader2]
        
        ChromaVectorStoreManager("test_manager", [loc1, loc2], MagicMock(), sync_interval=60)
        
        mock_loader1.watcher.assert_called()
        mock_loader2.watcher.assert_not_called()