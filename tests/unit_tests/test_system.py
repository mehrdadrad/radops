import unittest
from unittest.mock import MagicMock, patch

from services.tools.system.kb.kb_tools import create_kb_tools, _generic_retriever, VectorStoreError
from langchain_core.tools import StructuredTool
from langchain_weaviate.vectorstores import WeaviateVectorStore


class TestKBTools(unittest.TestCase):
    @patch("services.tools.system.kb.kb_tools.settings")
    def test_create_kb_tools(self, mock_settings):
        """Test creation of KB tools from settings."""
        # Mock settings
        mock_profile = MagicMock()
        mock_profile.name = "default"
        
        mock_location = MagicMock()
        mock_location.name = "My Docs"
        mock_location.collection = "my_docs"
        mock_location.metadata.structure = []
        mock_location.prompt_file = None
        mock_location.prompt = "Search my docs"
        
        mock_profile.sync_locations = [mock_location]
        mock_settings.vector_store.profiles = [mock_profile]
        
        # Mock VectorStoreManager
        mock_manager = MagicMock()
        mock_manager.name.return_value = "default"
        mock_manager.get_vectorstore.return_value = MagicMock()
        
        tools = create_kb_tools([mock_manager])
        
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertIsInstance(tool, StructuredTool)
        self.assertEqual(tool.name, "kb_My_Docs")
        self.assertEqual(tool.description, "Search my docs")

    @patch("services.tools.system.kb.kb_tools.retriever_weaviate")
    def test_generic_retriever_weaviate(self, mock_weaviate_retriever):
        """Test routing to Weaviate retriever."""
        # Create a mock object that is an instance of WeaviateVectorStore
        mock_store = MagicMock(spec=WeaviateVectorStore)
        
        _generic_retriever("query", mock_store, {})
        mock_weaviate_retriever.assert_called_once()

    @patch("services.tools.system.kb.kb_tools.retriever_chroma")
    def test_generic_retriever_chroma(self, mock_chroma_retriever):
        """Test routing to Chroma retriever (default fallback)."""
        mock_store = MagicMock() 
        
        _generic_retriever("query", mock_store, {})
        mock_chroma_retriever.assert_called_once()

    def test_generic_retriever_no_store(self):
        """Test error when vector store is None."""
        with self.assertRaises(VectorStoreError):
            _generic_retriever("query", None, {})

    def test_generic_retriever_invalid_query(self):
        """Test validation of query."""
        mock_store = MagicMock()
        result = _generic_retriever("", mock_store, {})
        self.assertIn("Query cannot be empty", result)