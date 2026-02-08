import unittest
from unittest.mock import MagicMock, patch, AsyncMock

from services.tools.system.kb.kb_tools import (
    create_kb_tools, _generic_retriever, VectorStoreError,
    create_dynamic_input, validate_retrieval_config, _sanitize_milvus_value,
    _format_results, _rerank_documents,
    retriever_chroma, retriever_qdrant, retriever_milvus,
    _validate_prompt_file_path
)
from services.tools.system.history.long_memory import memory__clear_long_term_memory
from services.tools.system.history.history_tools import (
    create_history_deletion_tool, create_history_retrieval_tool
)
from langchain_core.tools import StructuredTool
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

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

    def test_create_dynamic_input(self):
        """Test dynamic input model creation."""
        fields = {"category": "Category filter"}
        InputModel = create_dynamic_input(fields)
        schema = InputModel.schema()
        self.assertIn("category", schema["properties"])
        self.assertIn("query", schema["properties"])

    def test_validate_retrieval_config(self):
        """Test retrieval config validation."""
        # Test defaults
        config = {}
        validated = validate_retrieval_config(config)
        self.assertEqual(validated["k"], 3)
        self.assertEqual(validated["search_type"], "similarity")

        # Test corrections
        config = {"k": 1000, "score_threshold": 2.0, "search_type": "invalid"}
        validated = validate_retrieval_config(config)
        self.assertEqual(validated["k"], 100)  # Max results
        self.assertEqual(validated["score_threshold"], 0.25)  # Default
        self.assertEqual(validated["search_type"], "similarity")  # Default

    def test_sanitize_milvus_value(self):
        """Test Milvus value sanitization."""
        self.assertEqual(_sanitize_milvus_value('test"val'), 'test\\"val')
        self.assertEqual(_sanitize_milvus_value("test'val"), "test\\'val")

    def test_format_results(self):
        """Test result formatting."""
        docs = [
            Document(page_content="content1", metadata={"source": "doc1"}),
            Document(page_content="content2", metadata={"source": "doc2"})
        ]
        result = _format_results(docs)
        self.assertIn("Source: doc1", result)
        self.assertIn("content1", result)
        self.assertIn("---", result)
        
        self.assertEqual(_format_results([]), "No results found.")

    @patch("services.tools.system.kb.kb_tools.Path")
    def test_validate_prompt_file_path(self, mock_path):
        """Test prompt file path validation."""
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj
        mock_path_obj.resolve.return_value = mock_path_obj
        
        # Valid file
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = True
        self.assertTrue(_validate_prompt_file_path("valid.txt"))
        
        # Invalid: not exists
        mock_path_obj.exists.return_value = False
        self.assertFalse(_validate_prompt_file_path("missing.txt"))
        
        # Invalid: not a file
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False
        self.assertFalse(_validate_prompt_file_path("folder"))

    @patch("langchain_community.document_compressors.flashrank_rerank.FlashrankRerank")
    def test_rerank_documents(self, mock_flashrank_cls):
        """Test document reranking."""
        mock_compressor = MagicMock()
        mock_flashrank_cls.return_value = mock_compressor
        mock_compressor.compress_documents.return_value = [Document(page_content="reranked")]
        
        docs = [Document(page_content="original")]
        config = {"rerank": {"enabled": True, "provider": "flashrank"}}
        
        result = _rerank_documents(docs, "query", config)
        self.assertEqual(result[0].page_content, "reranked")
        mock_compressor.compress_documents.assert_called()

    def test_retriever_chroma_logic(self):
        """Test Chroma retriever logic."""
        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_store.as_retriever.return_value = mock_retriever
        mock_retriever.invoke.return_value = [Document(page_content="res", metadata={"source": "s"})]
        
        config = {"k": 5}
        # Pass filter kwargs
        result = retriever_chroma(mock_store, "query", config, category="docs")
        
        self.assertIn("res", result)
        # Check if filter was built
        _, kwargs = mock_store.as_retriever.call_args
        self.assertEqual(kwargs["search_kwargs"]["filter"], {"category": "docs"})

    def test_retriever_qdrant_logic(self):
        """Test Qdrant retriever logic."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [Document(page_content="res", metadata={"source": "s"})]
        
        config = {"k": 5}
        result = retriever_qdrant(mock_store, "query", config, category="docs")
        
        self.assertIn("res", result)
        mock_store.similarity_search.assert_called()
        _, kwargs = mock_store.similarity_search.call_args
        # Check filter
        self.assertIsNotNone(kwargs.get("filter"))

    def test_retriever_milvus_logic(self):
        """Test Milvus retriever logic."""
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [Document(page_content="res", metadata={"source": "s"})]
        
        config = {"k": 5}
        result = retriever_milvus(mock_store, "query", config, category="docs")
        
        self.assertIn("res", result)
        mock_store.similarity_search.assert_called()
        _, kwargs = mock_store.similarity_search.call_args
        # Check expr
        self.assertIn('category == "docs"', kwargs.get("expr"))


class TestLongMemoryTools(unittest.IsolatedAsyncioTestCase):
    @patch("services.tools.system.history.long_memory.get_mem0_client")
    async def test_memory_clear_long_term_memory_success(self, mock_get_client):
        """Test successful memory clearing."""
        mock_client = MagicMock()
        mock_client.delete_all = AsyncMock()
        mock_get_client.return_value = mock_client

        result = await memory__clear_long_term_memory.ainvoke({"user_id": "test_user"})

        mock_client.delete_all.assert_called_with(user_id="test_user")
        self.assertEqual(result, "Memory cleared")

    @patch("services.tools.system.history.long_memory.get_mem0_client")
    async def test_memory_clear_long_term_memory_failure(self, mock_get_client):
        """Test memory clearing failure."""
        mock_client = MagicMock()
        mock_client.delete_all = AsyncMock(side_effect=Exception("DB Error"))
        mock_get_client.return_value = mock_client

        result = await memory__clear_long_term_memory.ainvoke({"user_id": "test_user"})

        self.assertEqual(result, "Error clearing memory")


class TestHistoryTools(unittest.IsolatedAsyncioTestCase):
    async def test_deletion_tool_no_checkpointer(self):
        """Test deletion tool when checkpointer is missing."""
        tool = create_history_deletion_tool(None)
        result = await tool.ainvoke({"user_id": "u1"})
        self.assertIn("History is not being saved", result["content"])

    async def test_deletion_tool_success(self):
        """Test successful history deletion."""
        mock_cp = MagicMock()
        mock_cp.adelete_thread = AsyncMock()
        
        tool = create_history_deletion_tool(mock_cp)
        result = await tool.ainvoke({"user_id": "u1"})
        
        mock_cp.adelete_thread.assert_called_with("u1")
        self.assertIn("successfully deleted", result["content"])

    async def test_retrieval_tool_no_checkpointer(self):
        """Test retrieval tool when checkpointer is missing."""
        tool = create_history_retrieval_tool(None)
        result = await tool.ainvoke({"user_id": "u1"})
        self.assertIn("History is not being saved", result)

    async def test_retrieval_tool_no_history(self):
        """Test retrieval tool when no history exists."""
        mock_cp = MagicMock()
        # Mock alist to return empty async iterator
        async def empty_gen():
            if False: yield
        mock_cp.alist.return_value = empty_gen()
        
        tool = create_history_retrieval_tool(mock_cp)
        result = await tool.ainvoke({"user_id": "u1"})
        
        self.assertIn("No history found", result)

    async def test_retrieval_tool_success(self):
        """Test successful history retrieval with truncation."""
        mock_cp = MagicMock()
        
        # Create 25 messages to test truncation (limit is 20)
        messages = [HumanMessage(content=f"msg {i}") for i in range(25)]
        
        checkpoint = {"channel_values": {"messages": messages}}
        tuple_mock = MagicMock()
        tuple_mock.checkpoint = checkpoint
        
        async def gen():
            yield tuple_mock
            
        mock_cp.alist.return_value = gen()
        
        tool = create_history_retrieval_tool(mock_cp)
        result = await tool.ainvoke({"user_id": "u1"})
        
        self.assertIn("msg 24", result)
        self.assertNotIn("msg 4", result) # Should be truncated (0-4 are first 5, 5-24 are last 20)
        self.assertIn("Displaying the most recent part", result)