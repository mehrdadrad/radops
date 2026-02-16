import unittest
import asyncio
import sys
from unittest.mock import MagicMock, patch, AsyncMock

from pydantic import ValidationError
from core.auth import get_user, is_tool_authorized, UserSettings
from core.checkpoint import get_checkpointer
from core.llm import llm_factory, embedding_factory, LLMCallbackHandler, close_shared_client
from core.memory import Mem0Manager
from core.state import SupervisorAgentOutputBase
from core.vector_store import vector_store_factory
from core.mcp_client import MCPClient, sanitize_kwargs

class TestAuth(unittest.IsolatedAsyncioTestCase):
    @patch("core.auth.settings")
    async def test_get_user_local(self, mock_settings):
        mock_settings.get_user.return_value = UserSettings(role="admin")
        user = await get_user("test@example.com")
        self.assertEqual(user.role, "admin")

    @patch("core.auth.settings")
    async def test_is_tool_authorized(self, mock_settings):
        # Mock get_user_role via settings.get_user
        mock_settings.get_user.return_value = UserSettings(role="developer")
        mock_settings.role_permissions = {
            "developer": ["git.*", "kubectl"]
        }
        
        # Exact match
        self.assertTrue(await is_tool_authorized("kubectl", "user"))
        # Regex match
        self.assertTrue(await is_tool_authorized("git_commit", "user"))
        # No match
        self.assertFalse(await is_tool_authorized("delete_db", "user"))

    @patch("core.auth.httpx.AsyncClient")
    @patch("core.auth.settings")
    async def test_get_oidc_user(self, mock_settings, mock_client_cls):
        mock_settings.get_user.return_value = None
        mock_settings.oidc.enabled = True
        mock_settings.oidc.userinfo_endpoint = "http://oidc/userinfo"
        mock_settings.oidc.role_attribute = "role"
        
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client_cls.return_value.__aexit__.return_value = None
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "sub": "user1",
            "role": "admin",
            "given_name": "John",
            "family_name": "Doe"
        }
        mock_client.get.return_value = mock_response
        
        user = await get_user("user1")
        self.assertIsNotNone(user)
        self.assertEqual(user.role, "admin")

    @patch("core.auth.httpx.AsyncClient")
    @patch("core.auth.settings")
    async def test_get_auth0_user(self, mock_settings, mock_client_cls):
        mock_settings.get_user.return_value = None
        mock_settings.oidc.enabled = False
        mock_settings.auth0.enabled = True
        mock_settings.auth0.domain = "test.auth0.com"
        mock_settings.auth0.client_id = "cid"
        mock_settings.auth0.client_secret = "csec"
        mock_settings.auth0.role_source = "metadata"
        mock_settings.auth0.role_attribute = "app_metadata.role"
        
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client_cls.return_value.__aexit__.return_value = None
        
        # Mock token response
        mock_token_resp = MagicMock()
        mock_token_resp.status_code = 200
        mock_token_resp.json.return_value = {"access_token": "tok", "expires_in": 3600}
        
        # Mock user response
        mock_user_resp = MagicMock()
        mock_user_resp.status_code = 200
        mock_user_resp.json.return_value = [{
            "user_id": "auth0|123",
            "app_metadata": {"role": "editor"},
            "given_name": "Jane",
            "family_name": "Doe"
        }]
        
        mock_client.post.return_value = mock_token_resp
        mock_client.get.return_value = mock_user_resp
        
        user = await get_user("jane@example.com")
        self.assertIsNotNone(user)
        self.assertEqual(user.role, "editor")

    @patch("core.auth.httpx.AsyncClient")
    @patch("core.auth.settings")
    async def test_get_oidc_user_failure(self, mock_settings, mock_client_cls):
        mock_settings.get_user.return_value = None
        mock_settings.oidc.enabled = True
        mock_settings.oidc.userinfo_endpoint = "http://oidc/userinfo"
        mock_settings.oidc.cache_ttl_seconds = 300
        
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        mock_client_cls.return_value.__aexit__.return_value = None
        
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_client.get.return_value = mock_response
        
        user = await get_user("user_fail")
        self.assertIsNone(user)

    @patch("core.auth.settings")
    async def test_is_tool_authorized_wildcard(self, mock_settings):
        # Mock get_user_role via settings.get_user
        mock_settings.get_user.return_value = UserSettings(role="admin")
        mock_settings.role_permissions = {
            "admin": [".*"]
        }
        
        self.assertTrue(await is_tool_authorized("any_tool", "user"))
        self.assertTrue(await is_tool_authorized("another_tool", "user"))

    @patch("core.auth.settings")
    async def test_is_tool_authorized_no_permissions_for_role(self, mock_settings):
        mock_settings.get_user.return_value = UserSettings(role="guest")
        mock_settings.role_permissions = {
            "admin": [".*"]
        }
        self.assertFalse(await is_tool_authorized("any_tool", "user"))

    @patch("core.auth.settings")
    async def test_get_user_none(self, mock_settings):
        """Test get_user when no provider is enabled/found."""
        mock_settings.get_user.return_value = None
        mock_settings.oidc.enabled = False
        mock_settings.auth0.enabled = False
        
        user = await get_user("unknown@example.com")
        self.assertIsNone(user)

class TestCheckpoint(unittest.IsolatedAsyncioTestCase):
    @patch("core.checkpoint.settings")
    async def test_get_checkpointer_memory(self, mock_settings):
        mock_settings.memory.short_term = None
        async with get_checkpointer() as (cp, client):
            self.assertIsNotNone(cp)
            self.assertIsNone(client)

    @patch("core.checkpoint.settings")
    @patch("core.checkpoint.AsyncRedis")
    @patch("core.checkpoint.AsyncRedisSaver")
    async def test_get_checkpointer_redis(self, mock_saver, mock_redis, mock_settings):
        mock_settings.memory.short_term.provider = "redis"
        mock_settings.memory.short_term.config.url = "redis://localhost"
        
        mock_saver_instance = AsyncMock()
        mock_saver.return_value.__aenter__.return_value = mock_saver_instance
        
        async with get_checkpointer() as (cp, client):
            self.assertEqual(cp, mock_saver_instance)

class TestLLM(unittest.IsolatedAsyncioTestCase):
    @patch("core.llm.ChatOpenAI")
    @patch("core.llm.settings")
    def test_llm_factory_openai(self, mock_settings, mock_chat_openai):
        mock_settings.llm.profiles = {
            "default": MagicMock(provider="openai", model="gpt-4", api_key="key")
        }
        llm_factory("default")
        mock_chat_openai.assert_called_once()

    @patch("core.llm.ChatOpenAI")
    @patch("core.llm.settings")
    def test_llm_factory_deepseek(self, mock_settings, mock_chat_openai):
        mock_settings.llm.profiles = {
            "deepseek": MagicMock(provider="deepseek", model="deepseek-chat", api_key="key")
        }
        llm_factory("deepseek")
        mock_chat_openai.assert_called_once()

    @patch("core.llm.AzureChatOpenAI")
    @patch("core.llm.settings")
    def test_llm_factory_azure(self, mock_settings, mock_azure):
        mock_settings.llm.profiles = {
            "azure": MagicMock(
                provider="azure",
                model="gpt-4",
                api_key="key",
                api_base="endpoint",
                api_version="2023-05-15"
            )
        }
        llm_factory("azure")
        mock_azure.assert_called_once()

    @patch("core.llm.ChatAnthropic")
    @patch("core.llm.settings")
    def test_llm_factory_anthropic(self, mock_settings, mock_anthropic):
        mock_settings.llm.profiles = {
            "claude": MagicMock(provider="anthropic", model="claude-3", api_key="key")
        }
        llm_factory("claude")
        mock_anthropic.assert_called_once()

    @patch("core.llm.ChatGoogleGenerativeAI")
    @patch("core.llm.settings")
    def test_llm_factory_google(self, mock_settings, mock_google):
        mock_settings.llm.profiles = {
            "gemini": MagicMock(provider="gemini", model="gemini-pro", credentials="creds")
        }
        llm_factory("gemini")
        mock_google.assert_called_once()

    @patch("core.llm.ChatGroq")
    @patch("core.llm.settings")
    def test_llm_factory_groq(self, mock_settings, mock_groq):
        mock_settings.llm.profiles = {
            "groq": MagicMock(provider="groq", model="llama3", api_key="key")
        }
        llm_factory("groq")
        mock_groq.assert_called_once()

    @patch("core.llm.ChatMistralAI")
    @patch("core.llm.settings")
    def test_llm_factory_mistral(self, mock_settings, mock_mistral):
        mock_settings.llm.profiles = {
            "mistral": MagicMock(provider="mistral", model="mistral-large", api_key="key", base_url="url")
        }
        llm_factory("mistral")
        mock_mistral.assert_called_once()

    @patch("core.llm.ChatBedrockConverse")
    @patch("core.llm.settings")
    def test_llm_factory_bedrock(self, mock_settings, mock_bedrock):
        mock_settings.llm.profiles = {
            "bedrock": MagicMock(
                provider="bedrock", 
                model="anthropic.claude-3", 
                aws_region="us-east-1",
                aws_access_key_id="key",
                aws_secret_access_key="secret",
                aws_session_token="token"
            )
        }
        llm_factory("bedrock")
        mock_bedrock.assert_called_once()

    @patch("core.llm.ChatOllama")
    @patch("core.llm.settings")
    def test_llm_factory_ollama(self, mock_settings, mock_ollama):
        mock_settings.llm.profiles = {
            "ollama": MagicMock(provider="ollama", model="llama2", base_url="http://localhost:11434")
        }
        llm_factory("ollama")
        mock_ollama.assert_called_once()

    @patch("core.llm.settings")
    def test_llm_factory_unknown_provider(self, mock_settings):
        mock_settings.llm.profiles = {
            "unknown": MagicMock(provider="unknown_provider")
        }
        with self.assertRaises(ValueError):
            llm_factory("unknown")

    @patch("core.llm.settings")
    def test_llm_factory_missing_profile(self, mock_settings):
        mock_settings.llm.profiles = {}
        with self.assertRaises(Exception):
            llm_factory("missing")

    @patch("core.llm.OpenAIEmbeddings")
    @patch("core.llm.settings")
    def test_embedding_factory_openai(self, mock_settings, mock_openai_emb):
        mock_settings.llm.profiles = {
            "default": MagicMock(provider="openai", model="text-embedding-3", api_key="key")
        }
        embedding_factory("default")
        mock_openai_emb.assert_called_once()

    @patch("core.llm.OpenAIEmbeddings")
    @patch("core.llm.settings")
    def test_embedding_factory_deepseek(self, mock_settings, mock_openai_emb):
        mock_settings.llm.profiles = {
            "deepseek": MagicMock(provider="deepseek", model="deepseek-emb", api_key="key")
        }
        embedding_factory("deepseek")
        mock_openai_emb.assert_called_once()

    @patch("core.llm.AzureOpenAIEmbeddings")
    @patch("core.llm.settings")
    def test_embedding_factory_azure(self, mock_settings, mock_azure):
        mock_settings.llm.profiles = {
            "azure": MagicMock(
                provider="azure", model="ada", api_key="key", base_url="url", api_version="v1"
            )
        }
        embedding_factory("azure")
        mock_azure.assert_called_once()

    @patch("core.llm.GoogleGenerativeAIEmbeddings")
    @patch("core.llm.settings")
    def test_embedding_factory_gemini(self, mock_settings, mock_gemini):
        mock_settings.llm.profiles = {
            "gemini": MagicMock(provider="gemini", model="embedding-001", api_key="key")
        }
        embedding_factory("gemini")
        mock_gemini.assert_called_once()

    @patch("core.llm.BedrockEmbeddings")
    @patch("core.llm.settings")
    def test_embedding_factory_bedrock(self, mock_settings, mock_bedrock):
        mock_settings.llm.profiles = {
            "bedrock": MagicMock(
                provider="bedrock", model="titan", aws_region="us-east-1"
            )
        }
        embedding_factory("bedrock")
        mock_bedrock.assert_called_once()

    @patch("core.llm.OllamaEmbeddings")
    @patch("core.llm.settings")
    def test_embedding_factory_ollama(self, mock_settings, mock_ollama):
        mock_settings.llm.profiles = {
            "ollama": MagicMock(provider="ollama", model="nomic", base_url="url")
        }
        embedding_factory("ollama")
        mock_ollama.assert_called_once()

    @patch("core.llm.settings")
    def test_embedding_factory_unknown(self, mock_settings):
        mock_settings.llm.profiles = {
            "unknown": MagicMock(provider="unknown")
        }
        with self.assertRaises(ValueError):
            embedding_factory("unknown")

    @patch("core.llm.telemetry")
    def test_callback_handler(self, mock_telemetry):
        handler = LLMCallbackHandler(agent_name="test_agent")
        handler.on_llm_error(ValueError("oops"))
        mock_telemetry.update_counter.assert_called_with(
            "agent.llm.errors", attributes={"agent": "test_agent"}
        )
        
        output = MagicMock()
        output.llm_output = {"token_usage": {"total_tokens": 10}}
        handler.on_llm_end(output)
        mock_telemetry.update_counter.assert_called()

    @patch("core.llm.telemetry")
    def test_callback_handler_no_usage(self, mock_telemetry):
        handler = LLMCallbackHandler(agent_name="test_agent")
        output = MagicMock()
        output.llm_output = {}
        handler.on_llm_end(output)
        mock_telemetry.update_counter.assert_not_called()

    @patch("core.llm.telemetry")
    def test_callback_handler_errors(self, mock_telemetry):
        handler = LLMCallbackHandler(agent_name="test_agent")
        
        # Tool error
        handler.on_tool_error(ValueError("tool fail"))
        mock_telemetry.update_counter.assert_called_with(
            "agent.tool.errors", attributes={"agent": "test_agent"}
        )
        
        # Chain error
        handler.on_chain_error(ValueError("chain fail"))

    @patch("core.llm.telemetry")
    def test_callback_handler_cache_metrics(self, mock_telemetry):
        handler = LLMCallbackHandler(agent_name="test_agent")
        
        # Anthropic style
        output = MagicMock()
        output.llm_output = {
            "token_usage": {
                "total_tokens": 100,
                "cache_read_input_tokens": 50,
                "cache_creation_input_tokens": 20
            }
        }
        handler.on_llm_end(output)
        
        mock_telemetry.update_counter.assert_any_call(
            "agent.llm.tokens.cache_read", 50, attributes={"agent": "test_agent"}
        )
        mock_telemetry.update_counter.assert_any_call(
            "agent.llm.tokens.cache_creation", 20, attributes={"agent": "test_agent"}
        )

        # OpenAI style
        output_openai = MagicMock()
        output_openai.llm_output = {
            "token_usage": {
                "total_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 30}
            }
        }
        handler.on_llm_end(output_openai)
        mock_telemetry.update_counter.assert_any_call(
            "agent.llm.tokens.cache_read", 30, attributes={"agent": "test_agent"}
        )

    @patch("core.llm._shared_http_aclient")
    async def test_close_shared_client(self, mock_client):
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        
        await close_shared_client()
        mock_client.aclose.assert_called_once()

    @patch("core.llm._shared_http_aclient")
    @patch("core.llm.logger")
    async def test_close_shared_client_error(self, mock_logger, mock_client):
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock(side_effect=Exception("Close error"))
        
        await close_shared_client()
        mock_logger.error.assert_called()

class TestMemory(unittest.IsolatedAsyncioTestCase):
    @patch("core.memory.AsyncMemory")
    @patch("core.memory.settings")
    async def test_mem0_manager(self, mock_settings, mock_async_memory):
        # Reset singleton
        Mem0Manager._instance = None
        Mem0Manager._mem0_client = None
        
        mock_async_memory.from_config = AsyncMock()

        mock_settings.memory.long_term.config.llm_profile = "gpt4"
        mock_settings.memory.long_term.config.embedding_profile = "openai_emb"
        mock_settings.llm.profiles = {
            "gpt4": MagicMock(provider="openai"),
            "openai_emb": MagicMock(provider="openai")
        }
        
        manager = Mem0Manager()
        await manager.get_client()
        mock_async_memory.from_config.assert_called_once()

    async def test_mem0_manager_close(self):
        manager = Mem0Manager()
        # Mock the internal client
        mock_client = MagicMock()
        mock_client.vector_store.client.close = AsyncMock()
        manager._mem0_client = mock_client
        
        await manager.close()
        mock_client.vector_store.client.close.assert_called_once()

    async def test_mem0_manager_singleton_behavior(self):
        Mem0Manager._instance = None
        m1 = Mem0Manager()
        m2 = Mem0Manager()
        self.assertIs(m1, m2)

class TestVectorStore(unittest.TestCase):
    @patch("core.vector_store.settings")
    @patch("core.vector_store.embedding_factory")
    @patch("core.vector_store.ChromaVectorStoreManager")
    def test_vector_store_factory(self, mock_chroma, mock_emb_factory, mock_settings):
        profile = MagicMock()
        profile.provider = "chroma"
        profile.name = "test_chroma"
        mock_settings.vector_store.profiles = [profile]
        
        managers = vector_store_factory()
        self.assertEqual(len(managers), 1)
        mock_chroma.assert_called_once()

    @patch("core.vector_store.settings")
    @patch("core.vector_store.embedding_factory")
    def test_vector_store_factory_unknown_provider(self, mock_emb_factory, mock_settings):
        profile = MagicMock()
        profile.provider = "unknown"
        profile.name = "test_unknown"
        mock_settings.vector_store.profiles = [profile]
        
        with self.assertRaises(ValueError):
            vector_store_factory()

    @patch("core.vector_store.settings")
    @patch("core.vector_store.embedding_factory")
    @patch("core.vector_store.ChromaVectorStoreManager")
    def test_vector_store_factory_skip_sync(self, mock_chroma, mock_emb_factory, mock_settings):
        """Test vector_store_factory with skip_initial_sync."""
        profile = MagicMock()
        profile.provider = "chroma"
        profile.name = "test_chroma"
        mock_settings.vector_store.profiles = [profile]
        
        vector_store_factory(skip_initial_sync=True)
        
        mock_chroma.assert_called_with(
            "test_chroma", 
            profile.sync_locations, 
            mock_emb_factory.return_value, 
            skip_initial_sync=True
        )

class TestMCPClient(unittest.IsolatedAsyncioTestCase):
    @patch("core.mcp_client.ClientSession")
    @patch("core.mcp_client.stdio_client")
    async def test_mcp_client_connect(self, mock_stdio, mock_session_cls):
        config = {"transport": "stdio", "command": "cmd"}
        client = MCPClient("test", config)
        
        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session
        mock_session_cls.return_value.__aexit__.return_value = None
        
        # Mock tools
        mock_tool = MagicMock()
        mock_tool.name = "tool1"
        mock_tool.inputSchema = {"type": "object"}
        mock_session.list_tools.return_value.tools = [mock_tool]
        
        # Start client in background
        await client.start()
        
        # Wait a bit for connection
        await asyncio.sleep(0.1)
        
        tools = await client.get_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "test__tool1")
        
        await client.stop()

    @patch("core.mcp_client.ClientSession")
    @patch("core.mcp_client.stdio_client")
    async def test_mcp_client_get_tools_not_started(self, mock_stdio, mock_session_cls):
        config = {"transport": "stdio", "command": "cmd"}
        client = MCPClient("test", config)
        tools = await client.get_tools()
        self.assertEqual(tools, [])

    @patch("core.mcp_client.ClientSession")
    @patch("core.mcp_client.stdio_client")
    async def test_mcp_client_connection_error(self, mock_stdio, mock_session_cls):
        config = {"transport": "stdio", "command": sys.executable}
        client = MCPClient("test", config)
        
        mock_stdio.side_effect = Exception("Connection failed")
        
        # start() swallows connection errors and logs them
        await client.start()
        await asyncio.sleep(0.1)
        self.assertTrue(mock_stdio.called)

    @patch("core.mcp_client.ClientSession")
    @patch("core.mcp_client.stdio_client")
    async def test_mcp_client_call_tool(self, mock_stdio, mock_session_cls):
        """Test calling a tool via MCPClient."""
        config = {"transport": "stdio", "command": "cmd"}
        client = MCPClient("test", config)
        
        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session
        mock_session_cls.return_value.__aexit__.return_value = None
        
        # Mock list_tools to return a tool definition
        mock_tool_def = MagicMock()
        mock_tool_def.name = "tool1"
        mock_tool_def.inputSchema = {
            "type": "object",
            "properties": {
                "arg": {"type": "string"}
            }
        }
        mock_session.list_tools.return_value.tools = [mock_tool_def]

        mock_result = MagicMock()
        mock_result.isError = False
        content_item = MagicMock()
        content_item.type = "text"
        content_item.text = "result"
        mock_result.content = [content_item]
        mock_session.call_tool.return_value = mock_result
        
        await client.start()
        tools = await client.get_tools()
        result = await tools[0].ainvoke({"arg": "val"})
        
        mock_session.call_tool.assert_called_with("tool1", arguments={"arg": "val"})
        self.assertEqual(result, "result")
        
        await client.stop()

    @patch("core.mcp_client.ClientSession")
    @patch("core.mcp_client.stdio_client")
    async def test_mcp_client_tool_timeout(self, mock_stdio, mock_session_cls):
        """Test tool execution timeout."""
        config = {"transport": "stdio", "command": "cmd", "execution_timeout": 0.01}
        client = MCPClient("test", config)
        
        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session
        mock_session_cls.return_value.__aexit__.return_value = None
        
        # Mock tool def
        mock_tool_def = MagicMock()
        mock_tool_def.name = "tool1"
        mock_tool_def.inputSchema = {"type": "object"}
        mock_session.list_tools.return_value.tools = [mock_tool_def]
        
        # Mock call_tool to sleep longer than timeout
        async def slow_tool(*args, **kwargs):
            await asyncio.sleep(0.1)
            return MagicMock()
        
        mock_session.call_tool.side_effect = slow_tool
        
        await client.start()
        tools = await client.get_tools()
        
        with self.assertRaises(TimeoutError):
            await tools[0].ainvoke({})
            
        await client.stop()

    @patch("core.mcp_client.sse_client")
    @patch("core.mcp_client.ClientSession")
    async def test_mcp_client_sse_transport(self, mock_session_cls, mock_sse_client):
        config = {"transport": "sse", "url": "http://localhost:8000/sse"}
        client = MCPClient("test_sse", config)
        
        mock_session = AsyncMock()
        mock_session_cls.return_value.__aenter__.return_value = mock_session
        mock_session_cls.return_value.__aexit__.return_value = None
        mock_session.list_tools.return_value.tools = []

        await client.start()
        await asyncio.sleep(0.1)
        
        mock_sse_client.assert_called_with(
            url="http://localhost:8000/sse",
            headers={},
            timeout=10.0
        )
        await client.stop()

    async def test_mcp_client_invalid_transport(self):
        config = {"transport": "ftp"}
        client = MCPClient("test", config)
        with self.assertRaises(ValueError):
            await client._connect_session()

    def test_mcp_client_schema_generation(self):
        client = MCPClient("test", {})
        schema = {
            "properties": {
                "req": {"type": "string"},
                "opt": {"type": "integer", "description": "desc"}
            },
            "required": ["req"]
        }
        model = client._create_args_schema("tool_test", schema)
        self.assertTrue(model.model_fields["req"].is_required())
        self.assertFalse(model.model_fields["opt"].is_required())

    def test_sanitize_kwargs(self):
        # Test JSON parsing
        input_kwargs = {
            "str_arg": "value",
            "json_obj": '{"key": "value"}',
            "json_arr": '[1, 2, 3]',
            "invalid_json": '{key: value}'
        }
        sanitized = sanitize_kwargs(input_kwargs)
        self.assertEqual(sanitized["str_arg"], "value")
        self.assertEqual(sanitized["json_obj"], {"key": "value"})
        self.assertEqual(sanitized["json_arr"], [1, 2, 3])
        self.assertEqual(sanitized["invalid_json"], '{key: value}')

class TestState(unittest.TestCase):
    def test_validate_worker_instructions(self):
        # Valid
        output = SupervisorAgentOutputBase(
            current_step_id=1,
            current_step_status="pending",
            next_worker="system",
            response_to_user="msg",
            instructions_for_worker="Do something useful."
        )
        self.assertEqual(output.instructions_for_worker, "Do something useful.")
        
        # Invalid (too short)
        with self.assertRaises(ValidationError):
            SupervisorAgentOutputBase(
                current_step_id=1,
                current_step_status="pending",
                next_worker="system",
                response_to_user="msg",
                instructions_for_worker="Do"
            )
            
        # End worker doesn't require instructions
        output_end = SupervisorAgentOutputBase(
            current_step_id=1,
            current_step_status="completed",
            next_worker="end",
            response_to_user="msg",
            instructions_for_worker=""
        )
        self.assertEqual(output_end.next_worker, "end")
