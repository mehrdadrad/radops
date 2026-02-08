"""
Knowledge Base Tools Module.

This module provides tools for retrieving documents from multiple vector
databases including Weaviate, Qdrant, Milvus, and Chroma. It supports:

- Dynamic filter creation based on metadata structure
- Multiple search types (similarity, MMR, threshold-based)
- Document reranking using FlashRank or Cohere
- Configurable retrieval parameters

Supported Vector Stores:
    - Weaviate: Enterprise vector database with filtering and hybrid search
    - Qdrant: High-performance vector search engine
    - Milvus: Open-source vector database
    - Chroma: Lightweight embedding database
"""
import logging
from functools import partial
from pathlib import Path
from typing import Any, Optional, Dict, List
from langchain_core.tools import StructuredTool, BaseTool
from langchain_core.documents import Document
from pydantic.v1 import BaseModel, Field, create_model
from config.config import settings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models
from storage.protocols import VectorStoreManager
from weaviate.classes.query import Filter
from langchain_milvus import Milvus

logger = logging.getLogger(__name__)

# Constants
DEFAULT_K = 3
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_ALPHA = 0.25
DEFAULT_FETCH_K = 20
DEFAULT_LAMBDA_MULT = 0.5
DEFAULT_RERANK_TOP_N = 3

SEARCH_TYPE_SIMILARITY = "similarity"
SEARCH_TYPE_MMR = "mmr"
SEARCH_TYPE_THRESHOLD = "similarity_score_threshold"

RERANK_PROVIDER_FLASHRANK = "flashrank"
RERANK_PROVIDER_COHERE = "cohere"

QDRANT_METADATA_PREFIX = "metadata."
MAX_QUERY_LENGTH = 1000
MAX_RESULTS = 100


class VectorStoreError(Exception):
    """Base exception for vector store operations."""


class InvalidQueryError(VectorStoreError):
    """Raised when query is invalid."""


class ConfigurationError(VectorStoreError):
    """Raised when configuration is invalid."""


def create_dynamic_input(dynamic_fields: Dict[str, str]) -> type[BaseModel]:
    """
    Dynamically creates a Pydantic input model for kb tools.
    
    Creates a BaseModel subclass with a 'query' field and additional
    optional fields based on the provided dynamic_fields dictionary.
    
    Args:
        dynamic_fields: Dictionary mapping field names to descriptions.
                       Each becomes an optional filter field.
    
    Returns:
        A Pydantic BaseModel subclass with the specified fields.
    
    Example:
        >>> fields = {"device": "Device name filter"}
        >>> InputModel = create_dynamic_input(fields)
        >>> instance = InputModel(query="search term", device="router1")
    """
    fields = {
        "query": (str, Field(
            description="The main search query"
        )),
    }
    for name, description in dynamic_fields.items():
        fields[name] = (
            Optional[str],
            Field(default=None, description=f"The {name} to filter by. {description}")
        )

    return create_model(
        'DynamicSearchInput',
        **fields,
        __base__=BaseModel
    )


def validate_retrieval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and set defaults for retrieval configuration.
    
    Args:
        config: Retrieval configuration dictionary
        
    Returns:
        Validated configuration with defaults applied
    """
    validated = config.copy()
    
    # Validate search_type
    valid_search_types = [SEARCH_TYPE_SIMILARITY, SEARCH_TYPE_MMR, SEARCH_TYPE_THRESHOLD]
    search_type = validated.get("search_type", SEARCH_TYPE_SIMILARITY)
    if search_type not in valid_search_types:
        logger.warning("Invalid search_type '%s', using '%s'", search_type, SEARCH_TYPE_SIMILARITY)
        validated["search_type"] = SEARCH_TYPE_SIMILARITY
    else:
        validated["search_type"] = search_type
    
    # Validate k
    num_results = validated.get("k", DEFAULT_K)
    if not isinstance(num_results, int) or num_results < 1:
        logger.warning("Invalid k value '%s', using %d", num_results, DEFAULT_K)
        validated["k"] = DEFAULT_K
    elif num_results > MAX_RESULTS:
        logger.warning("Requested k=%d exceeds maximum %d", num_results, MAX_RESULTS)
        validated["k"] = MAX_RESULTS
    else:
        validated["k"] = num_results
    
    # Validate score_threshold
    if "score_threshold" in validated:
        threshold = validated["score_threshold"]
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            logger.warning("Invalid score_threshold '%s', using %.2f", threshold, DEFAULT_SCORE_THRESHOLD)
            validated["score_threshold"] = DEFAULT_SCORE_THRESHOLD
    
    return validated


def validate_query(query: str) -> str:
    """
    Validate search query.
    
    Args:
        query: Search query string
        
    Returns:
        Validated query string
        
    Raises:
        InvalidQueryError: If query is invalid
    """
    if not query or not query.strip():
        raise InvalidQueryError("Query cannot be empty")
    
    if len(query) > MAX_QUERY_LENGTH:
        raise InvalidQueryError(
            f"Query too long ({len(query)} chars). Maximum is {MAX_QUERY_LENGTH} characters."
        )
    
    return query.strip()


def _validate_prompt_file_path(file_path: str, base_dir: Optional[str] = None) -> bool:
    """
    Validate prompt file path is safe and within allowed directory.
    
    Args:
        file_path: Path to prompt file
        base_dir: Base directory to restrict paths to (optional)
        
    Returns:
        True if path is valid and safe
    """
    try:
        resolved_path = Path(file_path).resolve()
        
        # Check if file exists and is a file
        if not resolved_path.exists():
            logger.error("Prompt file not found: %s", file_path)
            return False
        
        if not resolved_path.is_file():
            logger.error("Prompt path is not a file: %s", file_path)
            return False
        
        # If base_dir specified, ensure path is within it
        if base_dir:
            allowed_path = Path(base_dir).resolve()
            if not resolved_path.is_relative_to(allowed_path):
                logger.error("Prompt file outside allowed directory: %s", file_path)
                return False
        
        return True
    except (ValueError, OSError) as e:
        logger.error("Error validating prompt file path %s: %s", file_path, e)
        return False


def _read_prompt_file(file_path: str) -> Optional[str]:
    """
    Safely read prompt file content.
    
    Args:
        file_path: Path to prompt file
        
    Returns:
        File content or None if read fails
    """
    if not _validate_prompt_file_path(file_path):
        return None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except (IOError, OSError) as e:
        logger.error("Failed to read prompt file %s: %s", file_path, e)
        return None


def _sanitize_milvus_value(value: str) -> str:
    """
    Sanitize value for Milvus expression to prevent injection.
    
    Args:
        value: Value to sanitize
        
    Returns:
        Sanitized value
    """
    # Remove or escape quotes and special characters
    return value.replace('"', '\\"').replace("'", "\\'")


def _build_search_kwargs(
    retrieval_config: Dict[str, Any],
    search_type: str
) -> Dict[str, Any]:
    """
    Build search kwargs based on search type.
    
    Args:
        retrieval_config: Retrieval configuration
        search_type: Type of search to perform
        
    Returns:
        Search kwargs dictionary
    """
    search_kwargs = {
        key: value 
        for key, value in retrieval_config.items() 
        if key not in ["search_type", "rerank"]
    }
    
    search_kwargs.setdefault("k", DEFAULT_K)
    
    if search_type == SEARCH_TYPE_MMR:
        search_kwargs.setdefault("fetch_k", DEFAULT_FETCH_K)
        search_kwargs.setdefault("lambda_mult", DEFAULT_LAMBDA_MULT)
    elif search_type == SEARCH_TYPE_THRESHOLD:
        search_kwargs.setdefault("score_threshold", DEFAULT_SCORE_THRESHOLD)
    elif search_type == SEARCH_TYPE_SIMILARITY:
        search_kwargs.setdefault("alpha", DEFAULT_ALPHA)
    
    return search_kwargs


def _format_results(docs: List[Document]) -> str:
    """
    Format document results consistently.
    
    Args:
        docs: List of documents to format
        
    Returns:
        Formatted string with document contents
    """
    if not docs:
        return "No results found."
    
    return "\n---\n".join([
        f"Source: {doc.metadata.get('source', 'N/A')}\n{doc.page_content}"
        for doc in docs
    ])


def _generic_retriever(
    query: str,
    vector_store: Any,
    retrieval_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any
) -> str:
    """
    Generic retriever that routes to specific vector store implementations.
    
    Args:
        query: Search query
        vector_store: Vector store instance
        retrieval_config: Optional retrieval configuration
        **kwargs: Additional filter parameters
        
    Returns:
        Formatted search results
        
    Raises:
        InvalidQueryError: If query is invalid
        VectorStoreError: If vector store is not available
    """
    # Validate inputs
    try:
        query = validate_query(query)
    except InvalidQueryError as e:
        logger.warning("Invalid query: %s", e)
        return str(e)
    
    if vector_store is None:
        logger.error("Vector store is None")
        raise VectorStoreError("Vector store not available")
    
    if retrieval_config is None:
        retrieval_config = {}
    
    retrieval_config = validate_retrieval_config(retrieval_config)
    
    logger.info(
        "Searching for query: '%s' with filters: '%s' and config: '%s'",
        query, kwargs, retrieval_config
    )
    
    try:
        if isinstance(vector_store, WeaviateVectorStore):
            return retriever_weaviate(vector_store, query, retrieval_config, **kwargs)
        elif isinstance(vector_store, QdrantVectorStore):
            return retriever_qdrant(vector_store, query, retrieval_config, **kwargs)
        elif isinstance(vector_store, Milvus):
            return retriever_milvus(vector_store, query, retrieval_config, **kwargs)
        else:
            return retriever_chroma(vector_store, query, retrieval_config, **kwargs)
    except (ValueError, KeyError, ConnectionError) as e:
        logger.error(
            "Knowledge base retrieval error: %s",
            e,
            exc_info=True
        )
        return (
            "An error occurred while searching. "
            f"Error: {type(e).__name__}. Please try again."
        )
    except Exception as e:
        logger.exception("Unexpected error in knowledge base retrieval")
        raise


def create_kb_tools(vector_store_managers: List[VectorStoreManager]) -> List[BaseTool]:
    """
    Creates a list of KB tools based on vector store profiles.
    
    Args:
        vector_store_managers: List of vector store manager instances
        
    Returns:
        List of created tools
    """
    tools = []
    
    for vector_store_profile in settings.vector_store.profiles:
        matching_vector_store_manager = None
        
        # Find matching vector store manager
        for vector_store_manager in vector_store_managers:
            if vector_store_manager.name() == vector_store_profile.name:
                matching_vector_store_manager = vector_store_manager
                break

        if not matching_vector_store_manager:
            logger.error(
                "Vector store manager for profile '%s' not found. "
                "Skipping tool creation for this profile.",
                vector_store_profile.name
            )
            continue

        for sync_location in vector_store_profile.sync_locations:
            collection_name = sync_location.collection
            sync_name = sync_location.name
            tool_name = f"kb_{sync_name.replace(' ', '_')}"

            dynamic_properties = sync_location.metadata.structure
            dynamic_fields = {
                prop.name: prop.description for prop in dynamic_properties
            }

            vector_store = matching_vector_store_manager.get_vectorstore(collection_name)
            retrieval_config = getattr(sync_location, "retrieval_config", {})

            retriever_func = partial(
                _generic_retriever,
                vector_store=vector_store,
                retrieval_config=retrieval_config
            )

            # Read docstring from file or use inline prompt
            docstring = tool_name  # Default fallback
            
            if sync_location.prompt_file:
                prompt_content = _read_prompt_file(sync_location.prompt_file)
                if prompt_content:
                    docstring = prompt_content
                else:
                    logger.warning(
                        "Failed to read prompt file for %s, using default", 
                        tool_name
                    )
            elif sync_location.prompt:
                docstring = sync_location.prompt

            dyn_tool = StructuredTool(
                name=tool_name,
                func=retriever_func,
                args_schema=create_dynamic_input(dynamic_fields),
                description=docstring,
            )
            
            logger.info(
                "Created tool: %s, collection: %s, sync_name: %s",
                tool_name, collection_name, sync_name
            )
            tools.append(dyn_tool)

    return tools


def _rerank_documents(
    docs: List[Document],
    query: str,
    retrieval_config: Dict[str, Any]
) -> List[Document]:
    """
    Rerank documents using specified provider.
    
    Args:
        docs: Documents to rerank
        query: Original search query
        retrieval_config: Configuration including rerank settings
        
    Returns:
        Reranked documents or original if reranking disabled/unavailable
    """
    rerank_config = retrieval_config.get("rerank")
    if not docs or not rerank_config or not rerank_config.get("enabled", False):
        return docs

    provider = rerank_config.get("provider", RERANK_PROVIDER_FLASHRANK)
    logger.info("Reranking documents by %s ...", provider)

    top_n = rerank_config.get("top_n", retrieval_config.get("k", DEFAULT_RERANK_TOP_N))

    if provider == RERANK_PROVIDER_COHERE:
        try:
            from langchain_cohere import CohereRerank
        except ImportError as e:
            logger.error(
                "CohereRerank not available: %s. "
                "Install with: pip install langchain-cohere",
                e
            )
            logger.info("Continuing without reranking")
            return docs

        model = rerank_config.get("model", "rerank-english-v3.0")
        compressor = CohereRerank(model=model, top_n=top_n)
    else:
        try:
            from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
        except ImportError as e:
            logger.error(
                "FlashrankRerank not available: %s. "
                "Install with: pip install flashrank",
                e
            )
            logger.info("Continuing without reranking")
            return docs
        
        FlashrankRerank.model_rebuild()
        kwargs = {"top_n": top_n}
        if "model" in rerank_config:
            kwargs["model"] = rerank_config["model"]
        try:
            compressor = FlashrankRerank(**kwargs)
        except Exception as e:
            logger.error("Failed to initialize Flashrank reranker: %s", e)
            return docs

    try:
        return compressor.compress_documents(docs, query)
    except Exception as e:
        logger.error("Error during reranking: %s", e, exc_info=True)
        logger.info("Returning original documents")
        return docs


def _build_chroma_filter(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build Chroma filter from kwargs.
    
    Args:
        kwargs: Filter parameters
        
    Returns:
        Chroma filter dict or None
    """
    filter_conditions = [
        {key: value}
        for key, value in kwargs.items()
        if value is not None
    ]
    
    if not filter_conditions:
        return None
    
    if len(filter_conditions) == 1:
        return filter_conditions[0]
    
    return {'$and': filter_conditions}


def retriever_chroma(
    vector_store: Any,
    query: str,
    retrieval_config: Dict[str, Any],
    **kwargs: Any
) -> str:
    """
    Retrieves documents from Chroma vector store.
    
    Args:
        vector_store: Chroma vector store instance
        query: Search query
        retrieval_config: Retrieval configuration
        **kwargs: Filter parameters
        
    Returns:
        Formatted search results
    """
    chroma_filter = _build_chroma_filter(kwargs)
    search_type = retrieval_config.get("search_type", SEARCH_TYPE_SIMILARITY)
    search_kwargs = _build_search_kwargs(retrieval_config, search_type)
    search_kwargs["filter"] = chroma_filter

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    
    docs = retriever.invoke(query)
    docs = _rerank_documents(docs, query, retrieval_config)
    return _format_results(docs)


def _build_weaviate_filters(kwargs: Dict[str, Any]) -> Optional[Filter]:
    """
    Build Weaviate filters from kwargs.
    
    Args:
        kwargs: Filter parameters
        
    Returns:
        Weaviate Filter object or None
    """
    all_filters = [
        Filter.by_property(key).equal(value)
        for key, value in kwargs.items()
        if value
    ]
    
    if not all_filters:
        return None
    
    if len(all_filters) == 1:
        return all_filters[0]
    
    # Combine multiple filters with AND
    combined_filter = all_filters[0]
    for single_filter in all_filters[1:]:
        combined_filter = combined_filter & single_filter
    
    return combined_filter


def retriever_weaviate(
    vector_store: Any,
    query: str,
    retrieval_config: Dict[str, Any],
    **kwargs: Any
) -> str:
    """
    Retrieves documents from Weaviate vector store.
    
    Supports multiple search types:
    - Similarity: Vector similarity search
    - MMR: Max Marginal Relevance for diversity
    - Similarity Score Threshold: Filtered by relevance score
    
    Args:
        vector_store: Weaviate vector store instance
        query: Search query
        retrieval_config: Retrieval configuration
        **kwargs: Filter parameters
        
    Returns:
        Formatted search results
    """
    filters = _build_weaviate_filters(kwargs)
    search_type = retrieval_config.get("search_type", SEARCH_TYPE_SIMILARITY)
    search_kwargs = _build_search_kwargs(retrieval_config, search_type)
    search_kwargs["filters"] = filters

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    docs = retriever.invoke(query)
    docs = _rerank_documents(docs, query, retrieval_config)
    return _format_results(docs)


def retriever_qdrant(
    vector_store: Any,
    query: str,
    retrieval_config: Dict[str, Any],
    **kwargs: Any
) -> str:
    """
    Retrieves documents from Qdrant vector store.
    
    Args:
        vector_store: Qdrant vector store instance
        query: Search query
        retrieval_config: Retrieval configuration
        **kwargs: Filter parameters
        
    Returns:
        Formatted search results
    """
    must_conditions = []
    for key, value in kwargs.items():
        if value:
            must_conditions.append(
                models.FieldCondition(
                    key=f"{QDRANT_METADATA_PREFIX}{key}",
                    match=models.MatchValue(value=value)
                )
            )

    qdrant_filter = None
    if must_conditions:
        qdrant_filter = models.Filter(must=must_conditions)

    num_results = retrieval_config.get("k", DEFAULT_K)
    search_params = {
        key: value 
        for key, value in retrieval_config.items() 
        if key not in ["k", "rerank"]
    }

    docs = vector_store.similarity_search(
        query=query,
        k=num_results,
        filter=qdrant_filter,
        **search_params
    )
    
    docs = _rerank_documents(docs, query, retrieval_config)
    return _format_results(docs)


def retriever_milvus(
    vector_store: Any,
    query: str,
    retrieval_config: Dict[str, Any],
    **kwargs: Any
) -> str:
    """
    Retrieves documents from Milvus vector store.
    
    Args:
        vector_store: Milvus vector store instance
        query: Search query
        retrieval_config: Retrieval configuration
        **kwargs: Filter parameters
        
    Returns:
        Formatted search results
    """
    expr_list = []
    for key, value in kwargs.items():
        if value:
            sanitized_value = _sanitize_milvus_value(str(value))
            expr_list.append(f'{key} == "{sanitized_value}"')

    expr = None
    if expr_list:
        expr = " and ".join(expr_list)

    num_results = retrieval_config.get("k", DEFAULT_K)
    search_params = {
        key: value 
        for key, value in retrieval_config.items() 
        if key not in ["k", "rerank"]
    }

    docs = vector_store.similarity_search(
        query=query,
        k=num_results,
        expr=expr,
        **search_params
    )
    
    docs = _rerank_documents(docs, query, retrieval_config)
    return _format_results(docs)