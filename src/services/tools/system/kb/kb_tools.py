import logging
from functools import partial
from langchain_core.tools import StructuredTool, BaseTool
from typing import Any, Optional
from pydantic.v1 import BaseModel, Field, create_model
from config.config import settings
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client.http import models
from storage.protocols import VectorStoreManager
from weaviate.classes.query import Filter
from langchain_milvus import Milvus

logger = logging.getLogger(__name__)

def create_dynamic_input(dynamic_fields: dict[str, str]) -> type[BaseModel]:
    """Dynamically creates the Pydantic input model for the router tool."""
    fields = {
        "query": (str, Field(
            description="The main search query for the router configuration, "
                        "excluding device and location information."
        )),
    }
    for name, description in dynamic_fields.items():
        fields[name] = (
            Optional[str],
            Field(default=None, description=f"The {name} to filter by. {description}")
        )

    return create_model(
        'DynamicRouterSearchInput',
        **fields,
        __base__=BaseModel
    )

# This is a placeholder for your actual knowledge base retrieval logic.
# You would replace this with your function that queries your KB.
def _generic_retriever(
    query: str,
    vector_store: Any,
    retrieval_config: Optional[dict] = None,
    **kwargs: Any
) -> str:
    if retrieval_config is None:
        retrieval_config = {}
    logging.info(
        f"Searching for query: '{query}' with filters: '{kwargs}' "
        f"and config: '{retrieval_config}'"
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
    except Exception as e:
        logging.error(f"An error occurred during knowledge base retrieval: {e}")
        return "Sorry, an error occurred while searching for the requested information. Please try again later."

def create_kb_tools(vector_store_managers: list[VectorStoreManager]) -> list[BaseTool]:
    tools = []
    for vector_store_profile in settings.vector_store.profiles:
        matching_vector_store_manager = None
        for vector_store_manager in vector_store_managers:
            if vector_store_manager.name() == vector_store_profile.name:
                matching_vector_store_manager = vector_store_manager
                break

        if not matching_vector_store_manager:
            logger.error(
                f"Vector store manager for profile '{vector_store_profile.name}' "
                "not found. Skipping tool creation for this profile."
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

            if sync_location.prompt_file:
                with open(sync_location.prompt_file, "r", encoding="utf-8") as f:
                    docstring = f.read()
                    f.close()
            elif sync_location.prompt:
                docstring = sync_location.prompt
            else:
                docstring = tool_name             

            dyn_tool = StructuredTool(
                name=tool_name,
                func=retriever_func,
                args_schema=create_dynamic_input(dynamic_fields),
                description=docstring,
            )
            logger.info(
                f"Created tool: {tool_name}, collection: {collection_name}, sync_name: {sync_name}"
            )
            tools.append(dyn_tool)

    return tools


def retriever_chroma(vector_store: Any, query: str, retrieval_config: dict, **kwargs: Any):
    filter_conditions = []

    for key, value in kwargs.items():
        if value:
            filter_conditions.append({key: value})

    filter = None
    if filter_conditions:
        filter = (
            {'$and': filter_conditions} if len(filter_conditions) > 1 else filter_conditions[0]
        )

    search_type = retrieval_config.get("search_type", "similarity")
    search_kwargs = {k: v for k, v in retrieval_config.items() if k != "search_type"}

    if "k" not in search_kwargs:
        search_kwargs["k"] = 3
    search_kwargs["filter"] = filter

    if search_type == "similarity_score_threshold" and "score_threshold" not in search_kwargs:
        search_kwargs["score_threshold"] = 0.25

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    docs = retriever.invoke(query)
    return "\n---\n".join([
        f"Source: {doc.metadata.get('source', 'N/A')}\n" + doc.page_content
        for doc in docs
    ])

def retriever_weaviate(vector_store: Any, query: str, retrieval_config: dict, **kwargs: Any):
    filters = None
    all_filters = []

    for key, value in kwargs.items():
        if value:
            all_filters.append(Filter.by_property(key).equal(value))

    if len(all_filters) > 1:
        # Combine multiple filters with a logical AND
        filters = all_filters[0]
        for filter in all_filters[1:]:
            filters = filters & filter
    elif all_filters:
        filters = all_filters[0]

    # Similarity Search: Retrieves documents based on vector similarity, ideal for finding the top k most similar documents.
    # Max Marginal Relevance (MMR): Balances relevance and diversity, useful for avoiding redundancy and ensuring diverse results.
    # Similarity Score Threshold: Retrieves only highly relevant documents based on a similarity score threshold, filtering out less relevant ones.    

    search_type = retrieval_config.get("search_type", "similarity")
    search_kwargs = {k: v for k, v in retrieval_config.items() if k != "search_type"}

    if "k" not in search_kwargs:
        search_kwargs["k"] = 3
    search_kwargs["filters"] = filters

    if search_type == "similarity" and "alpha" not in search_kwargs:
        search_kwargs["alpha"] = 0.25
    elif search_type == "mmr":
        if "fetch_k" not in search_kwargs: search_kwargs["fetch_k"] = 20
        if "lambda_mult" not in search_kwargs: search_kwargs["lambda_mult"] = 0.5
    elif (search_type == "similarity_score_threshold" and
          "score_threshold" not in search_kwargs):
        search_kwargs["score_threshold"] = 0.25
            
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    docs = retriever.invoke(query)

    return "\n---\n".join([
        f"Source: {doc.metadata.get('source', 'N/A')}\n" + doc.page_content
        for doc in docs
    ])

def retriever_qdrant(vector_store: Any, query: str, retrieval_config: dict, **kwargs: Any):
    must_conditions = []
    for key, value in kwargs.items():
        if value:
            must_conditions.append(
                models.FieldCondition(
                    key=f"metadata.{key}",
                    match=models.MatchValue(value=value)
                )
            )

    filter = None
    if must_conditions:
        filter = models.Filter(must=must_conditions)

    k = retrieval_config.get("k", 3)
    search_params = {k: v for k, v in retrieval_config.items() if k != "k"}

    docs = vector_store.similarity_search(
        query=query,
        k=k,
        filter=filter,
        **search_params
    )
    return "\n---\n".join([
        f"Source: {doc.metadata.get('source', 'N/A')}\n" + doc.page_content
        for doc in docs
    ])

def retriever_milvus(vector_store: Any, query: str, retrieval_config: dict, **kwargs: Any):
    expr_list = []
    for key, value in kwargs.items():
        if value:
            expr_list.append(f'{key} == "{value}"')

    expr = None
    if expr_list:
        expr = " and ".join(expr_list)

    k = retrieval_config.get("k", 3)
    search_params = {k: v for k, v in retrieval_config.items() if k != "k"}

    docs = vector_store.similarity_search(query=query, k=k, expr=expr, **search_params)

    return "\n---\n".join([
        f"Source: {doc.metadata.get('source', 'N/A')}\n" + doc.page_content
        for doc in docs
    ])