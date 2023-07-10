from typing import Annotated

from fastapi import Depends
from langchain.embeddings.openai import OpenAIEmbeddings
from pydantic import BaseSettings
from supabase import Client, create_client
from vectorstore.supabase import SupabaseVectorStore


class BrainSettings(BaseSettings):
    openai_api_key: str
    anthropic_api_key: str
    supabase_url: str
    supabase_service_key: str
    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_type: str = "open_ai"
    openai_gpt_deployment_id : str = None
    openai_embedding_deployment_id : str = None


class LLMSettings(BaseSettings):
    private: bool = False
    model_path: str = "gpt2"
    model_n_ctx: int = 1000
    model_n_batch: int = 8


def common_dependencies() -> dict:
    settings = BrainSettings()
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        deployment=settings.openai_embedding_deployment_id,
        openai_api_type=settings.openai_api_type,
    )
    supabase_client: Client = create_client(
        settings.supabase_url, settings.supabase_service_key
    )
    documents_vector_store = SupabaseVectorStore(
        supabase_client, embeddings, table_name="vectors"
    )
    summaries_vector_store = SupabaseVectorStore(
        supabase_client, embeddings, table_name="summaries"
    )

    return {
        "supabase": supabase_client,
        "embeddings": embeddings,
        "documents_vector_store": documents_vector_store,
        "summaries_vector_store": summaries_vector_store,
    }


CommonsDep = Annotated[dict, Depends(common_dependencies)]
