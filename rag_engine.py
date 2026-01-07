import nest_asyncio
from llama_index.core import (
    SimpleDirectoryReader,
    Settings,
    SummaryIndex,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

nest_asyncio.apply()


def build_query_engine(pdf_path: str):
    # Load document
    documents = SimpleDirectoryReader(
        input_files=[pdf_path]
    ).load_data()

    # Split text
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    # Models
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002"
    )

    # Indexes
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    summary_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    vector_engine = vector_index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        description="Summarization questions"
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_engine,
        description="Context retrieval questions"
    )

    return RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[summary_tool, vector_tool],
        verbose=True,
    )
