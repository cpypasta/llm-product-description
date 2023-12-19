"""
# LLM Task: Take a product ID or name (query) and create a product description suitable for an e-commerce website.

LLM will probably have limited knowledge about all products, so will have to provide knowledge via RAG.

Sources of data:
1. Internet
2. Product Manuals (PDFs)
"""
from timeit import default_timer as default_timer
from dotenv import load_dotenv
from web_research import WebResearchRetriever
from product_description import create_product_description_chain
from langchain_core.language_models.llms import LLM
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.schema import StrOutputParser, Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()
llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
llm_fast = OpenAI(model="text-davinci-003", temperature=0)


def _format_sources(docs: list[Document]) -> str:
  """Format sources as a string, with each source on a new line."""
  sources = set()
  for doc in docs:
    sources.add(doc.metadata.get("source", "unknown"))
      
  sources_str = "\n".join([f"* {s}" for s in sources])
  return f"Sources:\n{sources_str}"

def create_product_description(llm: LLM, product: str, docs: list[Document]) -> str:
  chain = create_product_description_chain(llm)   
  joined_docs = "\n\n".join([doc.page_content for doc in docs])
  description = chain.invoke({"product": product, "context": joined_docs})
  return f"{description}\n\n{_format_sources(docs)}"

product = "BUC10986"
query = f"what is the product description for product ID {product}"
chroma = Chroma(embedding_function=OpenAIEmbeddings(), collection_name="search_results")
retriever = WebResearchRetriever.from_llm(llm_fast, chroma)
search_docs = retriever.get_relevant_documents(query)
description = create_product_description(llm, product, search_docs)
print(description)

chroma.delete_collection()