import json
from timeit import default_timer
from typing import Callable, Any
from json_parser import JsonOutputParser
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.vectorstores import VectorStore
from langchain_core.pydantic_v1 import Field
from langchain.llms.base import BaseLLM
from langchain.utilities.google_serper import GoogleSerperAPIWrapper
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader, AsyncChromiumLoader
from langchain.document_transformers  import BeautifulSoupTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def timer(func: Callable) -> Any:
  def inner(*args, **kwargs):
    start = default_timer() 
    result = func(*args, **kwargs)
    end = default_timer()
    print(f"{func.__name__}:", end - start)
    return result
  return inner

class WebResearchRetriever(BaseRetriever):  
  """Will use Google search to find the top 3 relevant web pages and return the most relevant document results."""
  llm: BaseLLM = Field(..., description="LLM used to ask NLP questions")
  vector_store: VectorStore = Field(..., description="Vector store used to find relevant documents")
  
  @classmethod
  def from_llm(self, llm: BaseLLM, vector_store: VectorStore) -> "WebResearchRetriever":
    return self(llm=llm, vector_store=vector_store)
  
  def _parse_search_results(self, results: list[dict]) -> list[dict]:
      """Parse search results and return title, snippet, and link."""
      parsed_results = []
      for result in results:
          parsed_result = {
              "title": result["title"],
              "snippet": result["snippet"],
              "link": result["link"],
          }
          parsed_results.append(parsed_result)
      return parsed_results

  @timer
  def _web_search(self, query: str) -> list[dict]:
      """Search for query on Google and return results."""
      search_result = GoogleSerperAPIWrapper().results(query)
      return self._parse_search_results(search_result.get("organic"))

  @timer
  def _fine_best_web_sources(self, llm: BaseLLM, query: str, search_result: list[dict]) -> list[str]:
    """Find the best web sources from the search results."""
    search_dump = json.dumps(search_result, indent=2)
    best_search_prompt = PromptTemplate.from_template(
      """You are an expert at looking at search results and picking the best ones. You will be given the <query> and the top 10 <results>. Each result will include <title> and <description> and <link>. Using the search result details, you will pick the top results based on how likely the result will have information related to the <query>. Remember, only return the top 3 results, and respond with a JSON list of <link> values.
      QUERY:
      {query}

      RESULTS:
      '''json
      {results}
      '''

      TOP 3 RESULTS:                                                  
      """)

    chain = best_search_prompt | llm | JsonOutputParser()
    best_links = chain.invoke({"query": query, "results": search_dump}) 
    return best_links

  @timer
  def _fast_web_scrape(self, best_links: list[str]) -> list[Document]:
    """Fast web scraping using just the URL."""
    web_loader = WebBaseLoader(best_links)
    search_docs = web_loader.load()
    return search_docs

  @timer
  def _web_scrape(self, best_links: list[str]) -> list[Document]:
    """Web scraping using a headless browser."""
    web_loader = AsyncChromiumLoader(best_links)
    search_docs = web_loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
      search_docs, 
      tags_to_extract=["div", "span"]
    )
    return docs_transformed

  @timer
  def _find_relevant_search_docs(self, search_docs: list[Document], query: str) -> list[Document]:
    """Find the most relevant text within the search docs."""
    split_docs = RecursiveCharacterTextSplitter().split_documents(search_docs)  
    self.vector_store.add_documents(split_docs)
    related_docs = self.vector_store.similarity_search(f"what is the product description for {query}", k=10)
    return related_docs
  
  def get_relevant_documents(
    self, 
    query: str, 
    *, 
    run_manager: CallbackManagerForRetrieverRun
  ) -> list[Document]:
    """Search Google for documents related to the query input."""
    search_results = self._web_search(query)
    best_links = self._fine_best_web_sources(self.llm, query, search_results)
    search_docs = self._fast_web_scrape(best_links)
    search_docs = self._find_relevant_search_docs(search_docs, query)    
    return search_docs
