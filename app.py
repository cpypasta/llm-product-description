"""
# LLM Task: Take a product ID or name (query) and create a product description suitable for an e-commerce website.

LLM will probably have limited knowledge about all products, so will have to provide knowledge via RAG.

Sources of data:
1. Internet
2. Product Manuals (PDFs)
"""
import streamlit as st, re, os
from time import time
from dotenv import load_dotenv
from streamlit.elements.lib.mutable_status_container import StatusContainer
from streamlit.delta_generator import DeltaGenerator
from web_research import WebResearchRetriever, WebScrapeMethod
from product_description import create_product_description_chain
from langchain_core.language_models.llms import LLM
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.callbacks.base import BaseCallbackHandler

load_dotenv()

class StreamHandler(BaseCallbackHandler):
  def __init__(self, container: DeltaGenerator, initial_text: str = ""):
    self.container = container
    self.text=initial_text
  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)
        
def _get_sources(docs: list[Document]) -> list[str]:
  """Get sources from documents."""
  sources = set()
  for doc in docs:
    new_sources = doc.metadata.get("source", "unknown")
    new_sources = new_sources.split(";")
    for s in new_sources:
      sources.add(s)
  return list(sources)
        
def _format_sources(sources: list[str]) -> str:
  """Format sources as a string, with each source on a new line."""
  sources_str = "\n".join([f"* {s}" for s in sources])
  return f"Sources:\n{sources_str}"

def create_product_description(llm: LLM, product: str, docs: list[Document]) -> list[str]:
  chain = create_product_description_chain(llm)   
  joined_docs = "\n\n".join([doc.page_content for doc in docs])
  chain.invoke({"product": product, "context": joined_docs})
  return _get_sources(docs)

def run(
  llm: LLM, 
  llm_fast: LLM, 
  description_status: StatusContainer, 
  product: str,
  web_scrape_method: WebScrapeMethod,
  smart_top_links: bool,
  links_n: int,
  vector_n: int,
  url: str = None
) -> (list[Document], str):
  start_time = time()
  query = f"what is the product {product}"
  chroma = Chroma(embedding_function=OpenAIEmbeddings(), collection_name="search_results")
  with description_status as status:
    st.write("searching for information...")
    retriever = WebResearchRetriever.from_llm(
      llm_fast, 
      chroma, 
      web_scrape_method, 
      smart_top_links,
      links_n,
      vector_n
    )
    search_docs = retriever.get_relevant_documents(query, url=url)
    st.write("generating description...") 
    sources = create_product_description(llm, product, search_docs) 
    end_time = time()
    end_time = time()
    duration = end_time - start_time 
    duration = f"{duration:.2f} secs"
    status.update(label=f"Description Created ({duration})", state="complete", expanded=False)
  chroma.delete_collection()
  return search_docs, sources


if __name__ == "__main__":    
  openai_key = os.getenv("OPENAI_API_KEY")
  serper_key = os.getenv("SERPER_API_KEY")
  browserless_token = os.getenv("BROWSERLESS_TOKEN")
  
  st.title("Product Description Generator")
  st.info("Enter a product name or ID to generate a product description.")
  product_id = st.text_input("Product", value="Lemon Hint Water", key="product_id")
  product_url = st.text_input("Reference Link _(optional)_", value="", help="Will use the link to find product information.")
  generate_btn = st.button("Generate")
  
  # OPTIONS
  with st.sidebar:
    with st.expander("General Options"):
      chat_model = st.selectbox(
        "Chat Model",
        options=[
          "gpt-4-1106-preview", 
          "gpt-4", 
          "gpt-3.5-turbo-16k-0613", 
          "gpt-3.5-turbo-16k", 
          "gpt-3.5-turbo-1106",
          "gpt-3.5-turbo"
        ],
        index=0
      )
      openai_key_input = st.text_input("OpenAI API Key", value=openai_key, type="password")
      chat_temp = st.slider("Chat Creativity", 0.0, 2.0, 0.7, 0.1, help="This is the chat model temperature.")
    with st.expander("Web Search Options"):
      serper_key_input = st.text_input("Serper API Key", value=serper_key, type="password", help="Uses to perform Google search.")
      browserless_token_input = st.text_input("Browserless.io Token", value=browserless_token, type="password", help="Service to perform web page scraping.")
      web_scrape_method = st.selectbox(
        "Web Scrape Method", 
        options=["simple", "http", "browser", "screenshot"], 
        index=1,
        help="""
What method to use to scrape web pages, from simple to more complex.

_Note: will use Browserless.io if token provided._
      
1. `Simple`: Use the _answerBox_ summary.
2. `HTTP`: Use the HTTP response.
3. `Browser`: Use a headless browser.
4. `Screenshot`: Use a headless browser to take a screenshot.
"""
      )    
      links_n = st.slider("Search Links", 1, 10, 3, help="This is the number of search links to scrape.")
      vector_n = st.slider("Vector Results", 1, 15, 5, help="How many vector search results to inject with RAG.") 
      smart_top_links = st.toggle("Smart Links", value=False, help="Ask the LLM to find the top links.")
  
  # GENERATE DESCRIPTION
  if product_id and generate_btn:
    description_status = st.status("Generating description...", expanded=True)
    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box)
    if serper_key_input:
      os.environ["SERPER_API_KEY"] = serper_key_input
    if browserless_token_input:
      os.environ["BROWSERLESS_TOKEN"] = browserless_token_input
    llm = ChatOpenAI(
      model=chat_model, 
      temperature=chat_temp, 
      streaming=True, 
      callbacks=[stream_handler],
      api_key=openai_key_input
    )
    llm_fast = OpenAI(model="text-davinci-003", temperature=0)
    search_docs, sources = run(
      llm, 
      llm_fast, 
      description_status, 
      product_id, 
      WebScrapeMethod(web_scrape_method),
      smart_top_links,
      links_n,
      vector_n,
      product_url
    )
    st.write(_format_sources(sources))
    with st.expander("Search Documents"):
      for i, doc in enumerate(search_docs):
        st.text_area(
          doc.metadata["source"], 
          doc.page_content.replace("\n", " "), 
          height=200,
          key=f"doc{i}"
        )  
    
    if web_scrape_method == "screenshot":
      with st.expander("Screenshots"):
        domain = r"https?://([A-Za-z_0-9.-]+).*"
        source_domains = []
        for i, s in enumerate(sources):
          if re.match(domain, s):
            d = re.match(domain, s).group(1)
            d = d.replace("www.", "")
            source_domains.append(d)
          else:
            source_domains.append(f"Screenshot {i}")
            
        tabs = st.tabs(source_domains)
        for i, tab in enumerate(tabs):
          tab.image(f"screenshots/image{i}.png")