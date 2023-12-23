"""
# LLM Task: Take a product ID or name (query) and create a product description suitable for an e-commerce website.

LLM will probably have limited knowledge about all products, so will have to provide knowledge via RAG.

Sources of data:
1. Internet
2. Product Manuals (PDFs)
"""
import streamlit as st, re
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
  vector_n: int
) -> str:
  start_time = time()
  query = f"what is the product {product}"
  chroma = Chroma(embedding_function=OpenAIEmbeddings(), collection_name="search_results")
  with description_status as status:
    st.write("Searching internet")
    retriever = WebResearchRetriever.from_llm(
      llm_fast, 
      chroma, 
      web_scrape_method, 
      smart_top_links,
      links_n,
      vector_n
    )
    search_docs = retriever.get_relevant_documents(query)
    st.write("Processing description") 
    sources = create_product_description(llm, product, search_docs) 
    end_time = time()
    end_time = time()
    duration = end_time - start_time 
    duration = f"{duration:.2f} secs"
    status.update(label=f"Description Created ({duration})", state="complete", expanded=False)
  chroma.delete_collection()
  return sources


if __name__ == "__main__":    
  st.title("Product Description Generator")
  st.info("Enter a product ID to generate a product description.")
  product_id = st.text_input("Product ID", value="BUC10986", key="product_id")
  generate_btn = st.button("Generate")
  
  # OPTIONS
  with st.sidebar:
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
      index=1
    )
    chat_temp = st.slider("Chat temperature (creativity)", 0.0, 2.0, 0.7, 0.1)
    web_scrape_method = st.selectbox(
      "Web Scrape Method", 
      options=["simple", "http", "browser", "screenshot"], 
      index=1
    )
    smart_top_links = st.toggle("Smart top links", value=True)
    links_n = st.slider("Number of links to search", 1, 10, 3)
    vector_n = st.slider("Number of vector results", 1, 15, 5) 
  
  # GENERATE DESCRIPTION
  if product_id and generate_btn:
    description_status = st.status("Generating description...", expanded=True)
    chat_box = st.empty()
    stream_handler = StreamHandler(chat_box)
    print("using", chat_model)
    llm = ChatOpenAI(
      model=chat_model, 
      temperature=chat_temp, 
      streaming=True, 
      callbacks=[stream_handler]
    )
    llm_fast = OpenAI(model="text-davinci-003", temperature=0)
    sources = run(
      llm, 
      llm_fast, 
      description_status, 
      product_id, 
      WebScrapeMethod(web_scrape_method),
      smart_top_links,
      links_n,
      vector_n
    )
    st.write(_format_sources(sources))
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