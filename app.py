import streamlit as st
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="llama-2-13b-chat.Q4_0.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True, 
    n_ctx = 2048, 
    n_gpu_layers = 64
    # Verbose is required to pass to the callback manager
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def context_generator(query):
  retrieved_docs = retriever.get_relevant_documents(
    query
)
  context = ''
  
  for i in range(len(retrieved_docs)):
    
    context = context + str(retrieved_docs[i].page_content)
  return context



# Sidebar with file upload
#uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Main content
st.title("PDF Text Search App")

# Display uploaded file path
#if uploaded_file:
    #st.sidebar.write("Uploaded PDF File:")
    #st.sidebar.write(uploaded_file.name)

    # Store the path of the PDF file in a variable
    #pdf_path = uploaded_file.name

loader = PyPDFLoader('GenAI.pdf')
docs = loader.load()    
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
print(splits)
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()


# Display text box and search button
search_query = st.text_input("Enter search query:")
if st.button("Search"):
  if search_query:
      context = context_generator(search_query)
      template = f"""Use the following pieces of context to answer the question at the end.
      If you don't know the answer, just say that you don't know, don't try to make up an answer.
      Use three sentences maximum and keep the answer as concise as possible.

      Context: {context}
      Question: {search_query}
      Helpful Answer:"""
      result = llm(template)
      st.write(result)
  else:
      st.warning("Please enter a search query.")
#else:
#st.warning("Please upload a PDF file.")





