# -*- coding: utf-8 -*-
"""LLAMA_rag.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WTuxMAtqt5bxjaOtb0SfZ-glI5pUOhnX
"""

!CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python

!pip install langchain

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

!wget get "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="/content/llama-2-13b-chat.Q4_0.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx = 2048,
    n_gpu_layers = 64
    # Verbose is required to pass to the callback manager
)

!pip install pypdf
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("/content/GenAI.pdf")
#pages = loader.load_and_split()

docs = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_splitter

splits = text_splitter.split_documents(docs)

len(splits)

splits[0]

!pip install chromadb

!pip install sentence-transformers

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#chroma will be the vectordb and content is our splits which is basically our chunks we created above
# converting to embeddings and storing to vectordb
#out retriever is chromadb

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

!pip install langchainhub

from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

prompt

from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("what is generative ai")

rag_chain.invoke("what is the capital of india")

def context_generator(query):
  retrieved_docs = retriever.get_relevant_documents(
    query
)
  context = ''

  for i in range(len(retrieved_docs)):

    context = context + str(retrieved_docs[i].page_content)
  return context

query = "what is the capital of india"

context = context_generator(query)

context

template = f"""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Context: {context}
Question: {query}
Helpful Answer:"""

llm(template)

template

query = "what is generative ai"

context = context_generator(query)

context

template = f"""Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Context: {context}
Question: {query}
Helpful Answer:"""

llm(template)

