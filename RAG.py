from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import SingleStoreDB
import os

os.environ["SINGLESTOREDB_URL"] = "<REPLACE SINGLESTOREDB URL>"

# Loading Ingested Text Data
loader = TextLoader("llmdata.txt")
documents = loader.load()

# Splitting Text into Manageable Chunks
text_splitter = CharacterTextSplitter(chunk_size=1000,
                                      chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Setting Up Embeddings
embeddings = OpenAIEmbeddings()

# Storing Text in SingleStoreDB
docsearch = SingleStoreDB.from_documents(texts,
                                         embeddings,
                                         table_name = "pdf_docs3")

# Prompting and Retrival
prompt_template = "Use the following context to answer the question: {context}. Question: {question}"

PROMPT = PromptTemplate(template=prompt_template,
                        input_variables=["context", "question"])

qa_chain = load_qa_chain(OpenAI(), chain_type="stuff")
chain_type_kwargs = {"prompt": PROMPT}

qa = RetrievalQA.from_chain_type(llm=OpenAI(model_name='gpt-4-0613'),
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(),
                                 chain_type_kwargs=chain_type_kwargs)

