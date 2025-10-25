import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "api code"


pdf_path = "Good_Strategy_Bad_Strategy_PDF.pdf"   # file about business strategies
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# splitting documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# embeddings + FAISS vector search
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # retrieve top 3 matches only

# model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Anti-hallucination system prompt
template = """
You are a helpful assistant answering questions about the book *Good Strategy / Bad Strategy*.
You MUST use only the information in the provided context.

If the answer is not clearly found in the context, respond exactly with:
"I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)

# retrieval QA system
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=False
)

print("\n Strategy bot ready - ask about the book\n")

def ask(question):
    answer = qa.invoke(question)
    return answer


ask("what is strategy")