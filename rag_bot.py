from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def create_rag_bot(vector_store):
    retriever = vector_store.as_retriever()
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return qa_chain

def ask_question(qa_chain, query):
    custom_prompt = f"Please provide a detailed and comprehensive answer to the following question: {query}"
    response = qa_chain({"query": custom_prompt})
    answer = response["result"]
    sources = response["source_documents"]
    return answer, sources

'''def ask_question(qa_chain, query):
    response = qa_chain({"query": query})
    answer = response["result"]
    sources = response["source_documents"]
    return answer, sources'''