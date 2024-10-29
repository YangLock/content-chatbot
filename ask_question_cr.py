import pickle
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import argparse

_template = """Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about articles in the Strikingly support center. 
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure.".
Don't try to make up an answer. If the question is not about
Strikingly's product, politely inform them that you are tuned
to only answer questions about Strikingly.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA = PromptTemplate(template=template, input_variables=["question", "context"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Strikingly Q&A')
    parser.add_argument('question', type=str, help='Your question for strikingly.com')
    args = parser.parse_args()
    
    with open("faiss_store.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    
    llm = OpenAI(temperature=0)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        return_source_documents=True,
        qa_prompt=QA,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    )
    
    chat_history = []
    result = qa_chain({"question": args.question, "chat_history": chat_history})
    
    print(f"AI: {result['answer']}")
    print(f"Source: {result['source_documents']}")
    