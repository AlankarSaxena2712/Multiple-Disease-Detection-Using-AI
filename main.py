import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import Chat
import csv
import io
from langchain_google_genai import ChatGoogleGenerativeAI


OPENAI_API_KEY = "" #Pass your key here
# GOOGLE_API_KEY = "" #Pass your key here


#Upload PDF files
st.header("My first Chatbot")


with  st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions", type=["pdf", "csv"])


#Extract the text
if file is not None:
    text = ""
    if file.name.endswith(".pdf"):
      pdf_reader = PdfReader(file)
      for page in pdf_reader.pages:
          text += page.extract_text()
        #st.write(text)
    elif file.name.endswith(".csv"):
        decoded_file = io.StringIO(file.read().decode("utf-8"))
        reader = csv.reader(decoded_file)
        c = 0
        for row in reader:
            text += ",".join(row) + "\n"
            c += 1
            if c == 5:
                break


    st.write(text)
    # 
    print(text)

#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=500,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)




    # generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)


    # get user question
    user_question = st.text_input("Type Your question here")


    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)


        #define the LLM
        # llm = ChatOpenAI(
        #     openai_api_key = OPENAI_API_KEY,
        #     temperature = 0,
        #     max_tokens = 1000,
        #     model_name = "gpt-3.5-turbo"
        # )
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        )


        #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output

        chain = llm.invoke(match, question=user_question)
        response = chain.content

        # chain = load_qa_chain(llm, chain_type="stuff")
        # response = chain.run(input_documents = match, question = user_question)
        st.write(response)