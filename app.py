from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    VideoUnplayable as VideoRegionBlocked,
    InvalidVideoId
)
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

def get_youtube_transcript(video_url):
    try:
        video_id = YouTube(video_url).video_id
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = " ".join([d["text"] for d in transcript_data])
        return text
    except TranscriptsDisabled:
        st.error("Transcripts disabled for this video.")
        return None
    except NoTranscriptFound:
        st.error("No transcript found for this video.")
        return None
    except VideoUnavailable:
        st.error("Video is unavailable.")
        return None
    except VideoRegionBlocked:
        st.error("Video is blocked in your region.")
        return None
    except InvalidVideoId:
        st.error("Invalid video ID.")
        return None

def save_transcript(transcript):
    with open("transcript.txt", "w") as f:
        f.write(transcript)
    return "transcript.txt"

st.title("YouTube Transcript Chatbot")
st.write("Enter the YouTube video URL and click on the 'Get Transcript' button to retrieve the transcript.")

video_url = st.text_input("Enter the YouTube video URL")
if st.button("Get Transcript"):
    if video_url:
        transcript = get_youtube_transcript(video_url)
        if transcript:
            transcript_file = save_transcript(transcript)

            loader = TextLoader(transcript_file)
            documents = loader.load()
            
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)   
            chunks = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            retriever = vectorstore.as_retriever()

            chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
            st.session_state.chain = chain
            
            st.success("Transcript saved successfully.")

            # download and display transcript
            with open(transcript_file, "r") as f:
                transcript_content = f.read()
            st.download_button("Download Transcript", 
                              data=transcript_content,
                              file_name="transcript.txt",
                              mime="text/plain")
            st.text_area("Transcript", transcript_content, height=300)
        else:
            st.error("Failed to retrieve transcript.")
    else:
        st.error("Please enter a valid YouTube video URL.")

if "chain" in st.session_state:
    question = st.text_input("Enter your question")
    if st.button("Get Answer"):
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke(question)
            st.write(response["result"])