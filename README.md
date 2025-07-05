# YouTube Transcript Chatbot

A powerful AI-powered application that allows users to extract transcripts from YouTube videos and ask questions about the content. Built with Streamlit, LangChain, and Google's Gemini AI.

## 🚀 Features

- **YouTube Transcript Extraction**: Automatically fetch and process transcripts from any YouTube video
- **AI-Powered Q&A**: Ask questions about the video content using Google's Gemini AI
- **Document Processing**: Intelligent text splitting and embedding for better context understanding
- **Interactive UI**: User-friendly interface built with Streamlit
- **Transcript Download**: Save transcripts as text files for offline use

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI/ML**: 
  - LangChain
  - Google Gemini
  - HuggingFace Sentence Transformers
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **YouTube API**: youtube-transcript-api
- **Environment**: Python 3.8+

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/youtube-transcript-chatbot.git
   cd youtube-transcript-chatbot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   > **Note**: Never commit your `.env` file to version control.

## 🚀 Usage

1. **Run the application**
   ```bash
   streamlit run app.py
   ```

2. **Using the application**
   - Enter a YouTube video URL in the input field
   - Click "Get Transcript" to fetch and process the video transcript
   - Once processed, you can:
     - View the transcript in the text area
     - Download the transcript as a text file
     - Ask questions about the video content in the chat interface

## 🧠 How It Works

1. **Transcript Extraction**:
   - Extracts video ID from the YouTube URL
   - Fetches the English transcript using youtube-transcript-api
   - Processes and cleans the transcript text

2. **Document Processing**:
   - Splits the transcript into manageable chunks
   - Generates embeddings using HuggingFace's sentence transformers
   - Stores embeddings in a FAISS vector store for efficient similarity search

3. **Question Answering**:
   - Uses Google's Gemini model for generating answers
   - Retrieves relevant context from the transcript
   - Provides accurate answers based on the video content

## 📂 Project Structure

```
youtube-transcript-chatbot/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (gitignored)
└── README.md             # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing UI framework
- [LangChain](https://python.langchain.com/) for the LLM orchestration
- [Google Gemini](https://ai.google.dev/) for the powerful language model
- [HuggingFace](https://huggingface.co/) for the sentence transformers
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) for YouTube transcript extraction