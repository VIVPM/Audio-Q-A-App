# 🗣️ Audio-Q&A App

This project is an end-to-end **Audio-based Question Answering App** that enables users to upload an `.mp3` file, automatically transcribe the audio using **Amazon Transcribe**, generate embeddings via **Amazon Bedrock's Titan models**, and store the indexed chunks in **Pinecone** for semantic search. Users can then query the audio content and receive accurate answers using **context-aware LLM generation**.

---

## 🚀 Features

- 🔊 Upload any `.mp3` audio file
- 📝 Automatic transcription with speaker labels using **Amazon Transcribe**
- 🧠 Text embedding using **Amazon Titan Embedding model**
- 📦 Chunked vector storage in **Pinecone vector database**
- ❓ Ask questions and get answers from **Amazon Titan Text Express** model using retrieved context

---

## 🧰 Tech Stack

- **Frontend:** Streamlit
- **Transcription:** AWS Transcribe
- **Embedding & Generation:** Amazon Bedrock (Titan models)
- **Vector Store:** Pinecone
- **Cloud Storage:** Amazon S3

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/audio-qa-app.git
cd audio-qa-app
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Set Environment Variables

Create a `.env` file in the root directory and add the following keys:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_HOST=your_pinecone_index_host
BucketName=your_s3_bucket_name
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

---

## 📌 Workflow

1. **Upload MP3 File:** A user uploads an audio file via the Streamlit interface.
2. **Transcription:** The file is uploaded to S3 and transcribed using Amazon Transcribe with speaker labels.
3. **Chunking & Embedding:** The transcription is split into chunks, and each chunk is embedded using Amazon Titan.
4. **Indexing:** Embedded vectors are stored in Pinecone with metadata.
5. **Q&A Interface:** Users can ask questions, which are embedded and used to fetch relevant context from Pinecone. The final answer is generated using the Amazon Titan Text Express model.

---

## 📚 Example Usage

- Upload a meeting recording.
- Ask: "What did John say about the project deadline?"
- Get an accurate answer backed by contextual chunks from the audio.

---

## 🔐 Security

Make sure not to commit your `.env` file or any secrets. You can add `.env` to `.gitignore`.

---

## 🧾 License

MIT License

---

## 🤝 Contributions

PRs and feedback are welcome! Open an issue or submit a pull request to improve the project.
