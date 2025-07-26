import os
import uuid
import time
import json
import tempfile

import streamlit as st
import boto3
from pinecone import Pinecone
from dotenv import load_dotenv

# ——————————————————————————————————————————————
# Helpers (same as before)
# ——————————————————————————————————————————————

def init_pinecone():
    api_key       = os.getenv("PINECONE_API_KEY")
    pinecone_host = os.getenv("PINECONE_HOST")
    pc            = Pinecone(api_key=api_key)
    return pc.Index(host=pinecone_host)

def embed_text(bedrock_client, text: str) -> list:
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="*/*",
        body=json.dumps({"inputText": text})
    )
    body = json.loads(response["body"].read())
    return body["embedding"]

def chunk_text(text: str, chunk_size: int = 500) -> list:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

def transcribe_and_index(mp3_path, bucket):
    # 1️⃣ Upload
    s3 = boto3.client("s3", region_name="us-east-1")
    key = os.path.basename(mp3_path)
    s3.upload_file(mp3_path, bucket, key)

    # 2️⃣ Transcribe
    transcribe = boto3.client("transcribe", region_name="us-east-1")
    job_name = f"transcription-job-{uuid.uuid4()}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": f"s3://{bucket}/{key}"},
        MediaFormat="mp3",
        LanguageCode="en-US",
        OutputBucketName=bucket,
        Settings={"ShowSpeakerLabels": True, "MaxSpeakerLabels": 2}
    )
    # wait…
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status["TranscriptionJob"]["TranscriptionJobStatus"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(2)
    if status["TranscriptionJob"]["TranscriptionJobStatus"] != "COMPLETED":
        st.error("Transcription failed.")
        return None

    # fetch JSON
    s3_obj = s3.get_object(Bucket=bucket, Key=f"{job_name}.json")
    data   = json.loads(s3_obj["Body"].read())
    items  = data["results"]["items"]

    # assemble transcript
    transcript, current = "", None
    for it in items:
        spk = it.get("speaker_label")
        txt = it["alternatives"][0]["content"]
        if spk and spk != current:
            current = spk
            transcript += f"\n{spk}: "
        if it["type"] == "punctuation":
            transcript = transcript.rstrip()
        transcript += txt + " "

    # 3️⃣ Index in Pinecone
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    index   = init_pinecone()

    for i, chunk in enumerate(chunk_text(transcript)):
        vec = embed_text(bedrock, chunk)
        index.upsert(vectors=[(f"{job_name}-{i}", vec, {"text": chunk})])

    return {"bedrock": bedrock, "index": index, "job_name": job_name}

# ——————————————————————————————————————————————
# Streamlit UI
# ——————————————————————————————————————————————

st.title("🗣️ Audio Transcript Q&A App")

load_dotenv()
bucket = os.getenv("BucketName")

# Step 1: Upload
uploaded = st.file_uploader("Upload an MP3 file", type="mp3")

if uploaded:
    if "state" not in st.session_state:
        st.session_state.state = "processing"

    if st.session_state.state == "processing":
        st.info("Transcribing and indexing… this may take a minute.")
        # write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        result = transcribe_and_index(tmp_path, bucket)
        if result:
            st.session_state.bedrock   = result["bedrock"]
            st.session_state.index     = result["index"]
            st.session_state.job_name  = result["job_name"]
            st.session_state.state     = "ready"
            st.success("✅ Processing complete. You can now ask questions.")
        os.remove(tmp_path)

# Step 2: Ask questions once ready
if st.session_state.get("state") == "ready":
    question = st.text_input("Ask a question about your audio…")
    if question:
        bedrock = st.session_state.bedrock
        index   = st.session_state.index

        # Retrieval
        q_vec     = embed_text(bedrock, question)
        query_res = index.query(vector=q_vec, top_k=5, include_metadata=True)
        context   = "\n\n".join([m["metadata"]["text"] for m in query_res["matches"]])

        # Generation
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        resp = bedrock.invoke_model(
            modelId="amazon.nova-micro-v1:0",
            contentType="application/json",
            accept="*/*",
            body=json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {"maxTokenCount":512, "temperature":0, "topP":0.9}
            })
        )
        answer = json.loads(resp["body"].read())["results"][0]["outputText"]
        st.markdown(f"**Answer:** {answer}")
