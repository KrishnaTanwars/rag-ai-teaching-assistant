# RAG Based AI Teaching Assistant

A local retrieval-augmented teaching assistant that searches through video transcript chunks and answers questions with the most relevant video timestamp.

## What it does

1. Converts tutorial videos from `videos/` into MP3 files in `audios/`.
2. Transcribes audio files into JSON subtitle chunks with Whisper.
3. Merges transcript chunks into larger searchable passages.
4. Creates embeddings for the passages using a local Ollama embedding model.
5. Retrieves the most relevant chunks for a user question and asks a local LLM to answer with the matching video and timestamp.

## Requirements

- Python 3.10+
- FFmpeg installed locally
- Ollama running locally
- Ollama models:
  - `bge-m3` for embeddings
  - `llama3.2` for answering questions

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Setup

Copy the environment template if you plan to use OpenAI-based features later:

```bash
cp .env.example .env
```

The current workflow uses local Ollama endpoints, so no API key is required for the default RAG flow.

## Usage

1. Add video files to a local `videos/` folder.
2. Convert videos to MP3:

```bash
python video_to_mp3.py
```

3. Transcribe MP3 files:

```bash
python mp3_to_json.py
```

4. Merge transcript chunks:

```bash
python merge_chunks.py
```

5. Generate embeddings:

```bash
python preprocess_json.py
```

6. Ask a question:

```bash
python process_incoming.py
```

## Notes

Generated files such as media, transcripts, prompt/response outputs, and `embeddings.joblib` are intentionally ignored by Git to keep the repository lightweight and safe to publish.
