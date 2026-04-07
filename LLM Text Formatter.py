# pip install openai

import re
import json
import os
import time
import hashlib
from openai import OpenAI

# ================= Configuration =================
API_KEY = ""  # Or directly set your API key here
BASE_URL = "https://api.deepseek.com"
MODEL_ID = "deepseek-chat"

INPUT_FILE = "input.txt"
OUTPUT_FILE = "output.txt"

# Per-file progress tracking (key optimization: no need to manually delete JSON)
PROGRESS_FILE = f"progress_{os.path.basename(INPUT_FILE)}.json"

# Optimal parameters (800–1200 chars per chunk)
MAX_CHUNK_LEN = 1100       # Best balance of token efficiency and stability
REQUEST_INTERVAL = 1.5     # Rate limiting to avoid API throttling
TIMEOUT = 300.0

# Initialize client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# ================= Minimal Prompt (Save ~60% Tokens) =================
def get_system_prompt():
    # Short instruction = lower token cost + more stable output
    return (
        "Text formatting task: "
        "Fix paragraph structure, unify dialogue quotes to 「」, "
        "correct punctuation and unclosed quotes. "
        "Do NOT rewrite, delete, or polish content. "
        "Output only the corrected text."
    )

# ================= Smart Chunking (Optimal 800–1200 chars) =================
def split_text_smart(text):
    """
    Split text by paragraphs with length control

    Advantages:
    - No context carry-over (minimal token usage)
    - Reduces content moderation risks
    - Stable for large texts (100k+ characters)
    """
    paragraphs = text.split("\n")
    chunks = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) + 1 <= MAX_CHUNK_LEN:
            buffer += para + "\n"
        else:
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = para + "\n"

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks

# ================= Safe API Call (Core Stability Module) =================
def call_api(chunk):
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": chunk}
            ],
            temperature=0.1,      # Low temperature for correction tasks
            top_p=0.9,
            max_tokens=4096,      # Prevent output truncation
            timeout=TIMEOUT,
            stream=False
        )

        result = response.choices[0].message.content

        # ===== Truncation detection (Type-C failure protection) =====
        if not result or len(result.strip()) < len(chunk) * 0.4:
            print(" [Warning: Possible truncation → fallback to original]")
            return chunk

        return result.strip()

    except Exception as e:
        print(f"\n[API Error] {e} → fallback to original")
        return chunk  # Never interrupt the workflow

# ================= File Hash (Detect New Input Automatically) =================
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_progress(current_hash):
    if not os.path.exists(PROGRESS_FILE):
        return 0, True

    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("hash") == current_hash:
            print(f"Detected previous progress: resume from chunk {data['index']}")
            return data["index"], False
        else:
            print("\n[New file detected] Resetting progress and output file")
            if os.path.exists(OUTPUT_FILE):
                os.remove(OUTPUT_FILE)
            return 0, True
    except:
        return 0, True

def save_progress(index, current_hash):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "index": index,
            "hash": current_hash,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, ensure_ascii=False, indent=2)

# ================= Main Program (With Real-Time Progress) =================
def main():
    if not API_KEY:
        raise ValueError("Please set DEEPSEEK_API_KEY")

    if not os.path.exists(INPUT_FILE):
        print("Input file not found:", INPUT_FILE)
        return

    text = open(INPUT_FILE, "r", encoding="utf-8").read()
    chunks = split_text_smart(text)

    total_chunks = len(chunks)
    total_chars = sum(len(c) for c in chunks)

    current_hash = file_hash(INPUT_FILE)
    start_index, is_new = load_progress(current_hash)

    processed_chars = sum(len(chunks[i]) for i in range(start_index))
    mode = "a" if start_index > 0 else "w"

    print(f"\nStart processing: {total_chunks} chunks | Total chars {len(text):,}")
    print("Mode: Low-token optimization + chunk-based processing + real-time progress\n")

    with open(OUTPUT_FILE, mode, encoding="utf-8") as out:
        for i in range(start_index, total_chunks):
            chunk = chunks[i]
            chunk_len = len(chunk)

            # ===== Real-time progress display =====
            block_progress = (i + 1) / total_chunks * 100
            char_progress = (processed_chars / total_chars) * 100

            print(
                f"\n[Chunk {i+1}/{total_chunks}] "
                f"[Char Progress {char_progress:.2f}%] "
                f"[Length {chunk_len}]"
            )

            start_time = time.time()

            result = call_api(chunk)

            elapsed = time.time() - start_time
            speed = chunk_len / elapsed if elapsed > 0 else 0

            print(f"[Done] Time {elapsed:.2f}s | Speed {int(speed)} chars/s")

            out.write(result + "\n\n")
            out.flush()

            processed_chars += chunk_len
            save_progress(i + 1, current_hash)

            time.sleep(REQUEST_INTERVAL)  # Rate limiting

    print("\n=== Completed (Stable Low-Cost Mode) ===")
    print("Output file:", OUTPUT_FILE)

if __name__ == "__main__":
    main()