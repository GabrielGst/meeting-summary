import subprocess
import sys
import os
import gc
import torch
from faster_whisper import WhisperModel
import ollama

def extract_timestamp_seconds(markdown_line: str) -> float:
    """Helper to parse absolute seconds from our markdown table timestamps (`HH:MM:SS`)."""
    try:
        if "`" in markdown_line:
            ts_str = markdown_line.split("`")[1]
            h, m, s = map(int, ts_str.split(":"))
            return float(h * 3600 + m * 60 + s)
    except Exception:
        pass
    return -1.0

def summarize_chunk(chunk_index: int, time_range: str, chunk_text: str) -> str:
    """Generates an isolated, comprehensive analysis for a single 15-minute block."""
    print(f"    -> Running local LLM analysis on Segment #{chunk_index} ({time_range})...", flush=True)
    prompt = (
        f"Tu es un secrétaire de rédaction expert. Rédige un compte-rendu exhaustif, détaillé "
        f"et structuré en français pour la section #{chunk_index} ({time_range}) du webinaire.\n\n"
        "CONSIGNE : Restitue fidèlement les grands thèmes abordés, toutes les données chiffrées/montants, "
        "les dates clés, ainsi que les décisions prises et actions à mener dans cette partie spécifique.\n\n"
        f"Transcription de la section :\n{chunk_text}"
    )
    try:
        response = ollama.generate(
            model="llama3",
            prompt=prompt,
            options={
                "num_predict": 1024, 
                "temperature": 0.1,
                "keep_alive": 0
            }
        )
        return response.get("response", "").strip()
    except Exception as e:
        return f"⚠️ [Erreur Segment #{chunk_index}]: {e}"

def summarize_existing_file(input_file_path: str, output_file_path: str):
    """Exception Case: Processes an existing transcript file sequentially to prevent prompt truncation."""
    if os.path.abspath(input_file_path) == os.path.abspath(output_file_path):
        print("\n❌ Error: --output file cannot be the same as the --from-file input!", file=sys.stderr)
        return

    print(f"\n>>> Exception Pipeline: Processing pre-existing file: {input_file_path}...", flush=True)
    if not os.path.exists(input_file_path):
        print(f"❌ Error: The file '{input_file_path}' does not exist.", file=sys.stderr)
        return

    with open(input_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print("Parsing timestamps and grouping rows into 15-minute buckets...", flush=True)
    CHUNK_LIMIT_SECONDS = 900.0
    chunks = {}
    
    for line in content.splitlines():
        if "|" in line and "Horodatage" not in line and "---" not in line and "`" in line:
            parts = line.split("|")
            if len(parts) >= 3:
                text_content = parts[2].strip()
                seconds = extract_timestamp_seconds(line)
                
                bucket = int(seconds // CHUNK_LIMIT_SECONDS) if seconds >= 0 else 0
                if bucket not in chunks:
                    chunks[bucket] = []
                chunks[bucket].append(text_content)

    if not chunks:
        print("❌ Error: Could not parse any valid timestamped table rows from this file.", file=sys.stderr)
        return

    print("\n[Data Verification Check]:")
    for bucket_key in sorted(chunks.keys()):
        print(f" -> Bucket Minute {bucket_key*15}: Captured {len(chunks[bucket_key])} text segments.")

    # Write the summary incrementally to ensure no data is cut off
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(f"# 📝 Compte-Rendu Détaillé par Sections (Map-Iterate Architecture)\n\n")
        f.write(f"**Généré depuis :** {os.path.basename(input_file_path)}\n\n")
        f.write("Below is the comprehensive sequential synthesis of the entire webinar session, processed section by section:\n\n")
        f.write("---\n\n")
        
        print("\n>>> Commencing progressive sequential generation loop...", flush=True)
        for idx, bucket_key in enumerate(sorted(chunks.keys()), 1):
            raw_block_text = " ".join(chunks[bucket_key])
            if not raw_block_text.strip():
                continue
                
            time_range = f"Min {bucket_key*15} à {(bucket_key+1)*15}"
            chunk_summary = summarize_chunk(idx, time_range, raw_block_text)
            
            # Append each chunk summary directly to the file immediately
            f.write(f"## 📌 Section #{idx} ({time_range})\n\n")
            f.write(f"{chunk_summary}\n\n")
            f.write("---\n\n")
            
    print(f"\n✨ High-density master report successfully compiled and saved to: {output_file_path}")


def transcribe_video(video_url: str, browser: str, output_path: str, session_cookie: str = None):
    """End-to-End Pipeline: Transcribes, releases VRAM, then executes iterative chunked summarization."""
    temp_audio = "wsl_temp_audio.wav"
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
        
    print("\n>>> Phase 1: Downloading & converting remote stream to local audio...", flush=True)
    ydl_cmd = [
        "yt-dlp", "--no-playlist", "--extract-audio", "--audio-format", "wav",
        "--audio-quality", "0", "-o", "wsl_temp_audio.%(ext)s", video_url
    ]
    process_dl = subprocess.run(ydl_cmd, stdout=sys.stdout, stderr=sys.stderr)
    
    if process_dl.returncode != 0 or not os.path.exists(temp_audio):
        print("\n❌ Error: Local audio extraction failed.", file=sys.stderr)
        return

    print("\n>>> Phase 2: Commencing local GPU French transcription...", flush=True)
    CHUNK_LIMIT_SECONDS = 900.0
    chunks = {}
    markdown_lines = []

    try:
        model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
        segments, info = model.transcribe(temp_audio, language="fr", beam_size=5)
        
        for segment in segments:
            start_hours = int(segment.start // 3600)
            start_minutes = int((segment.start % 3600) // 60)
            start_seconds = int(segment.start % 60)
            timestamp = f"{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d}"
            
            clean_text = segment.text.strip()
            markdown_lines.append(f"| `{timestamp}` | {clean_text} |\n")
            print(f"[{timestamp}] {clean_text}", flush=True)
            
            bucket = int(segment.start // CHUNK_LIMIT_SECONDS)
            if bucket not in chunks:
                chunks[bucket] = []
            chunks[bucket].append(clean_text)
            
        print("\n>>> Releasing Whisper from VRAM to make room for Llama 3...", flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("\n[Data Verification Check]:")
        for bucket_key in sorted(chunks.keys()):
            print(f" -> Bucket Minute {bucket_key*15}: Captured {len(chunks[bucket_key])} text segments.")
            
        # Write the combined file output
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcription & Résumé de l'événement\n\n")
            f.write(f"**Source URL:** {video_url}\n\n")
            f.write(f"## 📝 Résumé par Sections (High-Fidelity Context)\n\n")
            
            for idx, bucket_key in enumerate(sorted(chunks.keys()), 1):
                raw_block_text = " ".join(chunks[bucket_key])
                if not raw_block_text.strip():
                    continue
                time_range = f"Min {bucket_key*15} à {(bucket_key+1)*15}"
                chunk_summary = summarize_chunk(idx, time_range, raw_block_text)
                
                f.write(f"### 📌 Section #{idx} ({time_range})\n\n")
                f.write(f"{chunk_summary}\n\n")
            
            f.write(f"---\n\n")
            f.write(f"## 📋 Transcription Horodatée\n\n")
            f.write("| Horodatage | Transcription |\n")
            f.write("| --- | --- |\n")
            f.writelines(markdown_lines)
                
        print(f"\n✨ Success! Chronological high-density summary saved to: {output_path}")

    except Exception as e:
        print(f"\n❌ Transcription pipeline engine failure: {e}", file=sys.stderr)
        raise e
    finally:
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
