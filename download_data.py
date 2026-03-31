"""
download_data.py — Data Download & Preparation Script
======================================================

Downloads training data from two sources and organizes it into the
folder structure that train.py expects:

  data/
    blog_corpus/          ← Blog Authorship Corpus (Hugging Face)
      author_id_1/
        post_0.txt
        post_1.txt
        ...
      author_id_2/
        ...
    demo/                 ← Project Gutenberg authors (optional)
      dickens/
        tale_of_two_cities.txt
      ...

Run this once before train.py:
  python3 download_data.py

Options (edit the CONFIG block below):
  - BLOG_MAX_AUTHORS    how many blog authors to download (default 30)
  - BLOG_POSTS_PER_AUTHOR  max posts per author (default 15)
  - DOWNLOAD_GUTENBERG  set to True to also grab Gutenberg books
  - GUTENBERG_AUTHORS   which authors to grab from Gutenberg

Requirements (installed automatically if missing):
  pip3 install datasets requests tqdm
"""

import os
import re
import sys
import time
import textwrap
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these to control what gets downloaded
# ─────────────────────────────────────────────────────────────────────────────

# How many unique blog authors to save.
# 30 is a good balance — enough for meaningful attribution, fast enough to download.
# Push to 50+ for a more serious experiment.
BLOG_MAX_AUTHORS = 30

# Max posts to save per blog author.
# More posts = better author model, larger disk usage.
BLOG_POSTS_PER_AUTHOR = 15

# Minimum word count for a blog post to be included.
# Short posts are noise — they don't have enough style signal.
BLOG_MIN_WORDS = 50

# Set to True to also download Project Gutenberg books.
# These make for a great demo since the authors are recognizable names.
DOWNLOAD_GUTENBERG = True

# Gutenberg authors to download.
# Format: { "folder_name": [ list of Gutenberg book IDs ] }
# Find more IDs at: https://www.gutenberg.org/browse/scores/top
# Each ID is the number at the end of a Gutenberg URL, e.g.:
#   https://www.gutenberg.org/ebooks/1342  →  ID is 1342
GUTENBERG_AUTHORS = {
    "dickens":    [98, 1400, 730],      # A Tale of Two Cities, Great Expectations, Oliver Twist
    "austen":     [1342, 161, 105],     # Pride and Prejudice, Sense and Sensibility, Persuasion
    "twain":      [74, 76, 245],        # Tom Sawyer, Huck Finn, A Connecticut Yankee
    "doyle":      [1661, 2097, 2852],   # Sherlock Holmes, The Sign of Four, Hound of Baskervilles
    "wilde":      [174, 844, 902],      # Picture of Dorian Gray, The Importance of Being Earnest, An Ideal Husband
}

# Output directories — must match what train.py expects
BLOG_OUTPUT_DIR      = Path("data/blog_corpus")
GUTENBERG_OUTPUT_DIR = Path("data/demo")

# ─────────────────────────────────────────────────────────────────────────────
# Dependency check — install missing packages automatically
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dependencies():
    """
    Check that the packages this script needs are installed.
    If any are missing, install them automatically using pip3.
    We do this before importing them so the script is self-contained.
    """
    required = ["datasets", "requests", "tqdm"]
    missing  = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"[setup] Installing missing packages: {', '.join(missing)}")
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--quiet", *missing
        ])
        print("[setup] Done.\n")

ensure_dependencies()

# Now safe to import
import requests
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Blog Authorship Corpus (via Hugging Face datasets)
# ─────────────────────────────────────────────────────────────────────────────

def download_blog_corpus():
    """
    Download the Blog Authorship Corpus from Hugging Face and save each
    author's posts as individual .txt files.

    The dataset has ~19,000 bloggers and ~681,000 posts. We only download
    what we need (controlled by BLOG_MAX_AUTHORS and BLOG_POSTS_PER_AUTHOR)
    so it doesn't take forever.

    Hugging Face streams the dataset row by row, so we never load the full
    700MB into memory at once — it downloads lazily as we iterate.
    """
    from datasets import load_dataset

    print("=" * 60)
    print("  Downloading Blog Authorship Corpus (Hugging Face)")
    print("=" * 60)
    print(f"  Target: {BLOG_MAX_AUTHORS} authors × {BLOG_POSTS_PER_AUTHOR} posts\n")

    BLOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # streaming=True means we download row-by-row instead of the full dataset
    # at once — much faster when we only want a subset
    print("  Loading dataset stream (this may take a moment on first run)...")
    try:
        dataset = load_dataset(
            "blog_authorship_corpus",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"\n  [!] Failed to load dataset: {e}")
        print("  Make sure you have run: pip3 install datasets")
        return False

    # We collect posts per author as we stream through the dataset.
    # Once an author has enough posts, we skip future posts from them.
    # Once we have enough authors, we stop streaming entirely.
    author_counts: dict[str, int] = defaultdict(int)
    authors_complete: set[str] = set()
    total_saved = 0

    print(f"  Streaming posts and saving to {BLOG_OUTPUT_DIR}/...\n")

    # tqdm gives us a live progress bar in the terminal
    with tqdm(desc="  Posts saved", unit=" posts") as pbar:
        for row in dataset:
            # Each row has: author (str), text (str), date, gender, age, topic, sign
            author_id = str(row["author"]).strip().replace("/", "_").replace(" ", "_")
            text      = str(row["text"]).strip()

            # Skip if this author already has enough posts
            if author_id in authors_complete:
                continue

            # Skip posts that are too short to be useful
            if len(text.split()) < BLOG_MIN_WORDS:
                continue

            # Save this post to disk
            author_dir = BLOG_OUTPUT_DIR / author_id
            author_dir.mkdir(exist_ok=True)

            post_index = author_counts[author_id]
            post_file  = author_dir / f"post_{post_index}.txt"
            post_file.write_text(text, encoding="utf-8")

            author_counts[author_id] += 1
            total_saved += 1
            pbar.update(1)

            # Mark author as complete when they hit the post limit
            if author_counts[author_id] >= BLOG_POSTS_PER_AUTHOR:
                authors_complete.add(author_id)

            # Stop streaming once we have enough authors
            if len(authors_complete) >= BLOG_MAX_AUTHORS:
                break

    print(f"\n  ✓ Saved {total_saved} posts across {len(authors_complete)} authors")
    print(f"  ✓ Output: {BLOG_OUTPUT_DIR}/\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Project Gutenberg downloader
# ─────────────────────────────────────────────────────────────────────────────

# Gutenberg serves plain text files at predictable URLs.
# We try two URL formats because older books use a different path structure.
GUTENBERG_URL_TEMPLATES = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
]

def fetch_gutenberg_text(book_id: int) -> str | None:
    """
    Try to download a plain-text book from Project Gutenberg.
    Returns the cleaned text, or None if the download fails.

    Gutenberg rate-limits aggressive scrapers, so we add a small
    delay between requests to be a polite downloader.
    """
    headers = {
        # Identify ourselves — Gutenberg blocks generic requests
        "User-Agent": "LinguisticFingerprintResearch/1.0 (academic project)"
    }

    for template in GUTENBERG_URL_TEMPLATES:
        url = template.format(id=book_id)
        try:
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 200:
                return clean_gutenberg_text(response.text)
        except requests.RequestException:
            continue  # Try the next URL format

    return None  # All formats failed


def clean_gutenberg_text(raw: str) -> str:
    """
    Strip the Gutenberg header and footer boilerplate from a downloaded book.

    Every Gutenberg file has a standard header ending with a line like:
      *** START OF THE PROJECT GUTENBERG EBOOK ... ***
    and a footer starting with:
      *** END OF THE PROJECT GUTENBERG EBOOK ... ***

    We want only the actual book text between those markers.
    """
    # These patterns match the start/end markers (case-insensitive)
    start_pattern = re.compile(r"\*\*\*\s*START OF.*?\*\*\*", re.IGNORECASE)
    end_pattern   = re.compile(r"\*\*\*\s*END OF.*?\*\*\*",   re.IGNORECASE)

    start_match = start_pattern.search(raw)
    end_match   = end_pattern.search(raw)

    if start_match and end_match:
        # Extract only the content between the markers
        text = raw[start_match.end() : end_match.start()]
    elif start_match:
        text = raw[start_match.end():]
    else:
        # No markers found — use the whole file (some older texts lack them)
        text = raw

    # Collapse excessive blank lines (Gutenberg texts often have 3-4 in a row)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def chunk_book_into_passages(text: str, chunk_size: int = 500) -> list[str]:
    """
    Split a full book into passages of roughly `chunk_size` words each.

    Why chunk instead of using the whole book?
    The stylometric engine was designed for blog-post-length texts.
    Feeding it a 200,000-word novel as a single sample would make the
    feature extraction unreliable and slow. Chunking gives us many
    shorter samples from each author — which is also better for training
    since the model sees more variety in how the author writes.

    Each chunk is saved as a separate .txt file for the author.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i : i + chunk_size]
        # Only keep chunks that are at least half the target size
        # (the last chunk is often shorter)
        if len(chunk_words) >= chunk_size // 2:
            chunks.append(" ".join(chunk_words))

    return chunks


def download_gutenberg_authors():
    """
    Download the configured Project Gutenberg books, chunk them into
    passages, and save each author's passages as individual .txt files.
    """
    print("=" * 60)
    print("  Downloading Project Gutenberg Books")
    print("=" * 60)
    print(f"  Authors: {', '.join(GUTENBERG_AUTHORS.keys())}\n")

    GUTENBERG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for author_name, book_ids in GUTENBERG_AUTHORS.items():
        author_dir = GUTENBERG_OUTPUT_DIR / author_name
        author_dir.mkdir(exist_ok=True)

        print(f"  → {author_name} ({len(book_ids)} books)")
        passage_index = 0

        for book_id in book_ids:
            print(f"      Book {book_id}... ", end="", flush=True)

            text = fetch_gutenberg_text(book_id)

            if text is None:
                print("FAILED (skipping)")
                continue

            # Split the book into manageable passage-length chunks
            passages = chunk_book_into_passages(text, chunk_size=500)

            for passage in passages:
                out_file = author_dir / f"passage_{passage_index}.txt"
                out_file.write_text(passage, encoding="utf-8")
                passage_index += 1

            print(f"✓  ({len(passages)} passages saved)")

            # Be polite to Gutenberg's servers — 1 second between requests
            time.sleep(1)

        print(f"      Total: {passage_index} passages for {author_name}\n")

    print(f"  ✓ Gutenberg output: {GUTENBERG_OUTPUT_DIR}/\n")


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    """
    Walk the output directories and print a summary of what was saved.
    Helps confirm everything looks right before running train.py.
    """
    print("=" * 60)
    print("  Download Summary")
    print("=" * 60)

    for label, directory in [("Blog Corpus", BLOG_OUTPUT_DIR), ("Gutenberg Demo", GUTENBERG_OUTPUT_DIR)]:
        if not directory.exists():
            continue

        author_dirs = [d for d in directory.iterdir() if d.is_dir()]
        if not author_dirs:
            continue

        total_files = sum(len(list(d.glob("*.txt"))) for d in author_dirs)
        print(f"\n  {label}  ({directory})")
        print(f"  {len(author_dirs)} authors · {total_files} text files")

        # Show a few example authors
        for d in sorted(author_dirs)[:5]:
            n = len(list(d.glob("*.txt")))
            print(f"    {d.name:<30s}  {n} files")
        if len(author_dirs) > 5:
            print(f"    … and {len(author_dirs) - 5} more")

    print(f"""
  Next steps:
    1. python3 train.py          ← train the model on this data
    2. uvicorn main:app --reload  ← start the API server
    3. Open frontend/index.html  ← use the demo UI
""")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n  Linguistic Fingerprinting Engine — Data Downloader\n")

    # Check we're in the right directory
    if not Path("train.py").exists():
        print("[!] Run this script from the project root directory")
        print("    (the folder containing train.py, main.py, etc.)\n")
        sys.exit(1)

    # Download blog corpus
    blog_ok = download_blog_corpus()

    if not blog_ok:
        print("[!] Blog corpus download failed.")
        print("    Check your internet connection and try again.\n")

    # Download Gutenberg books (optional)
    if DOWNLOAD_GUTENBERG:
        download_gutenberg_authors()
    else:
        print("  Skipping Gutenberg download (DOWNLOAD_GUTENBERG = False)\n")

    # Print what we got
    print_summary()


if __name__ == "__main__":
    main()
