#!/usr/bin/env python3
"""
Bot-Bin — Semantic Memory for Clawdbot
======================================
Converts markdown memory files to AIF-BIN format with vector embeddings
for semantic search across your bot's memory.

Usage:
    python3 botbin.py sync                    # Sync .md files to .aif-bin
    python3 botbin.py search "query"          # Semantic search
    python3 botbin.py status                  # Show sync status
    python3 botbin.py info <file.aif-bin>     # Show file info
    python3 botbin.py extract <file.aif-bin>  # Extract original content

Setup:
    1. Copy botbin.py to your Clawdbot workspace root
    2. Install dependencies: pip install -r requirements.txt
    3. Add sync command to HEARTBEAT.md
    4. Use memory_search to query past context
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime
from pathlib import Path

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
WORKSPACE = SCRIPT_DIR  # Assumes botbin.py is in workspace root

# Configurable paths
MEMORY_DIR = WORKSPACE / "memory"
AIFBIN_DIR = MEMORY_DIR / "aifbin"
SYNC_STATE_FILE = MEMORY_DIR / "botbin_sync_state.json"
MEMORY_MD = WORKSPACE / "MEMORY.md"

# Import the Pro CLI (should be in same directory)
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from aifbin_pro import (
        create_aifbin, load_aifbin, generate_embeddings,
        cosine_similarity, get_embedding_model, is_v2_format,
        print_header, print_success, print_error, print_warning, print_info
    )
    HAS_PRO = True
except ImportError:
    HAS_PRO = False
    print("Warning: aifbin_pro.py not found. Place it in the same directory as botbin.py")


def get_file_hash(path):
    """Get MD5 hash of file contents."""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_sync_state():
    """Load sync state from file."""
    if SYNC_STATE_FILE.exists():
        with open(SYNC_STATE_FILE) as f:
            return json.load(f)
    return {"files": {}, "last_sync": None}


def save_sync_state(state):
    """Save sync state to file."""
    state["last_sync"] = datetime.now().isoformat()
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNC_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def get_md_files():
    """Get all .md files that should be tracked."""
    files = []
    
    # Add MEMORY.md if it exists
    if MEMORY_MD.exists():
        files.append(MEMORY_MD)
    
    # Add all .md files in memory/ directory
    if MEMORY_DIR.exists():
        for f in MEMORY_DIR.glob("*.md"):
            files.append(f)
    
    return files


def cmd_sync(args):
    """Sync changed .md files to .aif-bin format."""
    if not HAS_PRO:
        print_error("aifbin_pro.py not found. Cannot sync.")
        return
    
    # Ensure output directory exists
    AIFBIN_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load previous sync state
    state = load_sync_state()
    
    # Get all tracked files
    md_files = get_md_files()
    
    if not md_files:
        print_warning("No .md files found to sync")
        return
    
    print(f"Checking {len(md_files)} memory files...")
    
    synced = []
    skipped = []
    
    for md_path in md_files:
        current_hash = get_file_hash(md_path)
        file_key = str(md_path.relative_to(WORKSPACE))
        
        # Check if file has changed
        if file_key in state["files"] and state["files"][file_key] == current_hash:
            skipped.append(md_path.name)
            continue
        
        # Sync this file
        output_path = AIFBIN_DIR / f"{md_path.stem}.aif-bin"
        
        try:
            print(f"  Syncing: {md_path.name}")
            create_aifbin(str(md_path), str(output_path), model_key='minilm')
            state["files"][file_key] = current_hash
            synced.append(md_path.name)
        except Exception as e:
            print_error(f"Failed to sync {md_path.name}: {e}")
    
    # Save updated state
    save_sync_state(state)
    
    # Report
    print()
    if synced:
        print_success(f"Synced: {len(synced)} files")
        for f in synced:
            print(f"   - {f}")
    if skipped:
        print(f"Skipped (unchanged): {len(skipped)} files")


def cmd_search(args):
    """Semantic search across memory files."""
    if not HAS_PRO:
        print_error("aifbin_pro.py not found. Cannot search.")
        return
    
    query = args.query
    top_k = args.k
    
    print_header("Bot-Bin — Semantic Search")
    print(f"Query: \"{query}\"")
    print(f"Searching in: {AIFBIN_DIR}")
    print()
    
    if not AIFBIN_DIR.exists():
        print_warning("No aifbin directory found. Run 'sync' first.")
        return
    
    aifbin_files = list(AIFBIN_DIR.glob("*.aif-bin"))
    
    if not aifbin_files:
        print_warning("No .aif-bin files found. Run 'sync' first.")
        return
    
    # Generate query embedding
    query_emb = generate_embeddings([query], 'minilm')[0]
    
    # Search all files
    results = []
    
    for filepath in aifbin_files:
        try:
            data = load_aifbin(str(filepath))
            
            for chunk in data.get('chunks', []):
                if 'embedding' not in chunk or chunk['embedding'] is None:
                    continue
                
                score = cosine_similarity(query_emb, chunk['embedding'])
                content = chunk.get('content', '')
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                
                results.append({
                    'score': score,
                    'file': filepath.name,
                    'content': content[:200],
                    'source': filepath.stem
                })
        except Exception as e:
            print_warning(f"Error reading {filepath.name}: {e}")
    
    # Sort and display
    results.sort(key=lambda x: x['score'], reverse=True)
    top_results = results[:top_k]
    
    if not top_results:
        print_warning("No results found")
        return
    
    for i, r in enumerate(top_results, 1):
        print(f"#{i} [{r['score']:.3f}] {r['file']}")
        print(f"   {r['content'][:100]}...")
        print()


def cmd_status(args):
    """Show sync status."""
    print_header("Bot-Bin — Status")
    
    state = load_sync_state()
    md_files = get_md_files()
    
    print(f"Workspace: {WORKSPACE}")
    print(f"Memory dir: {MEMORY_DIR}")
    print(f"AIF-BIN dir: {AIFBIN_DIR}")
    print()
    print(f"Tracked .md files: {len(md_files)}")
    print(f"Synced files: {len(state.get('files', {}))}")
    print(f"Last sync: {state.get('last_sync', 'Never')}")
    
    if AIFBIN_DIR.exists():
        aifbin_files = list(AIFBIN_DIR.glob("*.aif-bin"))
        total_size = sum(f.stat().st_size for f in aifbin_files)
        print(f"AIF-BIN files: {len(aifbin_files)} ({total_size:,} bytes)")


def cmd_info(args):
    """Show info about an AIF-BIN file."""
    if not HAS_PRO:
        print_error("aifbin_pro.py not found.")
        return
    
    filepath = Path(args.file)
    if not filepath.exists():
        # Try in aifbin directory
        filepath = AIFBIN_DIR / args.file
    
    if not filepath.exists():
        print_error(f"File not found: {args.file}")
        return
    
    print_header("Bot-Bin — File Info")
    
    data = load_aifbin(str(filepath))
    meta = data.get('metadata', {})
    chunks = data.get('chunks', [])
    
    print(f"File: {filepath.name}")
    print(f"Size: {filepath.stat().st_size:,} bytes")
    print(f"Format: {'v2 Binary' if is_v2_format(str(filepath)) else 'v1 JSON'}")
    print(f"Created: {meta.get('created_at', 'Unknown')}")
    print(f"Source: {meta.get('source_file', 'Unknown')}")
    print(f"Chunks: {len(chunks)}")
    
    has_embeddings = any(c.get('embedding') for c in chunks)
    print(f"Has embeddings: {'Yes' if has_embeddings else 'No'}")


def cmd_extract(args):
    """Extract original content from an AIF-BIN file."""
    if not HAS_PRO:
        print_error("aifbin_pro.py not found.")
        return
    
    filepath = Path(args.file)
    if not filepath.exists():
        filepath = AIFBIN_DIR / args.file
    
    if not filepath.exists():
        print_error(f"File not found: {args.file}")
        return
    
    data = load_aifbin(str(filepath))
    
    if data.get('original_raw'):
        content = data['original_raw']
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        print(content)
    else:
        print_error("No original content found in file")


def main():
    parser = argparse.ArgumentParser(
        prog='botbin',
        description='Bot-Bin — Semantic Memory for Clawdbot'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # sync
    p_sync = subparsers.add_parser('sync', help='Sync .md files to .aif-bin')
    
    # search
    p_search = subparsers.add_parser('search', help='Semantic search')
    p_search.add_argument('query', help='Search query')
    p_search.add_argument('-k', type=int, default=5, help='Number of results')
    
    # status
    p_status = subparsers.add_parser('status', help='Show sync status')
    
    # info
    p_info = subparsers.add_parser('info', help='Show file info')
    p_info.add_argument('file', help='AIF-BIN file')
    
    # extract
    p_extract = subparsers.add_parser('extract', help='Extract original content')
    p_extract.add_argument('file', help='AIF-BIN file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        'sync': cmd_sync,
        'search': cmd_search,
        'status': cmd_status,
        'info': cmd_info,
        'extract': cmd_extract,
    }
    
    commands[args.command](args)


if __name__ == '__main__':
    main()
