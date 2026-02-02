#!/usr/bin/env python3
"""
AIF-BIN Pro CLI — Full-Featured Command Line Tool
==================================================
Professional AI memory management with semantic search.

Now uses v2 Binary Format (compatible with Web Inspector).

Features:
- Pretty colored terminal output
- Batch parallel processing
- Watch mode (auto-sync)
- Multiple embedding models
- Search filters (date, type, tags)
- Diff versions
- Export to JSON, CSV, HTML, Markdown
"""

import os
import sys
import json
import time
import struct
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import IntEnum

# Suppress warnings
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import warnings
warnings.filterwarnings('ignore')

try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("Warning: msgpack not installed. Run: pip install msgpack")

# ============================================================
# COLORS & STYLING
# ============================================================

class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'
    
    @classmethod
    def disable(cls):
        cls.HEADER = cls.BLUE = cls.CYAN = cls.GREEN = ''
        cls.YELLOW = cls.RED = cls.BOLD = cls.DIM = cls.RESET = ''

if not sys.stdout.isatty():
    Colors.disable()

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 60}{Colors.RESET}\n")

def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")

def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")

def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")

def print_info(text: str):
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.RESET}")

def print_file(name: str, details: str = ""):
    print(f"  {Colors.DIM}📄{Colors.RESET} {Colors.BOLD}{name}{Colors.RESET} {Colors.DIM}{details}{Colors.RESET}")

# ============================================================
# V2 BINARY FORMAT
# ============================================================

MAGIC = b"AIFBIN\x00\x01"
HEADER_SIZE = 64
ABSENT_OFFSET = 0xFFFFFFFFFFFFFFFF
FORMAT_VERSION = 2


class ChunkType(IntEnum):
    TEXT = 1
    TABLE_JSON = 2
    IMAGE = 3
    AUDIO = 4
    VIDEO = 5
    CODE = 6


@dataclass
class ContentChunk:
    chunk_type: ChunkType
    data: bytes
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class AIFBINFile:
    metadata: Dict[str, Any]
    chunks: List[ContentChunk]
    original_raw: Optional[bytes] = None
    versions: List[Dict] = field(default_factory=list)


def write_aifbin_v2(aifbin: AIFBINFile, filepath: str) -> int:
    """Write an AIFBINFile to disk in v2 binary format."""
    if not HAS_MSGPACK:
        raise ImportError("msgpack is required for v2 format")
    
    # Prepare sections as bytes
    metadata_bytes = msgpack.packb(aifbin.metadata)
    
    original_raw_bytes = aifbin.original_raw or b""
    has_original = aifbin.original_raw is not None
    
    # Pack chunks
    chunks_parts = [struct.pack("<I", len(aifbin.chunks))]
    for chunk in aifbin.chunks:
        meta_with_emb = dict(chunk.metadata)
        if chunk.embedding:
            meta_with_emb['embedding'] = chunk.embedding
        meta_bytes = msgpack.packb(meta_with_emb)
        data_bytes = chunk.data if isinstance(chunk.data, bytes) else chunk.data.encode('utf-8')
        
        chunks_parts.append(struct.pack(
            "<I Q Q",
            int(chunk.chunk_type),
            len(data_bytes),
            len(meta_bytes)
        ))
        chunks_parts.append(meta_bytes)
        chunks_parts.append(data_bytes)
    chunks_bytes = b"".join(chunks_parts)
    
    # Calculate offsets
    metadata_offset = HEADER_SIZE
    metadata_section_size = 8 + len(metadata_bytes)
    
    if has_original:
        original_offset = metadata_offset + metadata_section_size
        original_section_size = 8 + len(original_raw_bytes)
    else:
        original_offset = ABSENT_OFFSET
        original_section_size = 0
    
    chunks_offset = metadata_offset + metadata_section_size + original_section_size
    chunks_section_size = len(chunks_bytes)
    
    versions_offset = ABSENT_OFFSET  # No versions for now
    
    footer_offset = chunks_offset + chunks_section_size
    
    # Build footer (index + checksum)
    footer_parts = [struct.pack("<I", len(aifbin.chunks))]
    current_offset = chunks_offset + 4
    for i, chunk in enumerate(aifbin.chunks):
        footer_parts.append(struct.pack("<I Q", i, current_offset))
        meta_with_emb = dict(chunk.metadata)
        if chunk.embedding:
            meta_with_emb['embedding'] = chunk.embedding
        meta_bytes = msgpack.packb(meta_with_emb)
        data_bytes = chunk.data if isinstance(chunk.data, bytes) else chunk.data.encode('utf-8')
        current_offset += 4 + 8 + 8 + len(meta_bytes) + len(data_bytes)
    
    checksum = sum(struct.unpack("<Q", footer_parts[i+1][4:])[0] for i in range(len(aifbin.chunks))) if aifbin.chunks else 0
    footer_parts.append(struct.pack("<Q", checksum & 0xFFFFFFFFFFFFFFFF))
    footer_bytes = b"".join(footer_parts)
    
    total_size = footer_offset + len(footer_bytes)
    
    # Build header
    header = struct.pack(
        "<8sII QQQQQQ",
        MAGIC,
        FORMAT_VERSION,
        0,  # Padding
        metadata_offset,
        original_offset,
        chunks_offset,
        versions_offset,
        footer_offset,
        total_size
    )
    
    # Write file
    with open(filepath, 'wb') as f:
        f.write(header)
        f.write(struct.pack("<Q", len(metadata_bytes)))
        f.write(metadata_bytes)
        if has_original:
            f.write(struct.pack("<Q", len(original_raw_bytes)))
            f.write(original_raw_bytes)
        f.write(chunks_bytes)
        f.write(footer_bytes)
    
    return total_size


def read_aifbin_v2(filepath: str) -> AIFBINFile:
    """Read an AIF-BIN file in v2 binary format."""
    if not HAS_MSGPACK:
        raise ImportError("msgpack is required for v2 format")
    
    filesize = os.path.getsize(filepath)
    
    with open(filepath, 'rb') as f:
        header = f.read(HEADER_SIZE)
        (
            magic, version, _,
            meta_off, raw_off, chunks_off, versions_off, footer_off, total_size
        ) = struct.unpack("<8sII QQQQQQ", header)
        
        if magic != MAGIC:
            raise ValueError(f"Invalid magic signature: {magic}")
        
        # Read metadata
        f.seek(meta_off)
        meta_len = struct.unpack("<Q", f.read(8))[0]
        metadata = msgpack.unpackb(f.read(meta_len))
        
        # Read original raw
        original_raw = None
        if raw_off != ABSENT_OFFSET and raw_off < filesize:
            f.seek(raw_off)
            raw_len = struct.unpack("<Q", f.read(8))[0]
            original_raw = f.read(raw_len)
        
        # Read chunks
        chunks = []
        if chunks_off < filesize:
            f.seek(chunks_off)
            chunk_count = struct.unpack("<I", f.read(4))[0]
            
            for _ in range(chunk_count):
                c_type, data_len, meta_len = struct.unpack("<I Q Q", f.read(20))
                c_meta = msgpack.unpackb(f.read(meta_len))
                c_data = f.read(data_len)
                
                embedding = c_meta.pop('embedding', None)
                
                chunks.append(ContentChunk(
                    chunk_type=ChunkType(c_type) if c_type in [e.value for e in ChunkType] else ChunkType.TEXT,
                    data=c_data,
                    metadata=c_meta,
                    embedding=embedding
                ))
        
        return AIFBINFile(
            metadata=metadata,
            chunks=chunks,
            original_raw=original_raw,
            versions=[]
        )


def is_v2_format(filepath: str) -> bool:
    """Check if file is v2 binary format."""
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(8)
            return magic == MAGIC
    except:
        return False


def load_aifbin(filepath: str) -> Dict[str, Any]:
    """Load an AIF-BIN file (auto-detect format)."""
    if is_v2_format(filepath):
        aifbin = read_aifbin_v2(filepath)
        return {
            'metadata': aifbin.metadata,
            'chunks': [
                {
                    'id': i,
                    'content': c.data.decode('utf-8', errors='replace') if isinstance(c.data, bytes) else c.data,
                    'embedding': c.embedding,
                    'type': c.chunk_type.name.lower(),
                    'metadata': c.metadata
                }
                for i, c in enumerate(aifbin.chunks)
            ],
            'original_raw': aifbin.original_raw.decode('utf-8', errors='replace') if aifbin.original_raw else None,
            'versions': aifbin.versions
        }
    else:
        # Fallback to v1 JSON
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)


# ============================================================
# EMBEDDING MODELS
# ============================================================

EMBEDDING_MODELS = {
    'minilm': {
        'name': 'all-MiniLM-L6-v2',
        'dim': 384,
        'description': 'Fast, good quality (default)'
    },
    'mpnet': {
        'name': 'all-mpnet-base-v2',
        'dim': 768,
        'description': 'Higher quality, slower'
    },
    'bge-small': {
        'name': 'BAAI/bge-small-en-v1.5',
        'dim': 384,
        'description': 'Optimized for retrieval'
    },
    'bge-base': {
        'name': 'BAAI/bge-base-en-v1.5',
        'dim': 768,
        'description': 'Best quality retrieval'
    },
    'e5-small': {
        'name': 'intfloat/e5-small-v2',
        'dim': 384,
        'description': 'Microsoft E5, fast'
    }
}

_model_cache = {}

def get_embedding_model(model_key: str = 'minilm'):
    """Get or load an embedding model."""
    if model_key not in _model_cache:
        from sentence_transformers import SentenceTransformer
        model_name = EMBEDDING_MODELS.get(model_key, EMBEDDING_MODELS['minilm'])['name']
        _model_cache[model_key] = SentenceTransformer(model_name)
    return _model_cache[model_key]

def generate_embeddings(texts: List[str], model_key: str = 'minilm') -> List[List[float]]:
    """Generate embeddings for a list of texts."""
    model = get_embedding_model(model_key)
    embeddings = model.encode(texts, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]

# ============================================================
# CORE OPERATIONS
# ============================================================

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks if chunks else [text]

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract basic entities from text."""
    import re
    
    dates = re.findall(r'\b\d{4}-\d{2}-\d{2}\b', text)
    emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
    urls = re.findall(r'https?://\S+', text)
    
    decisions = []
    for line in text.split('\n'):
        lower = line.lower()
        if any(kw in lower for kw in ['decided', 'decision', 'agreed', 'confirmed', 'set to', 'chose']):
            decisions.append(line.strip()[:100])
    
    return {
        'dates': list(set(dates))[:10],
        'emails': list(set(emails))[:5],
        'urls': list(set(urls))[:10],
        'decisions': decisions[:5]
    }

def create_aifbin(source_path: str, output_path: str, model_key: str = 'minilm') -> Dict[str, Any]:
    """Create an AIF-BIN file from a source file (v2 binary format)."""
    with open(source_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    text_chunks = chunk_text(content)
    embeddings = generate_embeddings(text_chunks, model_key)
    
    chunks = []
    for i, (text, emb) in enumerate(zip(text_chunks, embeddings)):
        chunks.append(ContentChunk(
            chunk_type=ChunkType.TEXT,
            data=text.encode('utf-8'),
            metadata={'chunk_id': i},
            embedding=emb
        ))
    
    aifbin = AIFBINFile(
        metadata={
            'source_file': str(source_path),
            'created_at': datetime.now().isoformat(),
            'model': model_key,
            'chunk_count': len(chunks),
            'entities': extract_entities(content)
        },
        chunks=chunks,
        original_raw=content.encode('utf-8'),
        versions=[]
    )
    
    size = write_aifbin_v2(aifbin, output_path)
    
    return {
        'source': source_path,
        'output': output_path,
        'chunks': len(chunks),
        'size': size
    }

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# ============================================================
# COMMANDS
# ============================================================

def cmd_migrate(args):
    """Migrate markdown files to AIF-BIN format (v2 binary)."""
    print_header("AIF-BIN Pro — Migrate (v2 Binary)")
    
    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if source.is_file():
        files = [source]
    else:
        files = list(source.glob('**/*.md')) if args.recursive else list(source.glob('*.md'))
    
    if not files:
        print_warning(f"No markdown files found in {source}")
        return
    
    print_info(f"Found {len(files)} file(s) to migrate")
    print_info(f"Model: {args.model} ({EMBEDDING_MODELS[args.model]['description']})")
    print_info(f"Output: {output_dir}")
    print_info(f"Format: v2 Binary (MessagePack)")
    print()
    
    results = []
    start_time = time.time()
    
    if args.parallel and len(files) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for f in files:
                out_path = output_dir / f"{f.stem}.aif-bin"
                futures[executor.submit(create_aifbin, str(f), str(out_path), args.model)] = f
            
            for future in as_completed(futures):
                f = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print_file(f.name, f"→ {result['chunks']} chunks, {result['size']} bytes")
                except Exception as e:
                    print_error(f"{f.name}: {e}")
    else:
        for f in files:
            try:
                out_path = output_dir / f"{f.stem}.aif-bin"
                result = create_aifbin(str(f), str(out_path), args.model)
                results.append(result)
                print_file(f.name, f"→ {result['chunks']} chunks, {result['size']} bytes")
            except Exception as e:
                print_error(f"{f.name}: {e}")
    
    elapsed = time.time() - start_time
    print()
    print_success(f"Migrated {len(results)} file(s) in {elapsed:.2f}s")
    total_chunks = sum(r['chunks'] for r in results)
    total_size = sum(r['size'] for r in results)
    print_info(f"Total: {total_chunks} chunks, {total_size:,} bytes")

def cmd_search(args):
    """Search AIF-BIN files semantically."""
    print_header("AIF-BIN Pro — Semantic Search")
    
    search_dir = Path(args.dir)
    files = list(search_dir.glob('*.aif-bin'))
    
    if not files:
        print_warning(f"No .aif-bin files found in {search_dir}")
        return
    
    print(f"{Colors.BOLD}🔍 Query:{Colors.RESET} \"{args.query}\"")
    print(f"{Colors.DIM}📁 Searching {len(files)} file(s) with model '{args.model}'{Colors.RESET}")
    
    if args.after:
        print(f"{Colors.DIM}📅 Filter: after {args.after}{Colors.RESET}")
    print()
    
    query_emb = generate_embeddings([args.query], args.model)[0]
    
    all_results = []
    for filepath in files:
        try:
            data = load_aifbin(str(filepath))
            
            if args.after:
                created = data.get('metadata', {}).get('created_at', '')
                if created and created < args.after:
                    continue
            
            for chunk in data.get('chunks', []):
                if 'embedding' not in chunk or chunk['embedding'] is None:
                    continue
                
                score = cosine_similarity(query_emb, chunk['embedding'])
                content = chunk.get('content', '')
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                
                all_results.append({
                    'score': score,
                    'file': filepath.name,
                    'content': content[:200],
                    'chunk_id': chunk.get('id', 0),
                    'metadata': data.get('metadata', {})
                })
        except Exception as e:
            print_warning(f"Error reading {filepath.name}: {e}")
    
    all_results.sort(key=lambda x: x['score'], reverse=True)
    top_results = all_results[:args.k]
    
    if not top_results:
        print_warning("No results found")
        return
    
    for i, r in enumerate(top_results, 1):
        score_color = Colors.GREEN if r['score'] > 0.5 else (Colors.YELLOW if r['score'] > 0.3 else Colors.DIM)
        
        print(f"┌{'─' * 58}┐")
        print(f"│ {Colors.BOLD}#{i}{Colors.RESET}  {score_color}Score: {r['score']:.3f}{Colors.RESET}  │  {Colors.CYAN}{r['file']}{Colors.RESET}")
        print(f"├{'─' * 58}┤")
        
        content = r['content'].replace('\n', ' ')[:150]
        print(f"│ {content}...")
        print(f"└{'─' * 58}┘")
        print()

def cmd_info(args):
    """Show detailed info about an AIF-BIN file."""
    print_header("AIF-BIN Pro — File Info")
    
    filepath = Path(args.file)
    if not filepath.exists():
        print_error(f"File not found: {filepath}")
        return
    
    is_v2 = is_v2_format(str(filepath))
    data = load_aifbin(str(filepath))
    meta = data.get('metadata', {})
    chunks = data.get('chunks', [])
    versions = data.get('versions', [])
    has_raw = data.get('original_raw') is not None
    
    file_size = os.path.getsize(filepath)
    
    print(f"  {Colors.BOLD}File:{Colors.RESET}       {filepath.name}")
    print(f"  {Colors.BOLD}Format:{Colors.RESET}     {'v2 Binary' if is_v2 else 'v1 JSON'}")
    print(f"  {Colors.BOLD}Size:{Colors.RESET}       {file_size:,} bytes")
    print(f"  {Colors.BOLD}Created:{Colors.RESET}    {meta.get('created_at', 'Unknown')}")
    print(f"  {Colors.BOLD}Source:{Colors.RESET}     {meta.get('source_file', 'Unknown')}")
    print(f"  {Colors.BOLD}Model:{Colors.RESET}      {meta.get('model', 'Unknown')}")
    print()
    print(f"  {Colors.BOLD}Chunks:{Colors.RESET}     {len(chunks)}")
    print(f"  {Colors.BOLD}Versions:{Colors.RESET}   {len(versions)}")
    print(f"  {Colors.BOLD}Has Raw:{Colors.RESET}    {'Yes' if has_raw else 'No'}")
    
    entities = meta.get('entities', {})
    if entities:
        print()
        print(f"  {Colors.BOLD}Entities:{Colors.RESET}")
        for etype, values in entities.items():
            if values:
                print(f"    {etype}: {', '.join(str(v)[:30] for v in values[:3])}")
    
    print()
    print(f"  {Colors.BOLD}Chunk Breakdown:{Colors.RESET}")
    total_chars = sum(len(str(c.get('content', ''))) for c in chunks)
    avg_chars = total_chars // len(chunks) if chunks else 0
    print(f"    Total characters: {total_chars:,}")
    print(f"    Avg per chunk: {avg_chars:,}")

def cmd_extract(args):
    """Extract original content from an AIF-BIN file."""
    filepath = Path(args.file)
    if not filepath.exists():
        print_error(f"File not found: {filepath}")
        return
    data = load_aifbin(str(filepath))
    
    if data.get('original_raw'):
        content = data['original_raw']
        if isinstance(content, bytes):
            content = content.decode('utf-8', errors='replace')
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(content)
            print_success(f"Extracted to {args.output}")
        else:
            print(content)
    else:
        print_error("No original content found in file")

def cmd_watch(args):
    """Watch directory and auto-sync changes."""
    print_header("AIF-BIN Pro — Watch Mode (v2 Binary)")
    
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_info(f"Watching: {source_dir}")
    print_info(f"Output: {output_dir}")
    print_info(f"Model: {args.model}")
    print_info(f"Format: v2 Binary")
    print()
    print(f"{Colors.DIM}Press Ctrl+C to stop{Colors.RESET}")
    print()
    
    file_hashes = {}
    
    def get_hash(filepath):
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def sync_file(filepath):
        out_path = output_dir / f"{filepath.stem}.aif-bin"
        try:
            result = create_aifbin(str(filepath), str(out_path), args.model)
            print_success(f"Synced: {filepath.name} → {result['chunks']} chunks")
        except Exception as e:
            print_error(f"Failed: {filepath.name} — {e}")
    
    for f in source_dir.glob('*.md'):
        file_hashes[str(f)] = get_hash(f)
        out_path = output_dir / f"{f.stem}.aif-bin"
        if not out_path.exists():
            sync_file(f)
    
    try:
        while True:
            time.sleep(args.interval)
            
            for f in source_dir.glob('*.md'):
                current_hash = get_hash(f)
                if str(f) not in file_hashes or file_hashes[str(f)] != current_hash:
                    file_hashes[str(f)] = current_hash
                    sync_file(f)
    except KeyboardInterrupt:
        print()
        print_info("Watch mode stopped")

def cmd_diff(args):
    """Show diff between two AIF-BIN files."""
    print_header("AIF-BIN Pro — Diff")
    
    file1 = load_aifbin(args.file1)
    file2 = load_aifbin(args.file2)
    
    chunks1 = {str(c.get('content', ''))[:50]: c for c in file1.get('chunks', [])}
    chunks2 = {str(c.get('content', ''))[:50]: c for c in file2.get('chunks', [])}
    
    keys1 = set(chunks1.keys())
    keys2 = set(chunks2.keys())
    
    added = keys2 - keys1
    removed = keys1 - keys2
    common = keys1 & keys2
    
    print(f"  {Colors.GREEN}+ Added:{Colors.RESET}   {len(added)} chunks")
    print(f"  {Colors.RED}- Removed:{Colors.RESET} {len(removed)} chunks")
    print(f"  {Colors.DIM}= Common:{Colors.RESET}  {len(common)} chunks")
    print()
    
    if added and args.verbose:
        print(f"{Colors.GREEN}Added chunks:{Colors.RESET}")
        for key in list(added)[:5]:
            print(f"  + {key}...")
        print()
    
    if removed and args.verbose:
        print(f"{Colors.RED}Removed chunks:{Colors.RESET}")
        for key in list(removed)[:5]:
            print(f"  - {key}...")

def cmd_export(args):
    """Export AIF-BIN to other formats."""
    print_header("AIF-BIN Pro — Export")
    
    data = load_aifbin(args.file)
    output = Path(args.output)
    
    if args.format == 'json':
        with open(output, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif args.format == 'csv':
        import csv
        with open(output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'content', 'type'])
            for chunk in data.get('chunks', []):
                content = chunk.get('content', '')
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                writer.writerow([chunk.get('id'), content[:500], chunk.get('type', 'text')])
    
    elif args.format == 'markdown':
        with open(output, 'w') as f:
            f.write(f"# {Path(args.file).stem}\n\n")
            meta = data.get('metadata', {})
            f.write(f"**Source:** {meta.get('source_file', 'Unknown')}\n")
            f.write(f"**Created:** {meta.get('created_at', 'Unknown')}\n\n")
            f.write("## Content\n\n")
            for chunk in data.get('chunks', []):
                content = chunk.get('content', '')
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                f.write(content + "\n\n---\n\n")
    
    elif args.format == 'html':
        with open(output, 'w') as f:
            f.write(f"""<!DOCTYPE html>
<html><head><title>{Path(args.file).stem}</title>
<style>body{{font-family:system-ui;max-width:800px;margin:0 auto;padding:20px;background:#0a0a0f;color:#e4e4e7}}
.chunk{{background:#12121a;padding:15px;margin:10px 0;border-radius:8px;border:1px solid #27272a}}</style></head>
<body><h1>{Path(args.file).stem}</h1>""")
            for chunk in data.get('chunks', []):
                content = chunk.get('content', '')
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='replace')
                f.write(f"<div class='chunk'>{content}</div>")
            f.write("</body></html>")
    
    print_success(f"Exported to {output} ({args.format})")

def cmd_models(args):
    """List available embedding models."""
    print_header("AIF-BIN Pro — Available Models")
    
    for key, info in EMBEDDING_MODELS.items():
        print(f"  {Colors.BOLD}{key}{Colors.RESET}")
        print(f"    Name: {info['name']}")
        print(f"    Dimensions: {info['dim']}")
        print(f"    {Colors.DIM}{info['description']}{Colors.RESET}")
        print()

# ============================================================
# INGESTOR — AI PROVIDERS
# ============================================================

import mimetypes
from abc import ABC, abstractmethod

CONFIG_DIR = Path.home() / ".aifbin"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "default_provider": "none",
    "providers": {
        "anthropic": {"api_key": ""},
        "openai": {"api_key": ""},
        "gemini": {"api_key": ""},
        "ollama": {"base_url": "http://localhost:11434", "model": "llama3.2"}
    }
}

def load_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG

def save_config(config: Dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

@dataclass
class ExtractionResult:
    title: str
    summary: str
    tags: List[str]
    chunks: List[Dict[str, str]]
    metadata: Dict[str, Any]

class AIProvider(ABC):
    @abstractmethod
    def extract(self, content: bytes, filename: str, mime_type: str) -> ExtractionResult:
        pass
    
    @abstractmethod
    def is_configured(self) -> bool:
        pass

class AnthropicProvider(AIProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def is_configured(self) -> bool:
        return bool(self.api_key)
    
    def extract(self, content: bytes, filename: str, mime_type: str) -> ExtractionResult:
        import anthropic
        import base64
        client = anthropic.Anthropic(api_key=self.api_key)
        
        if mime_type.startswith('text/') or mime_type in ['application/json', 'application/xml']:
            text_content = content.decode('utf-8', errors='replace')
            message_content = f"Analyze this file named '{filename}':\n\n{text_content[:50000]}"
        elif mime_type.startswith('image/'):
            b64 = base64.b64encode(content).decode('utf-8')
            message_content = [
                {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": b64}},
                {"type": "text", "text": f"Analyze this image named '{filename}'."}
            ]
        else:
            text_content = content.decode('utf-8', errors='replace')[:5000]
            message_content = f"Binary file '{filename}' ({mime_type}):\n\n{text_content}"
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": message_content}],
            system="Extract and return JSON: {title, summary, tags:[], chunks:[{label,content}], metadata:{}}"
        )
        
        try:
            result = json.loads(response.content[0].text)
            return ExtractionResult(result.get('title', filename), result.get('summary', ''),
                result.get('tags', []), result.get('chunks', []), result.get('metadata', {}))
        except:
            return ExtractionResult(filename, response.content[0].text[:500], [], [], {})

class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def is_configured(self) -> bool:
        return bool(self.api_key)
    
    def extract(self, content: bytes, filename: str, mime_type: str) -> ExtractionResult:
        import openai
        import base64
        client = openai.OpenAI(api_key=self.api_key)
        
        if mime_type.startswith('text/'):
            text_content = content.decode('utf-8', errors='replace')[:50000]
            messages = [{"role": "user", "content": f"Analyze '{filename}', return JSON:\n\n{text_content}"}]
        elif mime_type.startswith('image/'):
            b64 = base64.b64encode(content).decode('utf-8')
            messages = [{"role": "user", "content": [
                {"type": "text", "text": f"Analyze image '{filename}', return JSON"},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}
            ]}]
        else:
            text_content = content.decode('utf-8', errors='replace')[:5000]
            messages = [{"role": "user", "content": f"Analyze '{filename}':\n\n{text_content}"}]
        
        response = client.chat.completions.create(model="gpt-4o", messages=messages, response_format={"type": "json_object"})
        result = json.loads(response.choices[0].message.content)
        return ExtractionResult(result.get('title', filename), result.get('summary', ''),
            result.get('tags', []), result.get('chunks', []), result.get('metadata', {}))

class GeminiProvider(AIProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def is_configured(self) -> bool:
        return bool(self.api_key)
    
    def extract(self, content: bytes, filename: str, mime_type: str) -> ExtractionResult:
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"Analyze '{filename}', return JSON: {{title, summary, tags, chunks:[{{label,content}}], metadata}}"
        
        if mime_type.startswith('image/'):
            import PIL.Image, io
            image = PIL.Image.open(io.BytesIO(content))
            response = model.generate_content([prompt, image])
        else:
            text_content = content.decode('utf-8', errors='replace')[:50000]
            response = model.generate_content(f"{prompt}\n\nContent:\n{text_content}")
        
        try:
            text = response.text.strip()
            if text.startswith('```'): text = text.split('\n', 1)[1].rsplit('```', 1)[0]
            result = json.loads(text)
            return ExtractionResult(result.get('title', filename), result.get('summary', ''),
                result.get('tags', []), result.get('chunks', []), result.get('metadata', {}))
        except:
            return ExtractionResult(filename, response.text[:500], [], [], {})

class OllamaProvider(AIProvider):
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
    
    def is_configured(self) -> bool:
        try:
            import requests
            return requests.get(f"{self.base_url}/api/tags", timeout=2).status_code == 200
        except:
            return False
    
    def extract(self, content: bytes, filename: str, mime_type: str) -> ExtractionResult:
        import requests
        text_content = content.decode('utf-8', errors='replace')[:20000]
        prompt = f"Analyze '{filename}', return JSON: {{title, summary, tags, chunks, metadata}}\n\n{text_content}"
        response = requests.post(f"{self.base_url}/api/generate", json={"model": self.model, "prompt": prompt, "stream": False})
        try:
            text = response.json()['response']
            if '{' in text: text = text[text.index('{'):text.rindex('}')+1]
            result = json.loads(text)
            return ExtractionResult(result.get('title', filename), result.get('summary', ''),
                result.get('tags', []), result.get('chunks', []), result.get('metadata', {}))
        except:
            return ExtractionResult(filename, "Extraction failed", [], [], {})

class NoAIProvider(AIProvider):
    def is_configured(self) -> bool:
        return True
    
    def extract(self, content: bytes, filename: str, mime_type: str) -> ExtractionResult:
        text = content.decode('utf-8', errors='replace')
        lines = text.strip().split('\n')
        title = lines[0][:100].lstrip('#').strip() if lines else filename
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = [{"label": f"Section {i+1}", "content": p[:1000]} for i, p in enumerate(paragraphs[:10])]
        return ExtractionResult(title or filename, f"File: {filename} ({len(content)} bytes)", [], chunks, {})

def get_provider(name: str, config: Dict) -> AIProvider:
    pc = config.get('providers', {})
    if name == 'anthropic': return AnthropicProvider(pc.get('anthropic', {}).get('api_key', ''))
    elif name == 'openai': return OpenAIProvider(pc.get('openai', {}).get('api_key', ''))
    elif name == 'gemini': return GeminiProvider(pc.get('gemini', {}).get('api_key', ''))
    elif name == 'ollama':
        cfg = pc.get('ollama', {})
        return OllamaProvider(cfg.get('base_url', 'http://localhost:11434'), cfg.get('model', 'llama3.2'))
    return NoAIProvider()

def get_mime_type(filepath: Path) -> str:
    mime, _ = mimetypes.guess_type(str(filepath))
    if mime: return mime
    ext_map = {'.md': 'text/markdown', '.txt': 'text/plain', '.json': 'application/json',
               '.pdf': 'application/pdf', '.py': 'text/x-python', '.js': 'text/javascript'}
    return ext_map.get(filepath.suffix.lower(), 'application/octet-stream')

def ingest_file(filepath: Path, output_dir: Path, provider: AIProvider) -> Dict[str, Any]:
    with open(filepath, 'rb') as f:
        content = f.read()
    
    mime_type = get_mime_type(filepath)
    extraction = provider.extract(content, filepath.name, mime_type)
    
    # Generate embeddings
    embeddings = []
    if extraction.chunks:
        try:
            model = get_embedding_model('minilm')
            texts = [c.get('content', '')[:512] for c in extraction.chunks]
            embeddings = model.encode(texts, show_progress_bar=False).tolist()
        except:
            embeddings = [None] * len(extraction.chunks)
    
    chunks = []
    for i, chunk_data in enumerate(extraction.chunks):
        chunks.append(ContentChunk(
            chunk_type=ChunkType.TEXT,
            data=chunk_data.get('content', '').encode('utf-8'),
            metadata={'label': chunk_data.get('label', f'Chunk {i}')},
            embedding=embeddings[i] if i < len(embeddings) else None
        ))
    
    if not chunks:
        chunks.append(ContentChunk(chunk_type=ChunkType.TEXT, data=content[:10000], metadata={'label': 'Content'}, embedding=None))
    
    metadata = {
        'source_file': str(filepath), 'title': extraction.title, 'summary': extraction.summary,
        'tags': extraction.tags, 'mime_type': mime_type, 'file_size': len(content),
        'created_at': datetime.now().isoformat(), 'extracted_metadata': extraction.metadata
    }
    
    aifbin = AIFBINFile(metadata=metadata, chunks=chunks, original_raw=content, versions=[])
    output_path = output_dir / f"{filepath.stem}.aif-bin"
    size = write_aifbin_v2(aifbin, str(output_path))
    
    return {'source': str(filepath), 'output': str(output_path), 'title': extraction.title, 'chunks': len(chunks), 'size': size}

def cmd_ingest(args):
    """Ingest any file into AIF-BIN format with optional AI extraction."""
    print_header("AIF-BIN Pro — Ingest")
    
    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if source.is_file():
        files = [source]
    else:
        files = [f for f in source.glob('*.*') if f.is_file() and not f.name.startswith('.')]
    
    if not files:
        print_warning(f"No files found in {source}")
        return
    
    config = load_config()
    provider_name = args.provider or config.get('default_provider', 'none')
    provider = get_provider(provider_name, config)
    
    if provider_name != 'none' and not provider.is_configured():
        print_warning(f"Provider '{provider_name}' not configured. Using basic extraction.")
        print_info(f"Run: aifbin config --provider {provider_name} --api-key YOUR_KEY")
        provider = NoAIProvider()
        provider_name = 'none'
    
    print_info(f"Files: {len(files)}")
    print_info(f"Output: {output_dir}")
    print_info(f"Provider: {provider_name}")
    print()
    
    results = []
    for f in files:
        try:
            result = ingest_file(f, output_dir, provider)
            results.append(result)
            print_file(f.name, f"→ {result['title'][:40]}... ({result['chunks']} chunks)")
        except Exception as e:
            print_error(f"{f.name}: {e}")
    
    print()
    print_success(f"Ingested {len(results)} file(s)")
    print_info(f"Total size: {sum(r['size'] for r in results):,} bytes")

def cmd_config(args):
    """Configure AI providers."""
    config = load_config()
    
    if args.provider and args.api_key:
        if args.provider not in config['providers']:
            config['providers'][args.provider] = {}
        config['providers'][args.provider]['api_key'] = args.api_key
        save_config(config)
        print_success(f"Saved API key for {args.provider}")
        return
    
    if args.default:
        config['default_provider'] = args.default
        save_config(config)
        print_success(f"Default provider: {args.default}")
        return
    
    print_header("AIF-BIN Pro — Configuration")
    print(f"  Config file: {CONFIG_FILE}")
    print(f"  Default provider: {config.get('default_provider', 'none')}")
    print()
    print("  Providers:")
    for name, cfg in config.get('providers', {}).items():
        key = cfg.get('api_key', '')
        status = f"{Colors.GREEN}✓ configured{Colors.RESET}" if key else f"{Colors.DIM}✗ not set{Colors.RESET}"
        print(f"    {name}: {status}")

def cmd_providers(args):
    """List available AI providers."""
    print_header("AIF-BIN Pro — AI Providers")
    print(f"  {Colors.BOLD}anthropic{Colors.RESET}  — Claude (best for documents)")
    print(f"  {Colors.BOLD}openai{Colors.RESET}     — GPT-4 (good all-around)")
    print(f"  {Colors.BOLD}gemini{Colors.RESET}     — Gemini (good for images)")
    print(f"  {Colors.BOLD}ollama{Colors.RESET}     — Local models (free, private)")
    print(f"  {Colors.BOLD}none{Colors.RESET}       — Basic extraction (no AI)")
    print()
    print(f"  {Colors.DIM}Configure: aifbin config --provider anthropic --api-key sk-ant-...{Colors.RESET}")

# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        prog='aifbin',
        description=f'{Colors.BOLD}AIF-BIN Pro{Colors.RESET} — Professional AI Memory Management (v2 Binary Format)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--version', action='version', version='AIF-BIN Pro 1.0.0')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # migrate
    p_migrate = subparsers.add_parser('migrate', help='Convert files to AIF-BIN (v2 binary)')
    p_migrate.add_argument('source', help='Source file or directory')
    p_migrate.add_argument('-o', '--output', required=True, help='Output directory')
    p_migrate.add_argument('-m', '--model', default='minilm', choices=EMBEDDING_MODELS.keys(), help='Embedding model')
    p_migrate.add_argument('-r', '--recursive', action='store_true', help='Process subdirectories')
    p_migrate.add_argument('-p', '--parallel', action='store_true', help='Parallel processing')
    p_migrate.add_argument('-w', '--workers', type=int, default=4, help='Number of workers')
    
    # search
    p_search = subparsers.add_parser('search', help='Semantic search')
    p_search.add_argument('query', help='Search query')
    p_search.add_argument('-d', '--dir', required=True, help='Directory with .aif-bin files')
    p_search.add_argument('-k', type=int, default=5, help='Number of results')
    p_search.add_argument('-m', '--model', default='minilm', choices=EMBEDDING_MODELS.keys(), help='Embedding model')
    p_search.add_argument('--after', help='Filter: created after date (YYYY-MM-DD)')
    
    # info
    p_info = subparsers.add_parser('info', help='Show file info')
    p_info.add_argument('file', help='AIF-BIN file')
    
    # extract
    p_extract = subparsers.add_parser('extract', help='Extract original content')
    p_extract.add_argument('file', help='AIF-BIN file')
    p_extract.add_argument('-o', '--output', help='Output file (default: stdout)')
    
    # watch
    p_watch = subparsers.add_parser('watch', help='Watch and auto-sync')
    p_watch.add_argument('source', help='Source directory')
    p_watch.add_argument('-o', '--output', required=True, help='Output directory')
    p_watch.add_argument('-m', '--model', default='minilm', choices=EMBEDDING_MODELS.keys(), help='Embedding model')
    p_watch.add_argument('-i', '--interval', type=int, default=5, help='Check interval (seconds)')
    
    # diff
    p_diff = subparsers.add_parser('diff', help='Compare two files')
    p_diff.add_argument('file1', help='First AIF-BIN file')
    p_diff.add_argument('file2', help='Second AIF-BIN file')
    p_diff.add_argument('-v', '--verbose', action='store_true', help='Show chunk details')
    
    # export
    p_export = subparsers.add_parser('export', help='Export to other formats')
    p_export.add_argument('file', help='AIF-BIN file')
    p_export.add_argument('-o', '--output', required=True, help='Output file')
    p_export.add_argument('-f', '--format', choices=['json', 'csv', 'markdown', 'html'], default='json', help='Export format')
    
    # models
    p_models = subparsers.add_parser('models', help='List available models')
    
    # ingest
    p_ingest = subparsers.add_parser('ingest', help='Convert any file to AIF-BIN with AI')
    p_ingest.add_argument('source', help='File or directory to ingest')
    p_ingest.add_argument('-o', '--output', required=True, help='Output directory')
    p_ingest.add_argument('-p', '--provider', choices=['anthropic', 'openai', 'gemini', 'ollama', 'none'], help='AI provider')
    
    # config
    p_config = subparsers.add_parser('config', help='Configure AI providers')
    p_config.add_argument('--provider', help='Provider name')
    p_config.add_argument('--api-key', help='API key')
    p_config.add_argument('--default', help='Set default provider')
    
    # providers
    p_providers = subparsers.add_parser('providers', help='List AI providers')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        print()
        print(f"{Colors.CYAN}AIF-BIN Pro — Local-first AI memory management{Colors.RESET}")
        print(f"{Colors.DIM}Format: v2 Binary (compatible with Web Inspector){Colors.RESET}")
        print(f"{Colors.DIM}https://aifbin.dev{Colors.RESET}")
        return
    
    commands = {
        'migrate': cmd_migrate,
        'search': cmd_search,
        'info': cmd_info,
        'extract': cmd_extract,
        'watch': cmd_watch,
        'diff': cmd_diff,
        'export': cmd_export,
        'models': cmd_models,
        'ingest': cmd_ingest,
        'config': cmd_config,
        'providers': cmd_providers,
    }
    
    commands[args.command](args)

if __name__ == '__main__':
    main()
