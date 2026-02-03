# Bot-BIN

**Persistent Semantic Memory for AI Chatbots**

[![Patent Pending](https://img.shields.io/badge/Patent-Pending-blue.svg)](https://terronex.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Website](https://img.shields.io/badge/website-terronex.dev-cyan.svg)](https://terronex.dev)

---

Bot-BIN gives your AI chatbot persistent semantic memory. It converts markdown memory files into AIF-BIN format with vector embeddings, enabling search by meaning instead of keywords.

## How It Works

```
memory/2026-02-02.md  -->  [Bot-BIN Sync]  -->  memory/aifbin/2026-02-02.aif-bin
       |                         |                        |
   Raw markdown            Embeddings              Searchable vectors
                           (384-dim)
```

1. You write notes to `memory/*.md` files during sessions
2. Bot-BIN syncs changed files to `.aif-bin` with vector embeddings
3. Semantic search finds relevant past context by meaning
4. Your bot can recall "what did we decide about X" even across sessions

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Terronex-dev/bot-bin.git
cd bot-bin
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create memory directory

```bash
mkdir -p memory
```

---

## Commands

### Sync

Converts changed `.md` files to `.aif-bin` with embeddings.

```bash
python3 botbin.py sync
```

**Output:**
```
Checking 5 memory files...
  Syncing: 2026-02-02.md

Synced: 1 files
   - 2026-02-02.md
Skipped (unchanged): 4 files
```

### Search

Search your memory files by meaning.

```bash
python3 botbin.py search "what decisions did we make"
python3 botbin.py search "project timeline" -k 10
```

**Output:**
```
Query: "what decisions did we make"

#1 [0.542] 2026-02-02.aif-bin
   We decided to use the new API architecture. Budget approved for Q2...

#2 [0.387] 2026-01-28.aif-bin
   Decision: Switch to Tauri for the desktop app...
```

### Status

Show sync status and statistics.

```bash
python3 botbin.py status
```

### Info

Show details about an AIF-BIN file.

```bash
python3 botbin.py info memory/aifbin/2026-02-02.aif-bin
```

### Extract

Recover original markdown from an AIF-BIN file.

```bash
python3 botbin.py extract memory/aifbin/2026-02-02.aif-bin
```

---

## Directory Structure

```
your-workspace/
├── bot-bin/
│   ├── botbin.py           # Main script
│   ├── aifbin_pro.py       # AIF-BIN engine
│   ├── aifbin_spec_v2.py   # Binary format spec
│   └── requirements.txt
├── memory/
│   ├── 2026-02-02.md       # Your daily notes
│   └── aifbin/
│       └── 2026-02-02.aif-bin   # Synced with embeddings
└── MEMORY.md               # Long-term memory (also synced)
```

---

## Tracked Files

Bot-BIN automatically tracks:

- `MEMORY.md` in workspace root (if it exists)
- All `*.md` files in `memory/` directory

---

## Embedding Model

Uses `all-MiniLM-L6-v2` (384 dimensions) by default:

- Fast inference (~10ms per chunk)
- Good quality for document retrieval
- ~90MB model download on first run
- Cached locally after first use

---

## Related Projects

| Project | Description |
|---------|-------------|
| [AIF-BIN Lite](https://github.com/Terronex-dev/aifbin-lite) | Core library — create, read, convert |
| [AIF-BIN Pro](https://github.com/Terronex-dev/aifbin-pro) | CLI with semantic search, batch ingest, watch mode |

---

## License

MIT License. See [LICENSE](LICENSE) file.

---

© 2026 [Terronex.dev](https://terronex.dev) — Patent Pending
