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

#### Windows (PowerShell/CMD)

```powershell
pip install -r requirements.txt
mkdir memory
```

#### macOS

```bash
pip3 install -r requirements.txt
mkdir -p memory
```

#### Linux (Ubuntu/Debian)

```bash
pip3 install -r requirements.txt
mkdir -p memory
```

#### WSL (Ubuntu)

```bash
pip install -r requirements.txt --break-system-packages
mkdir -p memory
```

> **Note:** WSL Ubuntu uses externally-managed Python. The `--break-system-packages` flag is required, or use a virtual environment.

---

## Usage

### Windows (PowerShell/CMD)

```powershell
python botbin.py sync                              # Sync markdown to AIF-BIN
python botbin.py search "what decisions did we make"   # Semantic search
python botbin.py status                            # Show sync status
python botbin.py info memory\aifbin\file.aif-bin   # File details
python botbin.py extract memory\aifbin\file.aif-bin    # Extract original
```

### macOS / Linux / WSL

```bash
python3 botbin.py sync                              # Sync markdown to AIF-BIN
python3 botbin.py search "what decisions did we make"   # Semantic search
python3 botbin.py status                            # Show sync status
python3 botbin.py info memory/aifbin/file.aif-bin   # File details
python3 botbin.py extract memory/aifbin/file.aif-bin    # Extract original
```

---

## Commands

### Sync

Converts changed `.md` files to `.aif-bin` with embeddings.

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

### Info

Show details about an AIF-BIN file.

### Extract

Recover original markdown from an AIF-BIN file.

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

## Troubleshooting

### Windows

**"python is not recognized"**
- Install Python from [python.org](https://www.python.org/downloads/)
- Make sure to check "Add Python to PATH" during installation

**"pip is not recognized"**
- Run `python -m pip install -r requirements.txt` instead

### macOS

**"externally-managed-environment" error**
- Use `pip3 install --user -r requirements.txt`
- Or create a virtual environment: `python3 -m venv venv && source venv/bin/activate`

### Linux / WSL

**"externally-managed-environment" error (Debian/Ubuntu)**
- Use `pip install -r requirements.txt --break-system-packages`
- Or create a virtual environment: `python3 -m venv venv && source venv/bin/activate`

**Permission denied**
- Use `pip install --user -r requirements.txt`

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
