# Bot BIN

**Semantic Memory for Clawdbot**

Bot BIN gives your Clawdbot instance persistent semantic memory. It converts your markdown memory files into AIF-BIN format with vector embeddings, enabling search by meaning instead of keywords.

---

## How It Works

```
memory/2026-02-02.md  -->  [Bot BIN Sync]  -->  memory/aifbin/2026-02-02.aif-bin
       |                         |                        |
   Raw markdown            Embeddings              Searchable vectors
                           (384-dim)
```

1. You write notes to `memory/*.md` files during sessions
2. Bot BIN syncs changed files to `.aif-bin` with vector embeddings
3. Semantic search finds relevant past context by meaning
4. Your bot can recall "what did we decide about X" even across sessions

---

## Installation

### 1. Clone to your Clawdbot workspace

```bash
cd /path/to/your/clawd-workspace
git clone https://github.com/terronexdev/bot-bin.git botbin
```

### 2. Install dependencies

**Windows:**
```powershell
pip install -r botbin/requirements.txt
```

**Linux / macOS / WSL:**
```bash
pip3 install -r botbin/requirements.txt
```

### 3. Create memory directory

```bash
mkdir -p memory
```

### 4. Add to HEARTBEAT.md

Add this to your `HEARTBEAT.md` to auto-sync on every heartbeat:

```markdown
## Memory Sync (every heartbeat)
1. Run `python3 botbin/botbin.py sync` to sync changed .md files to .aif-bin
2. Only outputs if files were synced (silent if no changes)
```

---

## Commands

### Sync (Convert Memory Files)

Converts changed `.md` files to `.aif-bin` with embeddings.

**Windows:**
```powershell
python botbin/botbin.py sync
```

**Linux / macOS / WSL:**
```bash
python3 botbin/botbin.py sync
```

**Output:**
```
Checking 5 memory files...
  Syncing: 2026-02-02.md

Synced: 1 files
   - 2026-02-02.md
Skipped (unchanged): 4 files
```

---

### Search (Semantic Query)

Search your memory files by meaning.

**Windows:**
```powershell
python botbin/botbin.py search "what decisions did we make"
python botbin/botbin.py search "project timeline" -k 10
```

**Linux / macOS / WSL:**
```bash
python3 botbin/botbin.py search "what decisions did we make"
python3 botbin/botbin.py search "project timeline" -k 10
```

**Output:**
```
════════════════════════════════════════════════════════════
  Bot BIN — Semantic Search
════════════════════════════════════════════════════════════

Query: "what decisions did we make"
Searching in: memory/aifbin

#1 [0.542] 2026-02-02.aif-bin
   We decided to use the new API architecture. Budget approved for Q2...

#2 [0.387] 2026-01-28.aif-bin
   Decision: Switch to Tauri for the desktop app...
```

---

### Status

Show sync status and statistics.

**Windows:**
```powershell
python botbin/botbin.py status
```

**Linux / macOS / WSL:**
```bash
python3 botbin/botbin.py status
```

**Output:**
```
════════════════════════════════════════════════════════════
  Bot BIN — Status
════════════════════════════════════════════════════════════

Workspace: /home/user/clawd
Memory dir: /home/user/clawd/memory
AIF-BIN dir: /home/user/clawd/memory/aifbin

Tracked .md files: 5
Synced files: 5
Last sync: 2026-02-02T15:30:00
AIF-BIN files: 5 (125,000 bytes)
```

---

### Info

Show details about an AIF-BIN file.

**Windows:**
```powershell
python botbin/botbin.py info 2026-02-02.aif-bin
```

**Linux / macOS / WSL:**
```bash
python3 botbin/botbin.py info 2026-02-02.aif-bin
```

---

### Extract

Recover original markdown from an AIF-BIN file.

**Windows:**
```powershell
python botbin/botbin.py extract 2026-02-02.aif-bin
```

**Linux / macOS / WSL:**
```bash
python3 botbin/botbin.py extract 2026-02-02.aif-bin
```

---

## Directory Structure

After setup, your workspace will look like:

```
clawd-workspace/
├── botbin/
│   ├── botbin.py           # Main script
│   ├── aifbin_pro.py       # AIF-BIN Pro CLI
│   ├── aifbin_spec_v2.py   # Binary format spec
│   ├── requirements.txt
│   └── README.md
├── memory/
│   ├── 2026-02-02.md       # Your daily notes
│   ├── 2026-02-01.md
│   └── aifbin/
│       ├── 2026-02-02.aif-bin   # Synced with embeddings
│       ├── 2026-02-01.aif-bin
│       └── botbin_sync_state.json
├── MEMORY.md               # Long-term memory (also synced)
└── HEARTBEAT.md            # Contains sync command
```

---

## Tracked Files

Bot BIN automatically tracks:

- `MEMORY.md` in workspace root (if it exists)
- All `*.md` files in `memory/` directory

To add more locations, edit the `get_md_files()` function in `botbin.py`.

---

## Embedding Model

Bot BIN uses `all-MiniLM-L6-v2` (384 dimensions) by default:

- Fast inference (~10ms per chunk)
- Good quality for document retrieval
- ~90MB model download on first run
- Cached locally after first use

---

## Integration with Clawdbot

### Using memory_search

If your Clawdbot has the `memory_search` tool, Bot BIN files are automatically searchable. The semantic search will find relevant content from your `.aif-bin` files.

### Manual Search in Responses

You can also search directly in your bot's responses:

```python
# In your bot's code or prompts
result = exec("python3 botbin/botbin.py search 'pricing decisions'")
```

---

## Troubleshooting

### First run is slow

The embedding model (~90MB) downloads on first use. Subsequent runs are fast.

### ModuleNotFoundError

Install dependencies:
```bash
pip install -r botbin/requirements.txt
```

### No files synced

Make sure you have `.md` files in your `memory/` directory or a `MEMORY.md` in your workspace root.

### Search returns no results

Run `sync` first to create the `.aif-bin` files with embeddings.

---

## Related Projects

- **AIF-BIN Pro** — Full CLI with more features
- **AIF-BIN Studio** — Desktop app with AI Chat
- **AIF-BIN Lite** — Free format-only CLI

---

## License

MIT License. See LICENSE file.

---

(c) 2026 Terronex.dev
