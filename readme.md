# ⚙️ Jarvis L (50)
### The Ultimate Cross-Platform Personal AI Assistant — By Shrihan



A real-time voice AI that can hear, see, understand, and control your computer — on any OS. Supports Windows, macOS, and Linux. Built on the Gemini Live API for native audio streaming, delivering zero subscriptions and total digital autonomy.

---

## ✨ Overview

Jarvis L is where the assistant stops being a tool and starts being a presence. It remembers yesterday's conversation, watches the topics you care about, and speaks first when it has something worth saying. The goal of this build was continuity — JARVIS should feel like it never fully left, even after you close it.

It's not just an assistant — it's an extension of your digital life.

---

## 🚀 Capabilities

### Core Features
| Feature | Description |
|---|---|
| 🎙️ Real-time Voice | Ultra-low latency conversation in any language via Gemini Live API |
| 🖥️ System Control | Launch apps, adjust volume/brightness, WiFi, shortcuts, power — all by voice |
| 🧩 Autonomous Tasks | High-level planning for complex multi-step goals via agent mode |
| 👁️ Visual Awareness | Real-time screen capture and webcam vision piped into your main Gemini session |
| 🧠 Persistent Memory | Deeply remembers projects, preferences, and personal context across sessions |
| ⌨️ Hybrid Input | Seamlessly switch between keyboard typing and voice commands |
| 🌅 Morning Briefing | On first boot: greets you, reads the time, recaps yesterday, and fetches live news |
| 🔔 Proactive 2.0 | Time-aware, context-aware check-ins — knows the time of day, your projects, and what you've been discussing |
| 🗓️ Session Memory | Summarises each conversation and mentions it naturally next morning — consumed after use, never repeats |
| 👁️‍🗨️ Background Monitoring | User-configured topic watching — checks for new headlines once a day and alerts naturally |
| 📊 Hardware Monitoring | Continuous CPU, RAM, GPU and temperature telemetry with localized voice alerts |
| 🌤️ Weather Report | Live weather data for your city, personalized from memory |
| 🗺️ Dynamic Content Panel | Scrollable display layer beneath the HUD that renders web results, news, and search data |
| 🔍 Multi-Mode Web Search | `news` / `research` / `price` / `compare` / `search` — Gemini Grounded first, DDG fallback |
| ⏰ Smart Reminders | OS-native scheduled notifications (Windows Task Scheduler / macOS LaunchAgent / Linux systemd) |
| ✈️ Flight Finder | Live flight price and availability lookup |
| 🎮 Game Updater | Checks and triggers game updates on Steam and Epic Games on demand |
| 📂 File Processor | Read, summarize, and answer questions about local files |
| 💻 Code Helper | Inline code review, debugging, and generation |
| 🌐 Browser Control | Open URLs, navigate tabs, and interact with the browser by voice |
| 📨 Send Message | Compose and send messages through WhatsApp, Telegram, and more |
| 🎬 YouTube Control | Search, play, and control YouTube playback by voice |
| 🖱️ Desktop Control | Taskbar, window management, and desktop-level operations |
| 🧑‍💻 Silent Language Memory | Detects spoken language on first use — all future sessions adapt automatically |
| 📱 Remote Dashboard | Control the assistant from your phone via QR code pairing |
| ⚡ Auto-Start on Boot | Registers with the OS startup system (registry / LaunchAgent / .desktop) |
| 📋 Clipboard Intelligence | Copy any text → floating panel with Translate / Summarise / Explain / Fix |
| 🎨 Assistant Customization | Change the assistant name and your name from the UI — takes effect immediately |
| 🔁 Workflow Automation | Record, save, replay desktop workflows by voice (clicks, typing, hotkeys, app launches) |
| 🌐 Network Toolkit | WiFi control, port scan, connectivity checks, active connections, password retrieval |
| 🧹 System Optimizer | Clean temp files/caches, manage startup programs, kill processes, empty trash, disk usage |

---




## 🆕 What's New in Jarvis L

### 🗓️ Session Memory — JARVIS Remembers Yesterday
At the end of every session, JARVIS generates a 1-2 sentence summary of what was discussed and saves it to memory. The next morning, it's mentioned naturally in the briefing:
> *"Good morning, sir — it's 09:15. Yesterday you were working on the Jarvis L background monitoring feature. Fetching today's headlines now."*

The summary is consumed immediately after use — it never repeats in future briefings and adds zero long-term bloat to memory.

### 👁️‍🗨️ Background Monitoring — JARVIS Watches While You're Away
Tell JARVIS to monitor any topic and it checks for new developments once a day using DuckDuckGo news. When a headline changes, it reports back naturally in your language:
> *"Efendim, takip ettiğiniz yapay zeka haberlerinde bir gelişme var: Google yeni bir model duyurdu."*

Fully opt-in — JARVIS monitors nothing without being explicitly asked. Crypto, financial, and trading topics are blocked at the code level regardless of what is requested. Same headline never triggers twice.

### 🔔 Proactive System 2.0 — Context-Aware, Time-Aware, Non-Repetitive
The proactive engine was rebuilt from the ground up. Instead of a generic check-in after 15 minutes of silence, JARVIS now:
- Knows the **time of day** — morning tone differs from evening tone
- Knows your **active projects** from memory and can ask how something is going
- Knows your **monitored topics** and can bring one up naturally
- Knows **what you were just talking about** (last 8 conversation turns)
- **Rotates** between three focus areas so it never opens with the same line twice
- Has a 20-minute cooldown (up from 10) — less intrusive, more meaningful

### 👁️ Instant Vision Acknowledgment — No More Silent Waiting
When you ask JARVIS to look at your screen or camera, it no longer goes silent while processing. It immediately says something natural ("Looking at your screen now, sir" / "Ekrana bakıyorum efendim") while the capture runs. The actual analysis follows as the next response.

### 📰 Parallel News Search — First Result Wins
News queries now run Gemini Grounded Search and DuckDuckGo news simultaneously in two threads. Whichever delivers a valid result first is used; the other is silently discarded. A Gemini 503 error no longer delays results — the DDG fallback is already running in parallel.

### 🔁 Workflow Automation — Record & Replay Desktop Tasks
Tell JARVIS to record a workflow and it captures your mouse movements, clicks, typing, hotkeys, and app launches into a repeatable sequence. Save it with a name and replay it anytime with a single voice command. Great for morning routines, repetitive data entry, or any multi-step desktop task.

### 🌐 Network Toolkit — WiFi & Diagnostics by Voice
Full network control: list saved WiFi profiles, connect to new networks, retrieve saved WiFi passwords, check internet connectivity, scan ports, ping hosts, and view active connections. All from natural voice commands.

### 🧹 System Optimizer — Voice-Activated Maintenance
Keep your PC healthy: clean temp files and caches, list and disable startup programs, kill runaway processes, empty the recycle bin, find the largest files, and check disk usage. JARVIS reports exactly how much space was freed.

---



## 🗺️ Jarvis Roadmap

| Jarvis | Focus |
|---|---|
| **XLVIII** | Instant interrupt · parallel news · two-phase briefing · exponential backoff · vision cooldown |
| **XLIX** | Auto-start · clipboard intelligence · assistant customization |
| **L** | Session memory · background monitoring · proactive 2.0 · instant vision · parallel news search |
| **LI+** | Plugin system · email · quiz mode · calorie counter · calendar |

---

## 🚀 What's New — Advanced Upgrade

The plugin architecture (already present in `core/`) is now **fully wired into the
live assistant**: every `*.py` in `plugins/` with a `PLUGIN` dict becomes a
first-class tool the AI can call by voice. Drop in a new skill and it is picked up
on the next launch (hot-reload enabled, 3s poll). The model sees per-plugin tool
declarations generated automatically from each plugin's metadata.

Four advanced, production-ready plugins are included:

| Plugin | What it does | Try saying |
|---|---|---|
| 🧠 **Quiz Mode** (`quiz_mode.py`) | Generates a graded quiz on any topic via the local LLM, scores your answers, tracks history | *"Quiz me on World War 2"* → *"grade my quiz"* |
| 📅 **Calendar** (`calendar_integration.py`) | Real offline event store (no OAuth) — add / list / delete / today / upcoming | *"Add dentist on 2026-09-01 at 10:00"* |
| 📨 **Email** (`email_integration.py`) | Sends & reads email over SMTP/IMAP with an app password (stdlib only, no SDK) | *"Send email to x@y.com"* / *"check my inbox"* |
| 💪 **Habit Tracker** (`habit_tracker.py`) | Logs water, calories, workouts, mood, sleep; shows daily totals, streaks, summaries | *"log 2 water"* / *"my habit summary"* |
| 🧩 **Plugin Manager** (`plugin_manager_skill.py`) | List / enable / disable installed plugins by voice | *"list plugins"* / *"disable plugin quiz"* |
| 🌐 **Translator** (`translator.py`) | Translates text between languages (local LLM + Gemini fallback) | *"translate 'hello' to French"* |
| 📈 **Markets** (`markets.py`) | Live crypto prices & conversions via CoinGecko (no API key) | *"price of bitcoin"* / *"2 eth to usd"* |
| 📚 **Wikipedia** (`wikipedia.py`) | Instant encyclopedia summaries (free REST API) | *"tell me about black holes"* |
| 🛠️ **Utilities** (`utils.py`) | Offline password generator, safe calculator, unit converter | *"generate a password"* / *"convert 10 km to mi"* |

To add a provider email account, create `config/email_config.json` (see the
`email` plugin's `setup` action) — no OAuth provider SDK required.

> The plugin system, calendar, and habit tracker run fully offline. Quiz mode
> needs a local LLM (Ollama / OpenAI-compatible) configured in
> `config/api_keys.json`; email needs an app password. Both degrade gracefully
> with a clear spoken message if unavailable.

---

## 🛰️ Jarvis 6.0 — Clap-to-Wake, Sentinel & God-Tier Power Tools

The camera/gesture control was **removed**. JARVIS now wakes completely hands-free,
purely from the microphone, and keeps itself running without manual `python main.py`.

### 🤚➜🎙️ Clap to Activate (no camera, no hand tracking)
```
(asleep) ──clap twice──▶ ARMED (beep + orange orb) ──say "wake up"──▶ LISTENING
```
- **`ClapDetector`** keys on *attack sharpness + bright spectrum + fast decay*, so
  speech, music, typing and single/slapped claps are rejected — not plain loudness.
- **`WakePhraseDetector`** confirms the phrase via an offline chain:
  **Vosk (exact) → Porcupine → energy VAD fallback**. Vosk model is pre-downloaded
  once with `python sentinel.py --install-vosk`; nothing is fetched at startup.
- Config (`config/__init__.py`): `CLAP_ENABLED`, `CLAP_SENSITIVITY`, `CLAP_COUNT`,
  `CLAP_WINDOW`, `CLAP_COOLDOWN`, `WAKE_TIMEOUT`, `WAKE_WORDS`, `WAKE_REQUIRE_CLAP`,
  `WAKE_BEEP`. The camera stays only for vision/face recognition, never to wake.

### 🛡️ Sentinel — Boot & Keep-Alive Daemon
- `sentinel.py` auto-launches JARVIS at **Windows login** and restarts it if it ever
  exits (pauses itself when JARVIS is already running, hands off the wake request
  file on launch).
- `install_autostart.bat` drops a Startup shortcut + installs `sounddevice`/Vosk.
  `uninstall_autostart.bat` removes it. `Sentinel.bat` / `Clap-Test.bat` for manual use.

### 🦾 Power Tools — JARVIS actually does things on the PC
`power_tools` action covers clipboard (get/set/append/history), winget app
install/update/remove, window focus/min/max/close, top processes + kill, lightning-fast
file search across the PC, Windows services + scheduled tasks, environment variables,
and power state (shutdown/restart/sleep/lock). **Destructive actions require God Mode.**

### 🧠 Smarter Agents & Brain
- `core/brain.py` unifies local + Gemini LLM with caching and a `preferred_backend`
  selector; `core/agent_memory.py` records every mission so routing and quality improve.
- `core/agent_manager.py` adds `orchestrate` (review → revise), `orchestrate_parallel`,
  `plan_mission`/`run_mission` (coordinated squad), and `review_agent_output` (structural
  + LLM critique). New tools: **`run_mission`** (big goals split into a squad) and
  **`power_tools`** (PC control) are wired into `main.py`.

---

> ⚠️ **Installation Note:** Some OS-specific dependencies are not bundled in `requirements.txt` to keep the repo lightweight. If you hit a `ModuleNotFoundError`, install the missing package with `pip install <module_name>`.

---

## 📋 Requirements

| Requirement | Details |
| --- | --- |
| **OS** | Windows 10/11, macOS, or Linux |
| **Python** | 3.11 or 3.12 |
| **Microphone** | Required for voice interaction |
| **API Key** | Free Gemini API key (`config/api_keys.json`) |

---

## 🗂️ Project Structure

```
Jarvis L/
├── main.py                   # Core loop — Gemini Live session, audio I/O, tool dispatch
├── ui.py                     # PyQt6 HUD — waveform, log panel, interrupt button, camera feed
├── setup.py                  # First-run configuration wizard
├── actions/
│   ├── web_search.py         # Gemini + DDG parallel search (news, research, price, compare)
│   ├── screen_processor.py   # Screen capture & webcam vision via Gemini Live
│   ├── background_monitor.py # User-configured topic watching — daily DDG check, no crypto
│   ├── proactive.py          # Proactive 2.0 — time/context/rotation-aware check-ins
│   ├── reminder.py           # OS-native scheduled notifications
│   ├── system_monitor.py     # CPU / RAM / GPU / temperature telemetry
│   ├── computer_settings.py  # Volume, brightness, WiFi, power
│   ├── computer_control.py   # Keyboard shortcuts, mouse, window management
│   ├── open_app.py           # Application launcher
│   ├── browser_control.py    # Web browser control
│   ├── file_controller.py    # File system operations
│   ├── file_processor.py     # Document reading and summarization
│   ├── send_message.py       # Messaging integration
│   ├── weather_report.py     # Live weather data
│   ├── flight_finder.py      # Flight search
│   ├── youtube_video.py      # YouTube playback control
│   ├── game_updater.py       # Game update management (Steam / Epic)
│   ├── code_helper.py        # Code review and generation
│   ├── dev_agent.py          # Developer task agent
│   ├── desktop.py            # Desktop and taskbar control
│   ├── workflow.py           # Record, save, replay desktop workflows
│   ├── network_toolkit.py    # WiFi control, port scan, connectivity checks
│   └── system_optimizer.py   # Temp/cache cleanup, startup management, process control
├── memory/
│   ├── memory_manager.py     # Load/save long_term.json — sessions, monitors, identity
│   └── long_term.json        # Persistent store: identity, preferences, projects, sessions, monitors
├── core/
│   └── prompt.txt            # Assistant personality and tool-routing rules
└── config/
    └── api_keys.json         # API key, OS setting, assistant name, user name
```

---

## ⚠️ License

Personal and non-commercial use only.
Licensed under **[Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**.







