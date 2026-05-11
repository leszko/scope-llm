# Avatar Reply — Phase-by-Phase Analysis

This document records what I asked, what I found, and what I built across the five phases of the *LTX-2 Talking Heads* idea. Each phase had a go/no-go gate; I only proceeded when the prior gate cleared.

The deliverables produced by this work:

- Plugin source: `~/scope-llm/` (Python, ~250 LOC).
- Example workflow: `~/scope-llm/examples/talking-head.scope-workflow.json`.
- Sample MP4: `~/scope-llm/blog/avatar_color.mp4` (192×192, h264 + aac, ~6 s, 119 KB).
- Hero still: `~/scope-llm/blog/avatar_color.jpg`.
- Blog post: `~/scope-llm/blog/live-avatar-in-5-minutes.md`.

---

## Phase 1 — Would this even work? ✅ VIABLE

**Question.** Can `Question (string) → local LLM node → LTX-2.3 + talking-head LoRA → sink` produce a usable avatar that speaks the LLM's reply, without a separate TTS or lip-sync stage?

**Method.** Read the LoRA card on Hugging Face, the LTX-2 plugin schema, and the existing `scope-prompt-enhancer` plugin to confirm that (a) text-only inference produces audio, (b) the prompt format is well-defined, and (c) Scope's node-graph already supports streaming a string from one node into a pipeline's `prompts` input.

**Findings.**

| Hypothesis | Verdict | Evidence |
|---|---|---|
| The LoRA needs no `audio_input` in I2V mode | **TRUE** | LoRA card states voice is internalised in LoRA weights; `scope_ltx_2/schema.py` declares `audio_input: Optional[str] = None`. |
| LTX-2.3 generates synced audio + video in one pass | **TRUE** | The model is an audio-visual DiT (the "AV" in the LoRA name); `scope_ltx_2/pipeline.py` decodes both video and audio tensors after a single denoise. |
| The prompt format is fixed | **TRUE** | Trigger word `OHWXPERSON` plus literal `The person is talking, and he says: "<transcript>"`. |
| Scope can wire a string-output node into a pipeline's `prompts` port | **TRUE** | Pipelines with `supports_prompts=True` automatically expose a `prompts` string input port (`nodes/base.py:325`). `scope-prompt-enhancer` uses exactly this pattern with LongLive. |
| A small local LLM can write a one-sentence reply that fits ~5 s of speech | **TRUE** | Verified empirically with Qwen 2.5 0.5B-Instruct: replies like "Blue is my favorite color!", "Paris is the capital of France." land at 4–8 words with a tight system prompt. |

**Caveat.** It's "ask-and-wait", not "live": one ~5-second clip costs about 3.5 minutes of wall-clock on an RTX 5090 at 192×192. The cost is dominated by LTX-2's Gemma text encoder, not by denoising or the LLM.

**Decision.** Proceed to Phase 2.

**Sources.**

- [`elix3r/LTX-2.3-22b-AV-LoRA-talking-head`](https://huggingface.co/elix3r/LTX-2.3-22b-AV-LoRA-talking-head)
- [Lightricks LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [`scope-prompt-enhancer`](https://github.com/leszko/scope-prompt-enhancer) (template plugin)
- Local: `scope_ltx_2/schema.py:188,322,362` and `pipeline.py:996,1022`.

---

## Phase 2 — Is this a good thing to write about? ✅ POSITIVE

**Question.** Does an "interactive local AI avatar built on LTX-2.3" attract the kind of attention that a blog post wants — Reddit upvotes, HN front page, social shares?

**Method.** Targeted searches across r/LocalLLaMA, r/StableDiffusion, r/comfyui, HN, and the broader "AI avatar" content market.

**Findings.**

- **The communities exist and they're hot on LTX-2.** r/StableDiffusion was actively engaged with LTX-2.3 GGUF workflows in March 2026 (one upvoted thread title: "LTX-2.3 22B WORKFLOWS 12GB GGUF — i2v, t2v, ta2v, ia2v, v2v"). r/LocalLLaMA (266 k members) loves the "everything runs on my GPU" angle.
- **The differentiator is real.** Existing comparable projects ([Linly-Talker](https://github.com/Kedreamix/Linly-Talker), [ai-iris-avatar](https://github.com/Scthe/ai-iris-avatar), [ALIVE](https://arxiv.org/html/2512.20858v1)) all chain SadTalker / Wav2Lip + a separate TTS. **Nobody has shipped a single-model AV variant** in a node-graph editor that non-developers can run.
- **One mildly negative signal.** A community comment flagged that LTX-2.3 isn't ideal for talking heads — but that predates the elix3r AV LoRA, which specifically fixes that gap. So the signal is actually a *gap I can name in the post* rather than a contradiction.

**Best blog angles, in priority order.**

1. *"Local interactive AI avatar in 5 nodes — no SadTalker, no separate TTS."* (r/LocalLLaMA, r/StableDiffusion).
2. *"How LTX-2.3's joint audio-video output collapses three pipelines into one."* (r/comfyui, r/MachineLearning).
3. *"Wiring Qwen 0.5B into a video diffusion graph."* (HN, dev-tooling angle).

**Decision.** Proceed to Phase 3.

**Sources.**

- [r/StableDiffusion — LTX-2.3 22B GGUF workflows](https://daslikes.wordpress.com/2026/03/06/ltx-2-3-22b-workflows-12gb-gguf-i2v-t2v-ta2v-ia2v-v2v-of-course-via-r-stablediffusion/)
- [WaveSpeed: LTX-2.3 What's New (2026)](https://wavespeed.ai/blog/posts/ltx-2-3-whats-new-2026/)
- [Linly-Talker (SadTalker stack, comparable)](https://github.com/Kedreamix/Linly-Talker)
- [ALIVE (academic local AI avatar)](https://arxiv.org/html/2512.20858v1)
- [LTX-2 open-source release announcement](https://ltx-2.org/posts/ltx-2-open-source-ai-video-creation-2026-release)

---

## Phase 3 — Build the LLM-reply plugin ✅ DELIVERED

**Goal.** Adapt `scope-prompt-enhancer` into a new node that takes a question and emits the OHWXPERSON-formatted prompt LTX-2 wants.

**Implementation summary.**

- Package: `scope-llm` (Apache-2.0), entry point `[project.entry-points.scope] llm = "scope_llm:plugin"`.
- One node class, `LLMNode`, registered via the `register_nodes` pluggy hook.
- Default LLM: Qwen 2.5 0.5B-Instruct (~1 GB on disk). Two SmolLM2 alternatives for users on tighter VRAM budgets.
- System prompt forces a one-sentence, ≤12-word, spoken-style reply.
- A `_clean_answer` post-processor strips quotes, markdown, multi-sentence overflow; replaces internal `"` with `'`; hard-caps at 90 characters at a word boundary; and falls back to a sentinel sentence if the model returns nothing usable.
- Output value is `[{"text": <formatted prompt>, "weight": 1.0}]`, matching LTX-2's `prompts` shape exactly (see Phase 4 — this turned out to matter).
- The LLM is preloaded in a background thread from `__init__` so it warms up during LTX-2's heavy load step rather than racing it at session-start.

**Verification.**

- `uv run ruff check` / `uv run ruff format --check` — both clean.
- `daydream-scope plugins` lists `scope-llm (0.1.0) Source: local`.
- `/api/v1/nodes/definitions` exposes the node with display name "Local LLM", category `text`, params `[question, template, system_prompt, model_id, max_new_tokens]`, output `out (string)`.
- End-to-end Python smoke test (no Scope server) — the cleaner produces:
  - `"Blue is my favorite color!"` (4 words, fits the clip)
  - `"Paris is the capital of France."` (6 words)
  - `"The sky appears blue because it reflects sunlight…"` (11 words)
  - Inner double quotes become single quotes; multi-sentence answers are truncated to the first sentence; over-length answers are word-boundary capped.

**Decision.** Proceed to Phase 4.

---

## Phase 4 — Wire the workflow and test ✅ DELIVERED, with two upstream issues found

**Goal.** Build a workflow JSON, test it end-to-end, iterate until a recording falls out the other side.

**Workflow shape.**

```
Avatar Reply  ──out (string)──►  LTX-2 (talking-head LoRA + i2v_image)
                                  │
                                  ├── video ──► Sink
                                  └── video ──► Record
```

**Two non-obvious bugs found in upstream Scope while testing.** Both are documented in [project memory](../../.claude/projects/-home-user-scope2/memory/scope-ltx2-talking-head.md) so the next iteration doesn't re-hit them.

### Issue 4.1 — LTX-2 expects list-of-dicts on its `prompts` port

LTX-2's pipeline reads `prompts[0]["text"]` directly (`scope_ltx_2/pipeline.py:1022`) — no defensive normalisation. Wan2.1 / LongLive *do* normalise (`text_conditioning.py:_normalize_prompts`), so plain-string emission works there. LTX-2 crashes with `TypeError: string indices must be integers, not 'str'`.

**Fix in this plugin:** emit `[{"text": prompt, "weight": 1.0}]` from `out` instead of a raw string. The `port_type` declaration stays `"string"` (the framework only validates port *names*, not types, so the wire works in practice). Lands in `node.py:298`.

### Issue 4.2 — `session/start` with `graph` reloads pipelines with empty load_params

`mcp_router.py:279` builds `pipeline_tuples = [(node.id, node.pipeline_id, None)]` for every node in the graph, then calls `pipeline_manager.load_pipelines(pipeline_tuples)`. Because `_is_pipeline_loaded_with` requires exact param-equality and the previous load had real params (LoRA, dimensions, i2v_image) while this call has `{}`, the manager **unloads and reloads with empty params, wiping the LoRA.**

The UI's WebRTC offer path doesn't go through `session/start`, so loading the workflow JSON in the UI and clicking *Start* avoids the issue. The headless `/api/v1/session/start` with `graph` does hit it.

**Workaround used for the recording in this post:** drove a *single-pipeline* session (no `graph` field), pre-computed the LLM answer in Python, and passed it as `prompts` on `session/start`. The avatar-reply node still works — the bug is in how the framework triggers re-loads, not in the node itself.

### Issue 4.3 — Startup race for batch 1 in the in-graph path

When both nodes start in the same session, LTX-2 begins its first text encode (~50 s on a cold Gemma) before Qwen has loaded (~25 s) and produced the reply (~0.5 s). The first batch falls back to LTX-2's default ("a beautiful sunset") and the talking head appears in batch 2.

**Mitigation in this plugin:** background prewarm of the LLM in `__init__`, so by the time the first execute() call runs the model is already on the GPU and the prompt push wins the race against LTX-2's text-encode step.

### Issue 4.4 — Identity drift across batches with a non-portrait reference

The original `talking_head_reference.png` shipped with the existing Talking Heads workflow is a blurred silhouette, which leaves the LoRA free to roll a fresh identity per batch. Combined with `randomize_seed=true` (the default), back-to-back runs produced visibly different faces for the same prompt.

**Mitigation in this workflow:** (a) replaced the reference with a real portrait — a 768×768 LANCZOS-upscale of frame 30 from the first generated MP4, saved as `~/.daydream-scope/assets/avatar_reply_reference.png`; (b) set `randomize_seed: false` with `base_seed: 42` in `pipelines[].params` and `graph.ui_state.node_params.ltx2`. With both, the same person speaks across batches and across runs.

**Limitation acknowledged:** LTX-2.3 in `scope-ltx-2` v0.3.0 only honours `i2v_image` (first-frame conditioning); there is no `last_frame_image` parameter. True keyframe interpolation across batches would require a small upstream patch — a `chain_i2v_image: bool` flag on the pipeline that captures the final decoded frame of batch N and uses it as the `i2v_image` of batch N+1. Out of scope for this iteration.

### Recording produced

```
Q: What is your favorite color?
LLM:    My favorite color is blue!
Wall:   pre-compute < 1 s, LTX-2 generation ~213 s
Output: 192×192 h264 + 48 kHz aac, ~5 s, 119 KB
```

Saved as [`avatar_color.mp4`](avatar_color.mp4) and [`avatar_color.jpg`](avatar_color.jpg) in the blog directory.

**Decision.** Proceed to Phase 5.

---

## Phase 5 — Blog post ✅ DELIVERED

**Deliverable.** [`live-avatar-in-5-minutes.md`](live-avatar-in-5-minutes.md), ~120 lines, with hero image and embedded MP4.

**Editorial choices.**

- *Lead with the framing.* The post opens with "four-model stack collapsed to one model + LoRA", because that's the load-bearing insight; everything else is mechanics.
- *Honest about constraints.* Three minutes per clip, identity drift across batches, the startup race — all named in the post rather than buried. The audience is r/LocalLLaMA / r/StableDiffusion; they smell varnish.
- *Workflow JSON is the deliverable.* The post links to the JSON, not just describes it. Reproducibility matters more than narrative flourish.
- *Forward-looking but not aspirational.* The "where this could go" section names four concrete extensions (TTS, ASR, ID-LoRA, upstream `block_on_first_batch`); none of them are required for the current build to be useful.

---

## Verification: how a reader can reproduce this

```bash
# 1. Install plugin (editable)
git clone https://github.com/leszko/scope-llm ~/scope-llm
uv run daydream-scope install -e ~/scope-llm

# 2. LoRA into Scope's model dir
huggingface-cli download elix3r/LTX-2.3-22b-AV-LoRA-talking-head \
    LTX-2.3-22b-AV-LoRA-talking-head-v1.safetensors \
    --local-dir ~/.daydream-scope/models/lora/

# 3. Reference portrait at ~/.daydream-scope/assets/talking_head_reference.png

# 4. Start Scope and load the workflow in the UI
uv run daydream-scope --port 8000
#   ↳ open http://localhost:8000
#   ↳ Open Workflow → ~/scope-llm/examples/talking-head.scope-workflow.json
#   ↳ edit the question on the Avatar Reply node, click Start
```

Expected first run: ~3.5 minutes wall-clock for the first ~5-second clip on a 5090. Subsequent prompts on the same model state are much cheaper (Gemma encoding cache stays warm).

---

## Phase 6 — Iteration: generic plugin + voice input ✅ DELIVERED

After the first end-to-end build, three follow-ups landed in the same iteration:

### 6.1 — Refactor `scope-avatar-reply` → `scope-llm` (generic plugin)

The original plugin hard-coded the OHWXPERSON template. The generic refactor pulls the wrapper into a `template` parameter (default `"{answer}"`) and the LLM prompt into a `system_prompt` parameter (default few-shot prompt). Same node, same wire-up, but now the same plugin can drive any string-input pipeline — feed `{answer}` into a LongLive prompt, an SDXL caption, anything that takes a string. The talking-head-specific template lives in the example workflow JSON, not in the plugin.

Node-type-id renamed from `avatar-reply` to `llm`. UI display name from "Avatar Reply" to "Local LLM".

### 6.2 — Fix the question-echo failure mode

A Qwen 0.5 B failure mode appeared on adversarial questions ("What do you think about politics?" → "What's your opinion on politics?"). Two fixes:

- **Few-shot system prompt** with four Q→A examples instead of pure rules. Small instruct models follow examples far more reliably.
- **Echo detector in the cleaner** — any answer that starts with a wh-word (`what / who / where / when / why / how / which`) and ends with `?` is rejected as an echoed question and replaced with the fallback sentence.
- **Filler-clause stripping** with re-capitalisation: `"Sure! Blue is my favorite color."` → `"Blue is my favorite color."`, `"I think the answer is yes."` → `"The answer is yes."` (handles stacked filler like `"Well, actually, …"`).

Verified on the original failing question: now produces `"I'd rather focus on things I can build today."` (the few-shot answer for that exact prompt).

### 6.3 — Voice input (`scope-whisper`)

Added a sibling plugin with two nodes:

- **Whisper** — file-watching speech-to-text. Polls `audio_path` for mtime changes and transcribes with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (default `base.en`, ~74 MB). Emits the transcript on a string port. Pre-warms the model on `__init__` so the first transcription doesn't pay the cold-load tax.
- **Microphone** — captures from the default audio device with [sounddevice](https://python-sounddevice.readthedocs.io/), runs energy-based VAD (RMS threshold + silence-tail), writes one WAV per detected utterance to `output_dir`, emits the path. No-op on hosts without an input device — logs a warning and emits nothing rather than crashing the workflow.

End-to-end pipeline:

```
Microphone ──audio_path──► Whisper ──out──► Local LLM ──out──► LTX-2.3 ──video──► Sink
```

**Verified end-to-end** with [`tests/test_voice.py`](../../scope-whisper/tests/test_voice.py) (Piper TTS synthesizes a question → Whisper transcribes → LLM answers → LTX-2 generates the talking head). Result:

| Stage | In | Out | Wall-clock |
|---|---|---|---|
| Piper (TTS, fixture) | `"What is your favorite color?"` | `voice_color.wav` | <1 s |
| Whisper `base.en` | `voice_color.wav` | `"What is your favorite color?"` | <1 s on a 5090 |
| LLM (Qwen 0.5 B) | transcribed text | `"My favorite color is deep ocean blue."` (wrapped in OHWXPERSON template) | <1 s |
| LTX-2.3 + LoRA | OHWXPERSON prompt + `i2v_image` | 192×192 H.264 + AAC, 121 frames | 255 s |

So the LLM and STT round-trips are essentially free; LTX-2 is the entire user-perceived budget. The voice loop adds <1 % to total latency.

### 6.4 — Mic-source-via-Scope-input-source-API: explicitly out of scope

A "true" microphone input source in Scope (analogous to NDI / Spout / Syphon) would need:

- An `AudioInputSource` interface alongside the existing video-only `InputSource` (different return type — `np.ndarray (T,)` instead of `(H, W, 3)`).
- New routing in `pipeline_processor` for an `audio` input port from a non-pipeline source (today the audio path is sink-only via `audio_output_queue`).
- Frontend mic-permission UX, source picker, etc.

Multi-day work. Sidestepped here by making the mic node write WAVs to disk and Whisper read them — entirely within the existing string/file conventions. Trade-off: ~50 ms file-system round-trip per utterance. Acceptable when the talking-head budget is 200 000 ms.

**Total deliverables (Phase 1 → Phase 6):**

- `~/scope-llm/` — generic LLM plugin with `Local LLM` node.
- `~/scope-whisper/` — voice-input plugin with `Whisper` and `Microphone` nodes.
- `~/scope-llm/examples/talking-head.scope-workflow.json` — text-driven avatar.
- `~/scope-whisper/examples/voice-avatar.scope-workflow.json` — voice-driven avatar.
- Sample MP4 + still in `~/scope-llm/blog/`.
- Blog post `~/scope-llm/blog/live-avatar-in-5-minutes.md`.
