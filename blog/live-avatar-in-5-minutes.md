# Build a live AI avatar in 5 minutes — locally, on a single 5090

**TL;DR** — Two small Daydream Scope plugins (`scope-llm` and the LTX-2.3 talking-head AV LoRA), one workflow JSON, no cloud, no API keys. Type a question, the avatar speaks the answer. Setup-to-first-reply in about five minutes on a single consumer GPU.

![A frame of the avatar mid-reply](avatar_color.jpg)

> *Above: a still from the locally-generated clip. The MP4 ([`avatar_color.mp4`](avatar_color.mp4)) carries the spoken audio LTX-2.3 generated in the same forward pass — no separate TTS.*

## What you get

A node-graph in [Daydream Scope](https://daydream.live) that looks like this:

```
Local LLM (Qwen 0.5B)  ──out──►  LTX-2.3 (talking-head AV LoRA + reference portrait)  ──video──►  Sink
```

You type a question into the **Local LLM** node. A 0.5 B Qwen instruct model runs on the same GPU and writes a one-sentence reply. The reply gets wrapped in the LoRA's `OHWXPERSON, ...he says: "<answer>"` template and handed to LTX-2.3, which paints a person's face *and* generates the matching speech audio in a single denoising pass. Sink renders the result.

No SadTalker, no Wav2Lip, no Whisper-needed-on-the-output side, no cloud TTS. One LoRA quietly does three jobs: face, voice, lip-sync.

## Why now

- **LTX-2.3 came out open-source in January 2026.** It's a 22 B audio-visual diffusion transformer that produces synchronised audio + video in one pass.
- **The community AV talking-head LoRA** ([elix3r/LTX-2.3-22b-AV-LoRA-talking-head](https://huggingface.co/elix3r/LTX-2.3-22b-AV-LoRA-talking-head)) bakes the speaker's voice and identity straight into the LoRA weights, so I2V mode needs *only* a reference portrait and the prompt text. No external audio at inference.
- **A 5090 has the budget** to run LTX-2.3 (~22 GB transformer in FP8) and a 0.5 B-parameter LLM at the same time, with 16 GB of swap as breathing room during load.

## The five-minute path

```bash
# 1) clone + install both plugins (editable)
git clone https://github.com/leszko/scope-llm ~/scope-llm
git clone https://github.com/leszko/scope-whisper ~/scope-whisper        # only if you also want voice input
uv run daydream-scope install -e ~/scope-llm

# 2) drop the talking-head LoRA into Scope's lora dir
huggingface-cli download elix3r/LTX-2.3-22b-AV-LoRA-talking-head \
    LTX-2.3-22b-AV-LoRA-talking-head-v1.safetensors \
    --local-dir ~/.daydream-scope/models/lora/

# 3) put a reference portrait at
#    ~/.daydream-scope/assets/avatar_reply_reference.png
#    (I extracted one from a first-pass generated clip; any clean front-facing
#     headshot at ~512x512+ works.)

# 4) run Scope and load the workflow
uv run daydream-scope --port 8000
#    → http://localhost:8000
#    → Open Workflow → ~/scope-llm/examples/talking-head.scope-workflow.json
#    → click Start
```

That's it. The first Start triggers the Download Dialog for any missing weights (Qwen 0.5 B is ~1 GB, the LTX-2.3 base is already there if you've used Scope before). The first ~5-second clip lands in about 3.5 minutes wall-clock; subsequent answers on the same model state are much faster (encoder cache stays warm).

## How the plugin works

`scope-llm` is intentionally tiny — one file, ~250 lines. The key bits:

**The system prompt is few-shot, not just rules.** Qwen 0.5 B follows examples far more reliably than abstract instructions. The default system prompt includes four Q→A demonstrations, which fixed an early failure mode where it would echo the question back ("What's your opinion on politics?") instead of answering.

```text
You answer questions for a video avatar to speak aloud.

Rules:
- Output ONE short sentence of at most 12 words.
- Speak in first person, conversational tone.
- Do NOT repeat, rephrase, or echo the question back.
- ...

Examples:
Q: What is your favorite color?
A: My favorite color is deep ocean blue.
...
Q: What do you think about politics?
A: I'd rather focus on things I can build today.
```

**A defensive cleaner runs after generation.** Strips quotes (otherwise they collide with the LoRA template's literal `"…"` quoting), drops leading filler clauses (`"Sure! …"` → `"…"`), takes only the first sentence, and rejects answers that look like echoed questions (start with a wh-word, end with `?`).

**The template is a parameter.** The plugin doesn't know about LTX-2 specifically; it knows how to call an LLM and how to substitute `{answer}` into a user-supplied template. The talking-head workflow injects:

```text
OHWXPERSON, a portrait of a person facing the camera in a softly lit studio,
neutral background. The person is talking, and he says: "{answer}"
```

That's the only LTX-2-specific bit, and it lives in the workflow JSON, not in the plugin.

**The LLM warms up in the background.** When the node is instantiated as part of a session, `__init__` spawns a daemon thread that pre-loads Qwen onto the GPU. By the time LTX-2 finishes its initial text-encoding step (~50 s on the cold path), the LLM has been ready for 25 seconds and the first reply pushes onto the queue before LTX-2's first batch starts. No "first batch is sunset, second is the avatar" race.

**Artifacts hook auto-download.** `get_artifacts()` declares the Qwen weights via `HuggingfaceRepoArtifact`. Scope's Download Dialog reads this through `/api/v1/models/status?pipeline_id=llm` and pre-fetches the weights when you click Start, instead of making the user wait for a synchronous download mid-stream.

## Honest limitations

- **It's "ask-and-wait", not "live in the streaming sense".** The 22 B AV transformer is the budget; one ~5-second clip is ~3.5 min wall-clock at 192 × 192 on a 5090, dominated by the Gemma 3 12 B text encoder. Bigger resolutions cost more.
- **The trained voice belongs to the LoRA.** The community talking-head LoRA carries one specific identity + voice baked in. If you want your own face/voice you need to train an ID-LoRA, or wait for the upcoming `audio_mode: id_lora` path which takes a 5-second voice reference clip.
- **Identity stability across batches needs a real portrait + a fixed seed.** With the placeholder silhouette reference image and `randomize_seed=true`, you'll get a different face every batch. The example workflow ships a real portrait (`avatar_reply_reference.png`) and pins `base_seed: 42`. After both, the same person speaks across runs.
- **The graph-mode startup race is mitigated, not eliminated.** LTX-2 has no `block_on_first_batch` for its `prompts` input port today; the workaround is the LLM-prewarm trick above. A small upstream patch would make it bulletproof.

## Going further: voice in

With the [`scope-whisper`](https://github.com/leszko/scope-whisper) plugin installed, the workflow extends one node to the left:

```
Microphone → Whisper (base.en) → Local LLM → LTX-2.3 → Sink
```

The mic node uses energy-based VAD to write one WAV per detected utterance into `/tmp/scope-mic/`. Whisper watches that file and emits the transcribed string; the LLM consumes it as the question. End-to-end conversational latency is ~3.5 min per reply (still the LTX-2 budget). Shipped as a separate plugin so users who only want text input don't have to install faster-whisper.

## Try it

Plugin source: [`scope-llm`](https://github.com/leszko/scope-llm) (Apache-2.0). Workflow: [`talking-head.scope-workflow.json`](../examples/talking-head.scope-workflow.json). Built on [Daydream Scope](https://daydream.live), [Lightricks LTX-2](https://github.com/Lightricks/LTX-2), the [elix3r AV talking-head LoRA](https://huggingface.co/elix3r/LTX-2.3-22b-AV-LoRA-talking-head), [Qwen 2.5 0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), and the workflow pattern from [`scope-prompt-enhancer`](https://github.com/leszko/scope-prompt-enhancer).
