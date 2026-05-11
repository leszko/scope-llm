# Scope LLM

A [Daydream Scope](https://daydream.live) plugin that adds a **Local LLM** node — runs a tiny instruction-tuned LLM on the same GPU as your video pipeline, then wraps the answer in a user-supplied template before emitting it on a string port.

The example workflow uses it as the brain behind a single-model talking-head avatar:

```
Local LLM (Qwen 0.5B)  ─out─►  LTX-2.3 + talking-head LoRA  ─►  Sink
```

…but the plugin is generic. Any pipeline that consumes a string `prompts` port can be driven by it — change the `template` parameter and the same plugin can drive LongLive, SDXL captioning, or whatever else takes text.

By default it runs [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (~500M params, ~1GB on disk, sub-second answers on a modern GPU). [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) and [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) are available as lighter fall-backs.

## How it works

1. The question goes through the LLM with a few-shot `system_prompt` (default: rules + four Q→A demonstrations) that constrains the answer to one short, natural, ≤12-word sentence.
2. The answer is cleaned: outer quotes / markdown / backticks are stripped, leading filler clauses (`"Sure!"`, `"Well, actually,"`) are removed and the next sentence is re-capitalised, multi-sentence answers are clipped to the first sentence, inner `"` becomes `'`. Answers that look like echoed questions (start with a wh-word, end with `?`) are rejected.
3. The cleaned answer is substituted into your `template`. Default is the no-op `"{answer}"`; the talking-head workflow uses:
   ```
   OHWXPERSON, <visual_description>. The person is talking, and he says: "{answer}"
   ```
4. The result is emitted on the `out` port as `[{"text": <formatted>, "weight": 1.0}]` — matching LTX-2's `prompts` shape so it works without normalisation downstream.

The default model pre-warms on a background thread the moment the node is instantiated, so by the time LTX-2 finishes its initial Gemma text-encoding step the LLM is loaded and the first reply lands before LTX-2's first batch runs.

## Installation

From a Scope checkout:

```bash
uv run daydream-scope install -e ~/scope-llm
```

Or pull straight from GitHub:

```bash
uv run daydream-scope install https://github.com/leszko/scope-llm
```

The default LLM weights are auto-downloaded by Scope's Download Dialog on first Start (artifacts declared via `get_artifacts()`).

For the talking-head workflow you also need:

- The community talking-head LoRA: `elix3r/LTX-2.3-22b-AV-LoRA-talking-head` placed at `~/.daydream-scope/models/lora/`.
- A reference portrait at `~/.daydream-scope/assets/avatar_reply_reference.png` (any clean front-facing headshot, ≥256×256).

## Usage

1. Install the plugin and (re)start Scope.
2. Add a **Local LLM** node to your graph (category **text**).
3. Set the `template` to whatever your downstream pipeline expects — `{answer}` is replaced with the LLM's cleaned reply.
4. Wire `out` into the `prompts` input of any string-consuming pipeline (LTX-2, LongLive, etc.).
5. Type a question into the node's `Question` field, click Start.

A ready-to-load talking-head example is in [`examples/talking-head.scope-workflow.json`](examples/talking-head.scope-workflow.json).

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | `What is your favorite color?` | The question the LLM should answer. |
| `template` | string | `{answer}` | Wraps the cleaned answer; `{answer}` is the placeholder. |
| `system_prompt` | string | (few-shot avatar prompt) | LLM system instructions. Override for non-avatar use. |
| `model_id` | select | `Qwen/Qwen2.5-0.5B-Instruct` | Model to load. |
| `max_new_tokens` | number (8–256) | 64 | Generation cap. |

## Ports

| Port | Direction | Type | Description |
|------|-----------|------|-------------|
| `question` | input (optional) | string | Overrides the `Question` parameter when connected upstream. |
| `out` | output | string | The templated answer as `[{"text": ..., "weight": 1.0}]`. |

## Voice input

For a fully voice-driven version of the talking-head workflow, install the sibling plugin [`scope-whisper`](https://github.com/leszko/scope-whisper) and use its `voice-avatar.scope-workflow.json`:

```
Microphone → Whisper → Local LLM → LTX-2.3 → Sink
```

## License

Apache-2.0.
