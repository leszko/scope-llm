# Scope LLM

A [Daydream Scope](https://daydream.live) plugin that adds a **Local LLM** node — runs a tiny instruction-tuned LLM on the same GPU as your video pipeline and emits the cleaned answer on a string port, optionally wrapped in a user-supplied template.

The node is pipeline-agnostic. Any Scope node that consumes a string `prompts`-style port can be driven by it — change the `template` parameter and the same plugin can feed a talking-head video model, a live-captioning text-to-image model, or anything else that takes text.

By default it runs [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (~500M params, ~1GB on disk, sub-second answers on a modern GPU). [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) and [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) are available as lighter fall-backs.

## Example workflows

- **Talking-head avatar** — drive a portrait-animating model (e.g. LTX-2.3 + a talking-head LoRA) so that a still photo speaks the LLM's answer. A ready-to-load graph is in [`examples/talking-head.scope-workflow.json`](examples/talking-head.scope-workflow.json).
- **Dynamic prompting for text-to-video / text-to-image** — feed live, generated prompts into LongLive, SDXL captioning, or any model with a string prompt input.
- **Your own** — wire `out` into anything that accepts a string and let the template parameter shape the payload.

## How it works

1. The question goes through the LLM with a few-shot `system_prompt` (default: rules + four Q→A demonstrations) that constrains the answer to one short, natural, ≤12-word sentence.
2. The answer is cleaned: outer quotes / markdown / backticks are stripped, leading filler clauses (`"Sure!"`, `"Well, actually,"`) are removed and the next sentence is re-capitalised, multi-sentence answers are clipped to the first sentence, inner `"` becomes `'`. Answers that look like echoed questions (start with a wh-word, end with `?`) are rejected.
3. The cleaned answer is substituted into your `template`. Default is the no-op `"{answer}"`; the talking-head example uses:
   ```
   OHWXPERSON, <visual_description>. The person is talking, and he says: "{answer}"
   ```
4. The result is emitted on the `out` port as `[{"text": <formatted>, "weight": 1.0}]` — the `prompts` shape used by LTX-2 and similar nodes, so it flows downstream without normalisation.

The default model pre-warms on a background thread the moment the node is instantiated, so the first reply lands quickly — by the time a downstream video model finishes its initial text-encoding step, the LLM is already loaded and ready.

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

For the talking-head example workflow you additionally need:

- The community talking-head LoRA: `elix3r/LTX-2.3-22b-AV-LoRA-talking-head` placed at `~/.daydream-scope/models/lora/`.
- A reference portrait at `~/.daydream-scope/assets/avatar_reply_reference.png` (any clean front-facing headshot, ≥256×256).

## Usage

1. Install the plugin and (re)start Scope.
2. Add a **Local LLM** node to your graph (category **text**).
3. Set the `template` to whatever your downstream pipeline expects — `{answer}` is replaced with the LLM's cleaned reply.
4. Wire `out` into the `prompts` input of any string-consuming node.
5. Type a question into the node's `Question` field, click Start.

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

## License

Apache-2.0.
