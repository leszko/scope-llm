"""Local LLM node — runs a small instruction-tuned LLM and wraps its
answer in a user-supplied template before emitting it on a string port.

Generic enough to drive any text-input pipeline. The example talking-head
workflow plugs ``{answer}`` into the OHWXPERSON template required by the
LTX-2.3 AV talking-head LoRA, but the same node works for any pipeline
that consumes prompts through a string port — just change the template.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, ClassVar

from scope.core.config import get_model_file_path
from scope.core.nodes.base import BaseNode, NodeDefinition, NodeParam, NodePort
from scope.core.pipelines.artifacts import Artifact, HuggingfaceRepoArtifact

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Same small instruction-tuned models as scope-prompt-enhancer — fast,
# tiny VRAM footprint, sit comfortably next to a 22 B LTX-2 pipeline.
MODEL_OPTIONS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
]
DEFAULT_MODEL = MODEL_OPTIONS[0]

DEFAULT_QUESTION = "What is your favorite color?"

# Default template is a no-op passthrough so the node is generic. The
# example workflow injects the OHWXPERSON template required by the
# LTX-2.3 AV talking-head LoRA.
DEFAULT_TEMPLATE = "{answer}"

# Few-shot system prompt. Small models (Qwen 0.5 B) follow examples far
# more reliably than abstract rules; the "What do you think about politics?"
# echo bug went away once we stopped relying on rules alone.
DEFAULT_SYSTEM_PROMPT = (
    "You answer questions for a video avatar to speak aloud.\n"
    "\n"
    "Rules:\n"
    "- Output ONE short sentence of at most 12 words.\n"
    "- Speak in first person, conversational tone.\n"
    "- Do NOT repeat, rephrase, or echo the question back.\n"
    "- Do NOT begin with filler words like 'Sure', 'Well', 'I think', "
    "'Actually'.\n"
    "- Output ONLY the spoken sentence — no quotes, no markdown, no "
    "preamble, no explanation.\n"
    "\n"
    "Examples:\n"
    "Q: What is your favorite color?\n"
    "A: My favorite color is deep ocean blue.\n"
    "\n"
    "Q: Who built you?\n"
    "A: I'm a local AI avatar running on Daydream Scope.\n"
    "\n"
    "Q: What do you think about politics?\n"
    "A: I'd rather focus on things I can build today.\n"
    "\n"
    "Q: How are you?\n"
    "A: I'm doing great, thanks for asking!\n"
)

# Sentinel when nothing usable comes back.
FALLBACK_ANSWER = "Hmm, I'm not sure how to answer that."

# Filler clauses we drop from the start of the LLM's reply so the cleaner
# doesn't truncate at the filler's terminator (e.g. "Sure! Blue is..." →
# "Sure!"). Order matters: longer phrases first.
_FILLER_PREFIX = re.compile(
    r"^(actually|honestly|basically|literally|i think|i believe|"
    r"in my opinion|sure|well|okay|ok|hmm|right|yeah|yes|no|"
    r"of course|certainly)[\s,!.\-:]+",
    re.IGNORECASE,
)
_QUESTION_LEAD = re.compile(
    r"^(what|who|where|when|why|how|which|do you|are you|did you|"
    r"have you|is|can you|could you|would you)\b",
    re.IGNORECASE,
)


def _looks_like_echo(answer: str, question: str = "") -> bool:
    """True if the answer is a rephrased echo of the input question.

    Detection criteria, all required:
      1. The answer ends in ``?`` (it's a question back at us).
      2. It starts with a wh-word (``what``, ``who``, ``how`` …).
      3. ≥ 40 % Jaccard word-overlap with the input question — so a
         counter-question that shares no meaningful content (e.g.
         "What's up?" in response to "What do you think about this?")
         is *not* rejected; only "What's your opinion on politics?" in
         response to "What do you think about politics?" is.

    Without the third check the detector was over-eager and replaced
    perfectly fine counter-questions with the fallback sentinel.
    """
    a = answer.strip()
    if not a.endswith("?"):
        return False
    if not _QUESTION_LEAD.match(a):
        return False
    if not question:
        # No question text to compare against — fall back to the
        # rule-only behaviour (treat any wh-question answer as an echo).
        return True
    a_words = set(re.findall(r"[a-zA-Z]+", a.lower()))
    q_words = set(re.findall(r"[a-zA-Z]+", question.lower()))
    if not a_words or not q_words:
        return True
    overlap = len(a_words & q_words) / max(len(a_words | q_words), 1)
    # 0.18 keeps the original "What do you think about politics?" →
    # "What's your opinion on politics?" failure caught (Jaccard ≈ 0.20)
    # while letting innocent counter-questions like "What's up?" (≈ 0.13)
    # through.
    return overlap >= 0.18


def _clean_answer(raw: str, question: str = "") -> str:
    """Coerce LLM output into one short, speakable, non-echo sentence."""
    text = raw.strip().strip("\"'` \n")
    # Templates often wrap the spoken transcript in literal double quotes;
    # any inner '"' would break that, so collapse to single quotes.
    text = text.replace('"', "'")
    # Take just the first line — small models sometimes append a second
    # line with a meta-comment.
    text = text.split("\n", 1)[0].strip()

    # Strip leading filler clauses (possibly stacked, e.g. "Well, actually").
    stripped_filler = False
    while True:
        new = _FILLER_PREFIX.sub("", text, count=1).strip()
        if new == text:
            break
        text = new
        stripped_filler = True
    # Re-capitalise after stripping a filler so the spoken line still
    # reads as a complete sentence ("I think the answer is yes." →
    # "The answer is yes.").
    if stripped_filler and text and text[0].islower():
        text = text[0].upper() + text[1:]

    # Take just the first sentence.
    m = re.match(r"^(.+?[.!?])(\s|$)", text)
    if m:
        text = m.group(1)

    # Reject echoed questions outright.
    if _looks_like_echo(text, question):
        logger.warning("LLM: answer looks like an echoed question: %r", text)
        return FALLBACK_ANSWER

    if not text:
        return FALLBACK_ANSWER
    return text


def _apply_template(template: str, answer: str) -> str:
    """Substitute the LLM's answer into the user's template.

    We use a literal ``{answer}`` placeholder and string-replace rather
    than ``str.format`` so the template can contain other ``{`` / ``}``
    characters without escaping (e.g. JSON snippets, code).
    """
    if "{answer}" not in template:
        # No placeholder → user wants the answer alone (or the template
        # is malformed). Be forgiving and emit the answer.
        return answer
    return template.replace("{answer}", answer)


class LLMNode(BaseNode):
    """Runs a small local LLM and emits its answer on a string port."""

    node_type_id: ClassVar[str] = "llm"

    def __init__(self, node_id: str = "", config: dict[str, Any] | None = None):
        super().__init__(node_id, config)
        self._model = None
        self._tokenizer = None
        self._model_id: str | None = None
        self._device: str | None = None
        self._load_lock = threading.Lock()
        # Cache (question, template, system_prompt, model, tokens) ->
        # formatted prompt so we don't re-roll the LLM on every continuous
        # tick while the question is held steady.
        self._cache: dict[tuple[str, str, str, str, int], str] = {}
        # Once a question arrives via the input port (e.g. from a
        # Whisper transcription), it sticks: subsequent ticks keep using
        # it instead of falling back to the static `question` param.
        # Without this, every empty input tick would re-emit the cached
        # answer for the static default and overwrite a fresh upstream
        # answer in the downstream prompts queue.
        self._sticky_input_question: str | None = None
        self._last_emitted_key: tuple[str, str, str, str, int] | None = None
        # Background-load the default LLM during pipeline setup so the
        # first execute() doesn't pay the ~25 s load cost in the
        # foreground and lose the race against LTX-2's first-batch text
        # encoding.
        threading.Thread(
            target=self._prewarm,
            daemon=True,
            name=f"LLMNode[{node_id}]-prewarm",
        ).start()

    def _prewarm(self) -> None:
        """Best-effort background load of the default LLM."""
        try:
            self._ensure_model(DEFAULT_MODEL)
        except Exception:
            logger.exception("LLM: prewarm failed; will retry on first call")

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        return NodeDefinition(
            node_type_id=cls.node_type_id,
            display_name="Local LLM",
            category="text",
            description=(
                "Runs a tiny local LLM to answer a question, then wraps the "
                "answer in a template (e.g. an LTX-2 OHWXPERSON talking-head "
                "prompt) and emits it on a string port."
            ),
            continuous=True,
            inputs=[
                NodePort(
                    name="question",
                    port_type="string",
                    required=False,
                    description="Question (overrides the param when wired)",
                ),
            ],
            outputs=[
                NodePort(
                    name="out",
                    port_type="string",
                    description="Templated answer (list[{text, weight}])",
                ),
            ],
            params=[
                NodeParam(
                    name="question",
                    param_type="string",
                    default=DEFAULT_QUESTION,
                    description="Question",
                ),
                NodeParam(
                    name="template",
                    param_type="string",
                    default=DEFAULT_TEMPLATE,
                    description="Template (use {answer} as the LLM-output placeholder)",
                ),
                NodeParam(
                    name="system_prompt",
                    param_type="string",
                    default=DEFAULT_SYSTEM_PROMPT,
                    description="System prompt",
                ),
                NodeParam(
                    name="model_id",
                    param_type="select",
                    default=DEFAULT_MODEL,
                    description="Model",
                    ui={"options": MODEL_OPTIONS},
                ),
                NodeParam(
                    name="max_new_tokens",
                    param_type="number",
                    default=64,
                    description="Tokens",
                    ui={"min": 8, "max": 256, "step": 1},
                ),
            ],
        )

    @classmethod
    def get_artifacts(cls) -> list[Artifact]:
        """Surface the default LLM weights to Scope's Download Dialog so
        they are pre-fetched when a workflow that uses this node is
        loaded — no first-execute download stall.
        """
        return [
            HuggingfaceRepoArtifact(
                repo_id=DEFAULT_MODEL,
                files=[
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                    "merges.txt",
                    "generation_config.json",
                ],
            ),
        ]

    def _ensure_model(self, model_id: str) -> None:
        if self._model is not None and self._model_id == model_id:
            return
        with self._load_lock:
            if self._model is not None and self._model_id == model_id:
                return
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            local_dir = get_model_file_path(model_id.split("/")[-1])
            source = str(local_dir) if local_dir.is_dir() else model_id
            logger.info("LLM: loading %s on %s (%s)", source, device, dtype)
            tokenizer = AutoTokenizer.from_pretrained(source)
            # Avoid passing dtype/torch_dtype to from_pretrained — the
            # global default-dtype race seen in scope-prompt-enhancer
            # applies here too.
            model = AutoModelForCausalLM.from_pretrained(source)
            model = model.to(device=device, dtype=dtype)
            model.eval()
            self._tokenizer = tokenizer
            self._model = model
            self._model_id = model_id
            self._device = device
            logger.info("LLM: model ready")

    def _answer(
        self,
        question: str,
        system_prompt: str,
        model_id: str,
        max_new_tokens: int,
    ) -> str:
        import torch

        self._ensure_model(model_id)
        assert self._tokenizer is not None and self._model is not None

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.strip()},
        ]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _clean_answer(raw, question)

    def execute(self, inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        # Latch any new input-port question so subsequent ticks (which
        # arrive with empty inputs from NodeProcessor's continuous loop)
        # don't fall back to the static param and clobber the freshly
        # generated answer downstream.
        input_question = inputs.get("question")
        if isinstance(input_question, str) and input_question.strip():
            self._sticky_input_question = input_question.strip()
        question = (
            self._sticky_input_question
            or (
                kwargs.get("question")
                if isinstance(kwargs.get("question"), str)
                else ""
            )
            or ""
        )
        if not isinstance(question, str):
            question = str(question)
        question = question.strip()
        if not question:
            return {}

        template = kwargs.get("template")
        if not isinstance(template, str) or not template:
            template = DEFAULT_TEMPLATE
        system_prompt = kwargs.get("system_prompt")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            system_prompt = DEFAULT_SYSTEM_PROMPT

        model_id = kwargs.get("model_id") or DEFAULT_MODEL
        try:
            max_new_tokens = int(kwargs.get("max_new_tokens", 64))
        except (TypeError, ValueError):
            max_new_tokens = 64

        cache_key = (question, template, system_prompt, model_id, max_new_tokens)
        if cache_key == self._last_emitted_key:
            return {}
        cached = self._cache.get(cache_key)
        if cached is not None:
            # Cache hit on a previously-asked question — the LLM has
            # nothing new to say. Suppress the emit entirely so we don't
            # clobber a fresh value still in the downstream queue (e.g.
            # if the user re-asked something after speaking a different
            # question right after).
            self._last_emitted_key = cache_key
            logger.info(
                "LLM: cached answer for %r — NOT re-emitting %r",
                question[:80],
                cached[:80],
            )
            return {}

        logger.info(
            "LLM: answering (model=%s, tokens=%d): %r",
            model_id,
            max_new_tokens,
            question,
        )
        answer = self._answer(question, system_prompt, model_id, max_new_tokens)
        formatted = _apply_template(template, answer)
        self._cache[cache_key] = formatted
        self._last_emitted_key = cache_key
        logger.info("LLM: emit answer=%r prompt=%r", answer, formatted)
        # Emit the LTX-2 / `/session/parameters` prompts shape directly
        # (list of {text, weight}) so downstream pipelines that index the
        # list — e.g. LTX-2's `prompts[0]["text"]` — work without a
        # defensive normalizer.
        return {"out": [{"text": formatted, "weight": 1.0}]}

    def shutdown(self) -> None:
        self._model = None
        self._tokenizer = None
        self._cache.clear()
