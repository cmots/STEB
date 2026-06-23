"""LLM-based emotion preservation scorer."""

import asyncio
from typing import Any, Dict, List, Tuple

import aiohttp

from .base import BaseScorer, EvalRecord
from ._llm_utils import get_vllm_client, extract_json
from ._prompt_versions import DEFAULT_PROMPT_VERSION, select_prompt, versioned_name


EMOTION_SYSTEM_PROMPT_V4_CHOICE = """You are an expert evaluator for speech emotion similarity in speech-to-speech translation systems.

Your task: Compare the reference emotion description with the hypothesis emotion description using a fixed primary emotion pool. Judge only vocal expression, not semantic content.

## Fixed primary emotion pool
For source_label and hypothesis_label, choose exactly one label from:
neutral, happy, sad, angry, surprised, fearful.

You may additionally mention secondary nuance from this optional pool:
anxious, serious, calm, disgust, bored.

## Scoring Rubric (1-5)
5: same primary label and same secondary nuance/intensity.
4: same primary label, with minor intensity or nuance differences.
3: related or same broad direction, but primary label or intensity is uncertain.
2: different primary label with only coarse valence/arousal similarity.
1: opposite or completely different primary label.

## Output format
Return ONLY a JSON object with exactly these fields:
{"source_label": "...", "hypothesis_label": "...", "source_secondary": "...", "hypothesis_secondary": "...", "score": <int 1-5>, "reason": "..."}"""

EMOTION_SYSTEM_PROMPTS = {
    DEFAULT_PROMPT_VERSION: EMOTION_SYSTEM_PROMPT_V4_CHOICE,
}

# Backward-compatible constant name for external imports.
EMOTION_SYSTEM_PROMPT = EMOTION_SYSTEM_PROMPT_V4_CHOICE


def build_emotion_prompt(
    ref_text: str,
    ref_emotion: str,
    hyp_emotion: str,
    prompt_version: str | None = None,
) -> Tuple[str, str]:
    """Return (system_message, user_message) for emotion scoring."""
    user_msg = (
        f"Transcript: {ref_text}\n"
        f"Reference emotion: {ref_emotion}\n"
        f"Hypothesis emotion: {hyp_emotion}"
    )
    return select_prompt(EMOTION_SYSTEM_PROMPTS, prompt_version), user_msg


class LLMEmotionScorer(BaseScorer):
    name = "llm_emotion"
    requires_audio = False
    requires_reference = False

    def __init__(
        self,
        base_url: str,
        model_name: str,
        concurrency: int = 100,
        prompt_version: str | None = None,
    ) -> None:
        self._client = get_vllm_client(base_url, model_name)
        self._concurrency = concurrency
        self._prompt_version = prompt_version
        self.name = versioned_name("llm_emotion", prompt_version)

    def score(self, records: List[EvalRecord]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = [{"id": rec.id} for rec in records]

        async def _run() -> None:
            semaphore = asyncio.Semaphore(self._concurrency)
            connector = aiohttp.TCPConnector(limit=self._concurrency, limit_per_host=0)
            async with aiohttp.ClientSession(connector=connector) as session:
                tasks = []
                for idx, rec in enumerate(records):
                    if rec.error is not None:
                        continue
                    ref_emo = rec.ref_emotion
                    hyp_emo = rec.hyp_emotion or rec.hyp_emotion_text
                    if not ref_emo or not hyp_emo:
                        continue
                    tasks.append(
                        self._score_one(
                            session,
                            semaphore,
                            results,
                            idx,
                            rec.ref_text or "",
                            ref_emo,
                            hyp_emo,
                        )
                    )
                if tasks:
                    await asyncio.gather(*tasks)

        asyncio.run(_run())
        return results

    async def _score_one(
        self,
        session,
        semaphore,
        results,
        idx,
        ref_text,
        ref_emo,
        hyp_emo,
    ) -> None:
        system_msg, user_msg = build_emotion_prompt(
            ref_text,
            ref_emo,
            hyp_emo,
            self._prompt_version,
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        async with semaphore:
            resp = await self._client.chat_completions(messages, session)
        if resp and "choices" in resp:
            content = resp["choices"][0]["message"]["content"]
            parsed = extract_json(content)
            if parsed and isinstance(parsed.get("score"), (int, float)):
                results[idx]["emotion_score"] = float(parsed["score"])
                results[idx]["emotion_reason"] = parsed.get("reason")
                results[idx]["emotion_judgement"] = parsed
