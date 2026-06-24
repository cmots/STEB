"""LLM-based style preservation scorer."""

import asyncio
from typing import Any, Dict, List, Tuple

import aiohttp

from .base import BaseScorer, EvalRecord
from ._llm_utils import get_vllm_client, extract_json
from ._prompt_versions import DEFAULT_PROMPT_VERSION, select_prompt, versioned_name


STYLE_SYSTEM_PROMPT_DEFAULT = """You are an expert evaluator for speech style similarity in speech-to-speech translation systems.

Your task: Compare reference and hypothesis style descriptions using a fixed scene/media-genre pool. Judge from vocal delivery and production cues, not the text topic.

## Fixed scene/genre pool
For source_scene and hypothesis_scene, choose exactly one label from:
news_broadcast, interview, film_tv_drama, audiobook, advertisement, online_class, explainer, livestream.

## Scoring Rubric (1-5)
5: same pool label and strong confidence on both sides.
4: same pool label with minor uncertainty or framing differences.
3: related broad category or uncertain mapping to the pool.
2: different pool labels with only coarse similarity.
1: completely different pool labels and scene/genre.

## Output format
Return ONLY a JSON object with exactly these fields:
{"source_scene": "...", "hypothesis_scene": "...", "source_confidence": "...", "hypothesis_confidence": "...", "score": <int 1-5>, "reason": "..."}"""

STYLE_SYSTEM_PROMPTS = {
    DEFAULT_PROMPT_VERSION: STYLE_SYSTEM_PROMPT_DEFAULT,
}

# Backward-compatible constant name for external imports.
STYLE_SYSTEM_PROMPT = STYLE_SYSTEM_PROMPT_DEFAULT


def build_style_prompt(
    ref_text: str,
    ref_style: str,
    hyp_style: str,
    prompt_version: str | None = None,
) -> Tuple[str, str]:
    """Return (system_message, user_message) for style scoring."""
    user_msg = (
        f"Transcript: {ref_text}\n"
        f"Reference style: {ref_style}\n"
        f"Hypothesis style: {hyp_style}"
    )
    return select_prompt(STYLE_SYSTEM_PROMPTS, prompt_version), user_msg


class LLMStyleScorer(BaseScorer):
    name = "llm_style"
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
        self.name = versioned_name("llm_style", prompt_version)

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
                    ref_sty = rec.ref_style
                    hyp_sty = rec.hyp_style or rec.hyp_style_text
                    if not ref_sty or not hyp_sty:
                        continue
                    tasks.append(
                        self._score_one(
                            session,
                            semaphore,
                            results,
                            idx,
                            rec.ref_text or "",
                            ref_sty,
                            hyp_sty,
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
        ref_sty,
        hyp_sty,
    ) -> None:
        system_msg, user_msg = build_style_prompt(
            ref_text,
            ref_sty,
            hyp_sty,
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
                results[idx]["style_score"] = float(parsed["score"])
                results[idx]["style_reason"] = parsed.get("reason")
                results[idx]["style_judgement"] = parsed
