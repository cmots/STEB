"""LLM-based sound event preservation scorer."""

import asyncio
from typing import Any, Dict, List, Tuple

import aiohttp

from .base import BaseScorer, EvalRecord
from ._llm_utils import get_vllm_client, extract_json
from ._prompt_versions import DEFAULT_PROMPT_VERSION, select_prompt, versioned_name


EVENT_SYSTEM_PROMPT_DEFAULT = """\
You are an expert evaluator for sound event similarity in speech-to-speech translation systems.

Your task: Compare reference and hypothesis sound events using a fixed event
pool. Treat equivalent spellings such as [Breathing] and [inhale] as Breathing
when the audible event is a natural breath.

## Fixed event pool
Every listed event must be mapped to one of:
Breathing, Laughter, Cough, Sigh, Whispering, Pant, Crying.

Ignore text emotion labels such as [excited] or [disappointed] unless they
clearly correspond to an audible event in the fixed pool.

## Scoring Rubric (1-5)
5: all fixed-pool reference events are preserved with compatible type/count/order/position.
4: main events preserved with slight position/intensity deviations.
3: main events partly preserved, but there are clear omissions or shifts.
2: most fixed-pool events are missing or wrong, or salient false events are added.
1: reference events are clearly not preserved.

## Output format
Return ONLY a JSON object with exactly these fields:
{"reference_events": [...], "hypothesis_events": [...], "score": <int 1-5>, "reason": "..."}"""

EVENT_SYSTEM_PROMPTS = {
    DEFAULT_PROMPT_VERSION: EVENT_SYSTEM_PROMPT_DEFAULT,
}

# Backward-compatible constant name for external imports.
EVENT_SYSTEM_PROMPT = EVENT_SYSTEM_PROMPT_DEFAULT


def build_event_prompt(
    ref_events: str,
    hyp_events: str,
    prompt_version: str | None = None,
) -> Tuple[str, str]:
    """Return (system_message, user_message) for event scoring."""
    user_msg = (
        f"Reference (text with events): {ref_events}\n"
        f"Hypothesis (text with events): {hyp_events}"
    )
    return select_prompt(EVENT_SYSTEM_PROMPTS, prompt_version), user_msg


class LLMEventScorer(BaseScorer):
    name = "llm_event"
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
        self.name = versioned_name("llm_event", prompt_version)

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
                    ref_evt = rec.src_text_with_event or rec.ref_text_with_events
                    hyp_evt = rec.hyp_asr_text_with_event
                    if not ref_evt or not hyp_evt:
                        continue
                    tasks.append(self._score_one(session, semaphore, results, idx, ref_evt, hyp_evt))
                if tasks:
                    await asyncio.gather(*tasks)

        asyncio.run(_run())
        return results

    async def _score_one(self, session, semaphore, results, idx, ref_evt, hyp_evt) -> None:
        system_msg, user_msg = build_event_prompt(ref_evt, hyp_evt, self._prompt_version)
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
                results[idx]["event_score"] = float(parsed["score"])
                results[idx]["event_reason"] = parsed.get("reason")
                results[idx]["event_judgement"] = parsed
