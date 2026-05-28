from __future__ import annotations

import base64
import time
from typing import Any

from openai import AsyncOpenAI

from config.settings import ActorConfig
from logger_config import logger
from modules.base_agent import BaseAgent
from modules.judge.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE
from modules.judge.schema import JudgeVerdict
from utils.json_extract import extract_json_object
from utils.retry import async_retry


def _image_part(image_bytes: bytes, mime: str) -> dict[str, Any]:
    b64 = base64.b64encode(image_bytes).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


class JudgeAgent(BaseAgent):
    """Stateless pairwise visual judge. One call <-> one JudgeVerdict."""

    actor = "judge"

    def __init__(
        self,
        client: AsyncOpenAI,
        settings: ActorConfig,
        *,
        max_retries: int = 2,
    ) -> None:
        super().__init__(client, settings)
        self.max_retries = max_retries
        self.reasoning_effort = settings.reasoning_effort

    def _build_extra_body(self) -> dict[str, Any]:
        extra: dict[str, Any] = {}
        if self.reasoning_effort:
            if self.backend == "vllm":
                extra["chat_template_kwargs"] = {"enable_thinking": True}
            else:
                extra["reasoning"] = {"effort": self.reasoning_effort}
        for key, value in (
            ("top_k", self.top_k),
            ("min_p", self.min_p),
            ("repetition_penalty", self.repetition_penalty),
        ):
            if value is not None:
                extra[key] = value
        if self.providers is not None and self.backend != "vllm":
            extra["provider"] = self.providers.model_dump(exclude_none=True)
        return extra

    def _build_call_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        if self.presence_penalty is not None:
            kwargs["presence_penalty"] = self.presence_penalty
        return kwargs

    async def compare(
        self,
        *,
        task_id: str,
        match_label: str,
        reference_bytes: bytes,
        reference_mime: str,
        render_a: bytes,
        render_b: bytes,
    ) -> JudgeVerdict:
        extra_body = self._build_extra_body()
        call_kwargs = self._build_call_kwargs()
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    _image_part(reference_bytes, reference_mime),
                    _image_part(render_a, "image/png"),
                    _image_part(render_b, "image/png"),
                    {"type": "text", "text": JUDGE_USER_TEMPLATE},
                ],
            },
        ]

        prefix = f"[Judge {match_label}]"
        logger.info(
            f"{prefix} Started Task {task_id} | Model: {self.model} | "
            f"Ref KB: {len(reference_bytes) / 1024:.1f} | "
            f"A KB: {len(render_a) / 1024:.1f} | B KB: {len(render_b) / 1024:.1f}"
        )

        async def _call(_attempt: int, _last_err: str | None) -> JudgeVerdict:
            t0 = time.time()
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                seed=self.seed,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                extra_body=extra_body or None,
                **call_kwargs,
            )
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Judge returned empty response")
            raw = response.choices[0].message.content.strip()
            verdict = JudgeVerdict.model_validate_json(extract_json_object(raw))
            logger.info(
                f"{prefix} Finished Task {task_id} | Elapsed: {time.time() - t0:.1f}s | "
                f"Winner: {verdict.winner} | Confidence: {verdict.confidence:.2f} | "
                f"Reason: {verdict.reason[:120]}"
            )
            return verdict

        return await async_retry(_call, max_retries=self.max_retries)
