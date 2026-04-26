from __future__ import annotations

import os
import time

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError


class OpenAICompatibleTeacher:
    def __init__(
        self,
        api_base: str,
        api_key_env: str,
        model_name: str,
        system_prompt: str | None,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        max_retries: int,
        retry_backoff_seconds: float,
        retry_max_backoff_seconds: float,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable '{api_key_env}' is not set. Export it before generating teacher data."
            )

        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.retry_max_backoff_seconds = retry_max_backoff_seconds

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(self.retry_backoff_seconds * (2 ** max(attempt - 1, 0)), self.retry_max_backoff_seconds)
        time.sleep(max(delay, 0.0))

    def generate(self, prompt: str) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_new_tokens,
                )
                content = response.choices[0].message.content
                if not content:
                    raise RuntimeError("Teacher response was empty.")
                return content.strip()
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as error:
                last_error = error
                if attempt >= self.max_retries:
                    break
                self._sleep_before_retry(attempt)

        raise RuntimeError(
            f"Teacher request failed after {self.max_retries} attempts for model '{self.model_name}'."
        ) from last_error


class MultiTeacherRouter:
    def __init__(self, teachers: list[OpenAICompatibleTeacher]) -> None:
        if not teachers:
            raise ValueError("At least one teacher must be configured.")
        self.teachers = teachers

    def generate(self, prompt: str) -> tuple[str, str]:
        last_error: Exception | None = None
        for teacher in self.teachers:
            try:
                return teacher.generate(prompt), teacher.model_name
            except Exception as error:
                last_error = error
                continue

        raise RuntimeError("All configured teacher models failed for the current prompt.") from last_error
