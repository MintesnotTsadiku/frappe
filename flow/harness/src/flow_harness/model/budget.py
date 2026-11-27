"""Caller-side admission and settlement for one physical provider send."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from flow_harness.model.errors import ProviderError

if TYPE_CHECKING:
	from flow_harness.session.config import BudgetScope, CredentialBinding

_logger = logging.getLogger(__name__)


class BudgetError(ProviderError):
	def __init__(self, code: str, status: int | None = None):
		super().__init__(
			kind="invalid_request", retryable=False, provider="budget", detail=code, status_code=status
		)
		self.code = code


class AttemptJournal:
	"""The local outbox stores references, bounds and settlement only."""

	def __init__(self, path: str | Path):
		if str(path) != ":memory:":
			path = Path(path)
			path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
			fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
			os.close(fd)
		# Each job uses a short connection, except private in-memory journals used by tests.
		self.path = path
		self._memory = sqlite3.connect(path, check_same_thread=False) if str(path) == ":memory:" else None
		with self._connection() as db:
			db.execute(
				"""CREATE TABLE IF NOT EXISTS attempt (
				id TEXT PRIMARY KEY, run_ref TEXT NOT NULL, ordinal INTEGER NOT NULL,
				settlement TEXT, UNIQUE(run_ref, ordinal))"""
			)

	def begin(self, run_ref: str) -> str:
		attempt_id = str(uuid4())
		with self._connection() as db:
			db.execute(
				"""INSERT INTO attempt (id, run_ref, ordinal)
				SELECT ?, ?, coalesce(max(ordinal), 0) + 1 FROM attempt WHERE run_ref = ?""",
				(attempt_id, run_ref, run_ref),
			)
		return attempt_id

	def discard(self, attempt_id: str) -> None:
		with self._connection() as db:
			db.execute("DELETE FROM attempt WHERE id = ?", (attempt_id,))

	def store(self, attempt_id: str, payload: dict | None) -> None:
		with self._connection() as db:
			db.execute(
				"UPDATE attempt SET settlement = ? WHERE id = ?",
				(json.dumps(payload) if payload is not None else None, attempt_id),
			)

	def pending(self) -> list[dict]:
		with self._connection() as db:
			rows = db.execute("SELECT settlement FROM attempt WHERE settlement IS NOT NULL").fetchall()
		return [json.loads(row[0]) for row in rows]

	@contextmanager
	def _connection(self) -> Iterator[sqlite3.Connection]:
		db = self._memory or sqlite3.connect(self.path)
		try:
			with db:
				yield db
		finally:
			if self._memory is None:
				db.close()


@dataclass
class Budget:
	"""An optional runner callback pair. Reserve returns false for unmanaged traffic."""

	journal: AttemptJournal
	run_ref: str
	reserve: Callable[[dict], bool]
	settle: Callable[[dict], None]
	credential: CredentialBinding | None = None
	company: str | None = None
	project: str | None = None

	@contextmanager
	def attempt(self, candidate: dict | None) -> Iterator[Attempt | None]:
		self.flush()
		attempt = self.begin(candidate)
		try:
			yield attempt
		finally:
			if attempt is not None:
				self.finish(attempt)

	@asynccontextmanager
	async def async_attempt(self, candidate: dict | None) -> AsyncIterator[Attempt | None]:
		await asyncio.to_thread(self.flush)
		attempt = await asyncio.to_thread(self.begin, candidate)
		try:
			yield attempt
		finally:
			if attempt is not None:
				await asyncio.to_thread(self.finish, attempt)

	def begin(self, candidate: dict | None) -> Attempt | None:
		try:
			attempt_id = self.journal.begin(self.run_ref)
			granted = self.reserve(
				{
					"attempt_id": attempt_id,
					"run_ref": self.run_ref,
					"company": self.company,
					"project": self.project,
					"credential": self.credential.model_dump() if self.credential else None,
					"candidate": candidate,
				}
			)
		except BudgetError:
			raise
		except Exception:
			raise BudgetError("budget_unavailable") from None
		if granted:
			if candidate is None:
				raise BudgetError("budget_unpriced")
			return Attempt(attempt_id)
		self.journal.discard(attempt_id)
		return None

	def finish(self, attempt: Attempt) -> None:
		payload = {"attempt_id": attempt.id, **attempt.result}
		try:
			self.journal.store(attempt.id, payload)
			self._deliver(payload)
		except Exception:
			# The authority holds the reservation even if a local disk or network fails.
			_logger.warning("Could not record provider budget settlement for %s", attempt.id)

	def flush(self) -> None:
		try:
			for payload in self.journal.pending():
				self._deliver(payload)
		except Exception:
			_logger.warning("Could not read pending provider budget settlements")

	def _deliver(self, payload: dict) -> None:
		try:
			self.settle(payload)
			self.journal.store(payload["attempt_id"], None)
		except Exception:
			_logger.warning("Provider budget settlement still pending for %s", payload["attempt_id"])


type BudgetFactory = Callable[[BudgetScope, CredentialBinding | None], Budget]


class Attempt:
	def __init__(self, attempt_id: str):
		self.id = attempt_id
		self.result: dict = {"outcome": "uncertain"}

	def complete(self, usage: dict, response_id: str | None = None) -> None:
		self.result = {"outcome": "completed", "usage": usage, "provider_request_ref": response_id}


def post_budget(url: str, headers: Mapping[str, str], payload: dict) -> dict:
	"""Read the common Frappe envelope without leaking request or response text."""
	try:
		with httpx.Client(timeout=10, trust_env=False, follow_redirects=False) as client:
			response = client.post(url, headers=headers, json=payload)
		message = response.json()["message"]
		if not isinstance(message, dict):
			raise ValueError
	except (httpx.HTTPError, KeyError, TypeError, ValueError):
		raise BudgetError("budget_unavailable") from None
	if response.is_error or response.is_redirect or "code" in message:
		code = message.get("code")
		if not isinstance(code, str) or not code.startswith(("budget_", "attempt_")):
			code = "budget_unavailable"
		raise BudgetError(code, response.status_code)
	return message


def candidate_for(model: str, input_max: int | None, output_max: int | None) -> dict | None:
	if type(input_max) is not int or input_max <= 0 or type(output_max) is not int or output_max < 0:
		return None
	return {
		"model": model,
		"input_tokens_max": input_max,
		"output_tokens_max": output_max,
		"extra_units_max": {},
	}


def measured_usage(raw: Any, *, embedding: bool = False) -> dict:
	"""Keep missing usage distinct from a reported zero."""
	get = raw.get if isinstance(raw, dict) else lambda key: getattr(raw, key, None)
	prompt = get("prompt_tokens")
	output = 0 if embedding else get("completion_tokens")
	if type(prompt) is not int or type(output) is not int:
		return {}
	details = get("prompt_tokens_details")
	cached = (
		details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", 0)
	)
	return {"input_tokens": prompt, "output_tokens": output, "cached_input_tokens": cached or 0}


def local_model(model: str, api_base: str | None = None) -> bool:
	if model.split("/", 1)[0] not in {"ollama", "ollama_chat"}:
		return model.split("/", 1)[0] == "mock"
	return bool(api_base) and urlsplit(api_base).hostname in {"localhost", "127.0.0.1", "::1"}
