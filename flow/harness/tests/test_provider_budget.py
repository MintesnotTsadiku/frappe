from types import SimpleNamespace

import pytest

from flow_harness.events import UserMessage
from flow_harness.model import LiteLLMModel, ModelRequest, ProviderError


@pytest.mark.parametrize("stream", [False, True])
async def test_provider_send_requires_a_grant(monkeypatch, stream):
	import litellm

	from flow_harness.model.budget import AttemptJournal, Budget, BudgetError

	calls = []

	def reserve(payload):
		calls.append(("reserve", payload["attempt_id"]))
		return True

	def settle(payload):
		calls.append((payload["outcome"], payload["attempt_id"]))

	async def completion(**kwargs):
		assert calls[-1][0] == "reserve"
		assert kwargs["num_retries"] == 0
		assert kwargs["max_tokens"] == 2
		calls.append(("send", calls[-1][1]))
		usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2)
		if not stream:
			return SimpleNamespace(choices=[], usage=usage)

		async def chunks():
			yield SimpleNamespace(choices=[], usage=usage)

		return chunks()

	monkeypatch.setattr(litellm, "acompletion", completion)
	model = LiteLLMModel("openai/gpt-4o-mini", api_key="unused")
	model.budget = Budget(AttemptJournal(":memory:"), "run", reserve, settle)
	request = ModelRequest(messages=[UserMessage(content="Hello")], stream=stream, max_output_tokens=2)
	for _ in range(2):
		_ = [event async for event in model.run(request)]
	assert [verb for verb, _ in calls] == ["reserve", "send", "completed"] * 2
	assert len({attempt for _, attempt in calls}) == 2

	def denied(payload):
		raise BudgetError("budget_exhausted")

	model.budget.reserve = denied
	with pytest.raises(ProviderError, match="budget_exhausted"):
		_ = [event async for event in model.run(request)]
	assert len(calls) == 6


async def test_interrupted_stream_has_a_new_grant_and_preserves_the_hold(monkeypatch, tmp_path):
	import httpx
	import litellm

	from flow_harness.model.budget import AttemptJournal, Budget

	sends = []
	grants = []
	settlements = []

	def reserve(payload):
		grants.append(payload["attempt_id"])
		return True

	async def completion(**kwargs):
		sends.append(grants[-1])

		async def chunks():
			if len(sends) == 1:
				raise httpx.ReadError("interrupted")
			yield {
				"choices": [{"delta": {"content": "Done"}, "finish_reason": "stop"}],
				"usage": {"prompt_tokens": 5, "completion_tokens": 1},
			}

		return chunks()

	monkeypatch.setattr(litellm, "acompletion", completion)
	monkeypatch.setattr("flow_harness.model.retry._RETRY_BASE_DELAY_SECONDS", 0)
	model = LiteLLMModel("openai/gpt-4o-mini", api_key="unused")
	model.budget = Budget(AttemptJournal(tmp_path / "attempts.sqlite3"), "run", reserve, settlements.append)
	request = ModelRequest(messages=[UserMessage(content="Hello")], max_output_tokens=2)
	events = [event async for event in model.run(request)]
	assert len(set(sends)) == 2
	assert [row["attempt_id"] for row in settlements] == sends
	assert [row["outcome"] for row in settlements] == ["uncertain", "completed"]
	assert any(event.type == "retry" and event.status == "started" for event in events)


async def test_failed_settlement_replays_from_disk_without_replaying_the_send(monkeypatch, tmp_path):
	import litellm

	from flow_harness.model.budget import AttemptJournal, Budget

	path = tmp_path / "attempts.sqlite3"
	grants = []
	sends = []
	results = []

	def reserve(payload):
		grants.append(payload["attempt_id"])
		return True

	def settle(payload):
		results.append(payload)
		if len(results) == 1:
			raise TimeoutError

	async def completion(**kwargs):
		sends.append(grants[-1])
		return {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1}}

	monkeypatch.setattr(litellm, "acompletion", completion)
	request = ModelRequest(messages=[UserMessage(content="Hello")], stream=False, max_output_tokens=2)
	for _ in range(2):
		model = LiteLLMModel("openai/gpt-4o-mini", api_key="unused")
		model.budget = Budget(AttemptJournal(path), "run", reserve, settle)
		_ = [event async for event in model.run(request)]

	assert sends == grants
	assert len(sends) == 2
	assert results[0] == results[1]
	assert results[1]["attempt_id"] != results[2]["attempt_id"]
	assert AttemptJournal(path).pending() == []


async def test_unmanaged_model_keeps_standalone_transport(monkeypatch):
	import litellm

	calls = []

	async def completion(**kwargs):
		calls.append(kwargs)
		return {"choices": []}

	monkeypatch.setattr(litellm, "acompletion", completion)
	model = LiteLLMModel("openai/gpt-4o-mini")
	_ = [event async for event in model.run(ModelRequest(messages=[], stream=False))]
	assert len(calls) == 1
	assert "max_tokens" not in calls[0]
	assert "max_retries" not in calls[0]


async def test_unbounded_and_server_side_tools_are_refused(monkeypatch):
	import litellm

	from flow_harness.model.budget import AttemptJournal, Budget, BudgetError

	def reserve(payload):
		assert payload["candidate"] is None
		raise BudgetError("budget_unpriced")

	async def completion(**kwargs):
		pytest.fail("An unbounded attempt reached the provider")

	monkeypatch.setattr(litellm, "acompletion", completion)
	model = LiteLLMModel("openai/gpt-4o-mini", api_key="unused")
	model.budget = Budget(AttemptJournal(":memory:"), "run", reserve, lambda _: None)
	with pytest.raises(ProviderError, match="budget_unpriced"):
		_ = [event async for event in model.run(ModelRequest(messages=[], tools=[{"type": "web_search"}]))]


async def test_actual_openai_sdk_attempts_are_counted_at_http_transport(monkeypatch, tmp_path):
	import json

	import httpx

	from flow_harness.model.budget import AttemptJournal, Budget

	grants = []
	wire_sends = []

	def reserve(payload):
		grants.append(payload["attempt_id"])
		return True

	async def send(transport, request):
		assert request.url.host == "provider.example.invalid"
		assert json.loads(request.content)["max_tokens"] == 2
		wire_sends.append(grants[-1])
		if len(wire_sends) < 3:
			return httpx.Response(500, json={"error": {"message": "Busy"}}, request=request)
		return httpx.Response(
			200,
			json={
				"id": "answer",
				"object": "chat.completion",
				"created": 0,
				"model": "gpt-4o-mini",
				"choices": [
					{
						"index": 0,
						"message": {"role": "assistant", "content": "Ready"},
						"finish_reason": "stop",
					}
				],
				"usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
			},
			request=request,
		)

	monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
	monkeypatch.setattr("flow_harness.model.retry._RETRY_BASE_DELAY_SECONDS", 0)
	model = LiteLLMModel("openai/gpt-4o-mini", api_key="unused", api_base="https://provider.example.invalid")
	model.budget = Budget(AttemptJournal(tmp_path / "attempts.sqlite3"), "run", reserve, lambda _: None)
	request = ModelRequest(messages=[UserMessage(content="Hello")], max_output_tokens=2, stream=False)
	_ = [event async for event in model.run(request)]
	assert wire_sends == grants
	assert len(set(wire_sends)) == 3
