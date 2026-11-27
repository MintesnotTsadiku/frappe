# Copyright (c) 2026, Frappe Technologies and contributors
# License: MIT. See LICENSE

"""Embedding calls for the knowledge store, via litellm.

Credentials resolve the same way as chat models: the Flow Model row first,
falling back to the central Flow Provider store.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from flow_harness.model.budget import candidate_for, local_model, measured_usage
from flow_harness.model.litellm import managed_input_limit
from flow_harness.model.managed_transport import sync_client

import frappe
from frappe import _

BATCH_SIZE = 96  # smallest common provider batch limit (Cohere)
EMBED_TIMEOUT = 120
PROBE_TIMEOUT = 30


def embed_texts(texts: list[str]) -> list[list[float]]:
	"""Embed texts with the model from Flow Knowledge Settings, preserving order."""
	if not texts:
		return []
	config = _embedding_config()
	vectors: list[list[float]] = []
	for start in range(0, len(texts), BATCH_SIZE):
		vectors.extend(_embed_batch(texts[start : start + BATCH_SIZE], config, timeout=EMBED_TIMEOUT))
	return vectors


def probe_dimension(model: str) -> int:
	"""Vector width a Flow Model's embeddings come back with, via a one-input call."""
	config = _model_config(model)
	(vector,) = _embed_batch(["dimension probe"], config, timeout=PROBE_TIMEOUT)
	return len(vector)


def _embedding_config() -> dict[str, Any]:
	settings = frappe.get_cached_doc("Flow Knowledge Settings")
	if not settings.embedding_model:
		frappe.throw(
			_("Set an embedding model in Flow Knowledge Settings first."),
			title=_("Knowledge Not Configured"),
		)
	return _model_config(settings.embedding_model)


def _model_config(model: str) -> dict[str, Any]:
	from frappe.flow.daemon.provider_budget import direct_budget
	from frappe.flow.lib.model import litellm_route, model_credential, resolve_provider_credentials

	doc = frappe.get_doc("Flow Model", model)
	if not doc.enabled:
		frappe.throw(_("Flow Model {0} is disabled.").format(model), title=_("Model Disabled"))

	provider_creds = resolve_provider_credentials(doc.model_id, doc.provider)
	model_id, base_url = litellm_route(doc.model_id, doc.base_url or provider_creds.get("base_url"))
	key, reference = model_credential(doc, provider_creds)
	config: dict[str, Any] = {
		**doc.model_params(),
		"model": model_id,
		"api_key": key or "",
	}
	if base_url:
		config["api_base"] = base_url
	config["budget"] = direct_budget(reference)
	return config


def _embed_batch(texts: list[str], config: dict[str, Any], *, timeout: int) -> list[list[float]]:
	try:
		response = embed_request(texts, timeout, **config)
	except Exception as e:
		frappe.throw(str(e)[:500] or type(e).__name__, title=_("Embedding Failed"))

	entries = sorted(_get(response, "data"), key=lambda entry: _get(entry, "index") or 0)
	vectors = [[float(value) for value in _get(entry, "embedding")] for entry in entries]
	if len(vectors) != len(texts):
		frappe.throw(
			_("Embedding provider returned {0} vectors for {1} inputs.").format(len(vectors), len(texts)),
			title=_("Embedding Failed"),
		)
	return vectors


def embed_request(texts: list[str], timeout: int, *, budget=None, **kwargs) -> Any:
	import litellm

	route = kwargs["model"]
	provider, _, model = route.partition("/")
	if local_model(route, kwargs.get("api_base")) and not (kwargs.keys() - {"model", "api_key", "api_base"}):
		budget = None
	candidate = None
	if (
		budget is not None
		and provider in {"openai", "gemini"}
		and kwargs.get("api_key")
		and not (kwargs.keys() - {"model", "api_key", "api_base", "dimensions", "encoding_format"})
	):
		context = managed_input_limit(route)
		if context:
			candidate = candidate_for(model, context * len(texts), 0)
	with budget.attempt(candidate) if budget is not None else nullcontext() as attempt:
		if attempt is not None and provider == "openai":
			response = _openai_embedding(texts, timeout, **kwargs)
		else:
			if attempt is None:
				return litellm.embedding(input=texts, timeout=timeout, **kwargs)
			with sync_client(kwargs, timeout) as client:
				response = litellm.embedding(
					input=texts,
					timeout=timeout,
					client=client,
					num_retries=0,
					max_retries=0,
					caching=False,
					**kwargs,
				)
		if attempt is not None:
			attempt.complete(measured_usage(_get(response, "usage"), embedding=True), _get(response, "id"))
		return response


def _openai_embedding(texts, timeout, *, model, api_key, api_base=None, **params):
	# LiteLLM's OpenAI embedding adapter replaces zero SDK retries with its default.
	with sync_client({"model": model, "api_key": api_key, "api_base": api_base}, timeout) as client:
		return client.embeddings.create(model=model.partition("/")[2], input=texts, **params)


def _get(entry: Any, key: str) -> Any:
	return entry.get(key) if isinstance(entry, dict) else getattr(entry, key, None)
