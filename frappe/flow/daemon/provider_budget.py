"""Resolve Flow's credential reference and call the authority as this site."""

from __future__ import annotations

from urllib.parse import urlsplit
from uuid import uuid4

from flow_harness.model.budget import AttemptJournal, Budget, BudgetError, post_budget

import frappe


def settings():
	return frappe.get_single("Flow Settings")


def authority_url(value: str | None) -> str | None:
	try:
		url = urlsplit(value or "")
		if (
			url.scheme == "https"
			and url.hostname
			and not any((url.username, url.password, url.query, url.fragment))
		):
			return url.geturl().rstrip("/")
	except ValueError:
		pass
	return None


def credential_reference(reference: str | None) -> dict | None:
	config = settings()
	if not config.manage_paid_credentials and not config.provider_budget_bindings:
		return None
	binding = _binding(config, reference)
	return {"reference": reference or "", "binding": binding}


def direct_budget(reference: str | None) -> Budget | None:
	"""For synchronous calls made inside the site, such as knowledge embeddings."""
	from flow_harness.session.config import CredentialBinding

	from frappe.flow.daemon.lifecycle import site_data_dir

	credential = credential_reference(reference)
	if credential is None:
		return None
	return Budget(
		AttemptJournal(site_data_dir() / "budget-attempts.sqlite3"),
		f"flow-{uuid4().hex}",
		reserve,
		settle,
		CredentialBinding.model_validate(credential),
	)


@frappe.whitelist(allow_guest=True, methods=["POST"])  # nosemgrep: guest-whitelisted-method
def dispatch(operation: str, payload: dict) -> dict:
	"""Only the daemon's site token can use this route."""
	from frappe.flow.daemon.callback import _authenticate

	_authenticate()
	try:
		if operation == "reserve":
			if not reserve(payload):
				return {"managed": False}
			return {"attempt_id": payload["attempt_id"], "dispatch_granted": True}
		if operation == "settle":
			settle(payload)
			return {"attempt_id": payload["attempt_id"]}
		raise BudgetError("budget_unavailable")
	except BudgetError as error:
		frappe.response.http_status_code = error.status_code or 503
		return {"code": error.code}


def reserve(payload: dict) -> bool:
	config = settings()
	credential = payload.get("credential") or {}
	if (
		not config.manage_paid_credentials
		and not config.provider_budget_bindings
		and not credential.get("binding")
	):
		return False
	binding = _binding(config, credential.get("reference"))
	if not config.manage_paid_credentials or not binding or credential.get("binding") not in (None, binding):
		raise BudgetError("budget_unconfigured")
	candidate = payload.get("candidate")
	if not candidate:
		raise BudgetError("budget_unpriced")
	attempt_id = payload["attempt_id"]
	result = _call(
		config,
		"reserve",
		{
			"attempt_id": attempt_id,
			"origin_site": frappe.local.site,
			"company": payload.get("company"),
			"project": payload.get("project"),
			"run_ref": payload["run_ref"],
			"candidates": [{**candidate, "binding": binding}],
		},
	)
	if result.get("dispatch_granted") is not True or (
		result.get("attempt_id"),
		result.get("binding"),
		result.get("model"),
		result.get("state"),
	) != (attempt_id, binding, candidate["model"], "Held"):
		raise BudgetError("budget_unavailable")
	return True


def settle(payload: dict) -> None:
	result = _call(settings(), "settle", payload)
	if result.get("attempt_id") != payload["attempt_id"] or result.get("state") not in {
		"Uncertain",
		"Settled",
		"Released",
	}:
		raise BudgetError("budget_unavailable")


def _binding(config, reference: str | None) -> str | None:
	return next(
		(row.binding for row in config.provider_budget_bindings if row.credential_reference == reference),
		None,
	)


def _call(config, operation: str, payload: dict) -> dict:
	url = authority_url(config.authority_url)
	token = config.get_password("authority_credential", raise_exception=False)
	if not url or not token:
		raise BudgetError("budget_unconfigured")
	return post_budget(
		f"{url}/api/method/agent_plane.budgets.api.{operation}",
		{"Authorization": f"token {token}"},
		payload,
	)


def complete_request(model_id: str, reference: str | None, **kwargs):
	"""A synchronous Flow model probe uses the same admission as daemon chat."""
	import litellm
	from flow_harness.model.budget import local_model, measured_usage
	from flow_harness.model.litellm import apply_managed_completion, completion_candidate
	from flow_harness.model.managed_transport import sync_client

	budget = direct_budget(reference)
	if budget is None or (
		local_model(kwargs["model"], kwargs.get("api_base"))
		and not (kwargs.keys() - {"model", "api_key", "api_base", "messages", "max_tokens", "timeout"})
	):
		return litellm.completion(**kwargs)
	candidate = completion_candidate(kwargs, kwargs["model"], model_id)
	with budget.attempt(candidate) as attempt:
		if attempt is None:
			return litellm.completion(**kwargs)
		apply_managed_completion(kwargs, candidate)
		with sync_client(kwargs, kwargs["timeout"]) as client:
			response = litellm.completion(client=client, **kwargs)
		attempt.complete(measured_usage(getattr(response, "usage", None)), getattr(response, "id", None))
		return response
