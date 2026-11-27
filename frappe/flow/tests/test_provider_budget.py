"""Flow-to-authority admission on an isolated Frappe test site."""

import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import SkipTest
from unittest.mock import patch
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient
from flow_harness.server.app import create_app
from flow_harness.server.policy import DEPLOYED
from flow_harness.server.tenants import TenantTemplate
from flow_harness.session.store import SessionStore
from werkzeug.test import Client as WsgiClient

import frappe
from frappe.app import application
from frappe.core.doctype.user.user import generate_keys
from frappe.flow.daemon import lifecycle, provider_budget
from frappe.flow.knowledge import embedder
from frappe.flow.lib.model import model_config
from frappe.tests import IntegrationTestCase


class TestProviderBudget(IntegrationTestCase):
	def test_missing_binding_prevents_direct_sends(self):
		config = frappe.get_single("Flow Settings")
		config.manage_paid_credentials = 1
		with (
			patch.object(provider_budget, "settings", return_value=config),
			patch("litellm.completion") as provider,
		):
			with self.assertRaisesRegex(Exception, "budget_unconfigured"):
				provider_budget.complete_request("openai/gpt-4o-mini", None, model="openai/gpt-4o-mini")
			provider.assert_not_called()

	def test_daemon_continuation_retry_and_embedding_share_committed_budget(self):
		if not (frappe.local.site.endswith(".localhost") and frappe.conf.get("provider_budget_test_site")):
			raise SkipTest("Committed fixtures require an explicitly marked disposable local site")
		if "agent_plane" not in frappe.get_installed_apps():
			raise SkipTest("Install Agent Plane with provider budgets on the test site")
		budget = AuthorityFixture()
		self.addCleanup(budget.close)
		budget.create()
		self.addCleanup(budget.temp.cleanup)
		publish, options = self._daemon(budget)
		with (
			ThreadPoolExecutor(max_workers=2) as worker,
			patch.object(httpx.HTTPTransport, "handle_request", _local_http(worker, budget.site)),
			patch("litellm.acompletion", side_effect=budget.chat),
			patch("flow_harness.model.retry._RETRY_BASE_DELAY_SECONDS", 0),
			patch.object(embedder, "_openai_embedding", side_effect=budget.embed),
		):
			with publish() as daemon:
				self.assertEqual(daemon.post("/sessions", json=options).status_code, 200)
				reply = daemon.post("/sessions/wake-filter/run?wait=true", json={"message": "Read the note"})
				self.assertEqual(reply.status_code, 200)
				entries = daemon.get("/sessions/wake-filter/entries").json()["entries"]
				self.assertFalse([entry for entry in entries if entry["type"] == "run_error"])
				approval = next(entry for entry in entries if entry["type"] == "approval_required")

			# A new application and store read the original run and credential binding from disk.
			with publish() as daemon:
				reply = daemon.post(
					"/sessions/wake-filter/resume?wait=true",
					json={"approval_seq": approval["seq"], "answers": {"note-call": "Approve"}},
				)
				self.assertEqual(reply.status_code, 200)
				entries = daemon.get("/sessions/wake-filter/entries").json()["entries"]
				self.assertFalse([entry for entry in entries if entry["type"] == "run_error"])
			self.assertEqual(embedder.embed_texts(["Cedar warehouse", "Receiving desk"]), [[1.0], [1.0]])
			self.assertEqual(embedder.probe_dimension(budget.embedding_model), 1)
			self.assertEqual(len(set(budget.sends)), 5)
			rows = budget.attempts()
			self.assertEqual(
				[row.state for row in rows], ["Settled", "Uncertain", "Settled", "Settled", "Settled"]
			)
			self.assertEqual({row.run_ref for row in rows[:3]}, {"dispatch-741"})

			frappe.db.set_value("Provider Budget Limit", budget.limit, "monthly_usd", 0)
			frappe.db.commit()
			with self.assertRaisesRegex(frappe.ValidationError, "budget_exhausted"):
				embedder.embed_texts(["Dispatch note"])
			with self.assertRaisesRegex(frappe.ValidationError, "budget_exhausted"):
				frappe.get_doc("Flow Model", budget.chat_model).test_connection()
			self.assertEqual(len(budget.sends), 5)
			self.assertEqual([row.state for row in budget.attempts()][-2:], ["Denied", "Denied"])

	def _daemon(self, budget):
		root = budget.root
		(root / "receiving.txt").write_text("Gate two is open.")
		lifecycle.provision_site()
		token = frappe.get_site_config().flow_server_secret
		token_file = root / f"{budget.site}.json"
		token_file.touch(mode=0o600)
		token_file.write_text(json.dumps({"token": token}))
		tenant = TenantTemplate(
			token_file=str(root / "{site}.json"),
			token_key="token",
			allowed_roots=(str(root),),
			callback_headers={"X-Frappe-Site-Name": "{site}"},
			budget_url="http://127.0.0.1/api/method/frappe.flow.daemon.provider_budget.dispatch",
		)

		def publish():
			app = create_app(SessionStore(str(root / "daemon")), tenants=tenant, policy=DEPLOYED)
			return TestClient(app, headers={"Authorization": f"Bearer {token}", "X-Flow-Site": budget.site})

		return publish, {
			**model_config(budget.chat_model),
			"session_id": "wake-filter",
			"title": "Receiving dispatch",
			"cwd": str(root),
			"tools": ["read"],
			"confirm_tools": ["read"],
			"approval_mode": "ask",
			"skills_enabled": False,
			"max_turns": 2,
			"budget_scope": {"run_ref": "dispatch-741"},
		}


class AuthorityFixture:
	def __init__(self):
		self.site = frappe.local.site
		self.temp = TemporaryDirectory()
		self.root = Path(self.temp.name)
		self.created = []
		self.sends = []
		self.original = {}
		self.suffix = uuid4().hex[:8]
		self.principal = f"waverly-ledger-{self.suffix}@example.invalid"

	def create(self):
		config = frappe.get_single("Flow Settings")
		if config.manage_paid_credentials or config.provider_budget_bindings or config.authority_credential:
			raise SkipTest("Budget fixtures require an unconfigured Flow Settings record")
		self.original = {
			"Flow Settings": {
				"manage_paid_credentials": config.manage_paid_credentials,
				"authority_url": config.authority_url,
				"authority_credential": config.authority_credential,
			},
			"Flow Knowledge Settings": {
				"embedding_model": frappe.db.get_single_value("Flow Knowledge Settings", "embedding_model"),
			},
		}
		self.add(
			"User",
			email=self.principal,
			first_name="Waverly",
			send_welcome_email=0,
			roles=[{"role": "Provider Budget Caller"}],
		)
		account = self.add(
			"Provider Budget Account", label="Waverly Freight", owning_tenant="Waverly Freight"
		)
		self.limit = self.add(
			"Provider Budget Limit",
			account=account,
			provider="openai",
			scope_type="Account",
			scope_id=account,
			monthly_usd="7",
		)
		self.chat_model = self.model(config, account, "gpt-4o-mini", embedding=False)
		self.embedding_model = self.model(config, account, "text-embedding-3-small", embedding=True)
		config.manage_paid_credentials = 1
		config.authority_url = "https://authority.example.invalid"
		config.authority_credential = _authority_credential(self.principal)
		config.save()
		frappe.db.set_single_value("Flow Knowledge Settings", "embedding_model", self.embedding_model)
		frappe.db.commit()

	def model(self, config, account, model, *, embedding):
		name = self.add(
			"Flow Model",
			title=f"Waverly {model} {self.suffix}",
			model_id=f"openai/{model}",
			api_key=frappe.generate_hash(length=30),
			enabled=1,
			embedding=int(embedding),
			extra_params=None if embedding else '{"max_tokens":2}',
		)
		definition = self.add(
			"Model Definition",
			model_name=f"Waverly {model} {self.suffix}",
			provider="OpenAI",
			provider_model_name=model,
			prompt_cost_per_1k_tokens=0.01 if embedding else 0,
			completion_cost_per_1k_tokens=1000,
		)
		if not embedding:
			self.add(
				"Model Price Version",
				model_definition=definition,
				price_version="v2",
				effective_from=frappe.utils.now_datetime(),
				completion_cost_per_1k_tokens=1000,
				budget_unit_prices=[{"unit_code": "input_tokens", "usd_per_unit": 0}],
			)
		reference = f"Flow Model:{name}"
		binding = self.add(
			"Provider Budget Binding",
			account=account,
			provider="openai",
			principal=self.principal,
			origin_site=self.site,
			credential_reference=reference,
		)
		config.append("provider_budget_bindings", {"credential_reference": reference, "binding": binding})
		return name

	def add(self, doctype, **fields):
		doc = frappe.get_doc(dict(doctype=doctype, **fields)).insert()
		self.created.append((doctype, doc.name))
		return doc.name

	def attempts(self):
		frappe.db.rollback()
		return frappe.get_all(
			"Provider Budget Attempt",
			filters={"principal": self.principal},
			fields=["attempt_id", "state", "run_ref", "max_usd"],
			order_by="creation, name",
		)

	def record_send(self):
		# The provider sees the grant through a connection separate from the authority transaction.
		frappe.init(self.site)
		frappe.connect()
		try:
			rows = [
				row for row in self.attempts() if row.state == "Held" and row.attempt_id not in self.sends
			]
			assert len(rows) == 1
			self.sends.append(rows[0].attempt_id)
			assert Decimal(str(rows[0].max_usd)) > 0
		finally:
			frappe.destroy()

	async def chat(self, **request):
		import asyncio

		await asyncio.to_thread(self.record_send)
		assert request["max_retries"] == 0
		assert request["max_tokens"] == 2
		send = len(self.sends)

		async def chunks():
			if send == 2:
				raise httpx.ReadError("interrupted stream")
			delta = {"content": "Received."}
			if send == 1:
				delta = {
					"tool_calls": [
						{
							"index": 0,
							"id": "note-call",
							"type": "function",
							"function": {"name": "read", "arguments": '{"path":"receiving.txt"}'},
						}
					]
				}
			yield {
				"choices": [{"delta": delta, "finish_reason": "tool_calls" if send == 1 else "stop"}],
				"usage": {"prompt_tokens": 10, "completion_tokens": 1},
			}

		return chunks()

	def embed(self, texts, timeout, **kwargs):
		with ThreadPoolExecutor(max_workers=1) as thread:
			thread.submit(self.record_send).result()
		return {
			"data": [{"index": idx, "embedding": [1.0]} for idx, _ in enumerate(texts)],
			"usage": {"prompt_tokens": 10},
		}

	def close(self):
		if not self.original:
			return
		frappe.db.rollback()
		for doctype, values in self.original.items():
			frappe.db.set_single_value(doctype, values)
		from frappe.utils.password import remove_encrypted_password

		remove_encrypted_password("Flow Settings", "Flow Settings", "authority_credential")
		for doctype, filters in [
			("Flow Budget Binding", {"parent": "Flow Settings"}),
			("Provider Budget Attempt", {"principal": self.principal}),
			("Provider Budget Month", {"budget_limit": getattr(self, "limit", None)}),
		]:
			if all(filters.values()):
				for name in frappe.get_all(doctype, filters=filters, pluck="name"):
					frappe.db.delete(doctype, {"name": name})
		for doctype, name in reversed(self.created):
			if doctype == "Model Price Version":
				continue
			if doctype == "Model Definition":
				for version in frappe.get_all(
					"Model Price Version", {"model_definition": name}, pluck="name"
				):
					frappe.delete_doc("Model Price Version", version, force=True, delete_permanently=True)
			frappe.delete_doc(doctype, name, force=True, delete_permanently=True)
		frappe.db.commit()


def _local_http(worker, site):
	def post(transport, request):
		assert request.url.host in {"127.0.0.1", "authority.example.invalid"}
		response = worker.submit(
			WsgiClient(application).post,
			request.url.path,
			headers={**dict(request.headers), "x-frappe-site-name": site},
			json=json.loads(request.read()),
		).result(timeout=15)
		return httpx.Response(response.status_code, content=response.data, request=request)

	return post


def _authority_credential(principal: str) -> str:
	keys = generate_keys(principal)
	return f"{keys['api_key']}:{keys['api_secret']}"
