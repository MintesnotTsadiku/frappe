"""One HTTP send per managed LiteLLM invocation."""

from contextlib import asynccontextmanager, contextmanager

import httpx


class SingleSend:
	def __init__(self):
		self.sent = False

	def check(self, request: httpx.Request) -> None:
		if self.sent:
			raise httpx.RequestError("A provider retry needs its own budget grant", request=request)
		self.sent = True

	async def check_async(self, request: httpx.Request) -> None:
		self.check(request)


@asynccontextmanager
async def async_client(options, timeout):
	from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
	from openai import AsyncOpenAI

	hooks = {"request": [SingleSend().check_async]}
	if options["model"].startswith("openai/"):
		async with httpx.AsyncClient(event_hooks=hooks, follow_redirects=False) as wire:
			async with AsyncOpenAI(
				api_key=options["api_key"],
				base_url=options.get("api_base"),
				max_retries=0,
				timeout=timeout,
				http_client=wire,
			) as client:
				yield client
		return
	handler = AsyncHTTPHandler(timeout=timeout, event_hooks=hooks)
	try:
		handler.client.follow_redirects = False
		yield handler
	finally:
		await handler.close()


@contextmanager
def sync_client(options, timeout):
	from litellm.llms.custom_httpx.http_handler import HTTPHandler
	from openai import OpenAI

	hooks = {"request": [SingleSend().check]}
	with httpx.Client(event_hooks=hooks, follow_redirects=False, timeout=timeout) as wire:
		if options["model"].startswith("openai/"):
			with OpenAI(
				api_key=options["api_key"],
				base_url=options.get("api_base"),
				max_retries=0,
				timeout=timeout,
				http_client=wire,
			) as client:
				yield client
		else:
			yield HTTPHandler(client=wire)
