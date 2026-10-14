"""``python -m nem_forecast.serve``: run the API under uvicorn on $PORT (Render sets it)."""

from __future__ import annotations

import logging
import os

import uvicorn


def run(host: str = "0.0.0.0", port: int | None = None) -> None:  # noqa: S104 - container port
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for chatty in ("httpx", "httpx2", "huggingface_hub"):  # one line per HTTP call otherwise
        logging.getLogger(chatty).setLevel(logging.WARNING)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    uvicorn.run(
        "nem_forecast.serve.app:create_app",
        factory=True,
        host=host,
        port=port or int(os.getenv("PORT", "8000")),
        proxy_headers=True,
        forwarded_allow_ips="*",
        access_log=False,  # requests are counted in /metrics instead
    )


if __name__ == "__main__":
    run()
