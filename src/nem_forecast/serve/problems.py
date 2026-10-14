"""RFC 9457 problem details: every error the API returns is ``application/problem+json``.

Each problem has a stable ``type`` URI pointing at its explanation in docs/problems.md, a short
``title`` that never changes for that type, and a ``detail`` specific to the request. Unexpected
errors are reported as a generic 500 problem; internals never reach the client.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger(__name__)

PROBLEM_JSON = "application/problem+json"
DOCS = "https://github.com/proxykhan/nem-forecast/blob/main/docs/problems.md"

_HTTP_TITLES = {
    404: ("not-found", "Not found"),
    405: ("method-not-allowed", "Method not allowed"),
}


class Problem(Exception):
    def __init__(
        self,
        status: int,
        slug: str,
        title: str,
        detail: str,
        *,
        headers: dict[str, str] | None = None,
        **extensions: Any,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.slug = slug
        self.title = title
        self.detail = detail
        self.headers = headers or {}
        self.extensions = extensions


def problem_response(problem: Problem, instance: str) -> JSONResponse:
    body = {
        "type": f"{DOCS}#{problem.slug}",
        "title": problem.title,
        "status": problem.status,
        "detail": problem.detail,
        "instance": instance,
        **jsonable_encoder(problem.extensions),
    }
    return JSONResponse(
        body, status_code=problem.status, media_type=PROBLEM_JSON, headers=problem.headers
    )


def install_problem_handlers(app: FastAPI) -> None:
    @app.exception_handler(Problem)
    async def _problem(request: Request, exc: Problem) -> JSONResponse:
        return problem_response(exc, request.url.path)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {
                "location": list(error.get("loc", ())),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        summary = "; ".join(
            f"{'.'.join(str(p) for p in e['location'][1:]) or 'body'}: {e['message']}"
            for e in errors
        )
        return problem_response(
            Problem(422, "invalid-request", "Request failed validation", summary, errors=errors),
            request.url.path,
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        slug, title = _HTTP_TITLES.get(exc.status_code, ("http-error", "HTTP error"))
        return problem_response(
            Problem(exc.status_code, slug, title, str(exc.detail)), request.url.path
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return problem_response(
            Problem(
                500,
                "internal-error",
                "Internal error",
                "The server could not complete the request.",
            ),
            request.url.path,
        )
