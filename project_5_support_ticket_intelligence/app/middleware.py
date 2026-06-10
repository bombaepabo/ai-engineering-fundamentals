# app/middleware.py
import logging
import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("app.api")


class LoggingAndRequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Injects a unique X-Request-ID into request/response headers.
    2. Calculates request execution latency in milliseconds.
    3. Outputs structured JSON logs for observability.
    """
    async def dispatch(self, request: Request, call_next):
        # 1. Retrieve or generate Request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # Make request ID accessible during the request lifetime
        request.state.request_id = request_id
        
        # 2. Record start time
        start_time = time.perf_counter()
        
        # 3. Process the request
        try:
            response = await call_next(request)
        except Exception as e:
            # If request crashed, calculate latency and log the error before propagating
            latency = (time.perf_counter() - start_time) * 1000
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "extra_fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": 500,
                        "latency_ms": round(latency, 2),
                        "error": str(e)
                    }
                }
            )
            raise e
            
        # 4. Record end time and calculate latency
        latency = (time.perf_counter() - start_time) * 1000
        
        # Inject Request ID into response headers
        response.headers["X-Request-ID"] = request_id
        
        # 5. Log the HTTP request using JSON formatting
        logger.info(
            f"HTTP {request.method} {request.url.path} returned {response.status_code}",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(latency, 2)
                }
            }
        )
        
        return response