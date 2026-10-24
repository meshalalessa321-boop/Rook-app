"""Entrypoint that installs safe optional agent tools before serving requests."""
import os
import uvicorn
import app as meridian
from agent_extensions import install

install(meridian.__dict__)

if __name__ == "__main__":
    uvicorn.run(
        meridian.app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
