# Reference image for the Green SARC MCP server (KAOS integration).
#
# KAOS is the caller; this image runs Green SARC as an MCP server exposing the
# Pre-Action Gate and Post-Action Auditor as MCP tools. Register it with KAOS via
# the MCPServer custom resource in examples/kaos_mcp_adapter/kaos_manifests.yaml.
# The dependency runs one way: KAOS -> Green SARC.
#
#   docker build -t ghcr.io/besanson/green-sarc-mcp:0.1.0 -f deploy/Dockerfile .
#
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Install the package with the MCP adapter extra.
RUN pip install --no-cache-dir ".[mcp]"

# PAIS / MCP servers in KAOS listen on :8000.
EXPOSE 8000

# Configure via GREEN_SARC_* env vars (see examples/kaos_mcp_adapter/server.py).
CMD ["python", "examples/kaos_mcp_adapter/server.py"]
