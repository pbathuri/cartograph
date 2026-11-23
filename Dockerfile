# Runs the Cartograph MCP server over stdio. Mount the folders to index and a volume for the graph:
#   docker build -t cartograph .
#   docker run --rm -v ~/papers:/docs -v cartograph:/data cartograph ingest /docs
#   docker run --rm -i -v cartograph:/data cartograph mcp-server
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir ".[docs]"

ENV CARTOGRAPH_HOME=/data
VOLUME ["/data"]

ENTRYPOINT ["carto"]
CMD ["mcp-server"]
