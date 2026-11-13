# One Raft node: python -m raft.net.server. See docker-compose.yml.
FROM python:3.12-slim

# iproute2 for `ss -tin` (TCP retransmits, rto, backoff) when debugging.
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[net]"

# Peers dial each node by its compose service name (n0:7000, ...), which is
# its address in the cluster config; it listens on every interface.
EXPOSE 7000
VOLUME /data
ENTRYPOINT ["python", "-m", "raft.net.server", "--config", "/etc/raft/cluster.json", \
            "--data-dir", "/data", "--listen", "0.0.0.0:7000"]
