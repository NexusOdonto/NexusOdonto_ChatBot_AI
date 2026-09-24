#!/bin/bash
# Smoke latency for "quiero una cita" against local chatbot webhook.
set -euo pipefail
cd /home/ubuntu/nexusodonto/chatbot
set -a
# shellcheck disable=SC1091
. ./.env
set +a
SECRET="${WEBHOOK_SECRET:-$EVOLUTION_API_KEY}"
N="${1:-1}"
WAIT_S="${2:-55}"
TS=$(date +%s)
PREFIX="${3:-smoke}"
echo "latency_smoke n=$N wait=${WAIT_S}s ts=$TS model=$(grep -E '^GEMINI_MODEL=' .env || true)"
for i in $(seq 1 "$N"); do
  PHONE="57311${PREFIX:0:2}${i}${TS: -4}"
  # Keep phone digits-only length ~12
  PHONE="5731$((10000000 + (TS % 1000000) + i))"
  MID="${PREFIX}${TS}${i}"
  (
    code=$(curl -s -o "/tmp/${PREFIX}${i}.json" -w "%{http_code} %{time_total}" \
      -X POST "http://127.0.0.1:8000/webhook/whatsapp" \
      -H "Content-Type: application/json" \
      -H "apikey: ${SECRET}" \
      -H "Authorization: Bearer ${SECRET}" \
      -d "{\"event\":\"messages.upsert\",\"instance\":\"Nexus_Odonto\",\"data\":{\"key\":{\"remoteJid\":\"${PHONE}@s.whatsapp.net\",\"fromMe\":false,\"id\":\"${MID}\"},\"pushName\":\"LatSmoke${i}\",\"message\":{\"conversation\":\"quiero una cita\"}}}")
    echo "w${i} phone=${PHONE} ${code}s body=$(cat /tmp/${PREFIX}${i}.json)"
  ) &
done
wait
echo "--- waiting ${WAIT_S}s for graph ---"
sleep "$WAIT_S"
echo "--- latency logs ---"
docker logs --since $((WAIT_S + 30))s agente_python 2>&1 | grep -E "graph_timeout|llm_call_timeout|queue_wait|path=graph|path=cache|llm_turn|tools=|Skip L2|Fast-Path|max_concurrent|LatSmoke|5731" | tail -100 || true
echo "smoke_done"