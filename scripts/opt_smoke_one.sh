#!/bin/bash
set -eu
cd /home/ubuntu/nexusodonto/chatbot
set -a
. ./.env
set +a
SECRET="${WEBHOOK_SECRET:-$EVOLUTION_API_KEY}"
PHONE="${1:-573199988877}"
MSG="${2:-quiero una cita}"
MID="opt$(date +%s)"
echo "phone=$PHONE msg=$MSG"
curl -s -o /tmp/wopt.json -w "http=%{http_code} t=%{time_total}\n" \
  -X POST "http://127.0.0.1:8000/webhook/whatsapp" \
  -H "Content-Type: application/json" \
  -H "apikey: ${SECRET}" \
  -H "Authorization: Bearer ${SECRET}" \
  -d "{\"event\":\"messages.upsert\",\"instance\":\"Nexus_Odonto\",\"data\":{\"key\":{\"remoteJid\":\"${PHONE}@s.whatsapp.net\",\"fromMe\":false,\"id\":\"${MID}\"},\"pushName\":\"OptSmoke\",\"message\":{\"conversation\":\"${MSG}\"}}}"
echo "body=$(cat /tmp/wopt.json)"
sleep "${3:-8}"
docker logs --since 45s agente_python 2>&1 | grep -E "${PHONE}|Fast-Path|path=|Latency|llm_turn|Error|Traceback|Skip L2" | tail -50
