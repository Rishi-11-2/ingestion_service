# ingestion_service.py
import os
import json
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify

load_dotenv()

# ---------------- Config ----------------
REDIS_URL = os.getenv("REDIS_URL", None)
QUEUE_KEY = os.getenv("QUEUE_KEY", "ingest:queue")
PENDING_SET = os.getenv("PENDING_SET", "ingest:pending")
PROCESSING_SET = os.getenv("PROCESSING_SET", "ingest:processing")
RESULT_LIST = os.getenv("RESULT_LIST", "ingest:results")
MAX_RESULT_LOG = int(os.getenv("MAX_RESULT_LOG", "200"))
# BRPOP_TIMEOUT is no longer needed here

# NOTE: The import for search_and_ingest_papers is removed
#       as this service no longer runs the ingestion.

app = Flask(__name__)

# ---------- Redis client ----------
if not REDIS_URL:
    raise RuntimeError("REDIS_URL environment variable must be set (e.g. redis://:PASS@host:6379/0).")

try:
    import redis
    r = redis.from_url(REDIS_URL, decode_responses=True)
    # optional ping (will raise on auth/network issues)
    try:
        r.ping()
        print("[INFO] Redis connection successful.", flush=True)
    except Exception as e:
        print("[WARN] Redis ping failed at startup:", e, flush=True)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Redis client: {e}")

# ---------------- HTTP endpoints ----------------
def query_key(query: str) -> str:
    return str(hash(query.strip().lower()))

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}), 200

@app.route("/enqueue", methods=["POST"])
def enqueue():
    print("ENQUEUE")
    data = request.get_json(force=True, silent=True)
    print(data)
    if not data or "query" not in data:
        return jsonify({"status": "error", "message": "JSON payload with 'query' required"}), 400
    query = data["query"]
    if not isinstance(query, str) or not query.strip():
        return jsonify({"status": "error", "message": "non-empty string required for 'query'"}), 400

    key = query_key(query)
    try:
        added = r.sadd(PENDING_SET, key)
        if added == 1:
            payload = json.dumps({"task_id": key, "query": query})
            r.lpush(QUEUE_KEY, payload)  # LPUSH -> worker BRPOP yields FIFO
            queue_size = r.llen(QUEUE_KEY)
            return jsonify({"status": "queued", "task_id": key, "queue_size": queue_size}), 202
        else:
            if r.sismember(PROCESSING_SET, key):
                return jsonify({"status": "duplicate", "message": "Query already being processed"}), 200
            else:
                return jsonify({"status": "duplicate", "message": "Query already queued / pending"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    try:
        qlen = r.llen(QUEUE_KEY)
        pending = r.scard(PENDING_SET)
        processing = r.scard(PROCESSING_SET)
        recent = r.lrange(RESULT_LIST, 0, 9)
        return jsonify({
            "queue_len": qlen,
            "pending": pending,
            "processing": processing,
            "recent_queries": recent
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ---------------- Background worker (REMOVED) ----------------
# All worker code (_worker_thread, _stop_event, _process_payload,
# worker_loop, start_worker_thread, _shutdown_worker, atexit)
# has been moved to worker.py

# -------------- Run with gunicorn in production --------------
# Gunicorn command:
# gunicorn ingestion_service:app --bind 0.0.0.0:$PORT --workers 4 --threads 2 --timeout 120

if __name__ == "__main__":
    # local debug server (not for production)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5001)), debug=False)