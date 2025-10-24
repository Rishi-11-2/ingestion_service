# ingestion_service.py
import os
import json
import hashlib
from flask import Flask, request, jsonify
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = os.getenv("QUEUE_KEY", "ingest:queue")
PENDING_SET = os.getenv("PENDING_SET", "ingest:pending")
PROCESSING_SET = os.getenv("PROCESSING_SET", "ingest:processing")
RESULT_LIST = os.getenv("RESULT_LIST", "ingest:results")

r = redis.from_url(REDIS_URL, decode_responses=True)
app = Flask(__name__)

def query_key(query: str) -> str:
    q = query.strip().lower()
    return hashlib.sha256(q.encode("utf-8")).hexdigest()

@app.route("/enqueue", methods=["POST"])
def enqueue():
    """
    JSON body: {"query": "<text>"}
    Responses:
      202 queued  -> {"status":"queued","task_id": "<sha256>", "queue_size": N}
      200 duplicate -> {"status":"duplicate","message": "..."}
      400/500 error -> {"status":"error","message": "..."}
    """
    data = request.get_json(force=True, silent=True)
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
            # LPUSH + worker BRPOP => FIFO queue
            r.lpush(QUEUE_KEY, payload)
            queue_size = r.llen(QUEUE_KEY)
            return jsonify({"status": "queued", "task_id": key, "queue_size": queue_size}), 202
        else:
            # Already in pending or processing
            if r.sismember(PROCESSING_SET, key):
                return jsonify({"status": "duplicate", "message": "Query already being processed"}), 200
            else:
                return jsonify({"status": "duplicate", "message": "Query already queued / pending"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    """Return queue length and recent processed query strings."""
    try:
        queue_len = r.llen(QUEUE_KEY)
        pending = r.scard(PENDING_SET)
        processing = r.scard(PROCESSING_SET)
        recent_queries = r.lrange(RESULT_LIST, 0, 9)  # plain strings
        return jsonify({
            "queue_len": queue_len,
            "pending": pending,
            "processing": processing,
            "recent_queries": recent_queries
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    # For local testing only — on Render use gunicorn start command below
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5001")), debug=False)
