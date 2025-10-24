# worker.py
import os
import time
import json
import traceback
import redis
from dotenv import load_dotenv

# Import your ingestion routine from the module you placed it in
# Make sure rag_ingestion_service.py (containing search_and_ingest_papers) is in the same repo
from rag_ingestion_service import search_and_ingest_papers

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = os.getenv("QUEUE_KEY", "ingest:queue")
PENDING_SET = os.getenv("PENDING_SET", "ingest:pending")
PROCESSING_SET = os.getenv("PROCESSING_SET", "ingest:processing")
RESULT_LIST = os.getenv("RESULT_LIST", "ingest:results")
MAX_RESULT_LOG = int(os.getenv("MAX_RESULT_LOG", "200"))

r = redis.from_url(REDIS_URL, decode_responses=True)

def safe_parse_payload(payload_str):
    try:
        return json.loads(payload_str)
    except Exception:
        return None

def process_loop():
    print("Worker: starting loop, waiting for items...")
    while True:
        try:
            item = r.brpop(QUEUE_KEY, timeout=5)
            if not item:
                continue
            _, payload = item
            data = safe_parse_payload(payload)
            if not data or "task_id" not in data or "query" not in data:
                print("Worker: malformed payload, skipping:", payload)
                continue

            task_id = data["task_id"]
            query = data["query"]

            # Move from pending -> processing
            try:
                r.srem(PENDING_SET, task_id)
                r.sadd(PROCESSING_SET, task_id)
            except Exception as e:
                print("Worker: warning updating sets:", e)

            print(f"Worker: processing task {task_id} query={query!r}")
            try:
                # This is your ingestion routine; it returns a status string.
                result_msg = search_and_ingest_papers(query)
                status = "success"
            except Exception as e:
                status = "failed"
                print(f"Worker: ingestion error for task {task_id}: {e}")
                print(traceback.format_exc())

            # Per your request: store only the query string in the results list
            try:
                r.lpush(RESULT_LIST, query)
                r.ltrim(RESULT_LIST, 0, MAX_RESULT_LOG - 1)
            except Exception as e:
                print("Worker: failed to write to result list:", e)

            # Remove task from processing set so it can be re-enqueued later if needed
            try:
                r.srem(PROCESSING_SET, task_id)
            except Exception as e:
                print("Worker: warning removing from processing set:", e)

            print(f"Worker: finished task {task_id} status={status}")
        except KeyboardInterrupt:
            print("Worker: interrupted, exiting.")
            break
        except Exception as e:
            print("Worker: exception in main loop:", str(e))
            print(traceback.format_exc())
            time.sleep(1)  # small backoff on unexpected error

if __name__ == "__main__":
    process_loop()
