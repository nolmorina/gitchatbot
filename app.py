import json
import os

from flask import Flask, jsonify, request, send_from_directory

import config
config.validate()

from responder import Responder

app = Flask(__name__, static_folder="static")
responder = Responder()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def status():
    if not os.path.exists(config.SESSION_FILE):
        return jsonify({"loaded": False})
    try:
        with open(config.SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        repos = [
            {"repo_name": r["repo_name"], "chunk_count": r["chunk_count"]}
            for r in data.get("repos", [])
        ]
        return jsonify({"loaded": True, "repos": repos})
    except Exception as e:
        return jsonify({"loaded": False, "error": str(e)})


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        body      = request.get_json(force=True)
        message   = body.get("message", "").strip()
        namespace = body.get("namespace", "both")

        if not message:
            return jsonify({"error": "message is required"}), 400

        if namespace == "both":
            ns = [r["repo_name"] for r in responder.repos] if responder.repos \
                 else ["CTI-API", "Service-Portal-API"]
        else:
            ns = namespace

        result = responder.chat(message, ns)

        return jsonify({
            "answer":          result["answer"],
            "sources":         result["sources"],
            "rewritten_query": result["rewritten_query"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/clear", methods=["POST"])
def clear():
    responder.clear()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
