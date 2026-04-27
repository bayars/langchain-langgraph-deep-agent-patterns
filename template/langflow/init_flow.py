"""
One-shot Langflow initialization script.

Waits for Langflow to be ready, then creates the "Deep Agent Chat" flow
via the Langflow REST API. Safe to re-run — skips if already exists.

Queries Langflow's /api/v1/all endpoint to get exact component templates,
so the flow structure is always compatible with the installed Langflow version.
"""

import json
import os
import sys
import time

import requests

LANGFLOW_URL = os.getenv("LANGFLOW_URL", "http://langflow:7860")
AEGRA_URL    = os.getenv("AEGRA_URL",    "http://template-server:8001")
FLOW_NAME    = "Deep Agent Chat"
SUPERUSER    = os.getenv("LANGFLOW_SUPERUSER",          "admin")
PASSWORD     = os.getenv("LANGFLOW_SUPERUSER_PASSWORD", "admin")


def wait_for_langflow(max_retries: int = 40) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(f"{LANGFLOW_URL}/health", timeout=4)
            if r.status_code == 200:
                print(f"[init] Langflow ready (attempt {attempt})", flush=True)
                return True
        except Exception:
            pass
        print(f"[init] waiting… ({attempt}/{max_retries})", flush=True)
        time.sleep(5)
    return False


def get_auth_headers() -> dict:
    # auto_login endpoint (works when LANGFLOW_AUTO_LOGIN=true + LANGFLOW_SKIP_AUTH_AUTO_LOGIN=true)
    try:
        r = requests.get(f"{LANGFLOW_URL}/api/v1/auto_login", timeout=10)
        if r.status_code == 200:
            token = r.json().get("access_token", "")
            if token:
                return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    except Exception:
        pass
    # Fallback: password login
    try:
        r = requests.post(
            f"{LANGFLOW_URL}/api/v1/login",
            data={"username": SUPERUSER, "password": PASSWORD},
            timeout=10,
        )
        if r.status_code == 200:
            token = r.json().get("access_token", "")
            if token:
                return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    except Exception as exc:
        print(f"[init] login error: {exc}", flush=True)
    return {"Content-Type": "application/json"}


def flow_exists(headers: dict) -> bool:
    try:
        r = requests.get(f"{LANGFLOW_URL}/api/v1/flows/", headers=headers, timeout=10)
        if r.status_code == 200:
            flows = r.json()
            return any(f.get("name") == FLOW_NAME for f in (flows if isinstance(flows, list) else []))
    except Exception:
        pass
    return False


def get_all_components(headers: dict) -> dict:
    """Fetch Langflow's full component registry."""
    r = requests.get(f"{LANGFLOW_URL}/api/v1/all", headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def build_node(node_id: str, comp_type: str, comp_spec: dict, position: dict, overrides: dict = None) -> dict:
    """Build a genericNode dict from a component spec returned by /api/v1/all."""
    template = dict(comp_spec.get("template", {}))
    # Apply value overrides (e.g. set input_value default, aegra_url)
    if overrides:
        for key, val in overrides.items():
            if key in template:
                template[key] = dict(template[key])
                template[key]["value"] = val

    outputs = comp_spec.get("outputs", [])

    return {
        "id":   node_id,
        "type": "genericNode",
        "position": position,
        "data": {
            "id":   node_id,
            "type": comp_type,
            "node": {
                "template":     template,
                "description":  comp_spec.get("description", ""),
                "display_name": comp_spec.get("display_name", comp_type),
                "documentation": comp_spec.get("documentation", ""),
                "base_classes": comp_spec.get("base_classes", []),
                "outputs":      outputs,
                "beta":         comp_spec.get("beta", False),
                "error":        None,
                "edited":       False,
            },
            "outputs": outputs,
        },
    }


def build_edge(eid: str, src_id: str, src_type: str, src_port: str, src_types: list,
               tgt_id: str, tgt_type: str, tgt_field: str, tgt_input_types: list, tgt_ftype: str) -> dict:
    return {
        "id":           eid,
        "source":       src_id,
        "sourceHandle": f"{src_id}|{src_port}|{src_id}",
        "target":       tgt_id,
        "targetHandle": f"{tgt_id}|{tgt_field}|{tgt_id}",
        "animated":     False,
        "data": {
            "sourceHandle": {
                "dataType":    src_type,
                "id":          src_id,
                "name":        src_port,
                "output_types": src_types,
            },
            "targetHandle": {
                "fieldName":  tgt_field,
                "id":         tgt_id,
                "inputTypes": tgt_input_types,
                "type":       tgt_ftype,
            },
        },
    }


def build_flow_payload(all_comps: dict) -> dict:
    chat_in_spec  = all_comps["input_output"]["ChatInput"]
    aegra_spec    = all_comps["custom_components"]["AegraDeepAgent"]
    chat_out_spec = all_comps["input_output"]["ChatOutput"]

    nodes = [
        build_node("ChatInput-1",  "ChatInput",     chat_in_spec,  {"x": 50,  "y": 280}),
        build_node("AegraAgent-1", "AegraDeepAgent", aegra_spec,   {"x": 520, "y": 280},
                   overrides={"aegra_url": AEGRA_URL}),
        build_node("ChatOutput-1", "ChatOutput",    chat_out_spec, {"x": 990, "y": 280}),
    ]

    edges = [
        build_edge(
            "e1",
            "ChatInput-1",  "ChatInput",     "message",  ["Message"],
            "AegraAgent-1", "AegraDeepAgent","input_value", ["Message"], "str",
        ),
        build_edge(
            "e2",
            "AegraAgent-1", "AegraDeepAgent","output",   ["Message"],
            "ChatOutput-1", "ChatOutput",    "input_value", ["Message"], "str",
        ),
    ]

    return {
        "name":        FLOW_NAME,
        "description": "Chat with the Deep Agent — VFS + bash execution + HITL via Aegra API",
        "is_component": False,
        "data": {"nodes": nodes, "edges": edges, "viewport": {"zoom": 0.8, "x": 0, "y": 0}},
    }


def main():
    if not wait_for_langflow():
        print("[init] ERROR: Langflow never became ready.", flush=True)
        sys.exit(1)

    time.sleep(3)  # extra settle time
    headers = get_auth_headers()

    if flow_exists(headers):
        print(f"[init] '{FLOW_NAME}' already exists — skipping.", flush=True)
        return

    print("[init] Fetching component registry…", flush=True)
    try:
        all_comps = get_all_components(headers)
    except Exception as exc:
        print(f"[init] ERROR fetching components: {exc}", flush=True)
        sys.exit(1)

    # Verify required components are present
    missing = []
    for cat, name in [("input_output","ChatInput"), ("input_output","ChatOutput"), ("custom_components","AegraDeepAgent")]:
        if name not in all_comps.get(cat, {}):
            missing.append(f"{cat}/{name}")
    if missing:
        print(f"[init] ERROR: Components not found: {missing}", flush=True)
        sys.exit(1)

    payload = build_flow_payload(all_comps)

    try:
        r = requests.post(
            f"{LANGFLOW_URL}/api/v1/flows/",
            json=payload,
            headers=headers,
            timeout=30,
        )
        if r.status_code in (200, 201):
            flow_id = r.json().get("id", "?")
            print(f"[init] Created '{FLOW_NAME}' (id={flow_id})", flush=True)
        else:
            print(f"[init] WARN: {r.status_code} — {r.text[:400]}", flush=True)
    except Exception as exc:
        print(f"[init] ERROR: {exc}", flush=True)


if __name__ == "__main__":
    main()
