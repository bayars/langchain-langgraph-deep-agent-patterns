"""Playwright diagnostic — captures all console output after clicking Test SSE."""
import time
from playwright.sync_api import sync_playwright

URL = "http://10.0.0.172:8000"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    logs = []
    page.on("console",   lambda m: logs.append(f"[{m.type}] {m.text}"))
    page.on("pageerror", lambda e: logs.append(f"[PAGEERROR] {e}"))

    print(f"Loading {URL} ...")
    page.goto(URL, wait_until="networkidle", timeout=15000)

    # Clear any startup logs
    logs.clear()

    print("Clicking Test SSE...")
    page.click("button:has-text('Test SSE')")
    time.sleep(8)

    print(f"\n=== Console output ({len(logs)} lines) ===")
    for line in logs:
        print(line)

    ai_msgs = page.query_selector_all(".msg--ai")
    print(f"\nAI bubbles in DOM: {len(ai_msgs)}")
    chat = page.inner_text(".chat__messages")
    print("Chat text:", repr(chat[:300]))

    browser.close()
