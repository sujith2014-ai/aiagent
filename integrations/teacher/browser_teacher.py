"""A teacher reached through an AI service's WEB interface, driven by OpenClaw's browser tool (no model API key anywhere).
Rules: the operator must acknowledge the site's terms for automated use; the site must pass the policy engine (operator approval bound to the URL);
login, 2FA, captcha and consent screens are never automated: the teacher raises TeacherNeedsHuman with the exact manual action and the request is queued like any unavailable teacher;
only the minimal HelpRequest is typed; a fresh chat is opened per request; the reply is untrusted text that is parsed and validated downstream exactly like any other teacher output."""
from __future__ import annotations
import json, re, time
from dataclasses import dataclass, field
from integrations.teacher.provider import TeacherProvider, HelpRequest
from integrations.teacher.http_providers import SYSTEM_PROMPT, TeacherUnavailable
from integrations.openclaw.browser import OpenClawBrowser, BrowserError, parse_snapshot
from integrations.openclaw.action import ActionDenied, NeedsApproval


class TeacherNeedsHuman(TeacherUnavailable):
    """Manual action required (log in, solve a captcha, accept terms in the site UI). Nothing was sent."""
    def __init__(self, action: str): super().__init__(f"manual action required: {action}"); self.action = action


GENERIC_RESPONSES_JS = "() => Array.from(document.querySelectorAll('[data-message-author-role=\"assistant\"], .msg.assistant, [class*=\"assistant-message\"]')).map(e => e.textContent)"   # textContent, not innerText: innerText collapses runs of spaces, which destroys code indentation inside JSON strings
BLOCK_MARKERS = [r"\blog ?in\b", r"\bsign ?in\b", r"\bsign ?up\b", r"password", r"captcha", r"verify (that )?you are (a )?human", r"two-factor", r"verification code", r"unusual activity"]


@dataclass
class SiteAdapter:
    name: str
    url: str
    terms_acknowledged: bool = False            # the operator confirms that automated use of this site is permitted
    input_names: str = r"message|ask|prompt|chat|send a message"      # accessible name of the message box (regex, case-insensitive)
    submit: str = "Enter"                       # key to press, or "button:<regex>" to click a button by accessible name
    responses_js: str = GENERIC_RESPONSES_JS    # returns the list of assistant message texts on the page
    busy_js: str | None = None                  # returns true while the site is still generating
    block_markers: list = field(default_factory=lambda: list(BLOCK_MARKERS))
    max_wait_s: float = 90.0
    poll_s: float = 1.0
    stable_polls: int = 2


def extract_json(text: str) -> str:
    """The first balanced JSON object in a chat reply (which may be wrapped in prose or code fences)."""
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            if isinstance(obj, dict): return json.dumps(obj)
        except ValueError: continue
    raise TeacherUnavailable("the chat reply contained no JSON object")


def build_prompt(req: HelpRequest) -> str:
    """One line (a newline would submit the message box early)."""
    return " ".join(SYSTEM_PROMPT.split()) + " REQUEST: " + req.to_json()


class BrowserChat:
    """Transport: send one single-line message to the chat site through OpenClaw's browser and return the assistant's reply text."""
    def __init__(self, browser: OpenClawBrowser, site: SiteAdapter, broker=None, task_id: str = "browser-teacher"):
        self.browser, self.site, self.broker, self.task_id = browser, site, broker, task_id
        self.calls = 0

    def _blocked(self, snap: str) -> str | None:
        refs = parse_snapshot(snap)
        for m in self.site.block_markers:
            if re.search(m, snap, re.I):
                return f"the page at {self.site.url} shows a login/verification screen ({m!r}). Open the OpenClaw browser profile on this machine (`openclaw browser --browser-profile openclaw open {self.site.url}`), sign in yourself (do not give credentials to the agent), then retry."
        if not any(r.role in ("textbox", "searchbox", "combobox") and re.search(self.site.input_names, r.name, re.I) for r in refs) and not any(r.role == "textbox" for r in refs):
            return f"no message box was found at {self.site.url}; the site may need a manual step (consent, captcha, plan selection) or the adapter's input_names needs adjusting"
        return None

    def ask(self, prompt: str) -> str:
        if not self.site.terms_acknowledged:
            raise TeacherNeedsHuman(f"confirm that the site's terms permit automated use, then set terms_acknowledged=True for '{self.site.name}'")
        if self.broker is not None:
            try: self.broker.check_or_raise(self.task_id, "browser.chat", {"url": self.site.url})
            except NeedsApproval as e: raise TeacherNeedsHuman(f"operator approval required for browser access to {self.site.url}: {e}")
            except ActionDenied as e: raise TeacherUnavailable(f"browser access denied by policy: {e}")
        self.calls += 1; tab = ""
        try:
            tab = self.browser.open(self.site.url)
            snap = self.browser.snapshot()
            why = self._blocked(snap)
            if why: raise TeacherNeedsHuman(why)
            refs = parse_snapshot(snap)
            box = next((r for r in refs if r.role in ("textbox", "searchbox", "combobox") and re.search(self.site.input_names, r.name, re.I)), None) or next(r for r in refs if r.role == "textbox")
            before = len(self.browser.evaluate(self.site.responses_js) or [])
            self.browser.type(box.ref, prompt)
            if self.site.submit.startswith("button:"):
                pat = self.site.submit[7:]; btn = next((r for r in parse_snapshot(self.browser.snapshot()) if r.role == "button" and re.search(pat, r.name, re.I)), None)
                if not btn: raise TeacherNeedsHuman(f"send button /{pat}/ not found at {self.site.url}")
                self.browser.click(btn.ref)
            else:
                self.browser.press(self.site.submit)
            deadline = time.time() + self.site.max_wait_s; last, same = None, 0
            while time.time() < deadline:
                self.browser.wait_ms(int(self.site.poll_s * 1000))
                msgs = self.browser.evaluate(self.site.responses_js) or []
                busy = bool(self.browser.evaluate(self.site.busy_js)) if self.site.busy_js else False
                cur = msgs[-1] if len(msgs) > before else None
                if cur and cur.strip() and cur == last and not busy:
                    same += 1
                    if same >= self.site.stable_polls: return cur
                else: same = 0
                last = cur
            raise TeacherUnavailable(f"no stable reply from {self.site.name} within {self.site.max_wait_s}s")
        except BrowserError as e:
            raise TeacherUnavailable(f"browser error: {e}")
        finally:
            if tab: self.browser.close(tab)


class BrowserTeacher(TeacherProvider):
    def __init__(self, browser: OpenClawBrowser, site: SiteAdapter, broker=None, task_id: str = "browser-teacher"):
        self.chat = BrowserChat(browser, site, broker, task_id); self.site = site; self.name = f"browser-teacher:{site.name}"

    @property
    def calls(self) -> int: return self.chat.calls

    def advise(self, req: HelpRequest) -> str:
        return extract_json(self.chat.ask(build_prompt(req)))
