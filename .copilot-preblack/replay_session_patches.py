import json
from pathlib import Path, PureWindowsPath

TRANSCRIPT = Path(
    r"C:\Users\BBMM\AppData\Roaming\Code\User\workspaceStorage"
    r"\46a76ba303bf5532cebb592c2539aac7\GitHub.copilot-chat"
    r"\transcripts\75cda24f-ad36-4570-ae43-98a4d23e0832.jsonl"
)
SOURCE_ROOT = PureWindowsPath(r"D:\my_pro\LazyBull")
TARGET_ROOT = Path(r"D:\my_pro\LazyBull\.copilot-preblack")
FAILED_PATCH_CALLS = {
    "call_xcIvlvd3XUH4980AJ0n6E3oT",
    "call_OQyLIutSJjFL6bgc0iStmEVe",
}


def find_sequence(lines, needle, start_index):
    if not needle:
        return None
    matches = []
    width = len(needle)
    for index in range(start_index, len(lines) - width + 1):
        if lines[index : index + width] == needle:
            matches.append(index)
    if not matches:
        raise RuntimeError(
            f"补丁上下文未匹配: start={start_index}, first={needle[:2]}"
        )
    return matches[0]


def apply_hunk(target, hunk, start_index):
    body = [line for line in hunk if not line.startswith("@@")]
    if not any(line.startswith(("+", "-")) for line in body):
        anchor = [line[1:] if line.startswith(" ") else line for line in body]
        if not anchor:
            return start_index
        text = target.read_text(encoding="utf-8")
        lines = text.splitlines()
        index = find_sequence(lines, anchor, start_index)
        return index + len(anchor)
    old = [
        line[1:] if line[:1] in {"-", " "} else line
        for line in body
        if not line.startswith("+")
    ]
    new = [
        line[1:] if line[:1] in {"+", " "} else line
        for line in body
        if not line.startswith("-")
    ]

    text = target.read_text(encoding="utf-8")
    trailing_newline = text.endswith("\n")
    lines = text.splitlines()
    index = find_sequence(lines, old, start_index)
    lines[index : index + len(old)] = new
    target.write_text("\n".join(lines) + ("\n" if trailing_newline else ""), encoding="utf-8")
    return index + len(new)


def apply_patch(patch_text):
    current_target = None
    current_hunk = []
    current_cursor = 0

    def flush_hunk():
        nonlocal current_hunk, current_cursor
        if current_target is not None and current_hunk:
            current_cursor = apply_hunk(current_target, current_hunk, current_cursor)
        current_hunk = []

    for line in patch_text.splitlines():
        if line.startswith("*** Update File: "):
            flush_hunk()
            source_path = PureWindowsPath(line.removeprefix("*** Update File: "))
            relative_path = source_path.relative_to(SOURCE_ROOT)
            current_target = TARGET_ROOT.joinpath(*relative_path.parts)
            current_cursor = 0
        elif line.startswith("@@"):
            flush_hunk()
            current_hunk = [line]
        elif line in {"*** Begin Patch", "*** End Patch"}:
            flush_hunk()
        elif line.startswith("*** "):
            raise RuntimeError(f"不支持的补丁动作: {line}")
        elif current_target is not None:
            current_hunk.append(line)
    flush_hunk()


def main():
    entries = []
    with TRANSCRIPT.open("r", encoding="utf-8") as file:
        for line in file:
            entries.append(json.loads(line))

    successful_calls = {
        entry["data"]["toolCallId"]
        for entry in entries
        if entry.get("type") == "tool.execution_complete"
        and entry.get("data", {}).get("success") is True
    }
    patches = [
        entry
        for entry in entries
        if entry.get("type") == "tool.execution_start"
        and entry.get("data", {}).get("toolName") == "apply_patch"
        and entry["data"]["toolCallId"] in successful_calls
        and entry["data"]["toolCallId"] not in FAILED_PATCH_CALLS
        and ".copilot-preblack" not in entry["data"]["arguments"]["input"]
    ]

    for index, entry in enumerate(patches, start=1):
        explanation = entry["data"]["arguments"].get("explanation", "")
        print(f"[{index}/{len(patches)}] {explanation}")
        apply_patch(entry["data"]["arguments"]["input"])


if __name__ == "__main__":
    main()
