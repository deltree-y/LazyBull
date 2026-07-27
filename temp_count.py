with open('scripts/batch/batch_walk_forward.ps1', 'r', encoding='utf-8') as f:
    content = f.read()
o = content.count('{')
c = content.count('}')
print(f"Open: {o}, Close: {c}, Diff: {o-c}")
