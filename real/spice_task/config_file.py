"""Change one top-level line of config/real.yaml in place, keeping every comment."""
import re


def set_line(path, key, value_text):
    """`key: <anything>` -> `key: value_text` (appended if missing). Returns the new line."""
    with open(path) as f:
        text = f.read()
    line = f'{key}: {value_text}'
    new, n = re.subn(rf'^{re.escape(key)}:.*$', line, text, count=1, flags=re.M)
    if n == 0:
        new = text.rstrip('\n') + f'\n{line}\n'
    with open(path, 'w') as f:
        f.write(new)
    return line
