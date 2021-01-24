import pathlib
import textwrap


def tree(base):
    import textwrap
    lines = []
    files = sorted(base.iterdir(), key=lambda s: s.name.lower())

    for num, path in enumerate(files, start=1):
        prefix = '└── ' if num == len(files) else '├── '

        if path.name.startswith('.'):
            continue

        lines.append(prefix + path.name)

        if path.is_dir():
            indent = '   ' if num == len(files) else '|   '
            lines.append(textwrap.indent(tree(path), prefix=indent))

    return '\n'.join(lines)


print(tree(pathlib.Path("..")))