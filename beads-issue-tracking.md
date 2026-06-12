# Pattern: Issue Tracking with Beads

Beads (`bd`) is a graph-based CLI issue tracker built for AI coding agents. It stores issues in a local Dolt (version-controlled SQL) database and is designed to replace markdown TODO lists.

## Installation

```bash
curl -fsSL https://raw.githubusercontent.com/gastownhall/beads/main/scripts/install.sh | bash
```

## Session start

Run these two commands at the beginning of every work session:

```bash
bd prime    # loads workflow context and persistent project memory
bd ready    # lists unblocked issues — pick one and start there
```

## Core workflow

```bash
bd list                         # see all open issues
bd show <id>                    # read an issue's full detail
bd update <id> --claim          # claim it before starting work
bd close <id>                   # mark done when complete
bd create "Title of new issue"  # open an issue for something discovered mid-session
```

Issue IDs look like `<project-prefix>-a1b2` (the prefix is configured per repo). Reference them in commit messages.

## Before starting implementation

Break the work into small, focused issues first — one issue per module, feature, or logical unit. Create them all with `bd create` before writing any code. Then pull from the board with `bd ready` and work through them in dependency order.

## Rules

- **Never use markdown TODO lists** for task tracking. Beads is the single source of truth.
- **Claim before starting** (`--claim`) so parallel agents don't duplicate work.
- **Close immediately** when done — don't batch closures at the end of a session.
- **File new issues** for anything discovered mid-session rather than doing it inline and losing track.
- Use `bd remember "insight"` to store persistent project memory; don't create MEMORY.md files.

## Session end

Before stopping, file issues for any remaining work, update in-progress items, and close finished ones. Leave the board in a state the next agent can orient from with `bd prime` + `bd ready`.
