# Virtual Env + direnv — Setup Pattern

One `.envrc` per project. When you `cd` into the directory, direnv automatically
activates the venv and exports all secrets. No manual `source`, no `.env` files,
no separate secret management.

## Setup

```bash
# Create the venv
python3 -m venv .venv

# Install direnv if needed (mac)
brew install direnv

# Add to shell (zsh)
echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc

# Allow the .envrc for this directory
direnv allow
```

## .envrc

```bash
source .venv/bin/activate

# Add API keys and secrets your project needs, e.g.:
# export ANTHROPIC_API_KEY="..."
# export DATABASE_URL="postgresql://..."
# export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"
```

Use absolute paths for file-based credentials — relative paths break when direnv
resolves them from a different working directory.

## .gitignore

Always exclude `.envrc` from version control — it contains live API keys:

```
.envrc
.venv/
```

Add these before the first `git add`. If `.envrc` is ever accidentally committed,
rotate every key in it immediately.

## How it works

- `cd project/` → direnv sources `.envrc` → venv activates, secrets are in env
- `cd ..` → direnv unloads → venv deactivates, env vars are unset
- No need to manually activate or deactivate anything
