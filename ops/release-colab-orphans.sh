#!/usr/bin/env bash
# Release Colab VMs that the CLI can no longer see but that are still billing.
#
# Usage:
#   bash ops/release-colab-orphans.sh          # list what would be released
#   bash ops/release-colab-orphans.sh --yes    # actually release them
#
# Why this exists: each session's runtime-proxy token is issued with a one-hour
# expiry. When it lapses, `colab exec` gets a 404 and the CLI prunes the local
# record — but the VM keeps running and keeps consuming compute units. It then
# shows in `colab sessions` as `[?] <endpoint>` with no name, and `colab stop`
# refuses it ("Session not found") because stop resolves by *name*, and the name
# is exactly what was pruned. `colab new` does not reattach either; it allocates
# a second VM and leaves the first one orphaned.
#
# The server side still lists the assignment and still hands out a fresh token
# for it, so the assignment can be released directly through the same client the
# CLI uses. That is what this script does.

set -euo pipefail

CLI_HOME="${COLAB_CLI_HOME:-$HOME/.local/share/uv/tools/google-colab-cli}"
CLI_PYTHON="${CLI_HOME}/bin/python3"

if [ ! -x "${CLI_PYTHON}" ]; then
  echo "Could not find the colab-cli interpreter at ${CLI_PYTHON}." >&2
  echo "Set COLAB_CLI_HOME to the uv tool directory for google-colab-cli." >&2
  exit 2
fi

APPLY=0
case "${1:-}" in
  --yes|-y) APPLY=1 ;;
  "") ;;
  *) echo "Usage: $0 [--yes]" >&2; exit 2 ;;
esac

# The CLI's own session file is the list of endpoints that are *accounted for*.
# Anything the server lists that is not in there is an orphan.
APPLY="${APPLY}" "${CLI_PYTHON}" - <<'PY'
import json
import os
import pathlib
import sys

from colab_cli.auth import AuthProvider, get_credentials
from colab_cli.client import Client, Prod

apply_changes = os.environ.get("APPLY") == "1"

state = pathlib.Path.home() / ".config/colab-cli/sessions.json"
known = set()
if state.exists():
    try:
        known = {
            entry.get("endpoint")
            for entry in json.loads(state.read_text() or "{}").values()
            if isinstance(entry, dict)
        }
    except json.JSONDecodeError:
        print(f"warning: could not parse {state}; treating every VM as orphaned", file=sys.stderr)

client = Client(Prod(), get_credentials(provider=AuthProvider.OAUTH2))
assignments = client.list_assignments()
orphans = [a for a in assignments if a.endpoint not in known]

for a in assignments:
    label = "ORPHAN" if a.endpoint in {o.endpoint for o in orphans} else "tracked"
    print(f"[{label}] {a.endpoint}  {a.accelerator.value}")

if not orphans:
    print("\nNo orphaned VMs. Nothing to release.")
    sys.exit(0)

if not apply_changes:
    print(f"\n{len(orphans)} orphaned VM(s) still billing. Re-run with --yes to release them.")
    sys.exit(0)

for a in orphans:
    client.unassign(a.endpoint)
    print(f"released {a.endpoint}")

remaining = client.list_assignments()
print(f"\n{len(remaining)} assignment(s) left: {[a.endpoint for a in remaining] or 'none'}")
PY
