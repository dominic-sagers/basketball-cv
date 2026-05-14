# DVC Setup Guide

DVC tracks the `store/` directory (weights, dataset, footage, output) so large
binary files never enter git history. The git repo stays lightweight and public;
only people on your Tailscale network can reach the server to pull or push data.

## Current remote

The remote is already configured. The connection details are in `.dvc/config`
(committed to the repo). New machines only need Tailscale network access and
`pip install "dvc[ssh]"` — no other configuration required.

The server is only reachable over Tailscale — the remote URL in `.dvc/config`
is not usable without being on the network.

---

## Security model

| What | Where | Who sees it |
|---|---|---|
| File hashes (`store.dvc`) | Git repo | Everyone (public) |
| Remote URL (Tailscale hostname + path) | `.dvc/config` (committed) | Everyone (public — safe, host is unreachable off-network) |
| SSH credentials | None — Tailscale SSH handles auth | N/A |
| Actual data (weights, dataset, video) | Ubuntu server via Tailscale | Only people granted access in the Tailscale ACL |

**No SSH keys to generate, share, or rotate.** Authentication is handled entirely
by Tailscale identity. Access is granted or revoked from the Tailscale admin console.

---

## How access works

There are two ways to grant a collaborator access:

### Option A — Join the tailnet (simplest long-term)
```
Dominic generates a pre-auth key at tailscale.com/admin/settings/keys
      ↓
Collaborator runs: tailscale up --authkey=tskey-auth-XXXXXXXXXXXXXXXX
      ↓
They become a member of Dominic's tailnet
      ↓
autogroup:member covers them — dvc pull / dvc push work immediately
```

### Option B — Cross-tailnet node sharing (if they have their own tailnet)
```
Dominic shares scruffy via the Tailscale admin console
      ↓
Collaborator can see scruffy in their tailnet (network layer)
      ↓
Dominic adds their Tailscale identity to group:dvc-collaborators in the ACL
      ↓
dvc pull / dvc push work via Tailscale SSH auth
```

> **Note:** Node sharing alone is not enough — it only grants network connectivity,
> not SSH access. The ACL rule is required for Option B.

---

## Tailscale ACL structure

The ACL on Dominic's tailnet (`tailscale.com/admin/acls`) should look like this:

```json
{
    "groups": {
        "group:dvc-collaborators": [
            "collaborator@gmail.com",
        ],
    },

    "tagOwners": {
        "tag:dvc-server": ["autogroup:admin"],
    },

    "grants": [
        {"src": ["*"], "dst": ["*"], "ip": ["*"]},
    ],

    "ssh": [
        {
            "action": "check",
            "src":    ["autogroup:member"],
            "dst":    ["autogroup:self"],
            "users":  ["autogroup:nonroot", "root"],
        },
        {
            "action": "accept",
            "src":    ["autogroup:member"],
            "dst":    ["autogroup:self"],
            "users":  ["dvc", "autogroup:nonroot"],
        },
        {
            "action": "accept",
            "src":    ["group:dvc-collaborators"],
            "dst":    ["tag:dvc-server"],
            "users":  ["dvc"],
        },
    ],
}
```

The server (`scruffy`) must have the `tag:dvc-server` tag applied in the Tailscale
admin console under **Machines → scruffy → Edit tags**.

To add a new cross-tailnet collaborator: add their Tailscale identity to
`group:dvc-collaborators`. Their exact identity can be found by running
`tailscale status` on their machine.

---

## Server-side setup (do once)

### 1. Install Tailscale on the Ubuntu server

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --ssh
```

### 2. Note the server's Tailscale IP

```bash
tailscale status
```

Or find it in the Tailscale admin console at tailscale.com/admin/machines.

### 3. Create a dedicated user for DVC storage

```bash
sudo useradd --system --create-home --shell /bin/bash dvc
sudo mkdir -p /srv/dvc/basketball-cv
sudo chown dvc:dvc /srv/dvc/basketball-cv
```

---

## Install DVC (do once per machine)

DVC is included in `requirements.txt` but must be installed with the SSH extra:

```bash
pip install "dvc[ssh]"
```

Or if using the venv:
```bash
pip install -r requirements.txt   # includes dvc[ssh]
```

---

## Collaborator onboarding

### If joining Dominic's tailnet (Option A)

1. Dominic generates a pre-auth key at `tailscale.com/admin/settings/keys`
2. Collaborator installs Tailscale and runs:
```bash
tailscale up --authkey=tskey-auth-XXXXXXXXXXXXXXXX
dvc remote modify --local origin user dvc
dvc pull
```

### If using node sharing (Option B)

1. Dominic shares `scruffy` via the Tailscale admin console
2. Collaborator finds their exact Tailscale identity: `tailscale status`
3. Dominic adds that identity to `group:dvc-collaborators` in the ACL
4. Collaborator runs:
```bash
dvc remote modify --local origin user dvc
dvc pull
```

---

## Revoking access

**Option A (tailnet member):** Tailscale admin console → **Machines** → find their device → **Remove**.

**Option B (cross-tailnet):** Remove their email from `group:dvc-collaborators` in the ACL.

No key rotation or `authorized_keys` edits needed in either case.

---

## Updating the store after a game session

After a session where you've added footage, annotated outputs, or new weights,
run the post-session push script:

```bash
python scripts/push_session.py
```

Or manually:

```bash
dvc add store
git add store.dvc
git commit -m "dvc: <brief description>"
dvc push
git push
```

DVC only uploads files that have changed since the last push.

---

## Day-to-day commands

| Command | What it does |
|---|---|
| `dvc pull` | Download `store/` from the remote |
| `dvc push` | Upload local `store/` changes to the remote |
| `dvc status` | Check whether local and remote are in sync |
| `dvc add store` | Re-hash `store/` after adding new files (updates `store.dvc`) |
