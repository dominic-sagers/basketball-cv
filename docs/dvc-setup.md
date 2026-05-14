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
| Actual data (weights, dataset, video) | Ubuntu server via Tailscale | Only people on your Tailscale network |

**No SSH keys to generate, share, or rotate.** Authentication is handled entirely
by Tailscale identity. Access is granted or revoked from the Tailscale admin console.

---

## How access works

```
Collaborator installs Tailscale
      ↓
Joins your network with a pre-auth key you generate
      ↓
Their machine can reach the server over Tailscale
      ↓
dvc pull / dvc push work — Tailscale SSH authenticates them automatically
```

---

## Server-side setup (do once)

### 1. Install Tailscale on the Ubuntu server

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

Follow the auth link to connect the server to your Tailscale account.

### 2. Enable Tailscale SSH on the server

```bash
sudo tailscale up --ssh
```

This lets Tailscale handle SSH authentication — no `authorized_keys` needed.

### 3. Note the server's Tailscale hostname

```bash
tailscale status
# Look for your server's MagicDNS name, e.g.: myserver.tail12345.ts.net
```

Or find it in the Tailscale admin console at tailscale.com/admin/machines.

### 4. Create a dedicated user for DVC storage

```bash
sudo useradd --system --create-home --shell /bin/bash dvc
sudo mkdir -p /srv/dvc/basketball-cv
sudo chown dvc:dvc /srv/dvc/basketball-cv
```

### 5. Configure Tailscale SSH ACL to allow the dvc user

In your Tailscale admin console → **Access controls**, add a rule allowing your
network members to SSH as the `dvc` user on the server:

```json
{
  "ssh": [
    {
      "action": "accept",
      "src": ["autogroup:member"],
      "dst": ["tag:dvc-server"],
      "users": ["dvc"]
    }
  ]
}
```

Tag your server with `tag:dvc-server` in the admin console, or simplify by
allowing access to the specific server by its Tailscale IP.

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

## Configure the DVC remote (do once, when setting up a new server)

> **Already done** for this project — `.dvc/config` is committed and points at
> the active remote. Skip this section unless you are migrating to a new server.

First, confirm you're on the Tailscale network:

```bash
tailscale status
```

Add the remote using the server's Tailscale IP or MagicDNS hostname and the
storage path on that server:

```bash
dvc remote add origin ssh://dvc@<TAILSCALE_IP_OR_HOSTNAME>/<STORAGE_PATH>
dvc remote default origin
```

Commit the result:

```bash
git add .dvc/config
git commit -m "Configure DVC SSH remote over Tailscale"
```

No `--local` flags needed — there are no credentials to keep private.

---

## Push data to the remote

```bash
dvc push
```

---

## Updating the store after a game session

After a session where you've added footage, annotated outputs, or new weights:

```bash
# 1. Re-hash store/ and update the pointer file
dvc add store

# 2. Stage the updated pointer
git add store.dvc

# 3. Commit
git commit -m "Update store: <brief description of what changed>"

# 4. Push data to remote (make sure Tailscale is connected)
dvc push
```

DVC only uploads files that have changed since the last push — it won't re-upload existing footage.

---

## Collaborator onboarding

Granting access requires two steps, both done from the **Tailscale admin console**:

**Step 1 — Generate a pre-auth key**

1. Go to tailscale.com/admin/settings/keys
2. Click **Generate auth key**
3. Set it as **Reusable: No** (single-use per collaborator) and set an expiry
4. Share the key with the collaborator (Signal, encrypted email, etc.)

**Step 2 — They run two commands**

```bash
# Install Tailscale (Linux/Mac/Windows — see tailscale.com/download)
# Then join your network:
tailscale up --authkey=tskey-auth-XXXXXXXXXXXXXXXX

# Configure the DVC remote user (matches the dvc user on the server)
dvc remote modify --local origin user dvc
```

Then they can pull:

```bash
dvc pull
```

That's it. No SSH keys, no certificates, no `.dvc/config.local` secrets.

---

## Revoking access

In the Tailscale admin console → **Machines**, find their device and click
**Remove**. They immediately lose network access to the server.
No key rotation, no `authorized_keys` edits needed.

---

## Day-to-day commands

| Command | What it does |
|---|---|
| `dvc pull` | Download `store/` from the remote |
| `dvc push` | Upload local `store/` changes to the remote |
| `dvc status` | Check whether local and remote are in sync |
| `dvc add store` | Re-hash `store/` after adding new files (updates `store.dvc`) |

After `dvc add store`, commit the updated `store.dvc` so collaborators know
new data is available:

```bash
git add store.dvc && git commit -m "Update store: <what changed>"
```
