# DVC Setup Guide

DVC tracks the `store/` directory (weights, dataset, footage, output) so that large
binary files never enter git history. The git repo stays lightweight and public;
only people you give SSH access to can pull the actual data.

---

## How it works (security model)

| What | Where | Who sees it |
|---|---|---|
| File hashes (`store.dvc`) | Git repo | Everyone (public) |
| Remote URL (host + path) | `.dvc/config` (committed) | Everyone (public) |
| SSH credentials (user, key path) | `.dvc/config.local` (gitignored) | Local machine only |
| Actual data (weights, dataset, video) | Ubuntu server | Only people with the SSH key |

The `.dvc/config.local` file is gitignored by DVC automatically — credentials
never touch git history.

---

## Server-side setup (do once, on your Ubuntu server)

```bash
# 1. Create a dedicated user with a home dir but no login shell
sudo useradd --system --create-home --shell /usr/sbin/nologin dvc

# 2. Create the storage directory
sudo mkdir -p /srv/dvc/basketball-cv
sudo chown dvc:dvc /srv/dvc/basketball-cv

# 3. Generate a keypair for DVC access (on your local machine, not the server)
ssh-keygen -t ed25519 -C "basketball-cv-dvc" -f ~/.ssh/basketball_cv_dvc
# This creates:
#   ~/.ssh/basketball_cv_dvc       ← private key (share with collaborators out-of-band)
#   ~/.ssh/basketball_cv_dvc.pub   ← public key  (goes on the server)

# 4. Authorise the public key for the dvc user on the server
sudo -u dvc mkdir -p /home/dvc/.ssh
sudo -u dvc tee /home/dvc/.ssh/authorized_keys < ~/.ssh/basketball_cv_dvc.pub
sudo chmod 700 /home/dvc/.ssh
sudo chmod 600 /home/dvc/.ssh/authorized_keys
```

---

## Configure the DVC remote (do once, on your local machine)

```bash
# Add the remote (this writes to .dvc/config — safe to commit)
dvc remote add origin ssh://YOUR_SERVER_HOSTNAME/srv/dvc/basketball-cv

# Set your local SSH key (this writes to .dvc/config.local — gitignored)
dvc remote modify --local origin user dvc
dvc remote modify --local origin keyfile ~/.ssh/basketball_cv_dvc

# Set as default remote
dvc remote default origin
```

Replace `YOUR_SERVER_HOSTNAME` with your server's hostname or IP.

---

## Push data to the remote

```bash
dvc push
```

This uploads the contents of `store/` to the server. Run after training new weights
or adding new footage.

---

## Collaborator onboarding

Give a collaborator access in two steps:

**Step 1 — Share the private key** (out-of-band: Signal, encrypted email, etc.)

```
~/.ssh/basketball_cv_dvc   ← send this file securely, never via git or email plaintext
```

**Step 2 — They run two commands** after cloning the repo:

```bash
dvc remote modify --local origin user dvc
dvc remote modify --local origin keyfile /path/to/basketball_cv_dvc
```

Then they can pull all data:

```bash
dvc pull
```

---

## Revoking access

Remove a collaborator's public key from `/home/dvc/.ssh/authorized_keys` on the server.
No git changes needed — the remote URL stays the same.

---

## Day-to-day commands

| Command | What it does |
|---|---|
| `dvc pull` | Download `store/` from the remote (after cloning or when remote is updated) |
| `dvc push` | Upload local `store/` changes to the remote |
| `dvc status` | Check whether local and remote are in sync |
| `dvc add store` | Re-hash `store/` after adding new files (updates `store.dvc`) |

After `dvc add store`, commit the updated `store.dvc` to git so collaborators know
new data is available.
