# Server Setup — basketball-cv DVC Remote

You are Claude running on an Ubuntu server. Your job is to set this machine up as
the DVC data remote for a basketball computer vision project hosted on GitHub.

The person you are helping is setting up this server so that their development
machine (Windows 11, RTX 4080 Super) can push and pull large files (model weights,
training datasets, annotated video) via DVC over a Tailscale VPN.

---

## What you need to do

1. Install and configure Tailscale with SSH enabled
2. Create a dedicated `dvc` OS user
3. Configure the storage directory at `~/basketball-cv-data/` (already created)
4. Verify SSH over Tailscale works so DVC can connect

Do these steps in order. Check each one before moving on.

---

## Context

**Project:** Real-time basketball stat tracking using YOLOv11 + ByteTrack.

**DVC:** Data Version Control — tracks large binary files (model weights ~40MB,
Roboflow training dataset ~800MB, annotated video clips) separately from git.
The development machine runs `dvc push` to upload data here and `dvc pull` to
restore it. The server is purely a storage remote — it does not run any code.

**Tailscale SSH:** Instead of managing SSH keys, Tailscale handles authentication.
Once a machine is on the Tailscale network it can SSH directly — no `authorized_keys`
needed. Access is granted/revoked from the Tailscale admin console.

**Storage path:** `~/basketball-cv-data/` — this directory already exists.
DVC will read/write files here. Do not delete or move it.

---

## Step 1 — Install Tailscale

```bash
curl -fsSL https://tailscale.com/install.sh | sh
```

Start Tailscale and connect to the owner's account. The owner will give you an
auth key from their Tailscale admin console (tailscale.com/admin/settings/keys).

```bash
sudo tailscale up --authkey=<KEY_FROM_OWNER>
```

Confirm it connected:

```bash
tailscale status
# Should show this machine listed and connected
```

Note the MagicDNS hostname shown — it will look like `myserver.tail12345.ts.net`.
Tell the owner this hostname so they can configure the DVC remote on their end.

---

## Step 2 — Enable Tailscale SSH

This lets Tailscale handle SSH authentication so no SSH keys are needed:

```bash
sudo tailscale up --ssh
```

Confirm it is enabled:

```bash
tailscale status | grep -i ssh
# Should show SSH enabled
```

---

## Step 3 — Create the dvc user

This is a restricted system user that DVC will SSH in as. It only needs access
to the `basketball-cv-data/` directory.

```bash
sudo useradd --system --create-home --shell /bin/bash dvc
```

Find out your own username (the one you're logged in as now):

```bash
whoami
```

Transfer ownership of the storage directory to the dvc user. Replace `YOUR_USER`
with the result of `whoami` and confirm the path is correct:

```bash
# Confirm the directory exists
ls -la ~/basketball-cv-data/

# Hand it to the dvc user
sudo chown -R dvc:dvc /home/YOUR_USER/basketball-cv-data/
# Or if it's in a different location, use the full path:
# sudo chown -R dvc:dvc /path/to/basketball-cv-data/
```

Verify the dvc user can access it:

```bash
sudo -u dvc ls /home/YOUR_USER/basketball-cv-data/
# Should list the directory contents (or be empty — that's fine)
```

---

## Step 4 — Configure Tailscale SSH access for the dvc user

By default Tailscale SSH maps connections to OS users by Tailscale identity. To
allow the owner to SSH in specifically as the `dvc` user, the Tailscale ACL needs
a rule. The owner will do this from their admin console, but confirm with them
that the following is added to their Access Controls policy:

```json
{
  "ssh": [
    {
      "action": "accept",
      "src": ["autogroup:member"],
      "dst": ["autogroup:self"],
      "users": ["dvc", "autogroup:nonroot"]
    }
  ]
}
```

If the owner wants tighter control (specific machines only), they can replace
`"autogroup:self"` with a tag applied to this server machine.

---

## Step 5 — Verify SSH works

Ask the owner to try SSHing in from their development machine once Tailscale is
connected on both ends:

```bash
# Run this on the OWNER'S machine, not the server
ssh dvc@YOUR_SERVER_TAILSCALE_HOSTNAME
```

If it connects without asking for a password or key, the setup is complete.

---

## Step 6 — Report back to the owner

Tell the owner:
- The server's Tailscale MagicDNS hostname (from `tailscale status`)
- The full path to the storage directory (run: `sudo -u dvc realpath ~/basketball-cv-data/`)
- Whether any steps failed or required changes

The owner will then run on their development machine:

```bash
dvc remote add origin ssh://dvc@YOUR_HOSTNAME/FULL_PATH/basketball-cv-data
dvc remote default origin
```

And they can test with:

```bash
dvc push
```

---

## Troubleshooting

**Tailscale not connecting:** Check the auth key hasn't expired. Generate a new
one from tailscale.com/admin/settings/keys.

**SSH as dvc user fails:** Confirm Tailscale SSH is enabled (`sudo tailscale up --ssh`)
and the ACL rule is saved in the admin console.

**Permission denied on basketball-cv-data/:** Re-run the chown command and verify
with `ls -la` that dvc owns the directory.

**tailscale command not found:** The install script may not have updated PATH.
Try `sudo /usr/sbin/tailscaled &` or reboot and try again.
