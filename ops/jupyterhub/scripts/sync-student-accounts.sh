#!/usr/bin/env bash
# Sync Linux + Slurm accounts with the JupyterHub allowlist.
#
# Reads ../allowed_users.txt and ensures that every GitHub username listed
# there has a corresponding Linux user on Atlas, with:
#   - a stable UID allocated from /var/lib/atlas-jhub/uid-map.json
#   - home directory at /home/<username>
#   - membership in the 'jhub' group
#   - (if in ../gpu_users.txt) the 'gpu_users' supplementary group
#
# Users no longer in the allowlist are NOT deleted — their accounts are
# locked (passwd -l) and their shell is set to /sbin/nologin. Their home
# directories and any Slurm job records are preserved for audit.
#
# Run as root, idempotent. Safe to run repeatedly or from cron.
#
# Copyright (c) 2026 Andrew H. Bond. AGI-HPC Responsible AI License v1.0.

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${YELLOW}▶ $*${NC}"; }
ok()    { echo -e "${GREEN}✓ $*${NC}"; }
warn()  { echo -e "${YELLOW}⚠ $*${NC}"; }
die()   { echo -e "${RED}✗ $*${NC}"; exit 1; }

[[ $EUID -eq 0 ]] || die "Must be run as root."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLOWLIST="${SCRIPT_DIR}/../allowed_users.txt"
GPU_LIST="${SCRIPT_DIR}/../gpu_users.txt"
UID_MAP_DIR="/var/lib/atlas-jhub"
UID_MAP="${UID_MAP_DIR}/uid-map.json"
UID_POOL_START=10000
UID_POOL_END=19999

[[ -f "$ALLOWLIST" ]] || die "allowlist not found: $ALLOWLIST"

mkdir -p "$UID_MAP_DIR"
[[ -f "$UID_MAP" ]] || echo '{}' > "$UID_MAP"
chmod 640 "$UID_MAP"

# --------------------------------------------------------------------------
# Helpers using python for JSON manipulation (stdlib, no jq dep needed)
# --------------------------------------------------------------------------
python_bin="$(command -v python3 || command -v python)"
[[ -n "$python_bin" ]] || die "python3 not found."

read_map() {
    "$python_bin" - "$UID_MAP" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    m = json.load(f)
for k, v in sorted(m.items()):
    print(f"{k}\t{v}")
PY
}

write_map() {
    # Stdin: username\tuid lines
    "$python_bin" - "$UID_MAP" <<'PY'
import json, sys
path = sys.argv[1]
m = {}
for line in sys.stdin:
    line = line.rstrip('\n')
    if not line:
        continue
    u, uid = line.split('\t')
    m[u] = int(uid)
with open(path, 'w') as f:
    json.dump(m, f, indent=2, sort_keys=True)
PY
}

# --------------------------------------------------------------------------
# Parse allowlist + gpu list into shell arrays
# --------------------------------------------------------------------------
declare -a ALLOWED=()
while IFS= read -r raw; do
    line="${raw%%#*}"
    line="$(echo -n "$line" | xargs)"
    [[ -z "$line" ]] && continue
    ALLOWED+=("$(echo "$line" | tr '[:upper:]' '[:lower:]')")
done < "$ALLOWLIST"

declare -a GPU_ALLOWED=()
if [[ -f "$GPU_LIST" ]]; then
    while IFS= read -r raw; do
        line="${raw%%#*}"
        line="$(echo -n "$line" | xargs)"
        [[ -z "$line" ]] && continue
        GPU_ALLOWED+=("$(echo "$line" | tr '[:upper:]' '[:lower:]')")
    done < "$GPU_LIST"
fi

info "Allowlist has ${#ALLOWED[@]} users, ${#GPU_ALLOWED[@]} GPU-enabled."

getent group jhub >/dev/null || { info "Creating jhub group..."; groupadd jhub; }
getent group gpu_users >/dev/null || { info "Creating gpu_users group..."; groupadd gpu_users; }

# --------------------------------------------------------------------------
# Allocate next UID (scans existing map, returns lowest unused in pool)
# --------------------------------------------------------------------------
next_uid() {
    "$python_bin" - "$UID_MAP" "$UID_POOL_START" "$UID_POOL_END" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    m = json.load(f)
lo, hi = int(sys.argv[2]), int(sys.argv[3])
used = set(m.values())
for uid in range(lo, hi + 1):
    if uid not in used:
        print(uid)
        sys.exit(0)
print("ERR: UID pool exhausted", file=sys.stderr)
sys.exit(1)
PY
}

# --------------------------------------------------------------------------
# Add / update each allowlisted user
# --------------------------------------------------------------------------
# Load current map into an associative array
declare -A MAP
while IFS=$'\t' read -r u uid; do
    [[ -n "$u" ]] && MAP["$u"]="$uid"
done < <(read_map)

for gh_user in "${ALLOWED[@]}"; do
    uid="${MAP[$gh_user]:-}"
    if [[ -z "$uid" ]]; then
        uid="$(next_uid)"
        MAP["$gh_user"]="$uid"
        info "Allocating UID $uid for $gh_user"
    fi

    if ! id -u "$gh_user" >/dev/null 2>&1; then
        info "Creating Linux user: $gh_user (UID $uid)"
        useradd \
            --uid "$uid" \
            --gid jhub \
            --create-home \
            --home-dir "/home/$gh_user" \
            --shell /bin/bash \
            --comment "JHub student: github.com/$gh_user" \
            "$gh_user"
        chmod 700 "/home/$gh_user"
        ok "Created $gh_user (UID $uid)."
    else
        existing_uid="$(id -u "$gh_user")"
        if [[ "$existing_uid" != "$uid" ]]; then
            warn "UID drift: $gh_user has UID $existing_uid but map says $uid. Keeping $existing_uid."
            MAP["$gh_user"]="$existing_uid"
        fi
        # Ensure account is unlocked (could have been locked by a prior run)
        passwd -u "$gh_user" 2>/dev/null || true
        usermod -s /bin/bash "$gh_user" 2>/dev/null || true
    fi

    # GPU membership
    is_gpu=false
    for g in "${GPU_ALLOWED[@]}"; do
        [[ "$g" == "$gh_user" ]] && is_gpu=true && break
    done
    if $is_gpu; then
        if ! id -nG "$gh_user" | grep -qw gpu_users; then
            info "Adding $gh_user to gpu_users"
            usermod -aG gpu_users "$gh_user"
        fi
    else
        if id -nG "$gh_user" | grep -qw gpu_users; then
            info "Removing $gh_user from gpu_users"
            gpasswd -d "$gh_user" gpu_users >/dev/null
        fi
    fi
done

# --------------------------------------------------------------------------
# Lock users no longer in the allowlist
# --------------------------------------------------------------------------
for mapped_user in "${!MAP[@]}"; do
    still_allowed=false
    for a in "${ALLOWED[@]}"; do
        [[ "$a" == "$mapped_user" ]] && still_allowed=true && break
    done
    if ! $still_allowed; then
        if id -u "$mapped_user" >/dev/null 2>&1; then
            current_shell="$(getent passwd "$mapped_user" | cut -d: -f7)"
            if [[ "$current_shell" != "/sbin/nologin" ]]; then
                warn "Locking $mapped_user (no longer in allowlist; home dir preserved)"
                passwd -l "$mapped_user" >/dev/null
                usermod -s /sbin/nologin "$mapped_user"
            fi
        fi
    fi
done

# --------------------------------------------------------------------------
# Persist map
# --------------------------------------------------------------------------
{
    for u in "${!MAP[@]}"; do
        printf '%s\t%s\n' "$u" "${MAP[$u]}"
    done
} | write_map
chmod 640 "$UID_MAP"
chown root:jhub "$UID_MAP"
ok "UID map updated: $UID_MAP"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
cat <<EOF

$(echo -e "${GREEN}=========================================${NC}")
$(echo -e "${GREEN}  Student-account sync complete          ${NC}")
$(echo -e "${GREEN}=========================================${NC}")

Active users: ${#ALLOWED[@]}
GPU users:    ${#GPU_ALLOWED[@]}
UID pool:     ${UID_POOL_START}–${UID_POOL_END}
Map file:     ${UID_MAP}

If you've just added a new student:
  → They can now log in via JupyterHub.
  → Their notebook container will run as their Linux UID on Atlas.
  → Their Slurm jobs will be charged to their account.

If you've just removed a student:
  → Their account is locked (passwd -l) but not deleted.
  → Their home directory under /home/<user>/ is preserved for audit.
  → Restore with: sudo passwd -u <user> && sudo usermod -s /bin/bash <user>
EOF
