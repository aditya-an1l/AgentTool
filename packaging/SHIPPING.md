# Shipping Guide — AgentTool

Step-by-step instructions to publish AgentTool to each distribution channel.

---

## 1. AUR (Arch Linux)

### Prerequisites
- Arch Linux or Arch-based distro
- AUR account (register at https://aur.archlinux.org)
- SSH key added to AUR account

### Steps

```bash
# Clone the AUR repo
git clone ssh://aur@aur.archlinux.org/agenttool.git
cd agenttool

# Copy PKGBUILD from this repo
cp /path/to/AgentTool/packaging/aur/PKGBUILD .

# Test build
makepkg -si

# Push to AUR
git add PKGBUILD .SRCINFO
git commit -m "Initial AUR package"
git push
```

### Verify
- Users can install: `yay -S agenttool` or `paru -S agenttool`

---

## 2. Debian/Ubuntu (Launchpad PPA)

### Prerequisites
- Launchpad account (https://launchpad.net)
- GPG key for signing packages
- `debhelper`, `dh-python`, `python3-all`, `python3-setuptools`, `python3-wheel`

### Steps

```bash
# Install build dependencies
sudo apt install build-essential debhelper dh-python python3-all \
  python3-setuptools python3-wheel devscripts

# Set up signing key
gpg --full-generate-key
gpg --export --armor YOUR_KEY_ID > pubkey.asc

# Copy packaging files
cp -r /path/to/AgentTool/packaging/debian ./

# Build source package
debuild -S -sa

# Upload to PPA
dput ppa:aditya-an1l/ppa agenttool_0.2.0_source.changes
```

### Verify
- Users can install: `sudo add-apt-repository ppa:aditya-an1l/ppa && sudo apt install agenttool`

---

## 3. Fedora (Copr or Fedora Package Review)

### Option A: Copr (faster, self-service)

```bash
# Install copr-cli
sudo dnf install copr-cli

# Authenticate
copr-cli build aditya-an1l/agenttool packaging/fedora/agenttool.spec
```

### Option B: Official Fedora Repository (longer, requires review)

1. Install `fedora-packager`: `sudo dnf install fedora-packager`
2. Create Fedora Account System (FAS) account at https://src.fedoraproject.org
3. Clone Fedora SCM: `fedpkg clone agenttool`
4. Copy spec file: `cp packaging/fedora/agenttool.spec agenttool/`
5. Submit for review: `fedpkg request-reviews`
6. After approval: `fedpkg build`

### Verify
- Copr: `sudo dnf copr enable aditya-an1l/agenttool && sudo dnf install agenttool`
- Official: `sudo dnf install agenttool`

---

## Maintainer

Aditya Anil
GitHub: [@aditya-an1l](https://github.com/aditya-an1l)
Email: aditya.anil.productions@gmail.com
