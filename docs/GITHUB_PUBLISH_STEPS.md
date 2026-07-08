# GitHub publishing steps

This folder is prepared as the GitHub repository for **Quint Deep Flow evo**.

## Option A: Publish from GitHub Desktop

1. Open GitHub Desktop.
2. Choose `File` -> `Add local repository...`.
3. Select:

```text
<path-to-repo>\Quint-Deep-Flow-evo
```

4. If GitHub Desktop asks to create a repository, create it with:

```text
Quint-Deep-Flow-evo
```

5. Commit all files with a message such as:

```text
Initial public release of Quint Deep Flow evo
```

6. Click `Publish repository`.
7. Choose visibility. Use `Public` only after confirming that demo data and source code are acceptable for public release.

## Option B: Publish from command line

Install GitHub CLI first:

```powershell
winget install --id GitHub.cli
gh auth login
```

If this is your first Git commit on the PC, set your Git identity first:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

Then run:

```powershell
cd <path-to-repo>\Quint-Deep-Flow-evo
git init
git add .
git commit -m "Initial public release of Quint Deep Flow evo"
gh repo create Quint-Deep-Flow-evo --public --source . --remote origin --push
```

If you prefer a private repository first, replace `--public` with `--private`.

## Before making the repository public

- Confirm the demo image can be shared publicly.
- Decide the license. This package currently does not include an open-source license.
- Confirm whether Allen atlas derivative files under `atlas/ccf` may be redistributed for your intended audience.
- If you do not want to redistribute atlas files, remove `atlas/ccf/annotation_25.nrrd`, `labels.txt`, and `tree.json`, then update the README to ask users to provide their own atlas files.
