# Publishing To GitHub

Do not store GitHub passwords in this repository.

Recommended authentication methods:

- GitHub CLI: `gh auth login`
- SSH key: `git@github.com:<owner>/<repo>.git`
- HTTPS with a personal access token stored by the local credential manager

## Option 1: GitHub CLI

Install and authenticate:

```text
gh auth login
```

Create and push a new repository:

```text
scripts/publish_to_github.sh npu_arch_design
```

## Option 2: Existing Remote

If the repository already exists on GitHub:

```text
git remote add origin git@github.com:<owner>/npu_arch_design.git
git branch -M main
git push -u origin main
```

## Security Notes

- Never commit `.local/`.
- Never include passwords, personal access tokens, SSH private keys, or cookies
  in project files.
- If a password was shared in chat or copied into a terminal, rotate it before
  publishing.

