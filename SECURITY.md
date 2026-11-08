# Security

## Reporting a problem

Please report vulnerabilities privately through
[GitHub security advisories](https://github.com/rahulmandal0207/vrpr/security/advisories/new)
rather than in a public issue.

## Using VRPR safely

- **Only load weights you trust.** `.pt` files are Python pickles, and loading one can run code.
  Train your own, or verify where they came from.
- **Keep credentials out of config files you commit.** Stream URLs can contain a username and
  password. VRPR removes them from logs and `summary.json`, but not from your config file or shell
  history.
- **Keep API keys out of git.** `ROBOFLOW_API_KEY` is read from the environment or from `.env` in
  the folder you run `vrpr` from, and git ignores `.env`.
- **Protect the output folder.** It holds personal data: see [docs/privacy.md](docs/privacy.md).
