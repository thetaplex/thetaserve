# Security policy

## Supported versions

thetaserve is pre-release. Only the latest commit on `main` receives security fixes until 1.0.

## Reporting a vulnerability

Please do not open a public issue. Report privately through GitHub: on the repository page open **Security > Report a vulnerability** ([direct link](https://github.com/thetaplex/thetaserve/security/advisories/new)).

Include the affected version or commit, steps to reproduce, and the impact you expect. We aim to acknowledge reports within 3 working days and to agree a disclosure date with you once a fix is ready.

Issues of particular interest: leaked secrets (API keys, provider credentials, Hugging Face tokens) in output, logs or files, and endpoints reachable without the API key.
