# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.3.x   | ✅        |
| 0.2.x   | ❌        |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Email: **security@rasputin.to** or open a [private security advisory](https://github.com/jcartu/rasputin-memory/security/advisories/new) on GitHub.

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

## Response Timeline

- **Acknowledgment:** within 48 hours
- **Assessment:** within 7 days
- **Fix/disclosure:** within 30 days (coordinated)

## Scope

Security issues in a memory system include:
- **Data leakage** — unauthorized access to stored memories across tenants
- **Injection via memory content** — stored payloads that execute during retrieval or rendering
- **Unauthorized access** — bypassing tenant isolation or authentication
- **Denial of service** — queries that crash or exhaust resources

Out of scope: issues requiring local filesystem access on an already-compromised host.
