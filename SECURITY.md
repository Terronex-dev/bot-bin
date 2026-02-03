# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅        |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it privately:

**Email:** security@terronex.dev

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

We will respond within 48 hours and work with you to resolve the issue before any public disclosure.

## Security Considerations

Bot-BIN processes local files only. It does not:
- Transmit data to external servers (unless you configure an embedding API)
- Store credentials (API keys are read from environment variables)
- Execute untrusted code

When using external embedding providers, your document content is sent to those services. Review their privacy policies accordingly.
