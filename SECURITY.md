# Security

## Deployment

ONNX Quantization Lab is designed for local execution, not multi-tenant deployment. The default Docker Compose configuration binds the service to loopback. Do not expose it to untrusted networks without additional controls. Treat uploaded models and datasets as untrusted inputs; use suitable isolation and resource limits. The application does not provide a complete sandbox.

## Reporting a vulnerability

Please contact the repository maintainer privately. If GitHub private vulnerability reporting is enabled, use the Security tab. Do not include secrets, private datasets or exploitable details in public issues. Include the affected version, environment and a minimal non-sensitive reproduction.

## Scope

The current research preview has no guaranteed security support period or response SLA.
