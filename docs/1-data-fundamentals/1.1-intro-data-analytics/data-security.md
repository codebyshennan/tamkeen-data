# Data Security: Protecting Data Assets

**After this lesson:** You can describe the core goals of data security (confidentiality, integrity, availability), explain defense in depth at a high level, and read the code sketches as common patterns, not as copy-paste production systems.

## Introduction

**Data security** protects data from **unauthorized access**, **tampering**, and **loss** across storage, networks, and applications. Privacy asks whether you _should_ hold certain data; security asks how you _protect_ what you are allowed to hold.

### Video

_IBM Technology, Cybersecurity basics_

## Goals and layers

### Protection goals (CIA + N)

* **Confidentiality**: Only **authorized** people and systems can read the data. Leaks and oversharing break confidentiality.
* **Integrity**: Data is **accurate and complete**; changes are intentional and detectable.
* **Availability**: Authorized users can use systems and data when needed. Attacks and outages can target availability.
* **Non-repudiation**: Actors cannot plausibly deny sending or receiving data; important for audits and contracts.

_"Defense in depth" means one layer failing doesn't compromise the whole system. Each ring must be breached separately._

### Defense in depth (layers)

Teams stack controls so one failure does not mean total failure:

1. **Physical**: Facilities, badges, environmental controls. Still relevant in the cloud: providers run the buildings.
2. **Network**: Firewalls, segmentation, VPNs, intrusion detection, so one compromised laptop does not mean the whole network.
3. **Application**: **Authentication** (identity), **authorization** (permissions), **input validation** (block malicious input), **session management** (tokens, expiry).
4. **Data**: **Encryption**, **access controls**, **masking** in lower environments, **secure deletion** when data must go away.

## Implementation guide (illustrative)

### Encryption

**Symmetric** keys are fast for bulk data. **Asymmetric** (public/private) pairs help with **key exchange** and **signatures** but are expensive for large payloads. **Hybrid** encryption mixes both: encrypt the payload with a random session key, then encrypt that key for the recipient.

**Example:**

Key Initialisation

Generates both a symmetric AES key for bulk data and an RSA key-pair for key exchange at startup.

Symmetric Encryption

Uses AES-GCM mode which provides both confidentiality (ciphertext) and authenticity (tag), returning the nonce needed for decryption.

Asymmetric Encryption

RSA-OAEP encrypts small payloads (like session keys) with the recipient's public key so only they can decrypt.

Hybrid Encryption

Combines both: generates a random session key, encrypts the data symmetrically, then encrypts the session key asymmetrically, the pattern used in TLS.

**When each pattern shows up:** Symmetric encryption is typical for **disk or database** encryption when one system holds the key. Asymmetric is used for **establishing trust** between parties (TLS handshakes, signing). Hybrid is **TLS-like**: combine fast symmetric bulk encryption with asymmetric protection of the session key.

### Access control

**RBAC** assigns permissions to **roles** (analyst, admin), then assigns users to roles. **ABAC** (below) can depend on attributes (department, clearance, resource sensitivity) and context (time, location), more flexible, more complex to configure.

#### Role-based access control (RBAC)

**Example**:

State Stores

Three dicts hold roles, user-to-role assignments, and permissions, the minimal state for an RBAC system.

Role Creation

Records the permission set plus creation/modification timestamps so you can audit when a role was last changed.

Role Assignment

Validates the role exists before mapping a user to it, then logs the assignment for the audit trail.

Permission Check and Audit

Looks up the user's role and checks membership in its permission set; the audit method returns a snapshot for access reviews.

**Access Levels Example:**

```python
PERMISSION_LEVELS = {
    'admin': {
        'read': True,
        'write': True,
        'delete': True,
        'manage_users': True
    },
    'manager': {
        'read': True,
        'write': True,
        'delete': False,
        'manage_users': False
    },
    'user': {
        'read': True,
        'write': False,
        'delete': False,
        'manage_users': False
    }
}
```

***

#### Attribute-based access control (ABAC)

Policies can depend on **who** the user is, **what** the resource is, and **context** (device, time). Large enterprises use ABAC when RBAC alone is too coarse.

**Example**:

Policy Engine

Injects a policy engine (evaluates rules) and a context manager (supplies runtime context like time and device).

Attribute Bundle

Packages user attributes (department, clearance, location), resource attributes (classification, owner, type), the requested action, and environmental context into one dict for the policy engine.

Log and Return

Every access decision is logged before it is returned so security teams can review who accessed what and under which context.

## Security monitoring and incident response

**Monitoring** watches logs and metrics for suspicious patterns. **Incident response** is the playbook when something bad happens: contain, investigate, recover, document. Both are essential in production; detecting late is expensive.

### Security monitoring

**Example**:

Monitor Setup

Injects an alert manager for notifications and a threat detector for analysing collected metrics.

Monitoring Loop

Continuously collects metrics, analyses for threats, and dispatches handlers, sleeping between cycles to avoid CPU saturation.

Severity Dispatch

Maps severity levels to handler functions using a dict so adding a new severity level is a single-line change.

### Incident response

**Example**:

Response Setup

An incident manager coordinates the workflow; forensics tools handle evidence collection and analysis.

Four-Phase Response

Follows the standard IR playbook in order: contain → investigate → recover → document, building an evidence trail at each step.

Containment Dispatch

Maps incident type to a specific containment action (isolate, quarantine, DDoS protection, revoke tokens) with a safe default.

## Baselines and assessments (illustrative)

**Baseline** means a known-good configuration: patches, hardened settings, monitoring enabled. **Assessments** (scans, reviews) find gaps before attackers do.

### Security baseline

Baseline Setup

Injects a vulnerability scanner and a configuration manager so hardening and scanning are handled by dedicated services.

Hardening Steps

Applies secure configuration, installs updates, and sets access controls, the three pillars of a security baseline, in a defined order.

Monitor and Verify

Enables monitoring before verifying the baseline so the system is observable from the moment hardening is confirmed.

### Regular security assessment

Assessment Setup

Injects two specialist tools: a vulnerability scanner for automated checks and a penetration tester for adversarial simulation.

Four-Domain Scan

Runs vulnerability scanning, penetration testing, configuration review, and access control audit in parallel, gathering all results before reporting.

Report Generation

Passes the aggregated results to a report generator so findings from all four domains appear in one structured output.

## Common pitfalls

* **Shared passwords or API keys in notebooks**: Treat secrets like production; use environment variables and rotation.
* **Over-relying on perimeter security**: Insider risk and misconfigured buckets matter; layer controls and audit access.
* **Ignoring updates**: Unpatched dependencies are a common breach path.

## Next Steps

### In this submodule

Continue to [Workflow concepts](workflow-concepts.md). Then start [Introduction to Python](../1.2-intro-python/).

### Going deeper on your own

Specialists go further with **SIEM** tooling, **zero trust** architectures, **threat hunting**, and formal programs (ISO 27001, SOC 2, NIST CSF). For this course, focus on **hygiene**: least privilege, patching, secrets management, and logging, most incidents still exploit basics.

## Additional resources

* [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
* [OWASP Security Guidelines](https://owasp.org/www-project-security-guidelines/)
* [Cloud Security Alliance](https://cloudsecurityalliance.org/)
* [SANS Security Resources](https://www.sans.org/security-resources/)
* [ISO 27001 Standard](https://www.iso.org/isoiec-27001-information-security.html)
