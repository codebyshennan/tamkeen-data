# Data Privacy

**After this lesson:** You can explain what personal data and sensitive categories are, why privacy rules exist, and how common rights (access, correction, deletion, portability) show up in real projects.

## Introduction

**Data privacy** is about handling personal information in a way that respects people's autonomy and meets legal obligations. It is not only "security" (locks and encryption) and not only "ethics" (doing the right thing), it is the **rules and practices** for how data may be collected, used, and shared.

### Video

_Simplilearn, GDPR explained_

### Why privacy matters for analysts and scientists

* **People**: Laws and company policies exist because misuse of data can harm individuals (discrimination, fraud, embarrassment). Your work is more trustworthy when it respects those boundaries.
* **Trust**: Teams that are transparent about data use get better cooperation from customers and partners.
* **Compliance**: Regulations like GDPR, CCPA, and sector rules (e.g. HIPAA) set hard requirements. **Ignorance is not a defense**; know what applies to your data and region.
* **Risk**: Breaches and misuse lead to fines, lawsuits, and loss of reputation. Minimizing data and clarifying purpose reduces exposure.

## Key concepts

### Personally Identifiable Information (PII)

**PII** is any information that can identify a person, either alone or in combination with other data you hold. A name plus email is clearly PII; a "unique" customer ID can be PII if it maps to a real person in another table.

**Common examples:** Name, postal address, phone number, email, government ID numbers, account numbers, and many financial identifiers.

**Sensitive categories** (often called "special category" under GDPR) are a subset that can create **serious harm** if mishandled: racial or ethnic origin, political opinions, religious beliefs, health, sexual life, biometric data used to identify someone, and similar fields. **Do not collect these** unless you have a clear legal basis and safeguards.

### Protected Health Information (PHI)

**PHI** is health information that identifies an individual and is covered by **HIPAA** in the U.S. (other countries have similar health privacy laws). It includes clinical notes, insurance identifiers tied to health, billing, and much of what sits in EHR systems.

If you work with PHI, assume **stricter access controls, training, and agreements** than for ordinary business data.

### Data subject rights

Modern privacy laws give individuals **rights** over their data. Wording varies by law, but you will see these patterns:

* **Access**: See what data you hold about them.
* **Rectification**: Correct inaccurate or incomplete data.
* **Erasure**: Request deletion ("right to be forgotten") when the law allows.
* **Portability**: Receive a machine-readable copy to move to another service.

**For beginners:** Treat these as **product and process** requirements, not footnotes. Someone will eventually ask for an export or deletion; your pipelines should not make that impossible.

## Data privacy principles (plain language)

These ideas appear in GDPR and similar frameworks:

1. **Data minimization**: Collect only what you need for a **defined** purpose. "We might use it later" is a red flag.
2. **Purpose limitation**: Use data only for the purposes you disclosed (and that the law allows). A new use case may need new consent or analysis.
3. **Storage limitation**: Delete or anonymize when data is no longer needed for that purpose, per policy and law.
4. **Integrity and confidentiality**: Keep data accurate enough for its use, and protect it against unauthorized access (see [Data security](data-security.md)).

_Treat sensitive categories and PHI as their own tier, stricter access controls, minimal retention, and often explicit legal basis before you even touch the data._

## Legal frameworks (high level)

This is **not** legal advice. It is a map so you know **what to look up** with counsel or a compliance team.

### General Data Protection Regulation (GDPR)

**Where:** European Economic Area and often used as a global benchmark.

**What to remember:** Lawful basis for processing personal data, consent rules when consent is required, **data subject rights**, breach notification timelines, **data protection impact assessments** for risky processing, and **significant fines** for serious violations.

### California Consumer Privacy Act (CCPA)

**Where:** California residents (and similar laws exist in other U.S. states).

**What to remember:** Rights to **know** what is collected, to **delete** in many cases, to **opt out of sale** of personal information (as defined by the law), and **non-discrimination** for exercising rights.

### Health Insurance Portability and Accountability Act (HIPAA)

**Where:** U.S. covered entities and business associates handling PHI.

**What to remember:** Strict safeguards for **use and disclosure** of PHI, breach notification, and **minimum necessary** access.

## Best practices in practice

1. **Privacy impact assessments (PIAs)**: Before a new collection or model, ask: What data? Why? Who sees it? What could go wrong? Document answers.
2. **Privacy by design**: Build defaults that minimize data (e.g. short retention, role-based access) instead of bolting privacy on at the end.
3. **Training**: Everyone who touches data should know your classification rules and escalation paths.
4. **Anonymization and pseudonymization**: Remove or replace identifiers when analysis does not need names; understand that **anonymization is hard** when many fields are combined (see pitfalls below).

***

## Sensitive data: payments and health (illustrative code)

The examples below are **teaching sketches**-not production security advice. They show _why_ tokenization, encryption, and consent checks exist.

### Financial data

**Example: Payment processing**

Class Setup

The constructor injects a tokenization service and an encryption service so card data is never stored in plain text.

Tokenize Card

The raw card number is replaced by a token immediately, only the token is kept for future charges, limiting exposure.

Encrypt Transaction

Amount, currency, and timestamp are encrypted together so the payload is opaque to anyone without the key.

Return Receipt

Only the transaction ID, status, and token are returned, no card number or raw amount leaves this method.

**Why it matters:** Card data is toxic to store in plain text; tokenization and encryption limit blast radius if a system is compromised.

***

### Healthcare and special-category data

**Example: Healthcare data management**

Service Injection

Two services are injected: one for encryption and one for access control, following separation of concerns.

Category Gate

Medical and biometric data is routed to the stricter handler; everything else uses the standard path.

GDPR Consent Check

Explicit consent is verified before any special-category data is stored; raises an error if consent is missing, then encrypts and applies a strict access policy.

**Why it matters:** Special category data requires extra protection

## Privacy laws and code (illustrative)

The classes below show **how software might structure** consent checks and subject-rights handling. Real systems add auditing, identity verification, and legal review.

### GDPR (European Union)

**Technical sketch:**

Class Init

A consent manager and data processor are injected so rights requests and processing can be handled independently.

Subject Rights Dispatch

Routes the four GDPR subject rights, access, erasure, portability, rectification, to separate methods for clean separation.

Consent Verification

Checks that consent exists, is not expired, and is specific and informed, GDPR requires all three conditions before processing.

Processing Log

Records each processing activity with timestamp, legal basis, and data categories, required for GDPR's records of processing activities.

### CCPA (California)

**Technical sketch:**

CCPA Setup

Privacy notice and data mapper services handle disclosures and mapping what data is held about each consumer.

Consumer Rights Router

Routes the three CCPA rights, disclosure, deletion, and opt-out of sale, to their respective handlers.

Residency and Opt-Out

Verifies California residency before granting rights, and exposes the "do not sell" list for downstream ad/data pipelines.

## Practical tips (personal and work)

**Personal use:** Reduce attack surface (strong, unique passwords; MFA). **Work use:** Follow governance, classification, retention, and least-privilege access. The code below illustrates patterns, not a complete program.

### For personal use

#### Password security

**Sketch:**

Hasher Setup

A dedicated hasher service is injected so raw passwords are never stored directly by this class.

Strength Validation

Checks five requirements (length, upper, lower, digits, special characters) and raises an error if any fail.

Secure Generation

Uses the OS cryptographic random source via `secrets` to produce a URL-safe 16-byte token, suitable for temporary passwords.

**Habits that matter:** Use a password manager, turn on **two-factor authentication** where offered, and avoid reusing passwords across sites.

***

#### Data encryption (personal files)

**Sketch:**

Key Initialisation

Generates an encryption key and initialises the cipher suite at construction time so no unencrypted state is exposed later.

File Encryption

Reads the file in binary mode, encrypts the bytes, then writes a new `.encrypted` file, leaving the original for the caller to decide whether to delete.

Encrypted Backup

JSON-encodes the data, encrypts it, and delegates storage so backup media never holds a plaintext copy.

**In practice:** Encrypt sensitive files at rest, use encrypted backups, prefer HTTPS and trusted messengers for sensitive content, and use secure erase when disposing of storage.

### For business use

#### Data governance

**Sketch:**

Governance Setup

A policy manager and audit logger are injected so policy retrieval and logging are decoupled from this class.

Data Classification

Scores sensitivity and maps it to a protection level, the result drives downstream controls like encryption and access restrictions.

Retention Enforcement

Looks up the retention period for the data category and securely deletes records that have exceeded it, preventing stale data accumulation.

**In practice:** Classify data by sensitivity, enforce retention and access policies, audit periodically, and train anyone who handles personal data.

***

#### Privacy impact assessment

**Sketch:**

PIA Setup

Injects a risk assessor and mitigation planner so risk identification and remediation planning are handled by focused services.

Risk Assessment

Identifies risks, calculates impact scores, and returns them together with a mitigation plan, the three-part structure of a standard privacy impact assessment.

Compliance Monitoring

Delegates ongoing metric tracking to a compliance monitor so the project is continuously evaluated against its privacy commitments.

**Assessment Areas:**

* Data collection scope
* Processing purposes
* Security measures
* Data sharing
* User rights

## Common pitfalls

* **Collecting data "just in case"**: Extra fields increase risk and compliance scope; align collection to documented purposes.
* **Mixing purposes**: Using data for a new goal without notice or consent breaks trust and can break law.
* **Assuming anonymization is automatic**: Removing names is not always enough; combinations of fields can still identify people.

## Next Steps

### In this submodule

Continue to [Data security](data-security.md), then [Workflow concepts](workflow-concepts.md). After submodule 1.1, start [Introduction to Python](../1.2-intro-python/).

### Going deeper on your own

When you are ready to specialize, combine **technical depth** (encryption, identity, secure development) with **program management** (PIAs, vendor reviews, incident response) and **ongoing legal education**-privacy law changes, and your organization's counsel is the source of truth for obligations.

## Additional resources

* [GDPR Official Documentation](https://gdpr.eu/)
* [CCPA Compliance Guide](https://oag.ca.gov/privacy/ccpa)
* [NIST Privacy Framework](https://www.nist.gov/privacy-framework)
* [Privacy by Design](https://www.ipc.on.ca/wp-content/uploads/resources/7foundationalprinciples.pdf)
* [OWASP Privacy Risks](https://owasp.org/www-project-top-10-privacy-risks/)
