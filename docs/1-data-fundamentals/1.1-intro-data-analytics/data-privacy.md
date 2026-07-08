# Data Privacy

**After this lesson:** You can explain what personal data and sensitive categories are, why privacy rules exist, and how common rights (access, correction, deletion, portability) show up in real projects.

## Introduction

**Data privacy** is about handling personal information in a way that respects people's autonomy and meets legal obligations. It is not only "security" (locks and encryption) and not only "ethics" (doing the right thing), it is the **rules and practices** for how data may be collected, used, and shared.

### Video

<div class="video-embed">
<iframe width="560" height="315" src="https://www.youtube.com/embed/mdNQYU8Xj4E" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*Simplilearn, GDPR explained*

### Why privacy matters for analysts and scientists

- **People**: Laws and company policies exist because misuse of data can harm individuals (discrimination, fraud, embarrassment). Your work is more trustworthy when it respects those boundaries.
- **Trust**: Teams that are transparent about data use get better cooperation from customers and partners.
- **Compliance**: Regulations like GDPR, CCPA, and sector rules (e.g. HIPAA) set hard requirements. **Ignorance is not a defense**; know what applies to your data and region.
- **Risk**: Breaches and misuse lead to fines, lawsuits, and loss of reputation. Minimizing data and clarifying purpose reduces exposure.

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

- **Access**: See what data you hold about them.
- **Rectification**: Correct inaccurate or incomplete data.
- **Erasure**: Request deletion ("right to be forgotten") when the law allows.
- **Portability**: Receive a machine-readable copy to move to another service.

**For beginners:** Treat these as **product and process** requirements, not footnotes. Someone will eventually ask for an export or deletion; your pipelines should not make that impossible.

## Data privacy principles (plain language)

These ideas appear in GDPR and similar frameworks:

1. **Data minimization**: Collect only what you need for a **defined** purpose. "We might use it later" is a red flag.
2. **Purpose limitation**: Use data only for the purposes you disclosed (and that the law allows). A new use case may need new consent or analysis.
3. **Storage limitation**: Delete or anonymize when data is no longer needed for that purpose, per policy and law.
4. **Integrity and confidentiality**: Keep data accurate enough for its use, and protect it against unauthorized access (see [Data security](./data-security.md)).

{% include mermaid-diagram.html src="1-data-fundamentals/1.1-intro-data-analytics/diagrams/data-privacy-1.mmd" %}

*Treat sensitive categories and PHI as their own tier, stricter access controls, minimal retention, and often explicit legal basis before you even touch the data.*

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

---

## Sensitive data: payments and health (illustrative code)

The examples below are **teaching sketches**-not production security advice. They show *why* tokenization, encryption, and consent checks exist.

### Financial data

**Example: Payment processing**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class SecurePaymentProcessor:
    def __init__(self):
        self.tokenization_service = TokenizationService()
        self.encryption_service = EncryptionService()

    def process_payment(self, payment_info):
        """Securely process payment information"""
        # Tokenize card for future use
        payment_token = self.tokenization_service.tokenize(
            payment_info['card_number']
        )

        # Encrypt transaction details
        encrypted_transaction = self.encryption_service.encrypt({
            'amount': payment_info['amount'],
            'currency': payment_info['currency'],
            'timestamp': datetime.now(),
            'token': payment_token
        })

        return {
            'transaction_id': self.generate_transaction_id(),
            'status': 'processed',
            'token': payment_token
        }
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class Setup</span>
    </div>
    <div class="code-callout__body">
      <p>The constructor injects a tokenization service and an encryption service so card data is never stored in plain text.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-13" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Tokenize Card</span>
    </div>
    <div class="code-callout__body">
      <p>The raw card number is replaced by a token immediately, only the token is kept for future charges, limiting exposure.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="15-22" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Encrypt Transaction</span>
    </div>
    <div class="code-callout__body">
      <p>Amount, currency, and timestamp are encrypted together so the payload is opaque to anyone without the key.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="24-28" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Return Receipt</span>
    </div>
    <div class="code-callout__body">
      <p>Only the transaction ID, status, and token are returned, no card number or raw amount leaves this method.</p>
    </div>
  </div>
</aside>
</div>

**Why it matters:** Card data is toxic to store in plain text; tokenization and encryption limit blast radius if a system is compromised.

---

### Healthcare and special-category data

**Example: Healthcare data management**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class SensitiveDataManager:
    def __init__(self):
        self.encryption = EncryptionService()
        self.access_control = AccessControlService()

    def store_sensitive_data(self, data, category):
        """Store sensitive personal information"""
        if category in ['medical', 'biometric']:
            return self.handle_special_category_data(data)

        return self.handle_standard_data(data)

    def handle_special_category_data(self, data):
        """Handle special category data under GDPR"""
        if not self.verify_explicit_consent(data['user_id']):
            raise ConsentError("Explicit consent required")

        encrypted_data = self.encryption.encrypt_special_category(data)
        access_policy = self.create_strict_access_policy(data)

        return self.store_with_policy(encrypted_data, access_policy)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Service Injection</span>
    </div>
    <div class="code-callout__body">
      <p>Two services are injected: one for encryption and one for access control, following separation of concerns.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-12" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Category Gate</span>
    </div>
    <div class="code-callout__body">
      <p>Medical and biometric data is routed to the stricter handler; everything else uses the standard path.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="14-22" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">GDPR Consent Check</span>
    </div>
    <div class="code-callout__body">
      <p>Explicit consent is verified before any special-category data is stored; raises an error if consent is missing, then encrypts and applies a strict access policy.</p>
    </div>
  </div>
</aside>
</div>

**Why it matters:** Special category data requires extra protection

## Privacy laws and code (illustrative)

The classes below show **how software might structure** consent checks and subject-rights handling. Real systems add auditing, identity verification, and legal review.

### GDPR (European Union)

**Technical sketch:**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class GDPRCompliance:
    def __init__(self):
        self.consent_manager = ConsentManager()
        self.data_processor = DataProcessor()

    def handle_data_subject_request(self, user_id, request_type):
        """Handle GDPR data subject requests"""
        if request_type == 'access':
            return self.provide_data_copy(user_id)
        elif request_type == 'erasure':
            return self.delete_user_data(user_id)
        elif request_type == 'portability':
            return self.export_user_data(user_id)
        elif request_type == 'rectification':
            return self.correct_user_data(user_id)

    def verify_consent(self, user_id, processing_purpose):
        """Verify valid consent exists"""
        consent = self.consent_manager.get_consent(user_id, processing_purpose)

        if not consent or consent.is_expired():
            raise ConsentError("Valid consent not found")

        if not self.is_consent_specific_and_informed(consent):
            raise ConsentError("Consent must be specific and informed")

        return True

    def log_processing_activity(self, activity):
        """Maintain records of processing activities"""
        self.processing_log.append({
            'activity': activity,
            'timestamp': datetime.now(),
            'legal_basis': self.get_legal_basis(activity),
            'purpose': activity.get_purpose(),
            'categories': activity.get_data_categories()
        })
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Class Init</span>
    </div>
    <div class="code-callout__body">
      <p>A consent manager and data processor are injected so rights requests and processing can be handled independently.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Subject Rights Dispatch</span>
    </div>
    <div class="code-callout__body">
      <p>Routes the four GDPR subject rights, access, erasure, portability, rectification, to separate methods for clean separation.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-27" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Consent Verification</span>
    </div>
    <div class="code-callout__body">
      <p>Checks that consent exists, is not expired, and is specific and informed, GDPR requires all three conditions before processing.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="29-37" data-tint="4">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Processing Log</span>
    </div>
    <div class="code-callout__body">
      <p>Records each processing activity with timestamp, legal basis, and data categories, required for GDPR's records of processing activities.</p>
    </div>
  </div>
</aside>
</div>

### CCPA (California)

**Technical sketch:**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class CCPACompliance:
    def __init__(self):
        self.privacy_notice = PrivacyNotice()
        self.data_mapper = DataMapper()

    def handle_ccpa_request(self, consumer_id, request_type):
        """Handle CCPA consumer requests"""
        if request_type == 'disclosure':
            return self.disclose_data_collection(consumer_id)
        elif request_type == 'deletion':
            return self.delete_consumer_data(consumer_id)
        elif request_type == 'opt_out':
            return self.opt_out_of_sale(consumer_id)

    def verify_california_resident(self, consumer_id):
        """Verify California residency"""
        consumer_info = self.get_consumer_info(consumer_id)
        return self.residency_verifier.is_california_resident(consumer_info)

    def maintain_do_not_sell_list(self):
        """Maintain 'Do Not Sell My Personal Information' list"""
        return self.privacy_preferences.get_opt_out_list()
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">CCPA Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Privacy notice and data mapper services handle disclosures and mapping what data is held about each consumer.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-14" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Consumer Rights Router</span>
    </div>
    <div class="code-callout__body">
      <p>Routes the three CCPA rights, disclosure, deletion, and opt-out of sale, to their respective handlers.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="16-21" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Residency and Opt-Out</span>
    </div>
    <div class="code-callout__body">
      <p>Verifies California residency before granting rights, and exposes the "do not sell" list for downstream ad/data pipelines.</p>
    </div>
  </div>
</aside>
</div>

## Practical tips (personal and work)

**Personal use:** Reduce attack surface (strong, unique passwords; MFA). **Work use:** Follow governance, classification, retention, and least-privilege access. The code below illustrates patterns, not a complete program.

### For personal use

#### Password security

**Sketch:**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class PasswordManager:
    def __init__(self):
        self.hasher = PasswordHasher()

    def validate_password_strength(self, password):
        """Check password meets security requirements"""
        requirements = {
            'length': len(password) >= 12,
            'uppercase': any(c.isupper() for c in password),
            'lowercase': any(c.islower() for c in password),
            'numbers': any(c.isdigit() for c in password),
            'special': any(not c.isalnum() for c in password)
        }

        if not all(requirements.values()):
            raise WeakPasswordError(
                "Password must meet all security requirements"
            )

        return True

    def generate_secure_password(self):
        """Generate cryptographically secure password"""
        return secrets.token_urlsafe(16)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-3" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Hasher Setup</span>
    </div>
    <div class="code-callout__body">
      <p>A dedicated hasher service is injected so raw passwords are never stored directly by this class.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="5-19" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Strength Validation</span>
    </div>
    <div class="code-callout__body">
      <p>Checks five requirements (length, upper, lower, digits, special characters) and raises an error if any fail.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="21-23" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Secure Generation</span>
    </div>
    <div class="code-callout__body">
      <p>Uses the OS cryptographic random source via <code>secrets</code> to produce a URL-safe 16-byte token, suitable for temporary passwords.</p>
    </div>
  </div>
</aside>
</div>

**Habits that matter:** Use a password manager, turn on **two-factor authentication** where offered, and avoid reusing passwords across sites.

---

#### Data encryption (personal files)

**Sketch:**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class PersonalDataEncryption:
    def __init__(self):
        self.key = self.generate_key()
        self.cipher_suite = self.initialize_cipher()

    def encrypt_personal_file(self, file_path):
        """Encrypt personal files"""
        with open(file_path, 'rb') as file:
            data = file.read()

        encrypted_data = self.cipher_suite.encrypt(data)

        with open(f"{file_path}.encrypted", 'wb') as file:
            file.write(encrypted_data)

    def secure_backup(self, data):
        """Create encrypted backup"""
        encrypted_backup = self.cipher_suite.encrypt(
            json.dumps(data).encode()
        )
        return self.store_backup(encrypted_backup)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Key Initialisation</span>
    </div>
    <div class="code-callout__body">
      <p>Generates an encryption key and initialises the cipher suite at construction time so no unencrypted state is exposed later.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">File Encryption</span>
    </div>
    <div class="code-callout__body">
      <p>Reads the file in binary mode, encrypts the bytes, then writes a new <code>.encrypted</code> file, leaving the original for the caller to decide whether to delete.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-21" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Encrypted Backup</span>
    </div>
    <div class="code-callout__body">
      <p>JSON-encodes the data, encrypts it, and delegates storage so backup media never holds a plaintext copy.</p>
    </div>
  </div>
</aside>
</div>

**In practice:** Encrypt sensitive files at rest, use encrypted backups, prefer HTTPS and trusted messengers for sensitive content, and use secure erase when disposing of storage.

### For business use

#### Data governance

**Sketch:**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class DataGovernance:
    def __init__(self):
        self.policy_manager = PolicyManager()
        self.audit_logger = AuditLogger()

    def classify_data(self, data):
        """Classify data sensitivity level"""
        sensitivity_score = self.calculate_sensitivity(data)
        return self.assign_protection_level(sensitivity_score)

    def enforce_retention_policy(self, data, category):
        """Enforce data retention policies"""
        retention_period = self.policy_manager.get_retention_period(category)

        if self.is_past_retention(data, retention_period):
            return self.securely_delete_data(data)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Governance Setup</span>
    </div>
    <div class="code-callout__body">
      <p>A policy manager and audit logger are injected so policy retrieval and logging are decoupled from this class.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-9" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Data Classification</span>
    </div>
    <div class="code-callout__body">
      <p>Scores sensitivity and maps it to a protection level, the result drives downstream controls like encryption and access restrictions.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="11-17" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Retention Enforcement</span>
    </div>
    <div class="code-callout__body">
      <p>Looks up the retention period for the data category and securely deletes records that have exceeded it, preventing stale data accumulation.</p>
    </div>
  </div>
</aside>
</div>

**In practice:** Classify data by sensitivity, enforce retention and access policies, audit periodically, and train anyone who handles personal data.

---

#### Privacy impact assessment

**Sketch:**

<div class="code-explainer" data-code-explainer>
<div class="code-explainer__code">

{% highlight python %}
class PrivacyImpactAssessment:
    def __init__(self):
        self.risk_assessor = RiskAssessor()
        self.mitigation_planner = MitigationPlanner()

    def assess_privacy_risks(self, project):
        """Assess privacy risks of new project"""
        risks = self.risk_assessor.identify_risks(project)
        impact_scores = self.calculate_impact_scores(risks)

        return {
            'risks': risks,
            'impact_scores': impact_scores,
            'mitigation_plan': self.mitigation_planner.create_plan(risks)
        }

    def monitor_compliance(self, project):
        """Monitor ongoing privacy compliance"""
        return self.compliance_monitor.track_metrics(project)
{% endhighlight %}

</div>
<aside class="code-explainer__callouts" aria-label="Code walkthrough">
  <div class="code-callout" data-lines="1-4" data-tint="1">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">PIA Setup</span>
    </div>
    <div class="code-callout__body">
      <p>Injects a risk assessor and mitigation planner so risk identification and remediation planning are handled by focused services.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="6-15" data-tint="2">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Risk Assessment</span>
    </div>
    <div class="code-callout__body">
      <p>Identifies risks, calculates impact scores, and returns them together with a mitigation plan, the three-part structure of a standard privacy impact assessment.</p>
    </div>
  </div>
  <div class="code-callout" data-lines="17-19" data-tint="3">
    <div class="code-callout__meta">
      <span class="code-callout__lines"></span>
      <span class="code-callout__title">Compliance Monitoring</span>
    </div>
    <div class="code-callout__body">
      <p>Delegates ongoing metric tracking to a compliance monitor so the project is continuously evaluated against its privacy commitments.</p>
    </div>
  </div>
</aside>
</div>

**Assessment Areas:**
- Data collection scope
- Processing purposes
- Security measures
- Data sharing
- User rights

## Common pitfalls

- **Collecting data "just in case"**: Extra fields increase risk and compliance scope; align collection to documented purposes.
- **Mixing purposes**: Using data for a new goal without notice or consent breaks trust and can break law.
- **Assuming anonymization is automatic**: Removing names is not always enough; combinations of fields can still identify people.

## Next Steps

### In this submodule

Continue to [Data security](./data-security.md), then [Workflow concepts](./workflow-concepts.md). After submodule 1.1, start [Introduction to Python](../1.2-intro-python/README.md).

### Going deeper on your own

When you are ready to specialize, combine **technical depth** (encryption, identity, secure development) with **program management** (PIAs, vendor reviews, incident response) and **ongoing legal education**-privacy law changes, and your organization's counsel is the source of truth for obligations.

## Additional resources

- [GDPR Official Documentation](https://gdpr.eu/)
- [CCPA Compliance Guide](https://oag.ca.gov/privacy/ccpa)
- [NIST Privacy Framework](https://www.nist.gov/privacy-framework)
- [Privacy by Design](https://www.ipc.on.ca/wp-content/uploads/resources/7foundationalprinciples.pdf)
- [OWASP Privacy Risks](https://owasp.org/www-project-top-10-privacy-risks/)
