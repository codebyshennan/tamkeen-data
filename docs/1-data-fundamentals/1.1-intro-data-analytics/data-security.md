[Previous content remains the same...]

{% step %}
### Types of Encryption
**Implementation Example**:
```python
class EncryptionService:
    def __init__(self):
        self.symmetric_key = self.generate_symmetric_key()
        self.key_pair = self.generate_key_pair()
    
    def symmetric_encryption(self, data):
        """AES encryption for large data"""
        cipher = AES.new(self.symmetric_key, AES.MODE_GCM)
        ciphertext, tag = cipher.encrypt_and_digest(data)
        return {
            'ciphertext': ciphertext,
            'nonce': cipher.nonce,
            'tag': tag
        }
    
    def asymmetric_encryption(self, data):
        """RSA encryption for small data"""
        cipher = PKCS1_OAEP.new(self.key_pair.publickey())
        return cipher.encrypt(data)
    
    def hybrid_encryption(self, data):
        """Hybrid encryption for secure data transfer"""
        # Generate session key
        session_key = get_random_bytes(16)
        
        # Encrypt data with session key
        encrypted_data = self.symmetric_encryption(data)
        
        # Encrypt session key with recipient's public key
        encrypted_session_key = self.asymmetric_encryption(session_key)
        
        return {
            'encrypted_data': encrypted_data,
            'encrypted_session_key': encrypted_session_key
        }
```

**When to Use Each Type:**
- Symmetric: Large data sets, local encryption
- Asymmetric: Key exchange, digital signatures
- Hybrid: Secure communication channels
{% endstep %}
{% endstepper %}

### 2. Access Control Implementation
{% stepper %}
{% step %}
### Role-Based Access Control (RBAC)
**Implementation Example**:
```python
class RBACSystem:
    def __init__(self):
        self.roles = {}
        self.user_roles = {}
        self.permissions = {}
    
    def create_role(self, role_name, permissions):
        """Create a new role with specified permissions"""
        self.roles[role_name] = {
            'permissions': permissions,
            'created_at': datetime.now(),
            'modified_at': datetime.now()
        }
    
    def assign_role(self, user_id, role_name):
        """Assign role to user"""
        if role_name not in self.roles:
            raise ValueError(f"Role {role_name} does not exist")
        
        self.user_roles[user_id] = role_name
        self.log_role_assignment(user_id, role_name)
    
    def check_permission(self, user_id, permission):
        """Check if user has specific permission"""
        role = self.user_roles.get(user_id)
        if not role:
            return False
        
        return permission in self.roles[role]['permissions']
    
    def audit_access(self, user_id):
        """Audit user's access patterns"""
        return {
            'user_id': user_id,
            'role': self.user_roles.get(user_id),
            'permissions': self.get_user_permissions(user_id),
            'access_history': self.get_access_history(user_id)
        }
```

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
{% endstep %}

{% step %}
### Attribute-Based Access Control (ABAC)
**Implementation Example**:
```python
class ABACSystem:
    def __init__(self):
        self.policy_engine = PolicyEngine()
        self.context_manager = ContextManager()
    
    def evaluate_access(self, user, resource, action, context):
        """Evaluate access based on attributes"""
        policy_decision = self.policy_engine.evaluate({
            'user_attributes': {
                'department': user.department,
                'clearance_level': user.clearance_level,
                'location': user.location
            },
            'resource_attributes': {
                'classification': resource.classification,
                'owner': resource.owner,
                'type': resource.type
            },
            'action': action,
            'context': {
                'time': context.current_time,
                'location': context.location,
                'device': context.device_type
            }
        })
        
        self.log_access_decision(
            user, resource, action, policy_decision
        )
        
        return policy_decision
```
{% endstep %}
{% endstepper %}

## Security Monitoring and Incident Response

### 1. Security Monitoring
**Implementation Example**:
```python
class SecurityMonitor:
    def __init__(self):
        self.alert_manager = AlertManager()
        self.threat_detector = ThreatDetector()
    
    def monitor_system_activity(self):
        """Real-time security monitoring"""
        while True:
            # Collect security metrics
            metrics = self.collect_security_metrics()
            
            # Analyze for threats
            threats = self.threat_detector.analyze(metrics)
            
            # Handle detected threats
            for threat in threats:
                self.handle_threat(threat)
            
            time.sleep(self.monitoring_interval)
    
    def handle_threat(self, threat):
        """Handle detected security threat"""
        severity = self.assess_threat_severity(threat)
        
        response = {
            'high': self.emergency_response,
            'medium': self.standard_response,
            'low': self.log_and_monitor
        }[severity]
        
        return response(threat)
```

### 2. Incident Response
**Implementation Example**:
```python
class IncidentResponse:
    def __init__(self):
        self.incident_manager = IncidentManager()
        self.forensics = ForensicsTools()
    
    def handle_security_incident(self, incident):
        """Handle security incident"""
        # Containment
        self.contain_incident(incident)
        
        # Investigation
        evidence = self.collect_evidence(incident)
        analysis = self.analyze_incident(evidence)
        
        # Recovery
        recovery_plan = self.create_recovery_plan(analysis)
        self.execute_recovery(recovery_plan)
        
        # Documentation
        self.document_incident({
            'incident': incident,
            'evidence': evidence,
            'analysis': analysis,
            'recovery': recovery_plan,
            'lessons_learned': self.compile_lessons_learned()
        })
    
    def contain_incident(self, incident):
        """Implement containment measures"""
        containment_actions = {
            'data_breach': self.isolate_affected_systems,
            'malware': self.quarantine_infected_systems,
            'ddos': self.activate_ddos_protection,
            'unauthorized_access': self.revoke_access_tokens
        }
        
        action = containment_actions.get(
            incident.type, self.default_containment
        )
        return action(incident)
```

## Best Practices Implementation

### 1. Security Baseline
```python
class SecurityBaseline:
    def __init__(self):
        self.scanner = VulnerabilityScanner()
        self.config_manager = ConfigurationManager()
    
    def implement_baseline(self, system):
        """Implement security baseline"""
        # Secure configuration
        self.config_manager.apply_hardening(system)
        
        # Update management
        self.update_system(system)
        
        # Access control
        self.implement_access_controls(system)
        
        # Monitoring
        self.setup_monitoring(system)
        
        return self.verify_baseline(system)
```

### 2. Regular Security Assessment
```python
class SecurityAssessment:
    def __init__(self):
        self.vulnerability_scanner = VulnerabilityScanner()
        self.penetration_tester = PenetrationTester()
    
    def conduct_assessment(self, target):
        """Conduct security assessment"""
        results = {
            'vulnerability_scan': self.vulnerability_scanner.scan(target),
            'penetration_test': self.penetration_tester.test(target),
            'configuration_review': self.review_configuration(target),
            'access_control_audit': self.audit_access_controls(target)
        }
        
        return self.generate_assessment_report(results)
```

## Next Steps

After implementing these security measures:

1. **Advanced Security**
   - Implement AI-based threat detection
   - Deploy security information and event management (SIEM)
   - Develop automated incident response
   - Implement zero trust architecture

2. **Compliance and Standards**
   - ISO 27001 implementation
   - SOC 2 compliance
   - NIST framework adoption
   - Industry-specific regulations

3. **Security Operations**
   - Build security operations center
   - Develop incident response team
   - Implement threat hunting
   - Conduct regular red team exercises

4. **Continuous Improvement**
   - Regular security assessments
   - Threat intelligence integration
   - Security awareness training
   - Process automation

## Additional Resources

- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [OWASP Security Guidelines](https://owasp.org/www-project-security-guidelines/)
- [Cloud Security Alliance](https://cloudsecurityalliance.org/)
- [SANS Security Resources](https://www.sans.org/security-resources/)
- [ISO 27001 Standard](https://www.iso.org/isoiec-27001-information-security.html)
