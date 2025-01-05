# Data Collection

## Overview

Data collection is the process of gathering and assembling data from various sources. It is a crucial step in the data analytics process, as it provides the raw material for analysis.

## Types of Data Sources

1. First-Party Data
   - Definition: Data collected directly by a company from its own sources.
   - Examples:
     - Customer interactions (website visits, purchases)
     - CRM data
     - Surveys and feedback forms
2. Third-Party Data
   - Definition: Data collected by an entity that does not have a direct relationship with the user.
   - Examples:
     - Data aggregators (e.g., demographic data)
     - Social media insights
     - Market research reports
3. Second-Party Data
   - Definition: Data shared between two parties that have a direct relationship.
   - Examples:
     - Partner companies sharing customer data
     - Affiliate marketing data
4. Public Data
   - Definition: Data that is freely available to the public.
   - Examples:
     - Government databases
     - Open datasets from research institutions

## Data Collection Methods

1. **Data Logging**

   - Recording events or data points for analysis.
   - Essential for tracking user interactions and system performance.

2. **Surveys and Questionnaires**

   - Structured forms for gathering information.
   - Can be conducted online or offline.

3. **Interviews**

   - Direct conversations for detailed insights.
   - Can be structured, semi-structured, or unstructured.

4. **Observations**

   - Watching subjects in their natural environment.
   - Useful for qualitative data collection.

5. **Experiments**

   - Controlled studies to test hypotheses.
   - Often involves a control group and experimental group.

6. **Focus Groups**

   - Guided discussions for diverse opinions.
   - Typically involves 6-10 participants.

7. **Secondary Data Analysis**

   - Analyzing existing data from other sources.
   - Saves time and resources compared to primary data collection.

8. **Web Scraping**

   - Automated extraction of data from websites.
   - Useful for collecting large datasets quickly.

9. **Social Media Monitoring**

   - Analyzing data from social platforms.
   - Helps understand public sentiment and trends.

10. **Sensor Data Collection**

    - Gathering data from physical sensors.
    - Common in IoT applications and environmental monitoring.

### 3. Observational Data Collection

{% stepper %}
{% step %}
### In-Store Observation
**Example: Retail Store Layout Study**
What to Track:
- Customer walking patterns (heatmap)
- Time spent in each section (dwell time)
- Product interaction frequency (touch points)
- Purchase decision points (conversion zones)
- Traffic flow bottlenecks

**Implementation Example**:
```python
# Using computer vision for customer tracking
import cv2
import numpy as np
from tracking import CustomerTracker

class StoreAnalytics:
    def __init__(self):
        self.tracker = CustomerTracker()
        self.heatmap = np.zeros((store_height, store_width))
    
    def process_frame(self, frame):
        customers = self.tracker.detect_and_track(frame)
        for customer in customers:
            position = customer.get_position()
            self.update_heatmap(position)
            self.analyze_dwell_time(customer)
    
    def generate_insights(self):
        return {
            'high_traffic_areas': self.get_hotspots(),
            'avg_dwell_time': self.calculate_dwell_time(),
            'conversion_zones': self.identify_conversion_zones()
        }
```

**Methods:**
- Computer vision tracking
- IoT sensor networks
- RFID tracking
- WiFi positioning
{% endstep %}

{% step %}
### User Experience Testing
**Example: Website Usability Study**
Observations:
- Navigation patterns (click paths)
- Error encounters (frequency and type)
- Task completion time (efficiency)
- Facial expressions (emotional response)
- Mouse movement patterns

**Session Recording Example**:
```javascript
// Using FullStory-like session recording
class UserSession {
    constructor() {
        this.events = [];
        this.startTime = Date.now();
    }

    trackMouseMovement(event) {
        this.events.push({
            type: 'mouse_move',
            x: event.clientX,
            y: event.clientY,
            timestamp: Date.now() - this.startTime
        });
    }

    trackClick(event) {
        this.events.push({
            type: 'click',
            element: event.target.tagName,
            id: event.target.id,
            timestamp: Date.now() - this.startTime
        });
    }

    trackError(error) {
        this.events.push({
            type: 'error',
            message: error.message,
            stack: error.stack,
            timestamp: Date.now() - this.startTime
        });
    }
}
```

**Tools:**
- Screen recording software
- Eye tracking hardware
- Emotion recognition AI
- Session replay tools
{% endstep %}
{% endstepper %}

## Common Data Collection Challenges and Solutions

{% stepper %}
{% step %}
### 1. Data Quality Issues
**Challenge:** Incomplete or incorrect data
**Solution:** 
- Implement validation rules
- Use required fields
- Double-check data entry
- Regular data audits
- Automated data cleaning

**Data Validation Example**:
```python
class DataValidator:
    def __init__(self):
        self.rules = {
            'email': lambda x: re.match(r"[^@]+@[^@]+\.[^@]+", x),
            'phone': lambda x: re.match(r"^\+?1?\d{9,15}$", x),
            'age': lambda x: isinstance(x, int) and 0 <= x <= 120,
            'income': lambda x: isinstance(x, (int, float)) and x >= 0
        }
    
    def validate_record(self, record):
        errors = []
        for field, value in record.items():
            if field in self.rules:
                if not self.rules[field](value):
                    errors.append(f"Invalid {field}: {value}")
        return len(errors) == 0, errors

    def clean_data(self, dataset):
        return [record for record in dataset 
                if self.validate_record(record)[0]]
```
{% endstep %}

{% step %}
### 2. Privacy Concerns
**Challenge:** Collecting sensitive information
**Solution:**
- Clear consent forms
- Data anonymization
- Secure storage
- Privacy policy transparency
- Data minimization

**Data Anonymization Example**:
```python
class DataAnonymizer:
    def __init__(self):
        self.hash_key = os.urandom(16)
    
    def anonymize_pii(self, data):
        """Anonymize Personally Identifiable Information"""
        return {
            'user_id': self.hash_value(data['user_id']),
            'age_range': self.bucket_age(data['age']),
            'location': self.generalize_location(data['location']),
            'interests': data['interests']  # Non-PII can remain
        }
    
    def hash_value(self, value):
        return hashlib.sha256(
            f"{value}{self.hash_key}".encode()
        ).hexdigest()
    
    def bucket_age(self, age):
        ranges = [(0, 18), (19, 25), (26, 35), (36, 50), (51, float('inf'))]
        for start, end in ranges:
            if start <= age <= end:
                return f"{start}-{end if end != float('inf') else '+'}"
```
{% endstep %}

{% step %}
### 3. Sample Size and Representation
**Challenge:** Getting enough responses and ensuring representation
**Solution:**
- Multiple collection channels
- Incentive programs
- Extended collection period
- Follow-up reminders
- Stratified sampling

**Sample Size Calculator**:
```python
import math

def calculate_sample_size(population_size, confidence_level, margin_error):
    """
    Calculate required sample size for a given population
    
    Args:
        population_size: Total population size
        confidence_level: Desired confidence level (e.g., 0.95 for 95%)
        margin_error: Acceptable margin of error (e.g., 0.05 for 5%)
    """
    z_scores = {
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }
    
    z = z_scores.get(confidence_level, 1.96)
    
    sample_size = (
        (z**2 * 0.25 * population_size) /
        ((margin_error**2 * (population_size - 1)) + (z**2 * 0.25))
    )
    
    return math.ceil(sample_size)
```
{% endstep %}

{% step %}
### 4. Bias Management
**Challenge:** Skewed data collection
**Solution:**
- Random sampling techniques
- Diverse data sources
- Neutral question phrasing
- Representative demographics
- Bias detection algorithms

**Bias Detection Example**:
```python
class BiasDetector:
    def analyze_demographic_bias(self, data, protected_attributes):
        """
        Analyze dataset for demographic bias
        
        Args:
            data: DataFrame with survey responses
            protected_attributes: List of demographic columns to check
        """
        bias_metrics = {}
        
        for attribute in protected_attributes:
            # Calculate representation ratios
            population_dist = self.get_population_distribution(attribute)
            sample_dist = data[attribute].value_counts(normalize=True)
            
            # Compare with expected distribution
            bias_metrics[attribute] = {
                'representation_ratio': sample_dist / population_dist,
                'chi_square_test': self.chi_square_test(
                    sample_dist, population_dist
                )
            }
        
        return bias_metrics
```
{% endstep %}
{% endstepper %}

## Best Practices for Data Collection

### 1. Planning and Preparation
- Define clear objectives
- Choose appropriate methods
- Prepare necessary tools
- Set realistic timelines
- Create data governance framework

**Project Planning Template**:
```python
class DataCollectionProject:
    def __init__(self, name, objectives):
        self.name = name
        self.objectives = objectives
        self.timeline = {}
        self.methods = []
        self.tools = []
        self.team = []
    
    def add_phase(self, phase_name, duration, deliverables):
        self.timeline[phase_name] = {
            'duration': duration,
            'deliverables': deliverables,
            'status': 'planned'
        }
    
    def assign_team(self, role, person):
        self.team.append({
            'role': role,
            'person': person,
            'responsibilities': self.get_role_responsibilities(role)
        })
```

### 2. Quality Control
- Data validation frameworks
- Regular audits
- Error logging
- Quality metrics tracking
- Automated testing

### 3. Documentation
- Detailed methodology
- Data dictionary
- Collection procedures
- Quality control processes
- Ethics considerations

### 4. Technical Infrastructure
- Scalable storage solutions
- Backup systems
- Security measures
- Processing pipelines
- Monitoring tools

## Advanced Data Collection Methods

### 1. IoT and Sensor Networks
```python
class IoTDataCollector:
    def __init__(self):
        self.sensors = {}
        self.data_buffer = []
    
    def register_sensor(self, sensor_id, sensor_type, location):
        self.sensors[sensor_id] = {
            'type': sensor_type,
            'location': location,
            'status': 'active'
        }
    
    def collect_sensor_data(self, sensor_id, data):
        timestamp = datetime.now()
        self.data_buffer.append({
            'sensor_id': sensor_id,
            'timestamp': timestamp,
            'data': data,
            'metadata': self.sensors[sensor_id]
        })
```

### 2. API Integration
```python
class APIDataCollector:
    def __init__(self, api_key):
        self.api_key = api_key
        self.rate_limiter = RateLimiter()
    
    async def collect_data(self, endpoint, parameters):
        """Collect data from API with rate limiting"""
        await self.rate_limiter.acquire()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    endpoint,
                    params=parameters,
                    headers={'Authorization': f'Bearer {self.api_key}'}
                ) as response:
                    return await response.json()
        finally:
            self.rate_limiter.release()
```

### 3. Web Scraping
```python
class WebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.parser = BeautifulSoup
    
    def scrape_page(self, url, selectors):
        """Scrape specific elements from a webpage"""
        response = self.session.get(url)
        soup = self.parser(response.text, 'html.parser')
        
        data = {}
        for key, selector in selectors.items():
            elements = soup.select(selector)
            data[key] = [el.text.strip() for el in elements]
        
        return data
```

## Next Steps

After mastering these data collection methods:

1. **Data Processing**
   - Learn data cleaning techniques
   - Master ETL processes
   - Understand data warehousing

2. **Analysis Techniques**
   - Statistical analysis
   - Machine learning basics
   - Visualization methods

3. **Advanced Topics**
   - Real-time data collection
   - Distributed systems
   - Edge computing
   - Stream processing

4. **Professional Skills**
   - Data governance
   - Ethics and compliance
   - Project management
   - Team collaboration
