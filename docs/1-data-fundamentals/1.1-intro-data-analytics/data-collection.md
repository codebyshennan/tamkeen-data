# Data Collection

**After this lesson:** You can name common data sources and collection methods, explain how those choices affect later analysis, and spot where planning and documentation matter as much as tools.

## Overview

Data collection is how you obtain the raw material for any analysis. The way you collect, who is included, how often you sample, what you ask, and how you record consent, shows up later as **bias**, **gaps**, or **compliance risk**. This page introduces vocabulary and trade-offs; it does not try to teach every tool in depth.

### Video

_CrashCourse Statistics, Mathematical thinking (sampling and study design)_

## Why this matters

Poor collection (biased samples, wrong timestamps, missing consent) limits every downstream step. Thinking about collection early saves rework in cleaning and modeling.

## Prerequisites

* [Introduction to Data Analytics](./) (or equivalent context on the analytics lifecycle)
* No code required for this page; later examples use Python only as illustration

## Types of data sources

Organizations rarely use a single source. You should know **who generated the data** and **what relationship** they have to the people or systems described in it. That affects trust, legal rights, and how you interpret gaps.

_The closer to the source, the more you know about how data was gathered, and the fewer compliance surprises downstream._

### First-party data

**What it is:** Data your organization collects directly from customers, systems, or employees.

**Why it matters:** You usually know how it was captured (e.g. checkout logs, CRM fields). You still need clear definitions and timestamps, but you are not guessing a stranger's pipeline.

**Examples:** Website or app events, purchases, support tickets, in-house surveys, operational logs.

### Second-party data

**What it is:** Another organization's first-party data, shared with you under an agreement (often a partner or advertiser).

**Why it matters:** You inherit their definitions and collection rules. Contracts spell out **purpose**, **retention**, and **whether** the data can be combined with yours.

**Examples:** A retailer sharing purchase summaries with a brand; co-marketing lists governed by partner terms.

### Third-party data

**What it is:** Data collected by a vendor that typically **does not** have a direct relationship with the individuals in the dataset (aggregators, brokers, market research firms).

**Why it matters:** Useful for benchmarks or enrichment, but provenance can be opaque. Compliance (consent, opt-out) is stricter in many jurisdictions; always align use with license and law.

**Examples:** Demographic segments, credit or marketing databases where permitted, syndicated research.

### Public data

**What it is:** Data published for reuse: government open data, research repositories, competition datasets.

**Why it matters:** Often free to access, but **not** free of responsibility. Check licenses (commercial use, attribution), refresh cadence, and known limitations (geographic coverage, methodology changes).

**Examples:** Census tables, weather archives, open portals from cities or agencies.

## Data collection methods

Below are common **methods**, not a checklist for every project. Real work combines several (e.g. surveys plus server logs). For each method, ask: _What could go wrong if we only used this?_

_Start with the question you need to answer, that determines the method, not the other way around._

### 1. Data logging

**In plain terms:** Systems record events as they happen, clicks, errors, API calls, sensor readings.

**Good for:** High-volume, timestamped behavior; A/B tests; reliability monitoring.

**Watch out for:** Missing events when logging fails silently; inconsistent event names; **PII** in logs without a plan.

### 2. Surveys and questionnaires

**In plain terms:** You ask people the same questions in a structured way (online or on paper).

**Good for:** Opinions, demographics, satisfaction, things you cannot infer from clicks alone.

**Watch out for:** Non-response bias, leading questions, and scale mismatch (e.g. comparing two surveys with different wording).

### 3. Interviews

**In plain terms:** One-on-one or small conversations, structured, semi-structured, or open-ended.

**Good for:** Depth, nuance, and discovering variables you did not put in a survey yet.

**Watch out for:** Small **N**, hard-to-generalize quotes, and transcription cost. Often paired with quantitative methods.

### 4. Observations

**In plain terms:** You record what people or systems do in context (ethnography, field notes, session observation).

**Good for:** Workflows, physical spaces, and usability issues users cannot articulate.

**Watch out for:** Observer bias and privacy; get consent when observing people.

### 5. Experiments

**In plain terms:** You change one thing at a time (where possible) and compare outcomes to a **control**.

**Good for:** Causal claims ("this change _caused_ that effect") when randomized and well designed.

**Watch out for:** Selection into treatment, spillover between groups, and short windows that miss long-term effects.

### 6. Focus groups

**In plain terms:** A facilitator leads a small group discussion (often 6-10 people).

**Good for:** Exploring reactions to concepts, messaging, or prototypes.

**Watch out for:** Dominant participants and groupthink; not a substitute for a representative sample.

### 7. Secondary data analysis

**In plain terms:** You analyze data someone else already collected (reports, studies, internal databases).

**Good for:** Speed and cost; historical baselines.

**Watch out for:** Definition drift, missing documentation, and **not** assuming the original purpose matches yours.

### 8. Web scraping

**In plain terms:** Automated extraction of content from websites.

**Good for:** Public information at scale when APIs are missing.

**Watch out for:** Terms of service, robots.txt, rate limits, and fragile HTML. Prefer official APIs when available.

### 9. Social media monitoring

**In plain terms:** Collecting and analyzing posts, mentions, or trends from social platforms.

**Good for:** Brand sentiment, emerging topics.

**Watch out for:** Bots, sarcasm, sampling bias (who posts publicly), and platform policy changes.

### 10. Sensor data collection

**In plain terms:** Streams from devices: temperature, GPS, industrial equipment, wearables.

**Good for:** Physical world signals at high frequency.

**Watch out for:** Calibration drift, dropped packets, and aligning sensor time with business events.

***

## Observational studies in retail and product (examples)

These are **special cases** of observation and logging, shown so you see how "methods" become concrete metrics.

### In-store observation

**Example: Retail store layout**

Teams often want to know where people walk, how long they linger, and where they convert. Metrics might include:

* Customer walking patterns (heatmap)
* Time spent in each section (dwell time)
* Product interaction frequency
* Purchase decision points
* Traffic bottlenecks

#### Retail heatmap sketch (illustrative)

* **Purpose:** Show how in-store analytics code might hold a **2D accumulator** (`heatmap`) and expose **summary hooks** (`generate_insights`), not a real tracker, but the shape of the program.
* **Walkthrough:** `StoreAnalytics` stubs `process_frame` (where CV would update the heatmap) and returns dict-shaped "insights" for dashboards.

**Note:** `store_height` / `store_width` would match your camera frame.

```python
# Using computer vision for customer tracking (illustrative sketch)
import numpy as np

store_height, store_width = 480, 640  # example frame size in pixels

# from tracking import CustomerTracker  # hypothetical module

class StoreAnalytics:
    def __init__(self):
        # self.tracker = CustomerTracker()
        self.heatmap = np.zeros((store_height, store_width))

    def process_frame(self, frame):
        # customers = self.tracker.detect_and_track(frame)
        # for customer in customers:
        #     position = customer.get_position()
        #     self.update_heatmap(position)
        #     self.analyze_dwell_time(customer)
        pass

    def generate_insights(self):
        return {
            'high_traffic_areas': self.get_hotspots(),
            'avg_dwell_time': self.calculate_dwell_time(),
            'conversion_zones': self.identify_conversion_zones(),
        }
```

**Typical tooling:** Computer vision, IoT beacons, RFID, WiFi positioning, each with different cost and privacy implications.

### User experience testing

**Example: Website usability**

You might combine session replay, tasks, and optional moderated sessions. Signals include:

* Navigation paths and clicks
* Errors and dead ends
* Task completion time
* Qualitative notes or (with consent) emotion cues

#### Session event buffer (conceptual JavaScript)

* **Purpose:** Show how UX tooling often **buffers typed events** (move, click, error) with **relative timestamps** for replay or analytics, privacy and consent still apply before shipping anything like this.
* **Walkthrough:** `UserSession` appends small objects to `events`; production code would batch, compress, and redact PII.

**Illustrative pattern** (browser-side event capture):

```javascript
// Session recording style event buffer (conceptual)
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

**Tools:** Screen recording, eye tracking, session replay products, always governed by privacy notices and retention limits.

## Common challenges and how teams address them

### 1. Data quality issues

**What goes wrong:** Wrong types, duplicates, impossible values, or silent missingness.

**What teams do:** Define validation rules up front, require critical fields, audit samples regularly, and automate checks in pipelines, not only manual eyeballing.

#### Field-level validation rules

* **Purpose:** Centralize **per-column checks** (regex, bounds) in one `rules` map so pipelines can reject or quarantine bad rows consistently, instead of scattering one-off `if` statements.
* **Walkthrough:** Each lambda is one predicate; `validate_record` walks fields and collects error strings; `clean_data` keeps rows that pass every rule present in `rules`.

**Example, validation rules in code:**

```python
import re

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

### 2. Privacy concerns

**What goes wrong:** Collecting sensitive fields without legal basis, clear purpose, or secure handling.

**What teams do:** Minimize data, obtain meaningful consent where required, anonymize or pseudonymize where possible, and separate **identification** from **analysis** when you can.

#### Pseudonymization with a keyed hash (illustrative)

* **Purpose:** Replace direct identifiers with a **keyed digest**, **bucket** ages, and **generalize** location so downstream tables keep utility with lower re-identification risk.
* **Walkthrough:** `os.urandom` seeds `hash_key` (protect like any secret); `hash_value` concatenates value and key before SHA-256; `bucket_age` and `generalize_location` are placeholders for policy-defined coarsening.

**Example, pseudonymization sketch:**

```python
import hashlib
import os

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
        return "unknown"

    def generalize_location(self, location):
        # Illustrative: replace with real generalization rules
        return str(location)[:3] + "…"
```

### 3. Sample size and representation

**What goes wrong:** Too few responses, or a sample that looks like "who answered the survey" instead of "who we care about."

**What teams do:** Use multiple channels, sensible incentives, reminders, longer field periods, and **stratified** sampling when you need representation across known groups.

#### Finite-population sample size (illustrative)

* **Purpose:** Turn **confidence level**, **margin of error**, and **population size** into one planned $n$ using a normal approximation and \\(p(1-p)\approx 0.25\\) as a conservative proportion, useful for survey planning before fieldwork.
* **Walkthrough:** `z_scores` maps common \\(\alpha\\) to critical $z$; the fraction is a standard finite-population form; `math.ceil` rounds up. Confirm design and assumptions with a statistician for important decisions.

**Example, sample size helper** (classic formula sketch; confirm assumptions with a statistician for important decisions):

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

### 4. Bias management

**What goes wrong:** Over-representing easy-to-reach groups, asking loaded questions, or training on historical discrimination.

**What teams do:** Randomize where ethical and feasible, diversify sources, test question wording, and measure representation vs a reference population when you have one.

#### Sample vs reference mix (illustrative stub)

* **Purpose:** Sketch how teams compare **observed** category shares in a sample to a **reference** distribution (e.g. census) before trusting a survey or model training split.
* **Walkthrough:** For each `protected_attribute`, align `sample_dist` with `population_dist`, then store ratios and a test statistic. Implement `get_population_distribution` and `chi_square_test` with real baselines and `scipy`/domain packages, stubs here are not runnable end-to-end.

**Example, comparing sample mix to an expected distribution** (illustrative; needs domain-specific population baselines):

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
            population_dist = self.get_population_distribution(attribute)
            sample_dist = data[attribute].value_counts(normalize=True)

            bias_metrics[attribute] = {
                'representation_ratio': sample_dist / population_dist,
                'chi_square_test': self.chi_square_test(
                    sample_dist, population_dist
                )
            }

        return bias_metrics
```

## Best practices (what "good" looks like)

These are habits that separate fragile projects from auditable ones. You do not need every item on day one; you **do** need the mindset: **plan, document, validate, and govern.**

### Planning and preparation

Before you pull data, write down **why** you need it and **what decision** it will support. That drives method choice (survey vs log), timeline, and who must sign off.

* **Objectives**: Tie metrics to decisions, not to "more data."
* **Methods**: Match method to question (causal vs descriptive).
* **Tools**: Note systems of record, APIs, and access requests early.
* **Timelines**: Include pilot, validation, and buffer for rework.
* **Governance**: Who owns the data, retention rules, and approval for sensitive fields.

#### Project planning scaffold (code-shaped charter)

* **Purpose:** Keep **objectives, phases, timeline, and roles** in one structure so collection work stays traceable for engineering, compliance, or handoff.
* **Walkthrough:** `add_phase` records duration, deliverables, and status; `assign_team` pairs people to roles via `get_role_responsibilities` (define that method for your org, omitted here as a stub).

**Illustrative project scaffold:**

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

### Quality control

Treat quality as **continuous**, not a one-time scrub.

* **Validation**: Rules at ingest; quarantine bad rows instead of silently dropping without logs.
* **Audits**: Periodic manual review of samples and edge cases.
* **Error logging**: Know when pipelines fail or fields spike.
* **Metrics**: Completeness, timeliness, duplicate rate, whatever matches your risk.
* **Automated testing**: Especially for recurring extracts and transforms.

### Documentation

Future you is a stakeholder. Write for them.

* **Methodology**: How and when data was collected.
* **Data dictionary**: Field names, units, allowed values, and known issues.
* **Procedures**: Steps to reproduce a pull or survey wave.
* **Quality notes**: What you fixed, what you did not, and why.
* **Ethics**: Consent scope and sensitive fields.

### Technical infrastructure

Match investment to sensitivity and scale.

* **Storage**: Durable, permissioned, and cost-aware.
* **Backups**: Test restores, not only backups on paper.
* **Security**: Encryption, access control, secrets hygiene (see [Data security](data-security.md)).
* **Pipelines**: Repeatable jobs with monitoring.
* **Monitoring**: Volume, latency, and anomaly alerts for feeds you depend on.

## Advanced collection patterns (sketches)

The blocks below are **patterns** you may see in larger systems: sensors, APIs, scraping. Read them as architecture sketches, not copy-paste production code.

### IoT and sensor networks

#### IoT buffer pattern

* **Purpose:** **Register** sensors with metadata, then **append** timestamped readings to an in-memory buffer, typical first step before batching to a warehouse or stream.
* **Walkthrough:** `register_sensor` stores type/location; `collect_sensor_data` timestamps each row with `datetime.now()`.

```python
from datetime import datetime

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

### API integration

#### Async API pull with rate limiting

* **Purpose:** Show how **async** `GET` requests, **Authorization** headers, and a **rate limiter** fit together in a responsible API client, avoid hammering third-party endpoints.
* **Walkthrough:** `aiohttp` session context managers; `try`/`finally` releases the limiter-`RateLimiter` would be your shared implementation (queue, token bucket, etc.).

```python
# Illustrative: pip install aiohttp; implement RateLimiter for your policy
import aiohttp

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

### Web scraping

#### CSS selector extraction

* **Purpose:** Minimal **requests + BeautifulSoup** scrape: fetch HTML, parse, and pull text for each **CSS selector**-respect `robots.txt`, terms of service, and rate limits in real use.
* **Walkthrough:** `Session()` reuses connections; `soup.select(selector)` returns lists matching each named field in `selectors`.

```python
import requests
from bs4 import BeautifulSoup

class WebScraper:
    def __init__(self):
        self.session = requests.Session()

    def scrape_page(self, url, selectors):
        """Scrape specific elements from a webpage"""
        response = self.session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        data = {}
        for key, selector in selectors.items():
            elements = soup.select(selector)
            data[key] = [el.text.strip() for el in elements]

        return data
```

## Common pitfalls

* **Convenience sampling**: Data that is easy to reach (only active users, only one region) rarely represents the whole population. Ask who is missing.
* **Ignoring time and seasonality**: A snapshot taken in a holiday week or during an outage can mislead trends. Align windows and note anomalies.
* **Unclear consent or purpose**: Collecting fields "for later" without a use case increases privacy risk and rework. Collect what you need for known purposes.

## Next steps

In this submodule, continue with:

1. [Data privacy](data-privacy.md), legal and ethical constraints on what you collect
2. [Data security](data-security.md), protecting data after collection
3. [Workflow concepts](workflow-concepts.md), how collection fits into broader pipelines

Later in the course you will apply collection ideas in SQL, APIs, and engineering modules.
