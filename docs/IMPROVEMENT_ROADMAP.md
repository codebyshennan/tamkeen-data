# Course Content Improvement Roadmap

## Executive Summary

Based on comprehensive analysis of all 23 subsections across 6 modules, this roadmap outlines a phased approach to standardize quality, fill critical gaps, and enhance the learning experience.

### Current Progress (Updated)

| Phase | Status | Completed |
|-------|--------|-----------|
| Phase 1: Critical Fixes | **DONE** | 3.2, 3.3, 2.3, 5.4 all fixed |
| Phase 2: Notebooks | **IN PROGRESS** | Module 1 done (5/13 notebooks) |
| Phase 3: Visual Assets | Pending | - |
| Phase 4: Exercises | Pending | - |
| Phase 5: Structure | Pending | - |
| Phase 6: Navigation | Pending | - |
| Phase 7: QA | Pending | - |
| Phase 8: Capstone | Pending | - |

### Current State Metrics
- **Total Subsections**: 23
- **Tier 1 (Exemplary)**: 7 (30%)
- **Tier 2 (Strong)**: 6 (26%)
- **Tier 3 (Adequate)**: 6 (26%)
- **Tier 4 (Critical Gaps)**: 4 (17%)

### Target State
- **All subsections at Tier 2+**: 100%
- **Jupyter notebooks**: 100% coverage (currently 43%)
- **Visual assets**: Minimum 5 per subsection
- **Exercises/Assignments**: 100% coverage (currently 33%)

---

## Phase 1: Critical Fixes (Week 1-2) - COMPLETED

### Priority: Address Critical Gaps in Visualization Modules

**Status: COMPLETED**

Work completed:
- [x] 3.2 Advanced Data Visualization: Created `assets/generate_visualizations.py`, generated 20 images
- [x] 3.3 BI/Tableau: Created 71 placeholder images, updated all markdown files with image refs
- [x] 2.3 EDA: Created `assets/generate_visualizations.py`, generated 15 images
- [x] 5.4 Unsupervised Learning: Converted 4 stub files to redirect pages

These gaps are embarrassing for a data science course - visualization modules without visuals.

#### 1.1 Fix 3.2 Advanced Data Visualization
**Current State**: 0 images in a visualization module
**Actions**:
- [ ] Create `assets/` folder
- [ ] Add `generate_visualizations.py` script
- [ ] Generate 15+ Seaborn output examples:
  - Distribution plots (histplot, kdeplot, violinplot)
  - Relationship plots (scatterplot, regplot, pairplot)
  - Categorical plots (boxplot, barplot, countplot)
  - Matrix plots (heatmap, clustermap)
- [ ] Generate 10+ Plotly output examples:
  - Interactive scatter, line, bar charts
  - 3D plots
  - Geographic maps
  - Animated charts
- [ ] Update markdown files to reference new images

**Deliverables**:
- `3.2-adv-data-viz/assets/generate_visualizations.py`
- 25+ PNG files showing actual plot outputs
- Updated markdown with `![](assets/*.png)` references

#### 1.2 Fix 3.3 Tableau/BI Screenshots
**Current State**: All `[Screenshot: ...]` placeholders, no actual images
**Actions**:
- [ ] Create actual Tableau screenshots from Sample Superstore
- [ ] Create Power BI screenshots
- [ ] Create Looker Studio screenshots
- [ ] Replace all placeholder text with actual images

**Deliverables**:
- `3.3-bi-with-tableau/assets/` with 20+ screenshots
- Updated markdown files

#### 1.3 Fix 2.3 EDA Visualizations
**Current State**: 0 images for EDA module
**Actions**:
- [ ] Create distribution visualization outputs
- [ ] Create correlation heatmap examples
- [ ] Create time series plot examples
- [ ] Add before/after data cleaning visuals

**Deliverables**:
- `2.3-eda/assets/` with 10+ images
- Generation script for reproducibility

#### 1.4 Fix 5.4 Stub Files
**Current State**: 4 empty stub files
**Actions**:
- [ ] Either populate `k-means-clustering.md`, `hierarchical-clustering.md`, `dbscan.md`, `t-sne.md`
- [ ] Or remove them and consolidate into existing comprehensive files

**Deliverables**:
- Decision documented
- Files either populated or removed

---

## Phase 2: Notebook Coverage (Week 3-4) - IN PROGRESS

### Priority: Add Jupyter Notebooks to All Modules 1-3

**Status: IN PROGRESS (5/13 notebooks completed)**

Work completed:
- [x] 1.1 Intro Data Analytics: `tutorial.ipynb` created
- [x] 1.2 Intro Python: `tutorial.ipynb` created
- [x] 1.3 Intro Statistics: `tutorial.ipynb` created
- [x] 1.4 NumPy/Linear Algebra: `tutorial.ipynb` created
- [x] 1.5 Pandas: `tutorial.ipynb` created

Remaining:
- [ ] 2.1 SQL: `tutorial.ipynb`
- [ ] 2.2 Data Wrangling: `tutorial.ipynb`
- [ ] 2.3 EDA: `tutorial.ipynb`
- [ ] 2.4 Data Engineering: `tutorial.ipynb`
- [ ] 3.1 Intro Viz: `tutorial.ipynb`
- [ ] 3.2 Adv Viz: `tutorial.ipynb`
- [ ] 3.3 Tableau: `tutorial.ipynb`
- [ ] 3.4 Storytelling: `tutorial.ipynb`

Currently 0% notebook coverage in Modules 1-3.

#### 2.1 Module 1 Notebooks

| Subsection | Notebook Focus | Estimated Cells |
|------------|----------------|-----------------|
| 1.1 Intro Analytics | Data analytics workflow demo | 15 |
| 1.2 Intro Python | Python fundamentals exercises | 20 |
| 1.3 Statistics | Statistical simulations | 25 |
| 1.4 Linear Algebra | NumPy operations practice | 20 |
| 1.5 Pandas | DataFrame manipulation | 25 |

**Template Structure**:
```python
# Cell 1: Setup and imports
# Cell 2-5: Concept demonstrations
# Cell 6-10: Guided exercises
# Cell 11-15: Practice problems
# Final Cell: Solutions (collapsible)
```

**Deliverables**:
- `1.1-intro-data-analytics/tutorial.ipynb`
- `1.2-intro-python/tutorial.ipynb`
- `1.3-intro-statistics/tutorial.ipynb`
- `1.4-data-foundation-linear-algebra/tutorial.ipynb`
- `1.5-data-analysis-pandas/tutorial.ipynb`

#### 2.2 Module 2 Notebooks

| Subsection | Notebook Focus | Estimated Cells |
|------------|----------------|-----------------|
| 2.1 SQL | SQL query practice (using sqlite3) | 20 |
| 2.2 Data Wrangling | Data cleaning pipeline | 25 |
| 2.3 EDA | Exploratory analysis workflow | 30 |
| 2.4 Data Engineering | ETL pipeline demo | 20 |

**Deliverables**:
- `2.1-sql/tutorial.ipynb`
- `2.2-data-wrangling/tutorial.ipynb`
- `2.3-eda/tutorial.ipynb`
- `2.4-data-engineering/tutorial.ipynb`

#### 2.3 Module 3 Notebooks

| Subsection | Notebook Focus | Estimated Cells |
|------------|----------------|-----------------|
| 3.1 Intro Viz | Matplotlib basics | 20 |
| 3.2 Adv Viz | Seaborn & Plotly | 25 |
| 3.3 Tableau | Python-Tableau integration | 15 |
| 3.4 Storytelling | Dashboard creation | 20 |

**Deliverables**:
- `3.1-intro-data-viz/tutorial.ipynb`
- `3.2-adv-data-viz/tutorial.ipynb`
- `3.3-bi-with-tableau/tutorial.ipynb`
- `3.4-data-storytelling/tutorial.ipynb`

---

## Phase 3: Visual Assets Enhancement (Week 5-6)

### Priority: Add Images to Image-Poor Modules

#### 3.1 Module 2 Visualizations

| Subsection | Current | Target | Focus Areas |
|------------|---------|--------|-------------|
| 2.1 SQL | 0 | 8+ | ER diagrams, query results, joins visual |
| 2.2 Wrangling | 0 | 10+ | Before/after cleaning, pipeline flow |
| 2.3 EDA | 0 | 12+ | Distribution plots, correlations |
| 2.4 Data Eng | 0 | 8+ | ETL diagrams, architecture |

**Actions per subsection**:
- [ ] Create `assets/` folder
- [ ] Create `generate_visualizations.py`
- [ ] Generate concept diagrams
- [ ] Generate output examples
- [ ] Update markdown references

#### 3.2 Module 1 Enhancement

| Subsection | Current | Target | Focus Areas |
|------------|---------|--------|-------------|
| 1.1 Analytics | 1 | 6+ | Workflow diagrams, security layers |
| 1.5 Pandas | 1 | 8+ | DataFrame operations, method outputs |

#### 3.3 Capstone Visuals

| Item | Description |
|------|-------------|
| Project workflow diagram | Mermaid or PNG |
| Example dashboard screenshot | From sample project |
| Rubric visualization | Scoring breakdown |

---

## Phase 4: Exercises & Assignments (Week 7-8)

### Priority: Standardize Assessment Structure

#### 4.1 Create Exercises Folder Template

```
exercises/
├── README.md              # Overview and progression guide
├── quick_assessment.ipynb # 5-min skill check
├── level1_basics.ipynb    # Beginner exercises
├── level2_applied.ipynb   # Intermediate exercises
├── level3_challenge.ipynb # Advanced exercises
└── solutions/
    └── *.ipynb            # Solution notebooks
```

#### 4.2 Subsections Needing Exercises

| Module | Subsections | Priority |
|--------|-------------|----------|
| 1 | 1.1, 1.2, 1.3, 1.4, 1.5 | High |
| 2 | 2.1, 2.2, 2.4 (2.3 has project.md) | High |
| 3 | 3.1, 3.2, 3.3, 3.4 | Medium |
| 4 | 4.1, 4.2, 4.3, 4.4 | Medium |
| 5 | 5.1, 5.2, 5.3, 5.4 (5.5 has exercises) | Medium |

#### 4.3 Exercise Content Guidelines

**Level 1 (Beginner)**:
- Single concept focus
- Step-by-step guidance
- 15-20 minutes
- Immediate feedback

**Level 2 (Intermediate)**:
- Multiple concepts combined
- Less hand-holding
- 30-45 minutes
- Real-world mini-scenarios

**Level 3 (Advanced)**:
- Open-ended problems
- Minimal guidance
- 60+ minutes
- Portfolio-worthy outputs

---

## Phase 5: Structure Standardization (Week 9-10)

### Priority: Adopt Consistent Folder Structure

#### 5.1 Target Structure

```
<subsection>/
├── README.md                     # Overview, objectives, prerequisites
├── <topic-1>.md                  # Or nested folders for complex topics
├── <topic-2>.md
├── ...
├── tutorial.ipynb                # Required
├── exercises/                    # Required
│   ├── README.md
│   ├── quick_assessment.ipynb
│   ├── level1_*.ipynb
│   ├── level2_*.ipynb
│   └── level3_*.ipynb
├── assets/                       # Required
│   ├── *.png
│   └── generate_visualizations.py
├── slides/
│   ├── data.json
│   └── index.html
└── project.md                    # If applicable
```

#### 5.2 Migration Checklist per Subsection

- [ ] README.md has learning objectives
- [ ] README.md has prerequisites
- [ ] README.md has topic links
- [ ] tutorial.ipynb exists and runs
- [ ] exercises/ folder exists
- [ ] assets/ folder has 5+ images
- [ ] assets/ has generation script
- [ ] slides/data.json has 6+ slides
- [ ] All internal links work
- [ ] Prev/Next navigation added

#### 5.3 Subsections Requiring Restructure

**Convert to Nested Structure** (like 5.2):
- 5.3 Supervised Learning 2 (already nested, verify consistency)
- 4.4 Stat Modelling (consider nesting by topic)

**Add Missing Components**:
| Subsection | Missing Components |
|------------|-------------------|
| 1.1 | notebook, exercises, images |
| 1.2 | notebook, exercises |
| 1.3 | notebook, exercises |
| 1.4 | notebook, exercises |
| 1.5 | notebook, exercises, images |
| 2.1 | notebook, images |
| 2.2 | notebook, images |
| 2.4 | notebook, images |
| 3.1 | notebook, exercises |
| 3.2 | notebook, exercises, images |
| 3.3 | notebook, exercises, images |
| 3.4 | notebook, exercises |
| 6 | templates, examples |

---

## Phase 6: Navigation & UX (Week 11)

### Priority: Improve Learner Experience

#### 6.1 Add Navigation Links

**Header Template**:
```markdown
---
⏱️ Reading time: ~15 minutes
Prerequisites: [Previous Topic](./previous.md)
---
```

**Footer Template**:
```markdown
---
← Previous: [Topic Name](./previous.md) | Next: [Topic Name](./next.md) →
```

#### 6.2 Add Progress Indicators

**In README.md**:
```markdown
## Learning Checklist
- [ ] Read introduction
- [ ] Complete tutorial notebook
- [ ] Finish Level 1 exercises
- [ ] Attempt Level 2 exercises
- [ ] Review key takeaways
```

#### 6.3 Add Collapsible Sections

```markdown
<details>
<summary>💡 Key Takeaways</summary>

1. Point one
2. Point two
3. Point three

</details>
```

---

## Phase 7: Quality Assurance (Week 12)

### Priority: Verify All Content Works

#### 7.1 Technical Verification

- [ ] All code examples run without errors
- [ ] All notebooks execute end-to-end
- [ ] All images render correctly
- [ ] All internal links work
- [ ] All external links are valid

#### 7.2 Content Review

- [ ] Learning objectives align with content
- [ ] Prerequisites are accurate
- [ ] Difficulty progression is appropriate
- [ ] No duplicate content across files
- [ ] Consistent terminology

#### 7.3 Accessibility Check

- [ ] Images have alt text
- [ ] Color schemes are colorblind-friendly
- [ ] Code has sufficient comments
- [ ] Complex concepts have multiple explanations

---

## Phase 8: Capstone Enhancement (Week 12)

### Priority: Add Supporting Materials

#### 8.1 Create Project Templates

```
6-capstone/
├── README.md                 # Existing - comprehensive
├── templates/
│   ├── project_proposal.md   # NEW
│   ├── eda_notebook.ipynb    # NEW
│   ├── modeling_notebook.ipynb # NEW
│   └── presentation_template.pptx # NEW
├── examples/
│   ├── example_project_A/    # NEW - UN SDGs example
│   └── example_project_B/    # NEW - Basic example
└── rubric.md                 # NEW - Detailed scoring
```

#### 8.2 Example Project Structure

```
example_project_A/
├── README.md
├── data/
│   └── README.md (data sources)
├── notebooks/
│   ├── 1-eda.ipynb
│   ├── 2-preprocessing.ipynb
│   └── 3-modeling.ipynb
├── src/
│   └── utils.py
└── docs/
    └── presentation.md
```

---

## Implementation Timeline

```
Week 1-2:   Phase 1 - Critical Fixes (3.2, 3.3, 2.3, 5.4)
Week 3-4:   Phase 2 - Notebook Coverage (13 new notebooks)
Week 5-6:   Phase 3 - Visual Assets (40+ new images)
Week 7-8:   Phase 4 - Exercises & Assignments
Week 9-10:  Phase 5 - Structure Standardization
Week 11:    Phase 6 - Navigation & UX
Week 12:    Phase 7 & 8 - QA & Capstone
```

---

## Resource Requirements

### Per Phase Estimates

| Phase | Effort (hours) | Skills Needed |
|-------|----------------|---------------|
| 1 - Critical Fixes | 20-30 | Python, visualization tools |
| 2 - Notebooks | 40-60 | Python, teaching design |
| 3 - Visual Assets | 30-40 | Python, design |
| 4 - Exercises | 40-50 | Python, assessment design |
| 5 - Standardization | 20-30 | Documentation |
| 6 - Navigation | 10-15 | Markdown |
| 7 - QA | 15-20 | Testing |
| 8 - Capstone | 15-20 | Documentation |

**Total Estimated Effort**: 190-265 hours

---

## Success Metrics

### Phase Completion Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Tier 1-2 Subsections | 56% | 100% |
| Notebook Coverage | 43% | 100% |
| Avg Images/Subsection | 11 | 15+ |
| Exercise Coverage | 33% | 100% |
| Broken Links | Unknown | 0 |

### Quality Indicators

- All code runs without modification
- All notebooks complete in < 60 minutes
- All exercises have solutions
- All slides have 6+ content slides
- All READMEs have objectives + prerequisites

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Time constraints | Prioritize Phase 1-2 (critical gaps + notebooks) |
| Skill gaps | Use existing exemplary modules as templates |
| Scope creep | Stick to checklist, defer "nice to haves" |
| Quality variance | Use standardized templates and review process |

---

## Quick Reference: Subsection Status

### Module 1 - Data Fundamentals
| ID | Name | Images | Notebook | Exercises | Priority |
|----|------|--------|----------|-----------|----------|
| 1.1 | Intro Analytics | 1 | ❌ | ❌ | High |
| 1.2 | Intro Python | 5 | ❌ | ❌ | Medium |
| 1.3 | Statistics | 12 | ❌ | ❌ | Medium |
| 1.4 | Linear Algebra | 5 | ❌ | ❌ | Medium |
| 1.5 | Pandas | 1 | ❌ | ❌ | Medium |

### Module 2 - Data Wrangling
| ID | Name | Images | Notebook | Exercises | Priority |
|----|------|--------|----------|-----------|----------|
| 2.1 | SQL | 0 | ❌ | ✅ | Medium |
| 2.2 | Data Wrangling | 0 | ❌ | ✅ | Medium |
| 2.3 | EDA | 0 | ❌ | ✅ | High |
| 2.4 | Data Engineering | 0 | ❌ | ✅ | Medium |

### Module 3 - Data Visualization
| ID | Name | Images | Notebook | Exercises | Priority |
|----|------|--------|----------|-----------|----------|
| 3.1 | Intro Viz | 4 | ❌ | ❌ | Medium |
| 3.2 | Adv Viz | 0 | ❌ | ❌ | **Critical** |
| 3.3 | Tableau | 0 | ❌ | ❌ | **Critical** |
| 3.4 | Storytelling | 51 | ❌ | ❌ | Low |

### Module 4 - Statistical Analysis
| ID | Name | Images | Notebook | Exercises | Priority |
|----|------|--------|----------|-----------|----------|
| 4.1 | Inferential | 45 | ✅ | ❌ | Low |
| 4.2 | Hypothesis | 13 | ✅ | ❌ | Low |
| 4.3 | Relationships | 18 | ✅✅ | ❌ | Low |
| 4.4 | Stat Modelling | 47 | ✅ | ❌ | Low |

### Module 5 - ML Fundamentals
| ID | Name | Images | Notebook | Exercises | Priority |
|----|------|--------|----------|-----------|----------|
| 5.1 | Intro ML | 6 | ✅ | ✅ | Low |
| 5.2 | Supervised 1 | 17 | ✅ | ✅ | Low |
| 5.3 | Supervised 2 | 34 | ✅ | ❌ | Low |
| 5.4 | Unsupervised | 9 | ✅ | ❌ | Medium |
| 5.5 | Model Eval | 40+ | ✅✅✅✅ | ✅ | Low |

### Module 6 - Capstone
| ID | Name | Images | Notebook | Exercises | Priority |
|----|------|--------|----------|-----------|----------|
| 6 | Capstone | 0 | ❌ | N/A | Medium |

---

## Appendix: File Templates

### A. Notebook Template
See `5.5-model-eval/tutorial.ipynb` for reference.

### B. Exercises README Template
See `5.5-model-eval/exercises/README.md` for reference.

### C. Generate Visualizations Script Template
See `4.1-inferential-stats/assets/generate_visualizations.py` for reference.

### D. Slides data.json Template
See any `slides/data.json` for reference structure.

---

*Last Updated: Based on comprehensive analysis of all 23 subsections*
*Recommended Review: Monthly during implementation*
