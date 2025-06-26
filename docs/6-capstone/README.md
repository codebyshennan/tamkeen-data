# Data Science & AI Capstone Project

## Project Overview
This capstone project is designed to demonstrate industry-relevant data science and AI skills acquired throughout the course. Students will work on real-world datasets to solve practical problems using various data science techniques.

## Timeline
- Duration: 2 weeks (part-time)
- Deliverable: 5-minute video presentation

## Structured Project Options

Students can choose from three structured project briefs or propose their own project following the general requirements below.

### Example Project A: UN Sustainable Development Goals (SDGs) Data Analysis Pipeline

**Project Overview**
Develop a comprehensive data science pipeline to analyze progress towards specific UN Sustainable Development Goals using official UN datasets. Create visualizations and predictive models to assess global or regional SDG performance.

**Learning Objectives**
- Integrate data from multiple sources using APIs and direct downloads
- Apply data preprocessing and feature engineering techniques
- Create meaningful visualizations for policy insights
- Build predictive models for SDG progress forecasting

**Datasets & Resources**
1. **UN Data Commons for SDGs**: https://unstats.un.org/sdgs
2. **Global SDG Indicators Database**: https://unstats.un.org/sdgs/indicators/database/
3. **UN Statistics Division API**: Access 210+ SDG indicators for 193+ countries
4. **World Bank Open Data**: https://data.worldbank.org/ (complementary socioeconomic data)

**Implementation Guidelines**

*Week 1: Data Collection & Processing*
- Day 1-2: Explore UN SDG APIs and select 2-3 focused goals (e.g., SDG 1: No Poverty, SDG 3: Good Health)
- Day 3-4: Extract data using API calls or direct downloads
- Day 5-7: Data cleaning, preprocessing, and feature engineering

*Week 2: Analysis & Modeling*
- Day 8-10: Exploratory data analysis and visualization creation
- Day 11-12: Build predictive models (regression/classification)
- Day 13-14: Final presentation and documentation

**Technical Requirements**
```python
# Key Libraries
- pandas, numpy for data manipulation
- requests for API calls
- matplotlib, seaborn, plotly for visualizations
- scikit-learn for modeling
- jupyter notebooks for documentation
```

**Deliverables**
1. Jupyter notebook with complete pipeline
2. Interactive dashboard (using Plotly/Streamlit)
3. 5-minute presentation with policy recommendations

### Example Project B: Bahrain Vision 2030 Economic Development Analysis

**Project Overview**
Analyze Bahrain's economic development indicators to assess progress towards Vision 2030 goals. Work with official government data to create insights supporting policy decisions.

**Learning Objectives**
- Work with government open data portals
- Analyze economic trends and patterns
- Create policy-relevant visualizations
- Build forecasting models for economic indicators

**Datasets & Resources**
1. **Bahrain Open Data Portal**: https://www.data.gov.bh
   - General Economic Indicators: https://www.data.gov.bh/explore/dataset/01-annually-general-economic-indicators-by-cp/
   - Economic indicators for agricultural sector: https://www.data.gov.bh/explore/dataset/economic-indicators-for-the-agricultural-sector/
2. **National Summary Data Page**: https://www.data.gov.bh/pages/national-summary-data-page-nsdp/
3. **Ministry of Finance Reports**: Economic Quarterly reports for context
4. **World Bank Data on Bahrain**: For comparative analysis

**Key Indicators to Analyze**
- GDP growth and composition
- Private sector contribution
- Employment rates and sectoral distribution
- Trade statistics and diversification metrics
- Infrastructure development indicators

**Implementation Guidelines**

*Week 1: Data Acquisition & Exploration*
- Day 1-2: Register and explore data.gov.bh portal
- Day 3-4: Download and consolidate relevant datasets
- Day 5-7: Data cleaning and trend analysis preparation

*Week 2: Analysis & Insights*
- Day 8-9: Time series analysis and trend identification
- Day 10-11: Comparative analysis with regional benchmarks
- Day 12-13: Predictive modeling for 2030 targets
- Day 14: Final visualization dashboard and report

**Technical Requirements**
```python
# Key Libraries
- pandas, numpy for data manipulation
- plotly, matplotlib for economic visualizations
- statsmodels for time series analysis
- scikit-learn for predictive modeling
- streamlit for dashboard creation
```

**Deliverables**
1. Economic trends analysis report
2. Interactive dashboard showing Vision 2030 progress
3. Policy recommendations presentation

### Example Project C: Web Scraping to Data Pipeline Implementation

**Project Overview**
Build an end-to-end data pipeline that scrapes data from public websites/APIs, processes it, and creates actionable insights through machine learning models.

**Suggested Project Options**

*Option 1: Real Estate Market Analysis*
- **Data Source**: Scrape property listings from public real estate websites
- **Focus**: Price prediction and market trend analysis

*Option 2: Financial Markets Dashboard*
- **Data Source**: Yahoo Finance API, public financial data
- **Focus**: Stock performance analysis and prediction

*Option 3: Weather & Climate Analysis*
- **Data Source**: OpenWeatherMap API, climate data APIs
- **Focus**: Weather pattern analysis and forecasting

**Implementation Framework**

*Week 1: Data Collection Pipeline*
```python
# Core scraping workflow
import requests
import beautifulsoup4
import pandas as pd
import time

# Sample structure for API/scraping
def collect_data():
    # API calls or web scraping
    # Data validation and cleaning
    # Storage in structured format
    pass
```

*Week 2: Analysis & Modeling*
```python
# Analysis pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# EDA, feature engineering, modeling
```

**Ethical Guidelines**
- Respect robots.txt files
- Implement appropriate delays between requests
- Use only publicly available data
- Follow terms of service for APIs

**Technical Requirements**
- **Web Scraping**: BeautifulSoup4, requests, selenium (if needed)
- **APIs**: requests, json handling
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Modeling**: scikit-learn
- **Deployment**: streamlit or flask for demo

**Deliverables**
1. Complete data pipeline with documentation
2. Machine learning model with performance metrics
3. Web application or dashboard demonstrating insights

## General Project Requirements (For All Options)

### Core Project Requirements
1. Data Collection & Preparation
   - Source and clean dataset(s)
   - Document data quality issues and solutions
   - Implement proper data preprocessing techniques

2. Exploratory Data Analysis
   - Conduct thorough statistical analysis
   - Create meaningful visualizations
   - Extract key insights from the data

3. Modeling & Analysis
   - Apply appropriate machine learning techniques
   - Evaluate model performance
   - Document model selection rationale

4. Results & Insights
   - Present clear findings and recommendations
   - Support conclusions with data
   - Discuss practical implications

5. Technical Implementation
   - Use industry-standard tools (Python, pandas, scikit-learn, etc.)
   - Follow best practices for code organization
   - Include proper documentation

## Recommended Development Environments
1. Google Colab
   - Free GPU access
   - Pre-installed data science libraries
   - Easy sharing and collaboration

2. Deepnote
   - Real-time collaboration
   - Integrated version control
   - Rich markdown support

## Recommended Datasets from Kaggle

### Healthcare & Life Sciences
1. [COVID-19 Dataset](https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset)
   - Time series analysis
   - Geographical visualization
   - Predictive modeling opportunities

2. [Healthcare Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set)
   - Binary classification
   - Feature importance analysis
   - Medical diagnostic modeling

### Business & Finance
1. [E-commerce Customer Behavior](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
   - Customer segmentation
   - Purchase prediction
   - Time series analysis

2. [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Anomaly detection
   - Imbalanced classification
   - Risk modeling

### Environmental & Climate
1. [Global Temperature Time Series](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
   - Time series analysis
   - Trend prediction
   - Visualization challenges

2. [Air Quality Data](https://www.kaggle.com/datasets/fedesoriano/air-quality-data-set)
   - Multivariate analysis
   - Sensor data processing
   - Environmental impact assessment

### Technology & Social Media
1. [Twitter Sentiment Analysis](https://www.kaggle.com/datasets/kazanova/sentiment140)
   - Natural Language Processing
   - Sentiment classification
   - Text preprocessing

2. [Stack Overflow Questions](https://www.kaggle.com/datasets/stackoverflow/stackoverflow)
   - Text classification
   - Tag prediction
   - Trend analysis

### Urban & Transportation
1. [NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)
   - Regression analysis
   - Geospatial visualization
   - Feature engineering

2. [Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand)
   - Demand forecasting
   - Time series analysis
   - Weather impact analysis

## Additional Resources for Success

### Technical Resources
1. **Documentation & Tutorials**
   - [Pandas Documentation](https://pandas.pydata.org/docs/)
   - [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
   - [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
   - [Plotly Python Documentation](https://plotly.com/python/)

2. **API Integration Guides**
   - [Requests Library Documentation](https://docs.python-requests.org/en/latest/)
   - [Working with JSON in Python](https://realpython.com/python-json/)
   - [REST API Best Practices](https://restfulapi.net/)

3. **Data Visualization Resources**
   - [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
   - [Plotly Express Guide](https://plotly.com/python/plotly-express/)
   - [Data Visualization Best Practices](https://www.tableau.com/learn/articles/data-visualization)

4. **Machine Learning Resources**
   - [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)
   - [Cross-validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
   - [Model Selection and Evaluation](https://scikit-learn.org/stable/model_selection.html)

### Project Management Tools
1. **Version Control**
   - [Git Basics Tutorial](https://git-scm.com/docs/gittutorial)
   - [GitHub Desktop](https://desktop.github.com/) for GUI-based Git management
   - [Conventional Commits](https://www.conventionalcommits.org/) for commit message standards

2. **Documentation Tools**
   - [Jupyter Notebook Best Practices](https://jupyter-notebook.readthedocs.io/en/stable/)
   - [Markdown Guide](https://www.markdownguide.org/)
   - [README Template](https://github.com/othneildrew/Best-README-Template)

### Presentation Resources
1. **Video Creation Tools**
   - [OBS Studio](https://obsproject.com/) for screen recording
   - [Loom](https://www.loom.com/) for quick video creation
   - [Canva](https://www.canva.com/) for presentation slides

2. **Presentation Tips**
   - Keep technical explanations accessible to non-technical audiences
   - Use visual aids to support your narrative
   - Practice timing to stay within the 5-minute limit
   - Prepare for potential questions about methodology and results

## Common Pitfalls to Avoid

1. **Data Issues**
   - Not exploring data thoroughly before modeling
   - Ignoring missing values or outliers
   - Using inappropriate data splits
   - Data leakage in feature engineering

2. **Modeling Mistakes**
   - Not establishing baseline models
   - Overfitting without proper validation
   - Inappropriate metrics for the problem type
   - Not interpreting model results

3. **Presentation Problems**
   - Exceeding time limits
   - Too much technical jargon
   - Poor visual design
   - Not telling a coherent story

4. **Documentation Issues**
   - Insufficient code comments
   - Missing methodology explanations
   - No discussion of limitations
   - Unclear repository structure

## Project Structure
```
project/
├── data/               # Raw and processed data
├── notebooks/          # Jupyter notebooks
├── src/               # Source code
├── docs/              # Documentation
└── README.md          # Project documentation
```

## Evaluation Criteria
1. Technical Depth (40%)
   - Appropriate use of data science techniques
   - Code quality and implementation
   - Model performance and validation

2. Business Understanding (30%)
   - Problem formulation
   - Solution approach
   - Business value of insights

3. Communication (30%)
   - Video presentation clarity
   - Visualization quality
   - Documentation completeness

## Submission Guidelines

### GitHub Repository
1. Create a new public GitHub repository with the following structure:
```
capstone-project/
├── data/               # Data files or links to datasets
├── notebooks/          # Jupyter notebooks
│   ├── 1-eda.ipynb    # Exploratory Data Analysis
│   ├── 2-preprocessing.ipynb  # Data preprocessing
│   └── 3-modeling.ipynb      # Model development
├── src/               # Source code (if any)
├── docs/              # Documentation
│   └── presentation.md  # Presentation script/notes
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

2. README.md should include:
   - Project title and description
   - Problem statement
   - Dataset description and source
   - Methodology overview
   - Key findings and insights
   - Installation/setup instructions
   - Usage instructions

3. Code Requirements:
   - Well-documented notebooks with markdown explanations
   - Clean, commented code following PEP 8 style guide
   - Requirements.txt file with all dependencies
   - Data preprocessing and feature engineering steps
   - Model training and evaluation code
   - Visualization code

### Video Presentation
1. Problem Introduction (1 minute)
   - Context and motivation
   - Problem statement
   - Expected impact

2. Technical Approach (2 minutes)
   - Data processing methods
   - Analysis techniques
   - Model development

3. Results & Insights (1.5 minutes)
   - Key findings
   - Model performance
   - Business recommendations

4. Conclusion (0.5 minutes)
   - Summary of achievements
   - Future improvements
   - Lessons learned

### Submission Process
1. GitHub Repository:
   - Ensure all code and documentation is committed
   - Make the repository public
   - Submit the repository URL via the course platform

2. Video Presentation:
   - Upload to a video platform (YouTube, Vimeo, etc.)
   - Make the video unlisted or public
   - Submit the video URL via the course platform

3. Final Checks:
   - Repository is public and accessible
   - All code runs without errors
   - Documentation is complete and clear
   - Video is accessible and plays correctly

## Frequently Asked Questions

### Q: What if I can't access certain datasets?
A: If you encounter access issues with specific datasets, document the problem and use alternative sources. The Kaggle datasets section provides reliable alternatives.

### Q: Can I use pre-trained models?
A: Yes, but you must demonstrate understanding of the model, properly cite sources, and show how you adapted it to your specific problem.

### Q: What if my model performance is poor?
A: Focus on demonstrating proper methodology, data understanding, and clear communication of results. Document what you tried and why certain approaches didn't work - this shows critical thinking.

### Q: How technical should my presentation be?
A: Aim for a balance - explain your approach clearly but make it accessible to a general business audience. Use visualizations to support technical concepts.

### Q: Can I propose my own project idea?
A: Yes! If you have a specific domain interest or dataset in mind, you can propose your own project as long as it meets all the general requirements outlined above.

### Q: What programming languages can I use?
A: Python is strongly recommended as it's the primary language covered in the course. R is acceptable if you have strong justification for its use in your specific domain.

### Q: How do I handle large datasets that won't fit in memory?
A: Document the issue and use sampling techniques, data chunking, or cloud computing resources. Explain your approach and any limitations this introduces.

### Q: What if I finish early?
A: Use extra time to improve your analysis, add more sophisticated models, create better visualizations, or extend your analysis to additional research questions.

### Q: How important is the business context?
A: Very important! This project should demonstrate your ability to apply data science to solve real-world problems, not just technical skills in isolation.

## Getting Help

### During Development
- Use course discussion forums for technical questions
- Attend office hours for guidance on approach and methodology
- Consult documentation and online resources for implementation details

### Before Submission
- Review the assessment rubric to ensure you've addressed all criteria
- Test your code in a fresh environment to ensure reproducibility
- Have someone review your presentation for clarity and timing
- Double-check all submission requirements

Remember: This capstone project is your opportunity to showcase the skills you've developed throughout the course. Focus on demonstrating both technical competency and business acumen through a well-executed, clearly communicated data science project.

---

**Good luck with your capstone project!**
