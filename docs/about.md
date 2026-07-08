---
layout: default
title: About the Instructors
lesson_nav: false
---

<style>
.about-page {
  max-width: 780px;
  margin: 0 auto;
}

.instructor-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin: 2rem 0;
}

.instructor-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.75rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.instructor-card__header {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.instructor-card__avatar {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--link), #8250df);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  font-weight: 700;
  color: #fff;
  flex-shrink: 0;
}

.instructor-card__name {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text);
  margin: 0;
}

.instructor-card__role {
  font-size: 0.875rem;
  color: var(--text-muted);
  margin: 0;
}

.instructor-card__bio {
  font-size: 0.9375rem;
  line-height: 1.65;
  color: var(--text);
  margin: 0;
}

.instructor-card__tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-top: 0.25rem;
}

.instructor-card__tag {
  font-size: 0.775rem;
  font-weight: 500;
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  background: var(--code-bg);
  color: var(--text-muted);
  border: 1px solid var(--border);
}

.instructor-card__links {
  display: flex;
  gap: 0.75rem;
  flex-wrap: wrap;
  margin-top: auto;
  padding-top: 0.5rem;
  border-top: 1px solid var(--border);
}

.instructor-card__link {
  font-size: 0.875rem;
  font-weight: 500;
  color: var(--link);
  text-decoration: none;
}

.instructor-card__link:hover {
  text-decoration: underline;
}
</style>

<div class="about-page">

# About the Instructors

This course is taught by two practitioners who bring both industry depth and teaching experience to Data Science and AI.

<div class="instructor-grid">

<div class="instructor-card">
  <div class="instructor-card__header">
    <div class="instructor-card__avatar">SW</div>
    <div>
      <p class="instructor-card__name">Shen Nan, Wong (Shane)</p>
      <p class="instructor-card__role">Engineer · Educator · Builder</p>
    </div>
  </div>
  <p class="instructor-card__bio">
    Shennan is a software engineer and technical educator currently based in Ho Chi Minh City. He works as the sole engineer at <strong>Iterative</strong>, an early-stage VC fund across Southeast and South Asia, where he builds data infrastructure, internal tooling, and AI systems for investment operations - including an AI-powered deal-screening pipeline and a Voice AI system for founder outreach.
  </p>
  <p class="instructor-card__bio">
    He is also Founder and Technical Consultant at <strong>Fracxional</strong>, where he teaches AI and data engineering across Asian universities and corporate programs. Previous roles include Co-founder &amp; CTO of ZOLO (AI food supply chain, ~$350K pre-seed), Senior Software Engineer at Circles.Life, Senior DevSecOps Engineer at Partior, and Software Engineering Instructor at Rocket Academy.
  </p>
  <div class="instructor-card__tags">
    <span class="instructor-card__tag">Python</span>
    <span class="instructor-card__tag">Data Engineering</span>
    <span class="instructor-card__tag">Machine Learning</span>
    <span class="instructor-card__tag">AI Systems</span>
    <span class="instructor-card__tag">Next.js</span>
    <span class="instructor-card__tag">PostgreSQL</span>
  </div>
  <div class="instructor-card__links">
    <a class="instructor-card__link" href="https://byshennan.com/about" target="_blank" rel="noopener">Website ↗</a>
    <a class="instructor-card__link" href="https://github.com/codebyshennan" target="_blank" rel="noopener">GitHub ↗</a>
  </div>
</div>

<div class="instructor-card">
  <div class="instructor-card__header">
    <div class="instructor-card__avatar" style="background: linear-gradient(135deg, #1a7f37, #0969da)">ME</div>
    <div>
      <p class="instructor-card__name">Mariam Elmasry</p>
      <p class="instructor-card__role">Data Scientist · MSc · Educator</p>
    </div>
  </div>
  <p class="instructor-card__bio">
    Mariam is a data scientist with 6+ years of industry experience and a strong track record in both research and teaching. She holds an <strong>MSc in Advanced Analytics from NC State University</strong> and is a Certified Scrum Master.
  </p>
  <p class="instructor-card__bio">
    Her industry career spans the <strong>global R&amp;D team at Ericsson</strong> (Data Science &amp; Cloud Software Hub, Egypt) and <strong>Visa</strong>. On the education side, she has taught Data Science and AI bootcamps at <strong>Ironhack</strong> and advanced Python courses at the <strong>American University in Cairo</strong>, specialising in AI productization, data visualization, and applied analytics.
  </p>
  <div class="instructor-card__tags">
    <span class="instructor-card__tag">Python</span>
    <span class="instructor-card__tag">Advanced Analytics</span>
    <span class="instructor-card__tag">AI Productization</span>
    <span class="instructor-card__tag">Data Visualization</span>
    <span class="instructor-card__tag">Scrum</span>
  </div>
  <div class="instructor-card__links">
    <a class="instructor-card__link" href="https://www.linkedin.com/in/mariam-elmasry-/" target="_blank" rel="noopener">LinkedIn ↗</a>
  </div>
</div>

</div>

</div>
