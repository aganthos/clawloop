"""
Aganthos company knowledge base — used to generate tailored intake form answers.
"""

COMPANY_PROFILE = """
## AGANTHOS — Company Profile

**Tagline:** We enable agents to learn from experience.
**Website:** aganthos.com | info@aganthos.com
**Founded:** Q4 2025 (DE C-Corp) + Q1 2026 (Swiss GmbH, HQ)
**Stage:** Early market / first paying customers

---

### ONE-LINER
Aganthos provides the "learning layer" for AI agents — using Reinforcement Learning (RL) to enable agents to continuously improve from their own operational experience, increasing success rates, reducing costs, and ensuring data sovereignty.

### ELEVATOR PITCH (2 min)
Current AI agents are like employing a new intern every day. They don't learn from mistakes. They behave identically on day 100 as on day 1. Off-the-shelf frontier models achieve 50–60% on complex domain tasks — enterprises need 90%+.

Aganthos is the training program that fixes this. We feed back the errors and traces of an AI agent on day one into the improvement of that agent on day two — and so forth. This turns AI agents into experienced employees that become better over time, adapting to the specific tools, workflows, and constraints of each organisation.

We call this "Learning as a Service." Our platform wraps around any existing agent and adds a continuous RL improvement loop: Deploy & Collect (logs, traces) → Train Agent (adapt weights and harness) → improved model → repeat.

**Key results from our first paid pilot (large German University Hospital):**
- +30% performance improvement over o4-mini (state-of-the-art frontier model) on clinical benchmarks
- 60% cost reduction vs. frontier APIs via routing to our custom 7B parameter model (Qwen 2.5)
- Full data sovereignty: all data stays within the hospital's infrastructure
- Potential to save 5–15 FTE equivalents (>€1M per year)
- Paper submitted to ICML 2026

---

### THE PROBLEM (3 pain points)

1. **Unreliable Performance:** Off-the-shelf models achieve 50–60% on domain-specific tasks. Enterprises need 90%+. There is no scalable way to bridge this gap without custom training.

2. **Unsustainable Costs:** Frontier API costs (GPT-4o, Claude, Gemini) scale linearly with usage. At production volume, inference becomes the largest line item. Many deployments are economically unviable.

3. **Data Sovereignty:** Sensitive enterprise data — patient records, financial data, classified information — cannot leave the organisation. Vendor lock-in and regulatory compliance (GDPR, HIPAA, NIS2) are existential risks for many sectors.

---

### OUR SOLUTION: THE LEARNING FLYWHEEL

**Onboarding phase:**
1. Analyse: understand the organisation's existing workflows and agent setup
2. Setup: configure agents (model + harness: prompt, tools, memory)

**Continuous improvement phase (the core):**
3. Deploy & Collect: run in production, collect logs and traces
4. Train Agent: use RL to adapt model weights and harness based on real outcomes
→ Continuously improved model → repeat

This is not prompt engineering or RAG. This is genuine Reinforcement Learning post-training that improves the underlying model on the customer's specific tasks.

---

### MEASURED VALUE

| Dimension | Result | Mechanism |
|-----------|--------|-----------|
| Performance | +30% over o4-mini | Learning from Experience (RL) |
| Cost | -60% vs. frontier APIs | Routing + small trained models (7B) |
| Privacy | Data never leaves org | Knowledge transfer + local deployment |
| ROI | 5–15 FTE saved (>€1M/yr) | In our hospital case study |

---

### TARGET SECTORS & USE CASES

**Primary sectors (high-stakes, high action-volume):**
- **Healthcare:** Clinical documentation, QA agents, patient data queries (FHIR), coding, prior authorisation, hospital workflows
- **Financial Services / Banking:** Compliance monitoring, loan underwriting, fraud detection, regulatory reporting
- **Insurance:** Claims processing, underwriting, fraud detection, policy interpretation
- **Defence / National Security:** Software-defined operations, intelligence workflows, autonomous decision support

**Target customer profile:**
- Running multi-step, tool-calling AI agents in production
- Experiencing a "reliability wall" (agents not meeting required accuracy)
- High cost of failure OR high data sensitivity
- Want to reduce inference costs at scale

**Example use cases we can address:**
- Hospital QA agent (proven ✓)
- Document processing and extraction at scale
- Regulatory compliance checking
- Autonomous customer service agents
- Code review / software QA agents
- Intelligence analysis and summarisation
- Supply chain decision support

---

### DIFFERENTIATION

**Vs. Static Agent Platforms (N8N, Zapier, Sema4.ai, Lindy AI):**
They automate but do not learn. Agents are frozen after deployment. Aganthos adds the learning layer on top of any existing agent.

**Vs. Legacy BPO + LLM (Accenture, Deloitte, UiPath):**
They integrate AI but do not continuously improve it. They scale with headcount, not intelligence. Aganthos automates the improvement loop.

**Vs. Bespoke Intelligence (Palantir, InstaDeep, Applied Intuition):**
They build custom solutions but not scalable platforms. High implementation cost, slow iteration.

**Aganthos' unique position:** The only scalable "Learning Automation" platform — combining the intelligence of bespoke RL solutions with the scalability of SaaS. Our core IP is operationalising RL for tool-calling agents in enterprise environments.

**Comparable validated approaches:** OpenAI trains its models with RL (RLHF, o-series). We bring the same methodology to *your* domain-specific agent, running on *your* infrastructure.

---

### TRACTION & MILESTONES

**Q4 2025:**
- NeurIPS 2025 workshop paper (scientific validation)
- DE C-Corp incorporation
- 1st paid partnership secured
- Trained proprietary 1.5B/3B parameter models for tool orchestration
- R&D collaboration with first customer (large German University Hospital) and international researchers

**Q1 2026:**
- ~€28K revenue from completed projects (€3.3K + €15K + €10K milestone-based)
- RL Post-Trained Medical Model: +30% over o4-mini on clinical benchmarks
- Accepted into NVIDIA Inception program
- Swiss GmbH incorporation (HQ)
- Two ICML 2026 submissions
- Design partnership discussions with multinational companies

**Current pipeline:** 2–3 active design partnership discussions for Q2 2026

---

### TEAM

**Robert Mueller — Co-Founder (AI Research → Impact)**
- Decade of Reinforcement Learning research; AISTATS Best Paper nominee 2022
- Theory to production: games, high-speed robotics, electron microscopes, web agents
- Founding Research Scientist at Convergence AI (acquired by Salesforce, 2025)
- Background: TUM, Carnegie Mellon, Mila (Montreal AI Institute), Sony AI
- Full-time on Aganthos

**Tobias Schuster — Co-Founder (Operations & Strategy)**
- Designed and implemented a $3.3B national-scale COVID crisis management operation
- Built a 20–30 person team from scratch in <3 months within Swiss Federal Health Ministry (FOPH)
- Changed federal laws and ordinances in days (vs. years standard process)
- Generative AI in Public Health, Medicine & Climate
- Background: Switzerland (FOPH), London School of Hygiene & Tropical Medicine
- Transitioning to full-time (currently part-time, collaborating with major AI lab in London/Zurich)

**Technical Staff (Founding Engineer)**
- Data platforms on AWS/GCP at BMW Group & Levi Strauss
- Cloud deployment of custom RAG agents
- Background: TUM

---

### BUSINESS MODEL

**Revenue streams:**
1. Hourly setup fees (onboarding, workflow analysis, agent configuration)
2. Continuous improvement retainer / maintenance fees
3. Usage-based tiers for trained model inference

**Current pricing:** Design partnerships at €10K–€20K for initial pilots; scaling to larger contracts as value is demonstrated

**Unit economics:**
- GPU-compute driven (not headcount-driven)
- Learning loops scale linearly with compute, not human labor
- "Docking station" strategy: partner with existing agent platforms (B2B2B) for distribution at scale

---

### REGULATORY & COMPLIANCE

- GDPR-compliant (local deployment, no data egress)
- Supports on-premise / air-gapped deployment
- Healthcare: compatible with clinical data standards (FHIR)
- Defence: EWR-based entity (Swiss GmbH), capable of handling classified adjacent workflows
- No proprietary data is used for general model training

---

### KEY NUMBERS TO REFERENCE

- +30% performance over o4-mini (frontier model) on clinical tasks
- 60% cost reduction vs. frontier APIs
- 5–15 FTE equivalents saved per hospital deployment (>€1M/year potential)
- €28K revenue in first months of operation
- 7B parameter custom model (vs. 100B+ frontier)
- ICML 2026 — 2 paper submissions (peer-reviewed scientific credibility)
- NVIDIA Inception program member
"""

# Short versions for forms with character limits
ONE_LINER = "Aganthos enables AI agents to learn from experience using Reinforcement Learning, delivering +30% performance gains over frontier models, 60% cost reduction, and full data sovereignty."

PITCH_SHORT = """Aganthos is a "Learning as a Service" platform that enables AI agents to continuously improve through Reinforcement Learning. Unlike static automation platforms, our agents learn from every interaction — logs, traces, and outcomes — adapting to your specific workflows and constraints. In our first paid pilot with a large German University Hospital, we achieved +30% performance over o4-mini, 60% cost reduction, and full data sovereignty with a custom 7B model deployed on-premise. We solve the three fundamental blockers of enterprise AI at scale: unreliable performance, unsustainable costs, and data sovereignty constraints."""

DIFFERENTIATION_SHORT = """Our unique approach applies Reinforcement Learning post-training to any existing agent — not prompt engineering or fine-tuning on static data, but genuine RL from operational experience. This means agents continuously improve in production, compounding gains over time. No other scalable platform does this. Comparable methods power OpenAI's o-series models; we bring this to your domain-specific workflows running on your infrastructure."""

# Sector-specific value propositions
SECTOR_PITCHES = {
    "healthcare": """
In healthcare, our RL-powered agents address clinical documentation, QA workflows, and patient data management. In our flagship pilot with a large German University Hospital, we deployed a custom 7B-parameter model that achieved +30% performance over o4-mini on clinical benchmarks, 60% cost reduction, and full GDPR-compliant local deployment. The system integrates with FHIR servers and has the potential to save 5–15 FTE equivalents (>€1M/year) per hospital. Our ICML 2026 submission documents the methodology and results.
    """,
    "defense": """
For defence and national security applications, Aganthos provides RL-trained agents for software-defined operations, intelligence analysis, and autonomous decision support workflows. Our technology enables on-premise, air-gapped deployment — data never leaves the classified environment. We are a European entity (Swiss GmbH + DE C-Corp) and can work within EWR compliance requirements. Our Reinforcement Learning approach is particularly suited to dual-use scenarios where agents must adapt to rapidly changing operational constraints.
    """,
    "finance": """
In financial services, our RL agents tackle compliance monitoring, regulatory reporting, underwriting, and fraud detection. These workflows are ideal for RL post-training: they have high action volume, clear reward signals (correct/incorrect determinations), and massive cost-of-failure. Our 60% cost reduction vs. frontier APIs is particularly compelling at the transaction volumes typical in banking and insurance.
    """,
    "manufacturing": """
For manufacturing and industrial operations, Aganthos enables agents that learn from production-floor data — quality control, supply chain decisions, and process optimisation. Our platform adapts to proprietary tooling and APIs, with local deployment ensuring competitive IP never leaves the facility. The learning flywheel turns every production run into training data for more reliable autonomous decisions.
    """
}
