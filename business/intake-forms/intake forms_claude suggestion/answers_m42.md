# Intake Form Answers: M42 Digital Transformation Collaboration

---

**Q1: Your name**
Tobias Schuster

**Q2: Job title**
Co-Founder

**Q3: Company name**
Aganthos

**Q4: Phone number**
[Your number]

**Q5: Email address**
info@aganthos.com

**Q6: In what year was your organization founded?**
01/01/2025

---

**Q7: How would you describe your company's stage of maturity and experience?**

Aganthos is at the early market stage. Founded in Q4 2025, we have already secured paying partnerships and completed initial revenue-generating projects. In Q1 2026 we generated ~€28K in revenue across three milestones with a large German University Hospital — our first paid partner. In that engagement, we validated our core technology in a demanding production environment, achieving +30% accuracy over o4-mini (a state-of-the-art frontier model) on clinical benchmarks, while reducing inference costs by 60% and maintaining full on-premise data sovereignty.

We have been accepted into NVIDIA's Inception program and have submitted two papers to ICML 2026 documenting our methodology and results. We are now in active discussions for 2–3 additional design partnerships in Q2 2026, with a focus on healthcare and financial services. Our Swiss GmbH (HQ) and DE C-Corp are both operational, giving us a stable European legal structure for enterprise engagements.

---

**Q8: Briefly describe your proposal of collaboration.**

We propose a paid design partnership to deploy Aganthos' "Learning as a Service" platform within M42's clinical operations. The goal: take a high-value AI agent workflow — clinical documentation, patient data QA, care coordination routing, or similar — and make it measurably smarter over time through Reinforcement Learning.

Concretely: we analyse the target workflow, configure an agent (model + prompt + tools), deploy it in production, and then run our continuous RL loop — collecting interaction logs and traces, using them to retrain the underlying model on M42's specific clinical standards, and deploying an improved version. Agents compound in performance rather than stagnating.

In our comparable pilot with a German University Hospital we achieved:
- **+30% accuracy** over o4-mini on clinical question answering (FHIR-integrated)
- **60% cost reduction** by routing to our custom 7B model vs. frontier APIs
- **Full data sovereignty**: all patient data processed on-premise, zero cloud egress
- **Potential value**: 5–15 FTE equivalents saved per hospital (>€1M/year)

For M42, we envision a 3–6 month pilot focused on one specific high-volume clinical workflow, with measurable benchmarks agreed upfront and a clear path to scale.

---

**Q9: Which Focus Area does your solution address?**

✅ **AI** (primary)
✅ **Digital Health**
✅ **Health System Optimization**

Our platform sits at the intersection of all three: it is a pure AI infrastructure play (RL post-training), applied specifically to digital health workflows, with the end goal of optimising clinical and operational performance at scale.

---

**Q10: What is the current maturity level of your solution, and have you secured any regulatory approvals or certifications (if applicable)?**

Our platform has been validated in a production environment with a large German University Hospital. The core system — comprising our RL post-training pipeline, FHIR server integration, and local deployment infrastructure — is functional and battle-tested. Our custom 7B clinical model (Qwen 2.5 base, RL post-trained on clinical data) has been benchmarked against frontier models and outperforms o4-mini by +30% on the specific tasks it was trained for.

Regarding regulatory approvals: Aganthos is an AI infrastructure and learning layer — we are not a diagnostic tool, clinical decision support system, or medical device in the regulatory sense. As such, CE marking, FDA clearance, or MoHAP registration are not applicable to our platform. We operate fully within GDPR and data sovereignty requirements, and our on-premise deployment model ensures patient data never touches external infrastructure.

---

**Q11: Please list any relevant approvals (e.g. CE, FDA, ISO, UAE MoHAP, DoH) or state "Not applicable."**

Not applicable. Aganthos provides an AI infrastructure and learning layer (RL post-training pipeline), not a regulated medical device or clinical decision support tool. Our platform enables and improves AI agents built by our partners; regulatory classification depends on the downstream application, not our infrastructure layer.

---

**Q12: How novel or differentiated is your solution compared to others in the market?**

Aganthos is the only scalable platform that continuously improves AI agents through Reinforcement Learning post-training. Every competing platform — whether static orchestration tools (N8N, Zapier, Sema4.ai, Lindy AI) or bespoke integrators (Accenture, Palantir, InstaDeep) — delivers agents that are frozen after deployment. They behave identically on day 100 as on day 1, repeating the same errors, unable to adapt to shifting clinical standards or organisational constraints.

Our agents actually learn. Every interaction, error, and feedback signal is converted into training data for the next model iteration. The underlying mechanism is the same methodology OpenAI uses to build its o-series reasoning models (RL from feedback) — but applied to your domain-specific clinical workflows, running on your infrastructure with your data.

This creates a compounding performance advantage no static platform can replicate: the longer Aganthos runs in M42's environment, the more M42-specific the model becomes, and the higher the performance gap over any general-purpose tool grows. Our proprietary IP lies in making this RL loop computationally efficient and practically deployable on 7B-class models — delivering frontier-beating accuracy at 60% lower cost.

---

**Q13: Was the solution developed entirely by your team or in collaboration with external partners?**

The Aganthos platform — our RL post-training pipeline, training harness, and deployment infrastructure — was developed entirely in-house. The underlying base model (Qwen 2.5, 7B parameters) is open-source; all RL fine-tuning, reward modelling, and post-training is proprietary to Aganthos.

Our first deployment was developed in R&D collaboration with our launch partner (a large German University Hospital), whose clinical team provided domain expertise and feedback for reward signal design. This collaboration is documented in our ICML 2026 submission. We also collaborate with independent international AI researchers on methodology — a relationship that has produced a NeurIPS 2025 workshop paper and the aforementioned ICML submissions.

---

**Q14: What specific resources, input, or partnerships are you seeking from M42?**

We are seeking a focused, paid design partnership structured around four inputs from M42:

1. **Workflow access**: Identification of one high-volume clinical agent workflow where measurable performance improvement would deliver clear value (e.g., clinical documentation, patient query handling, QA over structured data). We need structured access to agent interaction logs and outcome labels for RL training.

2. **Clinical benchmark definition**: Collaboration with M42's clinical informatics or quality team to define what "correct" looks like in the target workflow — the reward signal that drives our RL loop. This is the most critical success factor and requires M42 domain expertise.

3. **Dedicated counterpart**: An internal contact (clinical informatics, innovation, or IT) for iterative feedback during the 3–6 month pilot, enabling rapid model iteration cycles.

4. **Longer-term**: Potential co-development and co-publication of a "M42-tuned" clinical model that demonstrates the value of domain-specific RL in a real-world integrated health system — a joint scientific and commercial asset for both parties.

---

**Q15: Which patient populations, clinical areas, or operational workflows will be impacted?**

For our proposed pilot, the most natural starting points — based on our existing work — are:

✅ **Clinical Documentation & Workflow** (primary — our proven use case)
✅ **Diagnostics & Imaging** (AI-assisted report QA and structured data extraction)
✅ **Care Coordination & Referrals** (routing and triage decision support)
✅ **Patient Access & Scheduling** (high-volume, repetitive agent workflows ideal for RL optimisation)

The specific populations and workflows would be defined jointly with M42 based on where the reliability gap is highest and the ROI most immediate. Our platform is workflow-agnostic — the RL loop adapts to whatever task is defined.

---

**Q16: Share case studies regarding feasibility, effectiveness, or ROI from previous pilots or deployments.**

**Case Study: Large German University Hospital — Clinical QA Agent (Q4 2025 / Q1 2026)**

We deployed a FHIR-integrated QA agent for clinical data queries, post-trained via RL on the hospital's own interaction logs and clinical feedback. The agent handles structured queries against patient records (demographics, admission histories, diagnostic data) and routes responses to appropriate clinical workflows.

**Results:**
| Metric | Result |
|---|---|
| Performance vs. o4-mini | **+30%** accuracy on clinical benchmarks |
| Inference cost reduction | **60%** vs. frontier API pricing |
| Model size | **7B parameters** (Qwen 2.5, RL post-trained) |
| Data sovereignty | **100%** on-premise, zero cloud egress |
| Projected FTE savings | **5–15 FTE equivalents** per hospital (>€1M/year) |
| Scientific output | ICML 2026 submission (under review) |

This pilot demonstrated that RL post-training on domain-specific data consistently outperforms general frontier models on specialist tasks — and does so at a fraction of the cost and with full data control. We are happy to share the full technical brief under NDA.

---

**Q17: Upload Company Profile/Product Brief/Other relevant documents**

[Attach: Aganthos Slides 110325.pdf]

---

**Q18: Are there any other additional details you would want us to know?**

A few additional points that may be relevant to M42's evaluation:

**Scientific credibility**: We have submitted two papers to ICML 2026 — one documenting the clinical RL model (the case study above) and one on our general RL post-training methodology for tool-calling agents. A prior NeurIPS 2025 workshop paper established early scientific validation of our approach. These publications demonstrate that what we are doing is not engineering heuristics but grounded in rigorous machine learning research.

**NVIDIA Inception**: We were accepted into NVIDIA's Inception program in Q1 2026, giving us access to compute resources and technical collaboration — relevant for scaling RL training loops for enterprise deployments.

**Team**: Our Lead Researcher (Robert Mueller) brings a decade of Reinforcement Learning research to this problem — formerly Founding Research Scientist at Convergence AI, recently acquired by Salesforce. Our Operations Lead (Tobias Schuster) built and led a 30-person team responsible for a CHF 3.3B national COVID-19 response operation for the Swiss Federal Health Ministry — a background that translates directly into the ability to deploy complex systems reliably in high-stakes, regulated environments.

**Why M42**: M42's ambition to build the future of integrated, intelligent healthcare across 24+ countries is precisely the environment where a learning agent platform provides compounding value. The more clinical pathways M42 automates, the larger the performance gap between a learning system and a static one. We believe this is a partnership with structural long-term potential, not just a point solution.
