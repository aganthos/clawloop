# Intake Form Answers: Siemens Startup Collaboration

---

**Your pitch**

Aganthos provides the learning layer for AI agents — enabling them to continuously improve through Reinforcement Learning rather than remaining static after deployment.

The problem: enterprise AI agents hit a reliability wall. Off-the-shelf frontier models achieve 50–60% accuracy on complex domain-specific tasks; production requires 90%+. And because agents don't learn, they degrade over time as tools, APIs, and workflows evolve. The result: brittle automation requiring constant manual maintenance — and no ROI.

Our solution: a "Learning as a Service" platform that wraps any existing agent with an RL improvement loop. We analyse the workflow, configure the agent, deploy in production, and continuously collect logs and traces to retrain the underlying model on the organisation's specific tasks and constraints. The agent compounds in performance over time.

In our first paid pilot with a large German University Hospital, this approach delivered:
- **+30% accuracy** over o4-mini (state-of-the-art frontier model) on domain-specific benchmarks
- **60% cost reduction** via routing to our custom 7B parameter model (vs. frontier API pricing)
- **Full data sovereignty**: all processing on-premise, zero external data egress

For Siemens, we see immediate application in high-volume, tool-calling agent workflows within IT and R&D — where agents interact with proprietary systems, documentation, and production data, and where consistent reliability is non-negotiable.

---

**Differentiation**

Aganthos is the only scalable platform that operationalises Reinforcement Learning post-training for enterprise agents. Three categories of competitors exist — and none solves the core problem:

1. **Static Agent Platforms** (N8N, Zapier, Sema4.ai, Lindy AI): These automate workflows but agents are frozen after deployment. They repeat the same errors indefinitely and cannot adapt to Siemens-specific systems or evolving requirements.

2. **Legacy BPO + LLM integrators** (Accenture, Deloitte, UiPath): They implement AI but scale with human headcount, not intelligence. The learning loop requires expensive engineering retainers.

3. **Bespoke Intelligence** (Palantir, InstaDeep): They build custom solutions but not scalable platforms. High implementation cost, slow iteration cycles, cannot be deployed across multiple Siemens business units cost-effectively.

**Aganthos' unique position**: we combine the intelligence of a bespoke RL system with the scalability of a SaaS platform. Our mechanism mirrors how OpenAI builds its o-series reasoning models — RL from feedback — but applied to Siemens' specific domain, running on Siemens' infrastructure. This creates a performance advantage that compounds over time and is structurally impossible for static platforms to close.

Our proprietary IP: making RL post-training computationally efficient for 7B-class models, delivering frontier-beating accuracy at 60% lower inference cost.

---

**Development stage of your solution**

✅ **Early market stage** (selling an early version of the solution; early adopters are the main clients)

We have completed paid engagements with our first enterprise customer (a large German University Hospital), generating ~€28K in revenue from three project milestones in Q1 2026. We are in active discussions for 2–3 additional design partnerships. Our core platform — RL training pipeline, deployment harness, and model infrastructure — is operational and validated in a production environment.

---

**Business model**

Aganthos monetises through a three-tier model:

1. **Setup / onboarding fee** (one-time): workflow analysis, agent configuration, initial RL training run. Typically €10K–€25K depending on scope.
2. **Continuous improvement retainer** (monthly/quarterly): ongoing RL retraining cycles as new production data accumulates, model updates, harness optimisation. This is the recurring revenue engine.
3. **Usage-based inference tiers**: as the trained model handles more requests, tiered pricing based on inference volume.

The model is designed to decouple revenue from headcount: RL training scales with GPU compute, not human engineers. For Siemens, a venture client engagement would likely begin with a defined pilot (fixed fee), transition to a retainer as value is demonstrated, and scale across additional workflows and business units.

---

**Reference customers**

- **Large German University Hospital** (anonymised): first paid partnership (Q4 2025 / Q1 2026). €28K revenue from completed milestones. Clinical QA agent with FHIR integration: +30% over o4-mini, 60% cost reduction, full on-premise deployment. ICML 2026 paper submission documenting results.
- 2–3 additional design partnership discussions in progress (Q2 2026, undisclosed).

---

**Which Siemens function will profit from your solution?**

✅ **IT** (primary — agent workflows interacting with internal systems, APIs, and data platforms)
✅ **Research & Development of Products, Services, etc.** (R&D productivity agents: literature review, code QA, documentation)
✅ **Production / Manufacturing** (quality control agents, anomaly detection workflows, process monitoring)

The most immediate fit is **IT and R&D**: these are the highest-volume environments for tool-calling agents where our RL loop can demonstrably improve reliability and reduce inference costs. Production / manufacturing workflows are a natural second step as agents mature.

---

**What type of partnership are you seeking?**

✅ **Collaborate: Venture Clienting**

A paid pilot with a specific Siemens business unit is the ideal starting point. We are not seeking equity or grant funding — we are looking for Siemens to be an enterprise customer for a defined 3–6 month pilot, with clear performance benchmarks agreed upfront. The structure would mirror our hospital engagement: scoped workflow, measurable outcome targets, fixed fee. If the pilot demonstrates the expected value, the natural path is a retainer for continuous improvement across additional workflows.
