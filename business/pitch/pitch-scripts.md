# Aganthos Pitch Scripts - Berkeley SkyDeck Batch 22

---

## 1-Minute Pitch (Elevator / Intro)

- **Hook:** Every enterprise is deploying AI agents, but they all hit the same wall: agents fail on domain-specific tasks, costs spiral at scale, and sensitive data can't leave the organisation.
- **What we do:** Aganthos is the learning layer for AI agents. We enable agents to learn from experience using reinforcement learning -- continuously improving from their own production data.
- **Proof:** We RL-post-trained a 7-billion-parameter model for a German university hospital that outperforms OpenAI's o4-mini by 30% on clinical benchmarks, at 90% lower inference cost, with full data sovereignty.
- **Traction:** Swiss and German entities incorporated, first paid partnerships, two ICML 2026 submissions, revenue-generating projects completed.
- **Ask:** We're joining SkyDeck to access US enterprise customers in insurance, healthcare, and financial services, and to close our pre-seed round by Demo Day.

---

## 3-Minute Pitch

### The Problem (30 sec)
- Enterprises are betting big on AI agents, but these agents hit a wall before becoming productive
- Three barriers: (1) unreliable performance on domain-specific tasks -- they get 50-60% when you need 90%+, (2) data sovereignty -- sensitive data can't go to external APIs, (3) unsustainable cost -- frontier model APIs scale linearly and become the largest line item at production volume

### Our Solution (45 sec)
- Aganthos is the learning layer for AI agents -- we enable agents to learn from experience
- Three value pillars: improve performance through RL and in-context learning, reduce cost through model distillation and smart routing to smaller models, enable privacy through knowledge transfer to locally deployed models
- Our Learning Flywheel: we analyse your workflows, set up and configure agents, deploy them to collect production data, then continuously train and improve -- creating a virtuous cycle where every interaction makes the agent smarter

### Case Study & Proof (45 sec)
- Concrete proof: we partnered with a large German university hospital to build a medical AI agent
- We RL-post-trained a 7B parameter model (FHIR-Agent) on clinical tasks
- Results: 30% improvement over o4-mini on clinical benchmarks, with 90% cost reduction and full data sovereignty -- patient data never leaves the hospital
- This demonstrates a universal optimisation engine for AI agents -- generalisable to any complex domain
- Two ICML 2026 papers submitted based on this work

### Traction & Team (30 sec)
- Founded by an RL researcher with a decade at the forefront (CMU, Mila, Sony AI, Convergence AI acquired by Salesforce) and an operator who built a $2.6B national crisis management system
- Incorporated in Switzerland (HQ) and Germany, first revenue-generating projects completed, NeurIPS workshop paper, proprietary 1.5B/3B models trained
- Targeting mid-market enterprises in insurance, healthcare, logistics, and financial services

### Ask (15 sec)
- SkyDeck gives us US market access to enterprise customers and pre-seed validation
- We will launch 2-3 paid design partnerships within 3 months, close pre-seed by Demo Day, and target Series A by Q1 2027
- We're always happy to help -- aganthos.com

---

## 8-Minute Pitch (SkyDeck First Round)

### Opening & Hook (45 sec)
- "Every enterprise CTO we talk to says the same thing: we've deployed AI agents, and they're... okay. They work for the easy stuff, but they hit a wall on anything domain-specific. And the costs are killing us."
- $24.8 billion in unrealised value from inefficient LLM inference (Linux Foundation estimate)
- The core problem: today's AI agents don't learn. They're static. Deploy them, and day 1000 is identical to day 1 -- same mistakes, same limitations, same costs.

### The Problem - Deep Dive (1 min)
- Three barriers holding back enterprise AI agents:
  1. **Unreliable performance:** Off-the-shelf frontier models get 50-60% on domain tasks where enterprises need 90%+. Not good enough for production.
  2. **Data sovereignty:** Healthcare, insurance, finance -- data can't leave the organisation. Vendor lock-in to OpenAI/Anthropic APIs is a compliance risk and a strategic risk.
  3. **Unsustainable costs:** At production volume, frontier API costs become the largest line item. They scale linearly with usage -- no learning curve benefit.

### Mission & Solution (1 min)
- **We enable agents to learn from experience.**
- Aganthos is the learning layer for AI agents -- we sit between your agent framework and the models that power it.
- Three value pillars:
  - **Improve Performance:** RL and in-context learning from production traces -- agents get measurably better at your specific tasks
  - **Reduce Cost:** Smart routing to the cheapest model that maintains quality, plus distillation to smaller purpose-built models -- 90% cost reduction
  - **Enable Privacy:** Train and deploy models on your own infrastructure -- data never leaves your organisation, full regulatory compliance

### How It Works - The Learning Flywheel (1 min 15 sec)
- Our approach is "Learning as a Service"
- Phase 1 -- Onboarding: We analyse your workflows, identify processes most suitable for agentic augmentation, define KPIs, set up agents and tooling
- Phase 2 -- Continuous Improvement: Deploy agents in production, collect logs and traces, then train and improve using RL, distillation, and in-context learning. This is a continuous cycle -- every interaction makes the system smarter.
- Multiple training modalities depending on what you have:
  - Got query logs? We distill to smaller models immediately (cost savings from day 1)
  - Got a training environment? We do full RL post-training (performance gains + cost savings)
  - Got tool traces? We build world models and train with simulated environments
- Key insight: we reduce AI cost immediately through smart routing, then continuously improve through learning -- the flywheel compounds

### Case Study (1 min 15 sec)
- Let me give you a concrete example: partnership with a large German university hospital
- Problem: clinical staff need to query electronic health records (FHIR) -- patient histories, medications, procedures. Frontier models fail on the domain-specific tool-calling required.
- What we did: RL post-trained a 7-billion-parameter open model (FHIR-Agent-7B)
- Results:
  - **30% improvement over o4-mini** (OpenAI's frontier model) on clinical benchmarks
  - The learning curve: from 50% to 65% accuracy over 270 training steps
  - Category-level improvements across conditions, encounters, medications, observations, patients, and procedures
  - **90% cost reduction** vs. frontier APIs
  - **Full data sovereignty:** trained and deployed within the hospital -- patient data never leaves
  - Potential to save **5-15 FTE equivalents** (>1M EUR annually) in medium-to-large hospitals
- This isn't just a healthcare result -- it demonstrates a **universal optimisation engine** for AI agents, generalisable to any complex domain
- Two ICML 2026 submissions based on this work

### Competition Landscape (45 sec)
- Two-axis view: scalable solutions vs. services, static automation vs. learning from experience
- **Bottom-left (Legacy BPO):** Accenture, Deloitte, UiPath -- they wrap LLMs with consulting, no learning, not scalable
- **Top-left (Static Platforms):** N8N, Zapier, 11x.ai, Lindy AI -- great for simple automation, but no learning from experience
- **Bottom-right (Bespoke Intelligence):** Palantir, InstaDeep, Applied Intuition -- they build learning systems, but bespoke and expensive
- **Top-right (Learning Automation Platform):** This is where we play -- Aganthos, alongside Orby AI, Adept AI, Imbue -- scalable solutions with learning built in
- Our edge: we're the only company focused specifically on the learning layer as infrastructure, not building the whole agent stack. We make any agent smarter.
- Training infrastructure companions (Tinker/Thinking Machines Lab, Serverless RL) provide compute; we provide the learning algorithms and customer deployment

### Business Model & Customer Journey (45 sec)
- **Customer journey:** Identify processes > Evaluate models > Optimise with training > Maintain with continuous learning
- **Monetisation:**
  - Hourly setup fees for initial assessment and configuration
  - Maintenance and continuous improvement fees (recurring)
  - Usage-based tiers for trained model inference
  - Joint ventures starting Q2 2026: equity + revenue share in AI-native products we co-build
- **Target customers:** Mid-market enterprises (500-15K employees, $100M+ revenue) in insurance, healthcare, logistics, and financial services -- companies with AI initiatives but no in-house research teams
- **Pricing:** CHF 50-250K pilots, CHF 200K-5M ACV for full contracts

### Team (30 sec)
- **Technical founder:** A decade at the forefront of RL research -- TU Munich, Carnegie Mellon, Mila, Sony AI. AISTATS Best Paper nominee. Founding Research Scientist at Convergence AI, leading the research team (acquired by Salesforce in Q2 2025). From theory to production: games, robotics, electron microscopes, web agents.
- **Business founder:** Designed and implemented a $2.6B national-scale crisis management system for the Swiss federal government. Complex coordination across disciplines, sectors, and government levels. Generative AI in public health, medicine, and climate.
- **Founding technical staff:** Experienced SWE building data platforms on AWS/GCP at BMW Group. Cloud deployment of custom RAG agents.

### Traction & Status (30 sec)
- Q4 2025: NeurIPS workshop paper, DE C-Corp, first paid partnership, proprietary 1.5B/3B models trained
- Q1 2026: Two ICML submissions, Swiss GmbH incorporation, first revenue projects completed, +30% medical AI result, multinational design partnership discussions

### The Ask & SkyDeck Milestones (30 sec)
- **$200K investment** funds first two hires and compute for enterprise pilots
- **US market access** through Berkeley network -- enterprise intros in insurance, healthcare, financial services
- **Pre-seed catalyst** -- SkyDeck validation to close round by Demo Day
- **Milestones:**
  - Month 1: 2 paid design partnerships, first hire, first US enterprise LOI
  - Month 3: 2-3 active partnerships generating revenue, multi-step training pipeline live
  - Month 6: Continuous improvement platform live, first joint venture, pre-seed closed
  - Series A by Q1 2027

### Close (15 sec)
- We empower professionals to break free from routine tasks. We orchestrate a world where human creativity thrives. We provide the learning layer for agents that drive performance.
- **aganthos.com -- we're always happy to help.**
