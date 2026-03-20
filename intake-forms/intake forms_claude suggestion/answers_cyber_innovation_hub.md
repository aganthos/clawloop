# Antworten: Cyber Innovation Hub der Bundeswehr (CIHBw)

---

**Vorname:** Tobias

**Nachname:** Schuster

**E-Mail Adresse:** info@aganthos.com

**LinkedIn URL:** https://www.linkedin.com/in/tmschuster/

---

**Startup Name:** Aganthos

**Webseite URL:** https://aganthos.com

**Hauptsitz:** Zürich, Schweiz (Swiss GmbH; zudem DE C-Corp, USA)

**Gründungsjahr:** 2025

**Unternehmensgröße:** 3 (Vollzeit: 1 Gründer + 1 Engineer; 1 Gründer in Transition zu Vollzeit)

**Unternehmensreife:** Frühe Marktphase — erster bezahlter Pilot erfolgreich abgeschlossen, erste Umsätze erzielt, aktive Gespräche für weitere Design-Partnerschaften

---

**Beschreibe die militärische Herausforderung**

Moderne Streitkräfte investieren erheblich in KI-gestützte Agentensysteme — für Lageauswertung, Planung, autonome Plattformsteuerung und Entscheidungsunterstützung. Das fundamentale Problem: Diese Systeme sind statisch. Sie lernen nicht aus Einsatzerfahrungen, passen sich nicht an neue operative Anforderungen an, und degradieren, sobald sich Tools, Schnittstellen oder Lagebilder verändern.

Konkret: Generelle Frontier-Modelle (z. B. GPT-4o, Llama) erreichen auf spezifischen Domänenaufgaben nur 50–60 % Genauigkeit — für militärische Anwendungen im Bereich Software-Defined Defence (SDD) ist das unzureichend. Die Alternative — vollständig maßgeschneiderte KI-Systeme — ist prohibitiv teuer, langsam in der Entwicklung, und skaliert nicht über einzelne Plattformen hinaus.

Das Kernproblem: KI-Agenten für SDD müssen sich an neue Lagebilder, veränderte Tool-APIs, neue taktische Anforderungen und einsatzinduzierte Randbedingungen anpassen — und das schnell. Ein statisches System, das nach der Erstentwicklung eingefroren ist, erfüllt diese Anforderung strukturell nicht.

---

**Beschreibe die Lösung(en)**

Aganthos stellt eine "Learning as a Service"-Schicht für KI-Agenten bereit, die auf Reinforcement Learning (RL) Post-Training basiert — die gleiche Methodik, die OpenAI für seine o-Serie Reasoning-Modelle verwendet, angepasst für domänenspezifische Einsatzszenarien und On-Premise-Deployment.

**Das Funktionsprinzip (Learning Flywheel):**
1. **Analyse**: Wir verstehen den Ziel-Workflow und die Anforderungen des Agenten (Tools, Entscheidungslogik, Erfolgsmetriken)
2. **Setup**: Konfiguration des Agenten (Modell + Prompt-Harness + Tools + Memory)
3. **Deploy & Collect**: Betrieb im Einsatz; kontinuierliche Sammlung von Logs, Traces und Outcomes
4. **Train Agent**: RL-basiertes Post-Training adaptiert Modellgewichte und Harness auf Basis realer Einsatzerfahrungen
→ Verbessertes Modell → kontinuierliche Iteration

**Für die Bundeswehr konkret:**
- **SDD-Anwendungsfall**: RL-trainierte Agenten für Lageauswertung, Planungsunterstützung oder Tool-Orchestrierung, die sich an neue Einsatzdoktrinen und API-Veränderungen adaptieren — ohne manuelle Neuentwicklung
- **On-Premise-Deployment**: Alle Modelle und Trainingsdaten verbleiben innerhalb der sicheren Infrastruktur — keine Cloud-Abhängigkeit, keine externen Datentransfers
- **Messbare Leistungssteigerung**: In unserem ersten bezahlten Piloten (großes deutsches Universitätsklinikum) erzielte dieses Verfahren **+30 % Genauigkeit** gegenüber o4-mini auf domänenspezifischen Benchmarks, **60 % Kostenreduktion** durch Routing auf ein custom 7B-Modell, und vollständige Datensouveränität

Das medizinische Einsatzszenario und militärische Anwendungen teilen eine strukturelle Gemeinsamkeit: hohe Anforderungen an Verlässlichkeit, sensible Daten, die das System nicht verlassen dürfen, und die Notwendigkeit, sich schnell an neue operative Realitäten anzupassen.

---

**Beschreibe deinen unfair Advantage**

Unser Wettbewerbsvorteil liegt in einer einzigartigen Kombination, die kein anderer Anbieter im "Learning Automation"-Segment vereint:

**1. Erstklassige RL-Forschungstiefe (Robert Mueller, Co-Founder):**
Ein Jahrzehnt an der Forefront des Reinforcement Learning — AISTATS Best Paper Nominee 2022, Founding Research Scientist bei Convergence AI (2025 von Salesforce übernommen). Dieser Hintergrund umfasst RL in Spielen, Hochgeschwindigkeitsrobotik, Elektronenmikroskopie und Web-Agenten — von der Theorie direkt in die Produktion. Unser technisches Fundament ist nicht die Anwendung fremder Frameworks, sondern eigene Forschung, die in laufende ICML-2026-Einreichungen mündet.

**2. Bewährte operative Skalierungskompetenz unter Druck (Tobias Schuster, Co-Founder):**
Aufbau einer 30-köpfigen Krisenorganisation innerhalb des Schweizer Bundesamts für Gesundheit (FOPH) innerhalb von Wochen, verantwortlich für eine 3,3-Milliarden-Franken-COVID-Testoperation. Entscheidungen mit nationalen Konsequenzen unter extremer Unsicherheit — und Umsetzungsgeschwindigkeit, die staatliche Standardprozesse um Faktor 10–100 übertraf (Gesetzesänderungen in Tagen statt Jahren). Diese Kombination aus Forschungstiefe und Ausführungskraft ist für einen Defence-Kontext von direkter Relevanz.

**3. Proprietäre Technologie mit nachgewiesenem Ergebnis:**
Wir haben proprietäre Modelle mit 1,5B und 3B Parametern für Tool-Orchestrierung trainiert und eingesetzt. Unser RL-Post-Training-Loop ist operationalisiert und produziert messbare Ergebnisse — nicht in der Lab-Umgebung, sondern im produktiven Einsatz eines regulierten Sektors.

**4. On-Premise-First-Architektur:**
Unsere Plattform wurde von Anfang an für souveräne Deployments entwickelt — alle Modellgewichte und Trainingsdaten verbleiben beim Kunden. Das ist kein nachträgliches Feature, sondern Architekturprinzip. Für den Defence-Kontext ist das nicht optional.

---

**Wie reif ist die Technologie auf der TRL Skala?**

**TRL 5–6**: Technologie in relevantem Umfeld validiert / Prototyp in relevantem Umfeld demonstriert.

Unser RL-Post-Training-System wurde in einem ersten bezahlten Piloten mit einem großen deutschen Universitätsklinikum erfolgreich eingesetzt — einem hochregulierten, datensensibler Umfeld mit hohen Anforderungen an Zuverlässigkeit und Datensouveränität. Die Kernkomponenten (RL-Trainingsloop, On-Premise-Deployment, Tool-Orchestrierung) sind betriebsbereit und haben reale Nutzer und messbare Ergebnisse produziert. Anpassungen für spezifische Bundeswehr-Anwendungsfälle (z. B. andere Tool-APIs, militärische Reward-Signale) würden im Rahmen eines 12–24-monatigen Prototypenprojekts erfolgen.

---

**Gibt es darüber hinaus noch Informationen, die du mit uns teilen möchtest (Referenzen, Traktion, etc.)?**

Einige zusätzliche Punkte, die für die Bewertung relevant sein könnten:

**Wissenschaftliche Validierung**: NeurIPS-2025-Workshop-Paper (erste wissenschaftliche Veröffentlichung), zwei Einreichungen bei ICML 2026 — eine zum klinischen RL-Modell, eine zur allgemeinen Methodik für Tool-Calling-Agenten. Unser Ansatz ist peer-review-fähig, nicht nur ein Engineering-Hack.

**NVIDIA Inception**: Aufnahme in das NVIDIA Inception-Programm (Q1 2026) — Bestätigung der technischen Substanz und Zugang zu Compute-Ressourcen für Trainingsläufe.

**Umsatz & Traktion**: ~€28K Umsatz aus abgeschlossenen Meilensteinen (Q1 2026). Erster bezahlter Pilot mit einem deutschen Universitätsklinikum erfolgreich abgeschlossen.

**EWR-Konformität**: Aganthos ist als Swiss GmbH (Zürich) und DE C-Corp strukturiert — vollständig innerhalb des EWR ansässig. Eine Zusammenarbeit mit der BWI GmbH und der Bundeswehr im Rahmen der geltenden Beschaffungsregeln ist damit möglich.

**Warum CIHBw**: Der Fokus des CIHBw auf Software-Defined Defence trifft den Kern dessen, was Aganthos löst: KI-Systeme, die sich dynamisch an neue softwarebasierte Anforderungen anpassen müssen. RL-Post-Training ist das technologische Fundament, das adaptive militärische KI-Agenten erst möglich macht — und genau das ist unser Kernprodukt.

---

**Wie bist du auf uns aufmerksam geworden?**

Durch eigene Recherche zu europäischen Defence-Innovation-Programmen im Rahmen unserer Marktentwicklung für Dual-Use-Anwendungsfälle.

---

**Pitch Deck**: Aganthos Slides 110325.pdf [hochladen]
