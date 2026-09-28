# Research brief: scientific due diligence for Olumi's decision-analysis engine

**For:** an AI research agent with web and GitHub search.

**Output:** `RESEARCH-FINDINGS.md` plus `evidence.json`, in the format set out in section 8.

**Language:** British English.

**Stance:** open-ended and evidence-first. Nothing in this brief is a conclusion you should confirm. Where the brief states a fact, it is labelled MEASURED with its source. Everything else is a question.

---

## 0. Your role and the one rule that matters most
You are doing independent due diligence, not advocacy. Two candidate engineering directions are described in section 2.4. **Do not assume either is right, and do not assume the problem is framed correctly.** It is an equally valid finding that:
- current best practice already matches what Olumi does;
- a different field has solved this better;
- the question itself should be reframed.

Never state a claim that you have not verified in a source you actually opened in this session.

## 1. Why this research is needed
Olumi is a decision-support product. A user describes a business decision in conversation. A language model then drafts a causal model of it: factors, options, a goal and limits. An engine analyses that model and reports:
- which option is more likely to reach the goal;
- which limits hold;
- where more information would change the decision.

Olumi is deciding how its analysis engine should evolve (section 2.4). Recent measurement (section 2.3) shows that the hard part is **partly or mostly scientific**, not only engineering. The open issues are:
- how to represent the decision quantitatively;
- how to obtain missing quantities honestly;
- how to handle time;
- how to compute and present information value when uncertainty is poorly specified.

We need to know what credible, current research and maintained open-source work say about these problems, including work that is tangential or that could make the current approach unnecessary.

## 2. What we know (facts) and what we do not know

### 2.1 How the system works today (MEASURED: code read of the Olumi services, September 2026)
- **Model drafting.** A language-model service drafts a graph from the user's brief.
  - **Nodes:** factors, options, a goal, and risks.
  - **Levels:** some nodes carry an observed level with a unit. The level comes either from the user or from a model estimate, and its provenance is recorded.
  - **Edges:** each carries a *normalised* strength (mean and standard deviation, roughly in [−1, 1]) and an existence probability. Some edges also carry an effect amount in the user's units, for example "−0.2 percentage points of churn per +10 perception score".
  - **Options** set levels on factor nodes.
  - **Goal and limits:** the goal has a threshold, and limits are thresholds on nodes.
- **The analysis engine** (Python, numpy/scipy) propagates effects by Monte Carlo.
  - Effects are additive and linear in the normalised space.
  - There is no representation of time, of a deadline horizon, or of stocks that accumulate.
  - Exact relationships such as `MRR = price × subscribers` were ignored until a change that is currently in review. That change evaluates declared product and sum relationships in the user's units.
- **Information value** is the expected value of partial perfect information (EVPPI).
  - It is estimated by a regression-based method: a Strong–Oakley-type polynomial fit.
  - Significance is tested against a noise floor, which is the maximum over 16 permutations.
  - Results are emitted at 6 decimal places.

### 2.2 The three test journeys (the real briefs behind the 12-model corpus)
- **A, pricing:** a subscription business at £75k monthly recurring revenue (MRR) wants £100k within 12 months, with monthly churn under 4%. The choice is £49 versus £59 per month for the Pro plan.
- **C, pricing vs advertising:** £100k MRR within 6 months, with a £20k budget.
- **E, hiring:** 2 senior versus 4 junior hires, salary spend under £400k a year, shipping "by Q3".

### 2.3 Measured results (MEASURED: an offline evaluation on 12 real drafted models, 4 per journey)
- **Quantities:** 116 were hand-audited.
  - Only 5 are stocks and 3 are flows.
  - 48 are unsupported or ambiguous: their role cannot be established from the model's own content.
  - 44 of the 67 levels the models hold are language-model estimates, not user facts.
- **Causal edges:** there are 142. The other 196 edges are structural, such as option → factor.
  - 101 carry no quantity usable in the user's units.
  - 31 carry a user-unit effect amount. **All 31 have the same template spread** (standard deviation = 0.5 × mean).
  - All 142 have existence probability 0.8, a default.
  - 10 are definitional identities.
- **Deadlines:** no model carries a typed horizon. Deadlines exist only in the brief text, and 4 of 12 are ambiguous ("by Q3" with no year or fiscal basis).
- **Strict simulation fails.** A strict monthly simulation that invents nothing produced **no** deadline verdict and **no** complete ranking of options for any of the 12 models.
- **Exploratory simulation flips verdicts.** When exploratory assumptions were added, the goal verdict or the winning option **flipped** in 3 of the 4 computable cases, depending on semantics the model does not carry:
  - whether "net new subscribers" is net or gross of churn;
  - the size of the price → churn response;
  - the competitive response;
  - whether an uplift compounds.
- **Break-even threshold** (journey A, £59 option, exploratory): the option still reaches the goal unless churn rises by more than about 2.1–2.35 percentage points per +£10. The user's 4% churn limit breaks first, above +1.5 points.
- **Across the programme:**
  - in 87% of 114 served analyses, no information-value result was above the noise floor;
  - in 24 of 120 served analyses, root quantities with no value were computed as zero.
- **The EVPPI noise floor is fragile.** It can collapse to exactly 0 when the permuted fits never cross, so tail artefacts read as "resolved". Example: £38.27 → £0.79 → £0 → £0.18 across seeds and draw counts, on a £98k outcome.

Traceability: the prototype, tests and results are in the Inference-Service-Layer repository, branch `r3b/sim-prototype`, folder `experiments/r3b_sim/`. See `EVALUATION.md`, `MAPPING.md`, `results.json` and `breakeven.json`. If you cannot access them, rely on the figures above.

### 2.4 Directions under consideration (NOT endorsed; evaluate, don't assume)
- **A:** keep the static engine and add exact relationships (identities) incrementally.
- **B:** a month-by-month stock-and-flow simulation over the deadline horizon.
- **MVP B:** a single stock with a gross inflow and a churn rate, reporting break-even thresholds rather than probabilities where a response is unknown.

### 2.5 What we do NOT know (the reason for this research)
- How credible fields turn a qualitative, machine-drafted causal graph into a quantitative decision model, and what they do about missing semantics.
- Whether explicit time dynamics change decisions often enough, over 6–12-month horizons, to justify their cost and elicitation burden.
- The most credible, lowest-burden way to obtain the missing quantities: baselines, rates, responses, onsets and uncertainty ranges.
- Whether EVPPI is meaningful when uncertainty is templated, and which information-value estimators and diagnostics are current best practice.
- Whether a different approach supersedes this framing.

## 3. Research questions
Answer every question. For each one, state: the answer; your confidence (high / medium / low) and why; the evidence IDs; and **what evidence would change the answer**.

1. **RQ1. From drafted causal graph to quantitative model.**
   - Which methods exist (in any field) for converting an informal or machine-generated causal diagram, with qualitative or normalised link strengths, into a quantitative model with units, exact identities and uncertainty?
   - What does each method require as input?
   - How does each detect, flag or handle missing or ambiguous semantics?
   - Include any evidence on the validity of treating normalised, signed link weights as quantitative effects.
2. **RQ2. When does time matter?**
   - What is the evidence on when explicit dynamics (stocks and flows, difference equations, cohort or retention models, state-space models, delays) change *decisions*, not just forecasts, compared with static or end-point models?
   - For subscription businesses specifically: what do credible customer-base, retention and pricing studies say about modelling churn and the price → churn response over 6–12 months? Is a constant churn rate adequate?
3. **RQ3. Missing quantities: elicitation and machine estimates.**
   - Which evidence-based methods obtain baselines, rates, responses, onset dates and uncertainty ranges from non-expert users with minimal burden?
   - What is the measured calibration and accuracy of language-model-generated numerical estimates and priors?
   - How should model estimates be combined with, labelled against, or replaced by user facts?
4. **RQ4. Deciding without credible probabilities.**
   - Which methods give decision-useful outputs when distributions are not credible?
   - Consider, among others: threshold, break-even and switching-value analysis; robust decision-making; info-gap; scenario discovery; imprecise or credal probabilities.
   - What evidence supports their validity and their comprehension by non-experts?
5. **RQ5. Information value: current best practice.**
   - What are the current recommended EVPPI and EVSI estimators? Cover their accuracy, bias, required sample sizes, convergence and stability diagnostics, and significance or noise-floor methods.
   - Are there documented failure modes of regression-based EVPPI with permutation-based floors?
   - Is information value meaningful when priors are templated? What do guidelines recommend?
6. **RQ6. Validating machine-built models.**
   - How do credible fields validate such models? Consider dimensional and unit consistency, identity-versus-data consistency, structural and behavioural validation tests, backtesting and face validity.
   - Which of these checks are automatable for models drafted by a language model?
7. **RQ7. Maintained open-source tooling.**
   - Which maintained libraries or repositories implement the credible methods from RQ1–RQ6 and could run inside a Python service? Areas include probabilistic programming, system-dynamics simulation, information value, elicitation, unit-aware computation and decision-making under deep uncertainty.
   - Assess maturity with the repository checklist in section 5.
8. **RQ8. Superseding or tangential approaches.**
   - Is there an approach that could make this framing unnecessary or clearly better?
   - Examples of the kind of thing to look for (not a list to confirm): language models that write executable probabilistic programs; simulation-based inference; learning effects from users' own data; agent-based models.
   - Assess each on evidence, not on claims.
9. **RQ9. Communicating results.**
   - What does credible risk-communication or decision-science evidence say about how non-expert decision makers understand and use probabilities, thresholds ("this holds unless churn rises more than X"), ranges and information-value statements?

**The null answer is acceptable.** If credible evidence shows nothing materially better than the directions in section 2.4, say so plainly and show what you checked.

## 4. Scope: go wide, then deep
- **Core:** work that directly addresses RQ1–RQ6 for business or economic decisions.
- **Adjacent:** the same problems in other domains. Examples: health-economic decision models, policy and climate models, engineering reliability, epidemiology, operations research, forecasting.
- **Tangential:** methods or tools that solve part of the problem in a different way, including human-factors and communication work.
- **Superseding:** anything that would make the current framing obsolete.

Do one breadth pass that covers every question and scope level before going deep on any one item. Record what you did not have time to cover.

## 5. Method and credibility rules
**Where to search:**
- scholarly indexes (Google Scholar, Semantic Scholar, OpenAlex, Crossref);
- arXiv, SSRN and RePEc;
- ACM DL and IEEE Xplore;
- PubMed, for the health-economics information-value literature;
- journal and conference proceedings in decision analysis, operations research and management science, system dynamics, marketing science, statistics, risk analysis, medical decision making, and machine learning and causal inference;
- GitHub, PyPI, CRAN and Papers with Code.

Do not restrict the search to venues you already know.

**How to search:**
- Build queries from the questions, not from candidate answers. Include synonyms from other fields.
- Snowball backwards (references) and forwards ("cited by") from every key item.
- Deliberately search for **disconfirming evidence and critiques** of every method you rate highly.

**Recency:**
- Prioritise 2023 to the present, and record the date of each search.
- Include older foundational work where it remains the authority, and say that this is why.

**Evidence grades.** Assign one to every item:

| Grade | Meaning |
|---|---|
| A | Systematic review or meta-analysis, or independent replications in peer-reviewed venues |
| B | A peer-reviewed study in a reputable venue with transparent methods and data |
| C | Peer-reviewed but weak design; or a preprint with public code and results you could reproduce |
| D | A preprint without code, a technical report, or a practitioner book |
| E | Vendor material, blogs, talks or forum posts. Context only: never the sole support for a claim |

**Flags.** Record any of the following that apply: retraction or expression of concern (check); a possible predatory venue; a conflict of interest (for example, a vendor evaluating its own tool); a small or synthetic sample; results that are only claimed, not demonstrated.

**Verification:**
- Every item must resolve through a DOI, arXiv identifier or URL that you opened in this session.
- Quote the supporting sentence(s) verbatim, with the section or page.
- If you saw only the abstract, mark the item "abstract only".
- **Never cite from memory. Never construct a citation.**

**Repository checklist.** Record for each repository:
- URL, licence, last commit date and latest release;
- tests and CI present;
- documentation quality;
- the maintaining organisation;
- open or closed issue ratio;
- Python compatibility;
- whether it is an academic companion to a paper.

Stars are context only, not evidence of quality.

**Separate** what a source *demonstrated* (the data, method and result) from what it *claims*.

## 6. Anti-bias rules
- Do not favour dynamic simulation, identities, information value, or any other method because this brief mentions it.
- For every method you recommend, give its strongest credible critique and the conditions under which it fails.
- Report negative and null results, and the leads you checked and rejected, with the reason.
- If terms differ across fields (for example "switching value", "tipping point" and "break-even"), map them to each other explicitly rather than treating them as different findings.

## 7. Constraints
- Use only this brief and public sources. Do not send any Olumi data or this brief's figures to third-party services beyond ordinary search queries.
- Do not call, test or probe any Olumi service or endpoint.
- Make no sign-ups or purchases. Where a source is paywalled, use legal open versions (arXiv, author pages, open-access copies); otherwise mark it "not accessed".
- Only run repository code inside an isolated sandbox, and never with credentials.

## 8. Deliverables
**`RESEARCH-FINDINGS.md`**, in this order:
1. **Answer.** At most 10 bullets, each tied to evidence IDs and a confidence level. Include the single most important thing Olumi does not currently know that it should.
2. **Per-question synthesis (RQ1–RQ9).** For each: the answer, confidence, evidence IDs, and what would change the answer.
3. **Superseding candidates.** For each: what it would replace, the evidence, maturity, and the risks.
4. **Tangential but valuable.**
5. **Checked and rejected,** with reasons.
6. **Evidence gaps:** the questions where no credible evidence exists.
7. **Proposed offline experiments** that could test the strongest candidates on Olumi's 12-model corpus without inventing data. For each: the hypothesis, the method, the pass/fail metric, the data needed, and an effort ESTIMATE.
8. **Search log:** date, source, query, hits screened and hits included.

**Evidence register (a table in the file, plus `evidence.json`).** One row per item with these fields: `id`, `authors`, `year`, `title`, `venue`, `doi_or_url`, `access_date`, `type`, `grade`, `flags`, `quote`, `location`, `demonstrated`, `limitations`, `research_questions`, `testable_on_corpus`.

**Repository register.** One row per repository, with the fields from the checklist in section 5, plus fit and risks.

## 9. Definition of done
Every research question has been answered with graded, verified evidence, or explicitly marked as an evidence gap. The breadth pass covers all four scope levels. The deliverables match section 8.

Olumi can then decide, on evidence, among:
- keeping the current plan;
- changing its representation, elicitation or information-value method;
- adopting a superseding approach.

---

### Appendix A: glossary
- **Stock:** a quantity that accumulates over time (for example, paying subscribers).
- **Flow:** a per-period change to a stock (for example, new subscribers per month).
- **Identity:** an exact relationship such as MRR = price × subscribers.
- **Normalised strength:** an edge weight rescaled from user units into roughly [−1, 1].
- **Natural effect:** an edge's effect stated in the user's units.
- **Template uncertainty:** a default spread applied whatever the evidence.
- **EVPPI:** the expected value of partial perfect information about one parameter.
- **Noise floor:** the threshold below which an EVPPI estimate is treated as indistinguishable from zero.
- **Horizon:** the deadline over which a goal is judged.

### Appendix B: non-binding search domains
This is an unverified starting list from the requester, for breadth only. It may be incomplete or wrong. It must not bound or steer the search, and nothing in it is a claim:
- system dynamics;
- fuzzy cognitive maps;
- structural causal models and dynamic causal models;
- customer-base and retention modelling for contractual subscriptions;
- price elasticity and churn;
- structured expert elicitation protocols;
- language-model priors and calibration;
- value-of-information estimation, especially in health economics;
- global sensitivity analysis;
- decision-making under deep uncertainty and scenario discovery;
- imprecise probability;
- probabilistic programming;
- unit and dimensional checking;
- automated generation of simulation models from text;
- risk communication.
