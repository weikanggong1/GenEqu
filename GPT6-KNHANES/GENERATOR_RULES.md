# GPT6-KNHANES: independently authored survey-population construction

## Scope and provenance

This generator represents South Korean adults aged 40–70 at a current survey measurement visit. Its binary target is a **self-reported history of physician diagnosis**, with the exact identities supplied in `disease_identities.csv`. Phenotype strings are stable task identifiers; their spelling does not override the Korean question. This is a knowledge-based synthetic construction, not a reconstruction of a sampled Korean cohort.

The only pre-existing project content used in authoring was `CONSTRUCTION_CONTRACT.md`, `allowed_schema.csv`, `disease_identities.csv`, and `schema_provenance.json` in this directory. No participant data, summary statistics, empirical missingness, disease counts, existing generators, classifier results, external retrieval, or memory informed the parameters. No citations are asserted. All numerical values—including intercepts, slopes, distribution shapes, diagnosis rates, treatment probabilities, error scales, supports and missingness probabilities—are independent construction choices. They are not measured Korean population estimates. Sex-specific stature and smoking patterns and a relatively lean adult BMI baseline are qualitative population choices, not estimates of contemporary Korean population proportions.

The recorded model configuration is **requested** `gpt-6-astra`, `ultra`, context fork `none`; it is not a verified backend build identity. The executable contains no fitting, feature selection, outcome balancing, performance objective or evaluation procedure. The test suite checks implementation and physiological consistency only.

## Exact targets

| Task identifier | Clinical history represented | Deliberate semantic boundary |
|---|---|---|
| I9_HYPTENSESS | DI1_dg physician-diagnosed hypertension | Secondary versus essential cause is not resolved. |
| E4_HYPERCHOL | DI2_dg physician-diagnosed dyslipidaemia | Includes high-LDL, high-triglyceride and mixed histories. |
| E4_DM2 | DE1_dg physician-diagnosed diabetes | Type is unspecified; latent susceptibility includes heterogeneous causes. |
| N14_CHRONKIDNEYDIS | DN1_dg physician-diagnosed kidney disease | Includes resolved episodes; neither chronicity nor CKD stage is assigned. |
| J10_ASTHMA_MAIN_EXMORE | DJ4_dg physician-diagnosed asthma | No main-diagnosis restriction or expanded exclusions. |
| I9_STR_SAH | DI3_dg physician-diagnosed stroke | No stroke subtype is assigned. |
| I9_ISCHHEART | DI4_dg physician-diagnosed myocardial infarction or angina | Represents the combined stated question, not all ischaemic heart disease. |
| M13_OSTEOPOROSIS | DM4_dg physician-diagnosed osteoporosis | Both sexes; no BMD or DXA cutoff generates the label. |
| G6_SLEEPAPNO | BP17_dg obstructive sleep apnoea diagnosed through sleep testing | A separate latent testing opportunity is required for diagnosis/report. |
| COPD_MODE | HE_PFTdr physician-diagnosed COPD, including chronic bronchitis/emphysema | No spirometric obstruction threshold or UKB ICD-mode rule. |

Every synthetic participant has latent histories for all ten conditions. The selected task determines which reported history is emitted. Non-cases may have undiagnosed disease, resolved disease, forgotten diagnoses, or other diagnoses. Cases may have managed measurements, inactive disease, or diagnostic/report uncertainty. The question-specific observation years are uniformly sampled among that target's permitted years solely for audit counts: 2022–2024 for the first five targets, 2024 for stroke/MI-or-angina/osteoporosis/COPD, and 2022–2023 for sleep apnoea. No year-specific empirical effects are claimed or applied.

## Executable dependency structure

1. Draw age, sex, household, social/access variables, premorbid lifestyle and physiology.
2. Draw correlated disease susceptibilities and lifetime histories from those prediagnosis variables.
3. Draw duration, activity/severity, clinical recognition, management/adherence, and recalled diagnosis.
4. Generate current physiology and behavior from shared baseline traits, active histories and management; add measurement variation.
5. Apply questionnaire/visit/assay missingness and documented censoring. Append only the requested binary reported target.

No measured laboratory value determines the target. The label is sampled through clinical history and ascertainment before current laboratory measurements exist. Effects from multiple active conditions add at the process level. Management is driven by latent clinical recognition, not by the final reported label. Disease effects are therefore not a deterministic case/non-case feature shift. Shared continuous susceptibility, incomplete recognition, disease activity, imperfect adherence and assay variation preserve physiological heterogeneity within either outcome group. No diagnostic, treatment, latent, probability, year or identifier column is exported as a predictor.

### Population and shared traits

All standard-normal terms are independent unless the equation explicitly shares a variable. Let `A=(age−55)/10`, `B=(premorbid BMI−24)/3`, and `S` be sex (1 male, 0 female). The executable is the exact specification, including draw order and clipping.

- Age is discrete uniform 40–70 and male probability is 0.49. Household size is `min(1+Poisson(exp(0.52−0.16 A)),6)`; six denotes six or more.
- Deprivation and inherited background are independent standard normals. Access is `0.80 N−0.30 deprivation+0.10 A`. Menopause is a latent Bernoulli with probability `sigmoid((age−50)/2.4)` in women.
- Lifetime cigarette exposure has probability `sigmoid(−1.90+2.25 S+0.14 A+0.25 deprivation)`. Among ever smokers, current smoking has probability `sigmoid(0.30−0.44 A+0.18 deprivation)`. Latent cumulative smoke burden is `clip(0.65+0.38 A+0.42 Gamma(1.8,0.6),0.2,3.5)` for ever smokers and zero otherwise. It is not a measured pack-year estimate.
- Ever drinking has probability `sigmoid(1.25+0.50 S−0.12 A)`; current drinking among ever drinkers has probability `sigmoid(1.70−0.33 A)`. Current drinking intensity is `Gamma(1.3,0.65)`; only never/previous/current is exported.
- Fitness is `N−0.18 deprivation−0.17 A`. Stature is `158.8+12.2 S−0.105(age−50)+5.1 N` cm. Premorbid BMI is `clip(23.7+0.65 S+0.34 A+0.45 deprivation−0.44 fitness+2.6 N,16,39)`.
- Insulin susceptibility is `0.32 B+0.18 inherited+0.92 N`; lipid susceptibility is `0.22 inherited+0.22 alcohol_intensity+0.90 N`; vascular susceptibility is `0.22 inherited+0.18 deprivation+0.16 smoke_burden+0.90 N`; renal susceptibility is `0.25 inherited+0.15 vascular+0.90 N`.
- Atopy, bone fragility, airway anatomy and other lung susceptibility are independent normal traits. A latent alternative diabetes-cause indicator is Bernoulli(0.025). It broadens the susceptibility model without assigning observed diabetes type.

### Disease history, recognition and current burden

For each disease, `history ~ Bernoulli(sigmoid(intercept + sum(risk_coefficient × named_trait)))`. `DISEASES` in the executable explicitly lists every risk coefficient, activity probability, treatment propensity and process effect; there are no inferred or fitted parameters. The automatically enumerated parameter tables below reproduce these constants for review.

Severity is `0.35+1.25 Beta(2,2.8)`; duration is `min(Gamma(1.6,4),age−18)` years. Current activity is an independent Bernoulli with the disease-specific probability, gated by history. Current burden is `history × active × severity × (0.82+0.18(1−exp(−duration/6)))`. Kidney burden has an additional 1.8 multiplier for a latent 8.5% severe-tail draw. Its 0.69 activity probability permits normal current renal measurements after a past episode. These represent mixtures at the visit, not follow-up trajectories or disease-stage estimates.

Clinical ascertainment probability is `sigmoid(ascertain_intercept+0.58 access+0.70(severity−0.8)+0.10 A+0.10 log(1+duration))`. A 0.001 false/suspected diagnosis probability applies to persons without the latent history. For obstructive sleep apnoea, recognition also requires a testing opportunity with probability `sigmoid(−0.85+0.70 access+0.55(severity−0.8))`. This is a generative proxy for testing access, not a measured test utilization rate.

Treatment is Bernoulli(`sigmoid(logit(treat)+0.28 access)`) conditional on diagnosis. Adherence is `0.35+0.65 Beta(3,1.8)`; management effect is diagnosis × treatment × adherence. Reported recall probability is `sigmoid(3.55−0.055 duration+0.25 active)`. A 0.0006 false recalled diagnosis probability applies when not diagnosed; sleep-apnoea reports remain gated by the testing opportunity. Thus false clinical recognition, forgotten diagnosis, and false reporting are distinct latent mechanisms. The model does not claim their rates are validated.

Dyslipidaemia subtypes are independently assigned high-LDL/high-TG/mixed with probabilities 0.42/0.25/0.33. The high-TG subtype multiplies the active LDL effect by 0.15; the high-LDL subtype multiplies active log-TG and HDL effects by 0.20. Other subtype/process multipliers are one. Management effects apply to the broad diagnosed condition. This implements the broader dyslipidaemia question explicitly.

Among current smokers, recognized diagnoses induce cessation with probability `1−exp(−0.20 cardiopulmonary_diagnosis_count−0.035 all_diagnosis_count)`; the cardiopulmonary set is stroke, MI/angina, asthma and COPD. Remaining current smokers are daily with probability 0.86 and occasional otherwise. Among current drinkers, cessation probability is `1−exp(−0.045 all_diagnosis_count)`. A reported diagnosis cannot itself trigger these changes.

## Measurement equations and units

All output values are already in the shared units. No KNHANES mg/dL conversion is applied to values constructed in mmol/L or µmol/L. Creatinine, urate, glucose, cholesterol and urea are constructed directly in the requested harmonized units. CRP is **mg/L** and never multiplied by ten. HbA1c alone is explicitly constructed on an NGSP-percentage intermediate, then transformed once by `(NGSP−2.152)/0.09148`. The water value uses the conventional numerical approximation 1 litre≈1 kg; this is not an exact physical-density identity.

Here `P_process` denotes the accumulated active-disease and management effects in `DISEASES`. All noise scales, min/max clips, and distribution parameters appear explicitly in `generate_population`; there is no generic normal fallback for unspecified columns.

| Exported fields | Construction and dependence |
|---|---|
| 21022, 31, 709 | Age, sex and top-coded household as above. |
| 50 | `clip(stature+P_height+0.15 N,140,192)` cm. |
| 21001, 21002 | Current BMI=`clip(premorbid BMI+P_bmi+0.30 N,15.5,42)`; measured weight= BMI×(height/100)² kg. |
| 48 | Waist=`clip(76+6 S+2.20(BMI−23)+0.08(age−50)+3.3 N,55,147)` cm. |
| 23099, 23100, 23101 | Body fat %= `clip(26−7.2 S+1.15(BMI−23)+0.085(age−50)+2.6 N,8,53)`; fat=weight×percentage/100; fat-free mass=weight−fat. |
| 23116, 23112, 23124, 23120, 23128 | Left/right leg, left/right arm, trunk fractions start at `(0.175−0.012S,0.175−0.012S,0.063+0.002S,0.063+0.002S,0.524+0.020S)`, multiply independent `exp(N(0,0.075))`, then normalize. Their sum is total fat×`(0.955+0.025 Beta(2,2))`; residual mass represents unreported regions and segmentation. |
| 23102 | Water=fat-free mass×`clip(0.735+0.007 N−0.00006 P_gfr,0.70,0.77)` kg-equivalent. |
| 864, 874 | Mobility=fitness+P_mobility. Days=`Binomial(7,sigmoid(0.40+0.50 mobility−0.14 B))`; duration is zero for zero days, otherwise rounded `clip(exp(3.35+0.20 mobility+0.57 N),10,240)` minutes per walking day. No multiplication by days. |
| 20116, 1239, 20117 | Current post-recognition smoking/tobacco and alcohol categories as described above; conservative former-smoking mapping is applied during missingness. |
| 4079, 4080 | DBP=`clip(74+0.13(age−50)+2.1 B+3.6 vascular+P_dbp+5 N,45,126)`; raw SBP=`116+0.49(age−50)+3.1 B+5 vascular+P_sbp+7.2 N`; final SBP=DBP+`clip(raw SBP−DBP,22,112)` mmHg. Noise represents the final survey reading rather than three exported readings. |
| 102 | Pulse=`clip(71−2.6 mobility+1.1 current_smoking+P_pulse+7 N,43,128)` bpm. An authored 85% measurement-protocol draw quantizes to even bpm (30-second count×2); the remainder quantizes to integer bpm (60-second count). |
| 30010 | RBC=`clip(4.42+0.48 S−0.035 A−0.48 iron−0.006 max(60−GFR,0)+P_rbc+0.29 N,2.4,6.4)` in 10¹²/L. |
| 30040, 30060 | MCV=`clip(89.5+1.5 alcohol−10 iron+3.2 N,62,112)` fL; MCHC=`clip(33.6−0.65 iron+0.63 N,29,37)` g/dL. Iron deficit has probability `sigmoid(−3+0.65 female×not_menopausal)` and magnitude Uniform(0.3,1). |
| 30030, 30020, 30050 | Haematocrit=RBC×MCV/10 %; haemoglobin=haematocrit×MCHC/100 g/dL; MCH=MCV×MCHC/100 pg. Shared noisy RBC/MCV/MCHC primitives preserve standard count/mass identities exactly before CSV precision. |
| 30080, 30000 | Platelets=`clip(exp(log(246)−0.025 A+0.075 inflammation+0.12 iron+0.22 N),65,780)`; WBC=`clip(exp(log(5.9)+0.09 current_smoking+0.12 inflammation+0.22 N),2,22)` in 10⁹/L. |
| 30620, 30650 | ALT=`clip(exp(log(21)+0.14(BMI−24)/3+0.13 alcohol+0.11 inflammation+liver+0.34 N),0.5,380)` U/L; AST=`clip(exp(log(23)+0.055(BMI−24)/3+0.10 alcohol+0.07 inflammation+0.70 liver+0.23 N),5,300)` U/L; shared liver noise=`0.30 N`. |
| 30710 | CRP=`clip(exp(log(0.85)+0.25(BMI−24)/3+0.12 A+0.50 inflammation+0.74 N),0.015,90)` mg/L. Inflammation=`max(−1.2,0.18 B+0.17 current_smoking+0.65 N+P_inflammation)`. |
| 30740, 30750 | Long-term glucose=`clip(5.20+0.24 B+0.12 A+0.32 insulin+P_glucose,3.2,19)` mmol/L. Visit glucose adds `0.38 N` and Uniform(0.3,1.5) if short fasting, then clips 2.8–24. NGSP HbA1c=`clip(5.35+0.47(long-term glucose−5.2)+0.11 iron−0.002 max(45−GFR,0)+0.20 N,3.5,14)`; transform once to mmol/mol. |
| 30780, 30760 | LDL=`clip(3.05+0.22 B+0.13 A+0.43 lipid+P_ldl+0.27 N,0.35,8)`; HDL=`clip(1.42−0.13 S−0.10 B+0.035 alcohol+P_hdl+0.18 N,0.40,3)` mmol/L. |
| 30870, 30690 | TG=`clip(exp(log(1.25)+0.21 B+0.13 insulin+0.13 alcohol+0.13 lipid+P_log_tg+0.34 N+0.15 short_fast),0.25,12)` mmol/L. Total cholesterol=LDL+HDL+remnant, with remnant=`max(0.06,0.43 min(TG,4.5)+0.12 max(TG−4.5,0)+0.07 N)`. The noisy nonnegative remnant represents component consistency; this is not a Friedewald-generated LDL cutoff or label. |
| 30700 | Creatinine=`clip(78(100/GFR)^0.85(0.90+0.14S)(FFM/(44+13S))^0.30 exp(0.10 N),28,800)` µmol/L. Latent filtration=`clip(99−0.45(age−50)−6 renal+P_gfr+6 N,12,135)`; no estimated GFR is exported or used as a case criterion. |
| 30880 | Urate=`clip(290+48 S+19 B+0.85(95−GFR)+9 alcohol+P_urate+39 N,105,790)` µmol/L. |
| 30670 | Urea=`clip((4.7+0.039(95−GFR)+0.16 S)exp(0.18 N),1.5,26)` mmol/L. These are urea units, not an unconverted BUN value. |

The algebraic mass/count identities intentionally describe harmonized measured constructs using shared noisy primitives. They do not claim every real instrument's separately reported values will satisfy exact equality. CSV uses eight significant digits; tests allow numerical rounding. Derived measurement supports and clipping values are engineering/physiology choices, not empirically observed hard bounds. Disease risk uses premorbid BMI, while body composition and liver/CRP equations use the current managed BMI. This avoids a circular disease-label/measurement calculation.

## Missingness and censoring

Missingness has a separate stable random stream. `apply_missingness` receives only predictors, age, access, deprivation and a fasting-process flag; it never receives disease probabilities, history, clinical diagnosis, treatment or target. Its shared demographic/access causes can produce ordinary indirect associations with diagnosis. It contains no outcome-specific missingness rates. Age and sex remain observed. The target is complete by construction.

- A missed visit has probability `sigmoid(−4+0.16 A−0.30 access+0.20 deprivation)`.
- Blood-panel absence is missed visit OR Bernoulli(`sigmoid(−3.7−0.25 access+0.13 deprivation)`). Every blood field additionally has an independent 0.004 assay failure probability.
- Anthropometry absence is missed visit OR Bernoulli(0.009). Height, weight, BMI and waist share this mask.
- Vital-sign absence is missed visit OR Bernoulli(0.012); BP and pulse share it.
- BIA absence is anthropometry absence OR Bernoulli(`sigmoid(−2.7+0.12 A−0.12 access)`). All nine BIA fields share it. This is a visit/participation process covering absent measurements, not a fitted implant/pregnancy exclusion classifier.
- Direct LDL has an additional absence probability `0.10+0.06 short_fast`; HbA1c has an additional 0.025 probability. Short fasting is independently Bernoulli(0.04) and also affects visit glucose/TG. Household missingness is 0.008.
- Both smoking questions share a refusal/mapping mask with probability `sigmoid(−4−0.15 access)`. Among observed previous smokers, an independent 0.035 fraction has a latent low-lifetime-cigarette mapping ambiguity: smoking-status becomes missing, but current tobacco remains zero. Current smoking is never mislabeled as previous or never.
- Alcohol missingness is `sigmoid(−4.1−0.15 access)`. Walking days and duration share `sigmoid(−3.7−0.15 access+0.08 A)`, preserving zero-day/zero-duration coherence.
- Before ordinary missingness, latent ALT below 5 is replaced by `5/sqrt(2)` U/L; CRP below 0.2 is replaced by `0.2/sqrt(2)` mg/L. These are censor replacements rather than missing values. Audit JSON contains pre-missingness counts, observed replacement counts and both counts by latent survey year. Other measurements are not assigned undocumented censoring rules.

## Exhaustive disease-feature rules

`feature_disease_rules.csv` has exactly 440 rows, one per disease and predictor, with unit, relationship category, direct process coefficients, shared susceptibility, and a prose rule. `direct` means an explicit disease/management process modifies that measurement or its physiological primitive; `indirect` means shared risk structure or recognition-mediated behavior links them; `neutral` means no disease-specific term is assigned after the listed shared background variables. Neutral is conditional, not a claim of zero marginal correlation. Correlated traits can create additional weak associations beyond a row's listed primary paths. No coefficient in this file was chosen by measured predictive performance.

## Interface, stable streams and audit

```
python3 generate_knhanes.py --schema allowed_schema.csv --disease-identities disease_identities.csv --output-dir OUTPUT --n 360000 --seed 20260831
python3 generate_knhanes.py --schema allowed_schema.csv --disease-identities disease_identities.csv --output-dir OUTPUT --n 360000 --seed 20260831 --diseases E4_DM2 G6_SLEEPAPNO
```

Each task writes `OUTPUT/<phenotype>/synthetic_<phenotype>_<n>.csv`, with the 44 columns in supplied schema order and one binary column named exactly as the phenotype. The executable validates the exact field set, types, units, category strings and disease target fields/years. It does not search for other files or datasets. It reads the four supplied inputs and four authored deliverables for content/hashes; task outputs are explicit command arguments. Audit files include counts, input/code/rules/tests/output SHA256 values and NumPy/pandas versions separately from the predictors.

Generation uses fixed 10,000-row blocks, releases each frame after writing, and processes tasks sequentially. Each RNG is `default_rng(SeedSequence([seed,block_index]+four_uint32_words_from_SHA256(phenotype+':'+purpose)))`. Physiology and missingness purposes are separate. Disease-list order, subset selection and Python hash randomization do not alter a task. There is no global NumPy random state. Byte-identical CSV output is expected for an unchanged generator and identical NumPy/pandas runtime; hashes/version metadata make runtime changes visible. A partial last block uses its requested size; prefix identity across different total sample sizes is not promised, because vectorized draw lengths differ.

The in-memory interface is `generate_population(n, phenotype, seed=20260831, block_index=0, fields=FIELDS, with_audit=False)`, with `1 <= n <= 10000`. It returns a DataFrame; `with_audit=True` returns `(DataFrame,audit_dict)`. Fixed-block generation bounds memory independently of the 360,000-row task size. Output publication reserves an exclusive `.csv.part` and uses a no-overwrite hard link plus exclusive audit creation. Existing CSV, audit or partial files are refused. Handled failures remove only files this invocation created; a hard process kill can leave a partial/output requiring explicit operator inspection, and a later invocation will refuse to overwrite it. This publication scheme targets the specified Linux/WSL environment.

The downstream fixed, label-independent permutation produces 300,000 training and 60,000 validation rows. Splitting, standardization, feature selection, classifier fitting, validation selection, calibration and thresholds are outside this generator. This code neither performs nor tunes that downstream procedure.

## Synthetic-only verification and freeze

```
python3 test_generator.py --report synthetic_test_report.json
```

The default run checks all ten diseases, all 44 fields, all 440 rule pairs, binary outcomes, categorical and questionnaire coherence, physiological identities, units/censoring, deterministic streams, subset/order/Python-hash invariance, label-invariant measurement masks, input read boundaries, and refusal to overwrite. An instrumented subprocess traps file-content reads outside the four supplied inputs, four authored deliverables and its explicit generated-output directory. An import/call allowlist additionally checks the generator has no dynamic code loading or external-service imports. These are auditable input-handling checks, not OS isolation.

The final test writes and rereads a full 360,000-row synthetic task with 10,000-row generation blocks and 20,000-row read chunks. It checks output hashes, exact counts/missing counts and physiological relationships across all chunks. In a fresh Linux subprocess, generator peak RSS must remain below 512 MiB; the entire test/read process must remain below 768 MiB. This is a resource acceptance check, not predictive-performance tuning. Temporary test populations are deleted; the JSON report retains hashes, byte count, row count and memory evidence.

`--skip-full-scale` produces an explicitly incomplete smoke-test status and cannot establish full acceptance. At initial authoring, local NumPy/pandas were unavailable; executable tests were therefore not claimed to have run locally. The coordinating task must run the default test in the stated scientific Python environment, resolve implementation errors only, and freeze all four input files, generator, rules CSV, this document, test source and actual test evidence **before** any final population generation or real-data evaluation. The test report captures the exact source hashes it tested. A generated test report alone is not a freeze manifest.

## Disease parameter tables

The following appended tables are rendered directly from the authored constants and are descriptive documentation, not runtime inputs.

| Target | Liability intercept | Liability coefficients | Ascertain intercept | Active probability | Treatment baseline |
|---|---:|---|---:|---:|---:|
| I9_HYPTENSESS | -1.45 | age +0.65, bmi +0.65, vascular +0.9, renal +0.2 | -0.1 | 0.95 | 0.76 |
| E4_HYPERCHOL | -1.4 | age +0.32, bmi +0.48, lipid +0.95, insulin +0.35 | -0.25 | 0.93 | 0.64 |
| E4_DM2 | -2.45 | age +0.55, bmi +0.7, insulin +1.05, diabetes_other +1.5 | 0 | 0.94 | 0.76 |
| N14_CHRONKIDNEYDIS | -3.6 | age +0.45, renal +1.1, vascular +0.3, insulin +0.25 | -0.55 | 0.69 | 0.64 |
| J10_ASTHMA_MAIN_EXMORE | -2.85 | atopy +1.15, bmi +0.22, female +0.18, smoke +0.12 | 0.05 | 0.63 | 0.63 |
| I9_STR_SAH | -4.05 | age +0.95, vascular +0.7, smoke +0.4, insulin +0.3 | 1.55 | 0.74 | 0.81 |
| I9_ISCHHEART | -3.55 | age +0.8, vascular +0.6, lipid +0.5, smoke +0.5, male +0.35 | 1 | 0.81 | 0.85 |
| M13_OSTEOPOROSIS | -2.65 | age +0.75, female +0.85, menopause +0.8, bmi -0.55, bone +0.85 | -0.65 | 0.96 | 0.56 |
| G6_SLEEPAPNO | -2.15 | age +0.22, bmi +0.9, male +0.65, airway +1 | 0.2 | 0.9 | 0.52 |
| COPD_MODE | -3.45 | age +0.72, smoke +1, lung +0.85, male +0.22 | -0.6 | 0.96 | 0.64 |

| Target | Active-burden process coefficients | Management/adherence process coefficients |
|---|---|---|
| I9_HYPTENSESS | sbp +23, dbp +11, gfr -3 | sbp -18, dbp -8, pulse -2.5, urate +8 |
| E4_HYPERCHOL | ldl +0.8, log_tg +0.43, hdl -0.1 | ldl -0.95, log_tg -0.22, hdl +0.025 |
| E4_DM2 | glucose +3.8, log_tg +0.19, hdl -0.07, gfr -4 | glucose -2.4, bmi -0.3, ldl -0.2 |
| N14_CHRONKIDNEYDIS | gfr -33, sbp +5, inflammation +0.25, rbc -0.16 | sbp -4, gfr +3 |
| J10_ASTHMA_MAIN_EXMORE | inflammation +0.38, pulse +2, mobility -0.3 | inflammation -0.2, pulse +1, bmi +0.1 |
| I9_STR_SAH | mobility -1, inflammation +0.15 | sbp -5, ldl -0.38 |
| I9_ISCHHEART | mobility -0.65, inflammation +0.2, pulse +1 | sbp -4, ldl -0.5, pulse -6 |
| M13_OSTEOPOROSIS | mobility -0.22, height -0.3 | none |
| G6_SLEEPAPNO | sbp +5, dbp +3, inflammation +0.25, mobility -0.22 | sbp -2, dbp -1, mobility +0.12 |
| COPD_MODE | inflammation +0.5, pulse +3, mobility -0.6, bmi -1.05 | inflammation -0.13, pulse +1.2 |

Process units are mmHg for sbp/dbp, bpm for pulse, kg/m² for bmi, cm for height, mmol/L for glucose/LDL/HDL, natural log units for log_tg, µmol/L for urate, 10¹²/L for rbc, and dimensionless latent units for inflammation/mobility. The gfr process uses a latent filtration scale numerically comparable to mL/min/1.73m²; it is neither an exported estimate nor a diagnostic rule. Multiple process pathways into the same measurement are evaluated together by the executable.
