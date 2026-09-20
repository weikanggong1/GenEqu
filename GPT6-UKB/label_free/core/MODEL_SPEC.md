# Dictionary-only synthetic population, version 1

This generator constructs 201 baseline measurements in exact dictionary order. No participant observations, empirical summaries, diagnoses, target identities, existing generator, prediction results, manuscript, memory, or external source were read. Every distribution, coefficient, prevalence, covariance mechanism, narrower support, code interpretation and missingness probability is an authored modeling assumption. The dictionary supplies names, units, types, bounds and codes. This is a physiologically structured synthetic population with a UK Biobank-like schema, not a calibrated reconstruction of that cohort.

Requested model: gpt-6-astra; requested reasoning: ultra. Hidden backend identity and reasoning execution were not verified. These values record the request only.

## Run and outputs

    python generate.py \
      --schema measurement_dictionary.csv \
      --output-dir NEWDIR --n 300000 --seed 20260831

NEWDIR must not exist. Outputs are synthetic_features_float32.npy with shape (n,201), feature_columns.json with ordered string Field IDs, and generation_parameters.json with seed, requested model, numerical environment, input/source/output hashes, provenance and measurement rules. No identifiers, diagnoses, latent factors, future events, targets or auxiliary feature columns are exported. The dictionary's baseline family-context and longstanding-problem self-reports are retained; they are not disease labels. NaNs require downstream missing-value handling. The exact permitted dictionary SHA-256 is enforced.

    python test_generator.py

Tests use 16,384 rows and a 257-row CLI roundtrip. They do not run production generation, train a representation or evaluate predictions. Memory grows linearly with n; double-precision intermediates precede float32 export.

## Notation and randomness

Z denotes an independent standard normal for each named stream. Reusing a named factor reuses the same draw. σ(t)=1/(1+exp(−t)); clip(x,l,h)=min(max(x,l),h). TN(m,s;l,h) means a normal with mean parameter m and standard deviation parameter s, truncated to [l,h] by inverse-CDF sampling; m and s are not claimed moments after truncation. Bernoulli(p) uses an independent named uniform stream. Round is nearest integer with ties to even.

O(q;c1,…,ck) returns code c(1+j), where j counts thresholds crossed by q+0.7 Z. The k−1 fixed thresholds are evenly spaced from −1.5 to +1.5. They are never sample percentiles. Codes remain within the allowed set; some codes need not occur. No substantive measurement uses an independent uniform placeholder. Equal conditional probabilities among four nominal food exclusions are an explicit discrete-choice assumption.

Each named stream seeds NumPy PCG64 using the first16 bytes of SHA256(version|seed|stream_name), interpreted little-endian. Uniform draws are clipped to [10^−12,1−10^−12] before inverse Gaussian CDF transformation. All operations are rowwise, with no fitting, centering, ranking or covariance estimation from a batch. Same code, seed and library versions give identical row prefixes for different n. Cross-version bitwise identity is not asserted.

## Shared factors and covariance assumptions

All primitive factors below are independent standard normals unless specified. Age=TN(55,8;39,70), sex∼Bernoulli(0.48), a=(age−55)/10. Assumed sex coding is0=female,1=male. Social access and behavioral drive are hypothetical independent continuous propensities, not observed socioeconomic variables.

- activity = 0.35 access − 0.20 a + 0.90 Z
- diet = 0.35 access + 0.25 activity + 0.85 Z
- adip = −0.30 activity − 0.18 diet + 0.18 a + 0.90 Z
- inflammation = 0.25 adip + 0.90 Z
- muscle = 0.30 activity + 0.85 Z
- hydration, bone turnover and pigmentation are independent standard normals.
- employed∼Bernoulli(σ(3.1−0.13(age−45)+0.20 access))
- digital = −0.55 a + 0.35 access + 0.60 Z
- sleep_quality = 0.20 activity − 0.25 drive − 0.20 adip + 0.80 Z
- chronotype = −0.30 a + 0.70 Z
- appetite = 0.30 sex + 0.20 activity + 0.50 Z
- family_longevity = 0.35 access + 0.80 Z
- functional_burden = 0.50 a + 0.40 adip − 0.40 activity + 0.80 Z
- iron = 0.25 diet + 0.80 Z; red-cell turnover is an independent standard normal.
- filtration = clip(100−0.7(age−45)+10 Z−3 adip,45,140), an internal nominal clearance scale.
- glycemic = 0.30 adip + 0.13 a + 0.50 Z; long_term_glucose=5.2 exp(0.10 glycemic).
- Lipid transport, lipid storage, hepatic turnover, bilirubin clearance, growth axis, heritable LpA, sex-hormone axis, cycle phase and season are independent standard normals.

The201 rows in measurement_rules.csv give every field equation and missingness rule. These shared-factor equations and that table jointly specify all covariance mechanisms. No covariance matrix or marginal distribution is fitted.

## Behavioral and demographic equations

Smoking propensity=−0.35 access+0.25 drive+0.20 Z. Current smoking probability=σ(−2+propensity−0.20 a). Among noncurrent smokers, former probability=σ(−0.75+0.30 a+0.30 propensity). Cigarettes/day=clip(exp(2.1+0.45 Z),1,45). Duration=clip(age−(18+2 Z)−former(8+4σ(Z)),0,age−12). Pack-years=(current+former) × duration × cigarettes/day /20. Never smokers have zero pack-years.

Alcohol current probability=σ(1.5+0.2 access+0.3 drive); former probability among noncurrent users=0.35. Amount=current × clip(exp(1.7+0.4 drive+0.2 sex+0.45 Z),0.2,45) native beverage servings/week; alcohol_score=ln(1+amount)−1.6. This score is a synthetic intake propensity, not grams of ethanol. The beverage softmax scores, in red wine, white wine/champagne, beer/cider, spirits and fortified-wine order, are:

1. 0.3+0.25 access+0.4 Z
2. 0.2+0.3(1−sex)+0.4 Z
3. 0.3+0.7 sex+0.4 Z
4. −0.5+0.3 drive+0.4 Z
5. −1.5+0.2 a+0.4 Z.

Each native beverage amount is total amount times its share; fortified wine is rounded to match its count type. This is not an ethanol conservation identity. Exercise durations are zero when corresponding days are zero. Sleep plus driving, computer and television hours cannot exceed24 hours/day. Exercise durations concern active days and may overlap outdoor time; summer and winter outdoor time are not summed. Employment items are structurally zero when employed=0.

Parental current age is participant age plus a truncated age gap. An assumed lifetime determines whether the current-age or age-at-death field applies at baseline. Internal lifetimes construct dictionary-required family context only; no future endpoint is exported. Family-context binary=Bernoulli(0.9) × indicator(either modeled parent deceased), without a disease-specific cause.

The dictionary provides allowed codes but lacks full value labels. Chosen meanings and ordinal directions are marked as assumptions in the table. Device use, activity ranks, food exclusions, tanning, appearance, protection and some other items are not externally certified questionnaire coding. The one-code food-exclusion field models one exclusion or none, not simultaneous exclusions. All values and types obey the supplied dictionary.

## Body accounting

Weight and height determine BMI. Weight equals fat plus fat-free mass. Fat fraction is constrained jointly so whole-body fat≥5.1 kg, lean mass is36–87.5 kg and segment supports remain valid. These constraints are modeling assumptions, not quantile matching.

Define l=0.006 tanh(Z_leg), r=0.004 tanh(Z_arm), s=0.002 tanh(Z_asymmetry), f=0.010 tanh(Z_leg_fat), g=0.005 tanh(Z_arm_fat). Lean shares are left/right leg0.185+l±s, left/right arm0.060+r±s, trunk0.460+0.008 tanh(Z). Fat shares are left/right leg0.180+f±0.5s, left/right arm0.045+g±0.5s, trunk0.500+0.010 tanh(Z). The five shares sum to less than one, leaving positive unreported head/other fat and lean compartments.

Segment fat percentage=100 fat/(fat+lean). Predicted mass is assumed muscle-like tissue=0.96 lean, not total segment weight. Body water is a hydration-dependent lean fraction. Higher hydration reduces albumin and urea concentration through dilution. Impedance increases with squared body length and decreases with conductive water/lean mass, with common conductivity variation. Basal metabolic rate uses assumed scaling4.184(370+21.6 lean) kJ/day.

## Blood counts and physiology

RBC, MCV and MCHC determine MCH, haemoglobin and haematocrit algebraically. Platelet count and volume determine platelet crit. Reticulocyte and high-scatter percentages determine counts using RBC. Nucleated red-cell percentage uses WBC as its assumed denominator. Continuous laboratory values retain precision despite display-decimal recommendations; independently rounding them would break identities.

White-cell stick-breaking fractions are:

- n=0.35+0.43σ(0.3+0.45 inflammation−0.3 activity+0.55 Z)
- l=(1−n)[0.55+0.30σ(0.3 Z−0.1 a)]
- m=(1−n−l)[0.60+0.12σ(0.4 Z)]
- e=(1−n−l−m)[0.70+0.20σ(Z)]
- b=1−n−l−m−e.

These represent neutrophils, lymphocytes, monocytes, eosinophils and basophils. Counts sum to WBC, percentages to100. The lowercase notation here is local, distinct from body-share symbols.

Systolic exceeds diastolic by at least25 mmHg. FEV1 is a fraction of FVC. The ratio Z score uses the model's age reference and assumed SD0.065, not a verified clinical reference equation. Total cholesterol=LDL+HDL+triglycerides/2.2 is idealized assumed accounting, not a universal exact physiological identity. Albumin contributes to protein and calcium; filtration affects creatinine, cystatin C, urea and urate; shared glycemic regulation affects glucose and HbA1c; hepatic factors affect enzyme activities.

Hormones depend on sex and age, including a reproductive transition whose probability varies smoothly with age. No disease state is generated. Three lower dictionary bounds are interpreted as technical reporting limits: oestradiol175 pmol/L, rheumatoid factor10 IU/mL and vitamin D10 nmol/L. Lower latent values are NaN. This reporting-limit interpretation is an assumption, not a verified laboratory specification.

## Missingness and validation

Age and sex are complete. Structural parental applicability and assay censoring apply even with random missingness disabled. Other prespecified missingness:

- Shared blood sample absence2.5%; independent laboratory-result absence0.8%.
- Shared physical-visit absence0.5%; shared impedance-session absence1.8%; independent other physical-item absence0.6%.
- Shared questionnaire absence1%; per-item probability0.015+0.01σ(−access).
- Additional independent nonresponse8% for each sexual-history item.

No probability is estimated from observations. Missingness does not alter latent values. Identities hold where the required observations are present, within float32 rounding tolerance.

Tests verify schema/order, values/bounds, field variability, identities, positive mass residuals, inequalities, applicability, broad prespecified shared-factor correlation directions, deterministic prefixes, CLI reload and hashes. These tests establish mathematical/software consistency of the assumed population. They do not establish cohort realism, clinical validity, representation quality or prediction performance.
