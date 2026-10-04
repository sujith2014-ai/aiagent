# Findings (including negative and engineering results)

## F1 (Phase 3): "no forgetting" is by construction, not a result
Independent modules with frozen parameters cannot interfere. The modular system's 0.000 forgetting drop is therefore uninformative about the hypothesis; the informative comparison is the naive sequential shared-MLP baseline, which collapsed (compare_numbers 1.000 -> 0.038 after learning the next task). A joint-trained shared MLP with fewer parameters (2,932 vs 3,817 total) matched the modular accuracies on all three single tasks. **On non-compositional tasks the modular system shows no accuracy advantage over a conventional tiny network; its advantage is incremental acquisition without retraining and portable packaging.**

## F2 (Phase 4): composition works on the tested tasks, and beats end-to-end baselines where the task is compositional
With human-authored plans, independently trained modules composed reliably: measured composite accuracy matched what independent module errors predict (count_inside 0.957 measured vs 0.968 predicted; mixed 0.987 vs 0.980; A->B->A->C 0.993 vs 0.980), i.e. no sign of representation mismatch or wiring errors. End-to-end MLPs given 3,000 examples were far worse on count_inside (0.57), mixed (0.83), path_ABAC (0.86), equal on sort4 (1.00; the modular version reuses a 1,251-parameter module versus a 6,040-parameter MLP).
Caveats that limit this claim: (a) the **plans are hand-written**; whether a system can *learn or obtain* plans is untested (Phase 5-6 teacher plans, Phase 7 learned routing); (b) modules received per-module labelled supervision from the teacher, the baseline only end-to-end labels, so the comparison measures the value of decomposition plus intermediate supervision, not only architecture; (c) one baseline configuration (MLP 64x64, 3,000 examples, 150 epochs); a stronger baseline (more data/tuning) might close the gap; (d) compare_numbers has a finite 400-pair domain, which all of train/val/test cover, so composites over it test composition rather than generalization of that module.

## F3 (Phase 4): engineering bug found by the overhead benchmark
First measurement: 16-node sort plan overhead 1,443 us vs 156 us of module compute. Cause: `Registry::record_call` rewrote `registry.json` on every inference. Fix: buffer usage statistics in memory and flush on command end/drop (structural changes still persist immediately). After the fix: overhead 30-55 us for the same plan (26x lower). Trade-off: usage statistics can be lost on a crash.

## F4 (Phase 4): latency cost of modularity
Through the same ONNX backend a monolithic end-to-end MLP takes 7-9 us per task; the modular plans take 66-78 us (about 9x), of which module compute is roughly half. Intent routing adds ~10-25 us per plan over id routing. Absolute numbers are small (<150 us), but the modular design is slower than a monolithic network wherever a monolithic network is accurate enough. Energy was not measured.

## F5 (Phase 4): scheduling
Independent branches are identified (`parallel_levels`) but executed sequentially; no parallel speedup is claimed or measured.

## F6 (Phase 5): what the detection signals do and do not catch
Ablation on 1,500 labelled cases (benchmarks/reports/phase5.*):
- **Input-novelty (training-range stats shipped in the package) removes out-of-domain fabrication**: in-scope intent with OOD inputs was answered KNOWN 100% of the time by the keyword router alone and 0% with the novelty signal.
- **Keyword routing is the weak point.** Out-of-scope tasks that share words with a capability were answered KNOWN 6.7% (generic out-of-scope set) and **35% (adversarial keyword overlap set)**, e.g. "compare prices of two shops" routed to compare_numbers. None of the numeric signals can catch these because their inputs are in-distribution. Fixing this needs a better routing signal (learned router, evaluator prediction, or teacher verification), not more thresholds. Not solved.
- **Paraphrase brittleness**: 25% of in-scope paraphrases had no keyword overlap and were sent to the teacher as unknown. In the task stream 2 of 12 related re-encounters re-triggered the teacher (and the simulator could not help), so teacher dependency on related tasks was 17%, not 0%.
- **Calibration**: temperature scaling reduced held-out ECE for all three capabilities (e.g. compare_numbers 0.058 -> 0.006), but two fits landed on the lower grid edge (T=0.05): these models are accuracy-saturated and early-stopped on accuracy, so their raw logits are under-confident. At a fixed 0.7 threshold, calibrated confidence flagged fewer boundary cases (5%) than raw confidence (15%), giving 4.9% vs 2.4% error among accepted answers at 94.7% vs 85% coverage. These are different operating points, not a clear win; a risk-coverage curve was not computed.
- Confidence gating helps only inside the learned domain (boundary points); it did nothing for unknown tasks.

## F7 (Phase 5): escalation-loop behaviour
Worked as designed: first encounters 3/3 resolved via teacher -> LearningPackage -> new module -> retry; related re-encounters mostly local; hostile or malformed teacher output (non-JSON, unknown action, bad package, reroute to nonexistent capability) activated nothing (4/4); offline unknown tasks were queued without contacting the teacher while known capabilities kept working; research/tool requests are reported as NEEDS_EXTERNAL (OpenClaw does not exist yet) instead of being faked. A bug found in my teacher simulator during the run ("send email to my colleague" classified as a memory update because of the word "my") was fixed; it also shows that nothing downstream validates the teacher's *choice of action* beyond schema checks.

## F8 (Phase 5): information classification is not solved
The rule baseline scores 20/20 on the set written alongside it (meaningless) and 5/10 on a harder set ("The sort function in this library is stable" -> CAPABILITY, "Improve at spotting fraudulent invoices" -> KNOWLEDGE, ...). It is only a placeholder gate; the schema still blocks neural training for MEMORY/KNOWLEDGE labelled packages, but a wrong label would bypass it.

## F9 (Phase 5): privacy filter limits
Regex-based redaction (emails, key=value secrets, token-like strings, paths, long numbers) plus whitelisted request fields; raw input values are never sent. Unknown secret formats or free-text personal data in an intent would pass through. The model never sees credentials only if they are not typed into the intent.
