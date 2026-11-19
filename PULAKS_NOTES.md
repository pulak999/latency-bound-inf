# tracker

[x] = done, [ ] = still to do. issues are big stuff i can hand to people, prs are small fixes i push myself.
when something lands, tick it and add the date. full drafts live in ~/Desktop/gh-drafts/.

## prs (small fixes -- local branches off deadline-harness, not pushed yet)
- [ ] `record-scheduled-arrival` -- feeder now logs when a frame was *due*, not just when it fired. before this, mailbox cells threw the lateness away unless it was > 10% of T (2 s at T=20)!
- [ ] `split-frame-counts` -- `frames_delivered` counted overwritten frames too (said 15 delivered when 6 were served). now: served / overwritten / error.
- [ ] `probe-separate-passes` -- probe was putting a 4-token decode before every timed request. gate-2 latency spread was measured like that.
- [ ] `arch-doc-controller` -- ARCH.md still said runner.py is the main path. now describes controller + mailbox.

all four merge together cleanly, tests + ruff pass. push with `gh pr create --base deadline-harness` (bodies in gh-drafts/prs/).

## issues (arch-level / new stuff -- on github, assign people)

gate 3 (alone cells) -- do these first:
- [ ] #3 decide how deadline-harness and main merge -- **me**. blocks #4, #6, #12, #13.
- [ ] #11 when is the GPU quiet enough for an alone cell? -- most urgent. in gate 2 a labmate's job hit 9 of 50 requests and nothing flagged it.
- [ ] #4 port main's latency breakdown (stages, slow tail, GPU clock) into deadline-harness -- after #3.
- [ ] #5 say *why* each missed frame missed (overwritten / queued / model was slow + which part) -- after #4.
- [ ] #7 find where stack A's ~180 ms outside the engine goes -- 11% of latency, stack B only loses 0.3 ms. so it's llama-server, not our asyncio.
- [ ] #8 does the harness slow down what it measures? (sampler thread, nvidia-smi per request, feeder lateness under load) -- wait for review round 2.

gate 4 (co-runners):
- [ ] #12 co-runners light + heavy, as in DESIGN.md -- after #3.

gate 5 (power mode):
- [ ] #10 power modes + energy per frame -- blocked on sudo. wait for review round 4.

no gate yet:
- [ ] #6 run stack B (smolvla, pi0.5) through the controller -- the "clean loop" for the phys ai side. after #3.
- [ ] #9 profile requests end to end: nsys (system), ncu (all kernels) -- ncu needs root on the thor.
- [ ] #13 containers + multi-tenant cells -- after #3, #6, #12. not before gate 4.

## experiment: space + multitenancy (branch `space-multitenancy-v1`)
quick results for #12 / #13. each VLM tenant is its own llama-server container, tenant 0 is measured at T = 3 s.
- [x] container smoke test -- same speed as host (1.52 s back to back)
- [x] MPS works on the thor. private daemon so labmates can't join it
- [x] fair SM split = static MPS partitions, 2-SM chunks. 2 or 5 tenants split the 20 SMs exactly, 3 or 4 leave SMs idle
- [x] harness: tenants / load / sharing per cell, background load in its own process -- 2 smoke cells pass
- [x] found + fixed: periodic background load phase-locks with tenant 0 (same overlap every frame!). now random arrivals
- [ ] agree the matrix (configs/space_multitenancy.toml, ~2.5 h of GPU)
- [ ] write the predictions down before running
- [ ] run it, make graph 1 (misses vs tenants) and graph 2 (misses vs deadline)

first numbers: on a 6-SM partition (1/3 of the GPU) a frame takes 4.3 s instead of 1.5 s, even when the neighbours are idle.

## waiting on the review (rounds 2-5)
- [ ] finish rounds 2-5
- [ ] check every issue above against what the rounds say; close or rewrite anything that's wrong
- [ ] `--max-minutes` estimate counts warm-ups at the cadence (real bug) -- fix after round 2 says what `--max-minutes` should mean
- [ ] should deadlines start at the scheduled tick or the actual one? -- round 2. the first pr logs both, so old runs can be redone either way
- [ ] rename `throughput_rps` in paced runs (it's not throughput there) -- round 4
- [ ] action-chunk deadlines for VLAs (deadline = actions left, not next frame) -- round 5, then a follow-up to #6

## housekeeping
- [ ] `uv run ruff format CODE_REVIEW_NOTES.md` when the review is done (a code block fails `ruff format --check`)
- [ ] delete branches `drop-ci` and `repo-workflow-agent-notes` (already merged as #1, #2)
- [ ] ask admin for sudo: nvpmodel, jetson_clocks, ncu (TODO.md step 0) -- blocks #9, #10
- [ ] GPU booking convention with labmates (TODO.md step 0) -- the people side of #11

## parked
- ZMQ instead of asyncio -- only if #7 or #8 says our side is the problem. right now it isn't.

## where my original notes went
- import the logging granularity from main into latency-bound inference --> #4
- extend to containerised and multi-tenant scenarios --> #12, #13
- use the clean loop from deadline-harness and add the phys ai angle --> #3, #6
- extend profiling: ncu (kernel level) and nsys (system level) --> #9
- measure the overhead of our CPU side (asyncio); explore ZMQ if significant --> #7, #8. turned out to be llama-server, not asyncio
- flame graphs per session/request: where each process spends its time --> #9
- more verbose logging for deadline misses: why did it miss, where did it go off-course? --> #5
- power matters for both the space and physical AI angles --> #10
- feeder wakeup could be an issue, make it more deterministic --> pr `record-scheduled-arrival` (measure it) + #8 (fix it if needed). so far ~1 ms late, 2 ms worst, but only measured with nothing else running
