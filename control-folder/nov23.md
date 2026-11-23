# Space experiment plan for November 23, 2026

Use this plan for the space experiments on the Jetson Thor. Work from the `deadline-harness` branch. Use the [GitHub issues](https://github.com/pulak999/latency-bound-inf/issues) for acceptance checks and [AGENTS.md](../AGENTS.md) for issue writing.

Run the steps **in order**, with one active work package and one experiment on the Thor at a time. If another agent takes over, give it the commit, commands, results path, and next unfinished step. Planned tenants and co-runners run together within their experiment.

## Choose the run length after the rate

For a result cell, let T be the time between scheduled input frames and N the number of measured input frames. Decide the offered rate (1/T) first. Then choose N using the observation time the experiment needs at that rate. Sixty frames is a candidate. Record T, N, the deadline D, and the reason for the run length in the cell plan. Count warm-up frames separately. Count scheduled inputs when reporting misses, including overwritten frames. For comparisons, record whether the cells share a frame count or an observation time. Keep repeated runs identifiable. A short smoke test or profiling capture can use its own length.

## Execution order

1. **Integrate and verify the run path (#3).** Inspect the branch differences and open the saved November 19 pilot. Reuse the existing controller, MPS support, and visualizer. Run a short Stack A cell on the integrated branch. Match the capture to its log and open it in the viewer. Link the commit, command, and result in #3 before closing it.
2. **Produce the first space results for P3 (November 25).** Choose the rate and N before each result series. Run S1 first, then the planned S3 subset with two and five tenants. Run S2 if time permits. Check GPU load before starting planned tenants, pin and record the clock, and record the power mode. Regenerate the P3 figure and Table I from the raw records. Label the 12-frame pilot separately.
3. **Verify logging (#8, #16).** Follow an answered frame and an overwritten frame through the logs and the report. Alternate runs with each measurement source off and on. Use #9 and #15 for profiling overhead and feeder lateness. Investigate outliers and record any unresolved effect on the measurements. Save the verified timing settings. Revisit affected P3 results if verification changes their interpretation.
4. **Measure power and performance (#10).** Compare matched runs at two power modes. Verify restoration after success, failure, and interruption. Report clock, frame time, misses, board power, and energy. If no frame has an answer, report total energy and mark energy per answered frame as unavailable.
5. **Explain the S4 slowdown (#13, #14).** Verify the partition setting with the owner of #13. Sweep 20, 10, 6, and 4 SMs with one tenant. At 6 SMs, vary neighbor load. Compare the busy partitioned and time-sliced runs. Repeat the cells and save separate profiling captures. Report the observed difference alongside run-to-run variation.
6. **Verify the visualizer with the saved S4 captures (#17).** Open two captures, select a frame, and compare the same kernel and input shape. Show the deadline, kernel times, assigned SM count, and one memory comparison. Label the source and scope of the metric. Save the demo links with S4. This completes the M4 check when it passes. The final demo and full memory view remain on the roadmap.
7. **Finish the M4 evidence and reconcile the trackers (December 7).** Use the results to select the ablation and boundary case. S7 is a candidate ablation. Repeat the relevant cells. Include S6 and the applicable power comparisons in the schedule. Confirm ownership of #12 and #19 before counting their output. After the #16 logging checks pass, review #20, update its run-length rule to reflect T and N, move S9 to Part 2, and keep the approved main claim. Defer Stack B (#6) until after the space work.

## Code and evidence for closure

Keep each change focused on one issue. Preserve existing run readers and test the normal controller and report paths. Save each run in its own directory with the commit, configuration, input sequence, clock, power mode, and partition settings. Keep interrupted runs identifiable. For each closure, link the change, reproduction command, raw results, and a figure or table that answers the issue. State which tenants were measured.

Use the [issue closure rule](../AGENTS.md#issue-closure) and the [results comment template](../.github/issue-results-comment.md) for each closure decision.
