<div class="value-panel" markdown>

<div class="value-panel__figure value-panel__figure--raw" markdown>
<span class="value-panel__label">The judge flagged</span>
<span class="value-panel__number">14.8%</span>
<span class="value-panel__note">of 1,000 traces</span>
</div>

<div class="value-panel__figure value-panel__figure--honest" markdown>
<span class="value-panel__label">Calibrated Agent Evaluation reports</span>
<span class="value-panel__number">6.0%</span>
<span class="value-panel__note">likely between 0.0% and 12.6%</span>
</div>

<div class="value-panel__figure value-panel__figure--key" markdown>
<span class="value-panel__label">The true rate is</span>
<span class="value-panel__number">6.0%</span>
<span class="value-panel__note">known here because the sample is synthetic</span>
</div>

<div class="value-panel__why" markdown>
The judge is right on 90% of failures and 90% of passes. On a sample where few traces fail, the
10% of passes it gets wrong outnumber the real failures, so its raw rate is more than double the
truth. The toolkit measures the judge's mistakes against your labels and takes them back out.
</div>

</div>
