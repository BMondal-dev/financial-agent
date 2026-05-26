🚨 Biggest Weakness #1
LSTM Is Not Truly Sequential

This is the biggest technical caveat.

You wrote:

X already contains lag features

Then:

create_sequences()

So effectively:

your LSTM sees:

lagged tabular features

AND rolling sequential windows of lagged features

This creates:

duplicated temporal structure.
Why This Matters

Your LSTM is NOT operating on:

raw sequential time series

It is operating on:

sequence of engineered lag snapshots

This is more like:

"sequence-over-tabular"

not:

pure temporal modeling.
Is This Wrong?

No.

Actually your paper honestly states this.

That’s good.

But reviewers may say:

“This is not a canonical LSTM forecasting setup.”

And they’d be partially right.

Is It Fatal?

No.

Because:
your paper explicitly frames:

same feature pipeline
different backbones

So your goal is:

controlled comparison

NOT:

optimal LSTM architecture.

That saves you.