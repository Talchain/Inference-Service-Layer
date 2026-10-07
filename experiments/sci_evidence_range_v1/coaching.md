# SCI-EVIDENCE range elicitation: coaching example

Case: the frozen R3-B MRR case (£49 → £59, £100k MRR by month 12, churn ≤ 4%). All numbers come from
`results/r3b.json`. **The user's range below is HYPOTHETICAL, an experimental stand-in only.**

**1. Olumi names the gap**
> Monthly churn matters here twice: it's what your 4% limit is about, and it feeds how Pro subscribers grow.
> The 3% I've used is my own estimate, with no range, so everything I've said about the churn limit quietly
> assumes churn is exactly 3%. I can't yet tell you how much room that limit really has.

**2. Olumi asks for the smallest useful input**
> What would you consider a plausible low-to-high range for your monthly churn today, before any price change?
> A rough "somewhere between X% and Y%" is enough. A recent figure you trust works too.

**3. The user answers** *(HYPOTHETICAL, experimental control input)*
> Somewhere between 2% and 4%.

**4. Olumi explains what the re-run shows**
> Under the current model, the £100k goal verdict of every option Olumi can compute stays the same anywhere in your 2–4% range: £49 with feature release and £59 with feature release reach it, Keep current pricing does not.
> £59 with feature release stays within your 4% churn limit across the whole range, but at 4% it sits exactly on the limit with no room to spare (at Olumi's 3% it had 1 point).
> The earlier 'what would change this' point, 1.5 points of extra churn per £10, assumed 3%. Across your range it runs from 2.5 (at 2%) to 0.5 (at 4%), and 0.5 is the model's own working guess for that response.

**5. What changed, and what is still uncertain**
> So how churn responds to the £10 rise is the check that could change this, and it matters most if today's churn is near the top of your range.
> This is a model result, not a forecast. It covers only the 3 options I have figures for, and the
> churn response to the price rise is still my working guess.
