Reading is a strange ability.

Evolution never designed the human brain to read. Writing is about 5,000 years old, far too recent for that. Yet culture found an input system the brain learns so well that it becomes automatic.

That made me wonder: how many other learnable abilities are sitting inside the brain's constraints, undiscovered simply because nobody has invented the right interface or training environment?

And instead of guessing them, could we search for them?

I built a small computational framework to explore this. I call it Cognitive Capability Search (CCS).

The idea: take a learner with a fixed architecture. An inner loop trains it. An outer loop searches over interfaces, objectives and plasticity schedules: which new input channel to give it, how to encode it, which brain-like "regions" to unlock and for how long. The score rewards new world-structure the learner can now read cheaply and penalizes damage to what it could already do. It never names a target ability.

The question isn't "how do I teach it X?" It's "what can this learner become?"

In a deliberately small toy model (a network that "grows up" seeing only half of its world's hidden causes):

→ Search found environments that only 0.4% of random ones match.
→ The new abilities transferred to tasks the search never saw. Performance on tasks needing previously invisible information roughly doubled (R² 0.28 → ~0.60).
→ The most interesting result was a principle I didn't design in. Every winner learned the new channel while the old sense was silent. Change only that setting and the damage to existing abilities jumps from ~1% to 35–97%.
→ And one honest failure: my favorite hypothesis, that smart region-level plasticity control would win, didn't hold up. The search also gamed my interference metric, which is a lesson in itself.

This says nothing about human brains yet. It's a toy network, not a neuroscience result. But I think it's the right shape of question, and it can be posed as a search.

Paper + code (pure NumPy, runs in 15 minutes on a laptop): [link]

#MachineLearning #Neuroscience #CognitiveScience #BCI #Research
