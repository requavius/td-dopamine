Conceptual Design Decisions

Agent parameters (model does not see only estimates):
f: Sensitivity to learning progress  
k: Effort aversion  
b: Boredom rate 

An episode represents one full completion of an activity like reading a chapter in a book. The stages within an episode are the steps that must be gone through before considering to disengaging like reading a paragraph in that chapter. The person can disengage at any stage, quitting before completing the activity.
Engagement is evaluated as a per stage decision every episode. A between episode reengagement decision exists in the full intended model but is not yet implemented. In the current simulation the person always attempts the next episode, which is a simplification that removes a real behavioral variable for data collection purposes.
Effort and boredom costs are zero at the start of each attempt. This reflects the assumption that a person begins a fresh activity in a neutral state before any effort accumulates.
Skill grows per stage rather than per episode. A person who quits at stage 1 gains partial skill rather than none. This prevents agents with high effort aversion from being permanently frozen at their initial skill level and represents even small efforts can improve skill, although without a strong learning signal progress is limited.
Reward is only available at the terminal stage. Value propagates backwards through the stage sequence via TD learning, which means early stages only acquire value because they predict eventual completion.
RPE drives the engagement signal. How motivated a person is to continue at any stage is proportional to how surprising the current outcome is relative to their expectation, scaled by their sensitivity parameter f. A fully predicted outcome has no RPE and therfore no motivation to continue.
Three latent parameters govern engagement behavior: f, sensitivity to learning progress; k, effort aversion; b, boredom rate. These cannot be directly observed. The model infers them using the Bayesian particle inference.
Engagement and disengagement are the only observable signals available to the model. The true parameters are never revealed. Everything the model knows about the person comes from whether they continued or stopped.