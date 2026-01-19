# Instructions
Given the specified directory look for `Recipe.md` which will contain the current content for the methods section of our proposal. 

First check that the format is correct (see `Format` below). If there is an issue with the format, note it and skip rule checking. The author will need to fix the format first.

Assuming the format is correct please look over the `Core Rules` and `Audience` to fill out your context.

Then ask what level the author would like to work on (1, 2, or 3 corresponding to sections, topics, or paragraphs). 

### Level 1
Check `Level Rules:Level 1` and raise any issues. Work the section organization until you and the author are happy or the author has clear next steps.

At the conclusion ask if the author would like to move onto level 2 (topics)

### Level 2
Request which section they wish to review. Check the Level 2 rules (`Level Rules:Level 2`). Work the topic sentences until you and the author are happy with the results or the author has decided they good next steps and want to move on. 

At the conclusion ask if the author would like to move onto another section.

### Level 3
Request which paragraph they wish to review and which sub-level:
- **Level 3a (Base Recipe Completeness)**: Ensure the recipe is complete and actionable - someone could follow it exactly
- **Level 3b (Base Recipe Justification)**: Ensure choices and techniques are justified - explain why each approach was selected

Check the corresponding rules (`Level Rules:Level 3a` or `Level Rules:Level 3b`) and work with the author until you and the author are happy with the content or the author has clear next steps.

At the conclusion ask if the author would like to move onto another paragraph or switch between 3a and 3b.

# Format
The content should be formatted as:

```markdown
## Section Heading
**Lead:** Topic sentence for the first paragraph.
**Development:**
- First development sentence for the first paragraph
- Second development sentence for the first paragraph

**Lead:** Topic sentence for the second paragraph.
**Development:**
- First development sentence for the second paragraph
- Second development sentence for the second paragraph
```

# Audience

Proposals are intended for those familiar with fisheries science and fisheries management. This means some level of familiarity with modeling, ecology, fish biology, resource management, etc. 

# Core Rules

## Core Requirements
- Describes a complete recipe (no gaps)
- Identifies risks and illustrates how such risks will be handled
- Clarifies why the techniques being used have been chosen (note this is not why the outcomes are what they are - that is handled in the introduction which is separate)
## Core Structure
- **Organize around pinnacles**: Major sections represent significant achievements or "aha moments," not individual actions
- **Pinnacles are**: Knowledge milestones, functional achievements, or key deliverables that represent major progress
- **Use three-level hierarchy**: Pinnacles → Steps → Actions/Specifications
## Managing Complexity
- **If 6-12+ steps in one section**: Identify sub-pinnacles (intermediate achievements) to give readers mental breaks and highlight especially valuable outputs
- **Rule of thumb**: Two levels (Pinnacles → Steps) is usually sufficient
- Only add subsections when needed to reduce density
# Level Rules

## Level 1: Major Sections (Pinnacles)
- Should capture major achievements your project will produce
- Create one top-level section per pinnacle
- Each section should represent a complete, meaningful outcome
## Level 2: Steps Within Sections
- **Steps produce independently useful outputs** that could be reused elsewhere
- **Steps are NOT individual actions** (e.g., "dice onions" is an action within a step)
- **Write topic sentences as recipe-style actions** that describe what you'll do to produce the useful output
- Use active voice with action verbs (e.g., "Sauté the onions until translucent" not "Onions are sautéed" or just "Sautéed onions")
- Each topic sentence should clearly indicate both the action AND the resulting output it produces
- **Verify completeness** - The steps should fully achieve the section heading. Check for missing steps in the logical flow from start to finish.
- **Show your work** - Include steps that demonstrate key outputs through tables, figures, or validation metrics. Every model needs performance assessment, every analysis needs visualization, every dataset needs quality checks.
## Level 3a: Base Recipe Completeness
- Provide the detail required for someone to follow your process exactly
- Specify concrete parameters: thresholds, sample sizes, time windows, distance limits
- Name specific tools, algorithms, or frameworks being used
- Define data structures and formats
- Identify who performs expert tasks
- Acknowledge when parameters will be determined from data (rather than specified upfront)
- Ensure no gaps exist - every step should be actionable

## Level 3b: Base Recipe Justification
- Explain why each technique or approach was chosen
- Justify parameter choices (e.g., why 5% false negative rate? why one week window?)
- Clarify advantages of selected methods over alternatives
- Address potential risks and how they're mitigated
- Connect choices to project goals or constraints
- Note: This is NOT explaining why outcomes occur (that's Introduction) - this explains why methodological choices were made
