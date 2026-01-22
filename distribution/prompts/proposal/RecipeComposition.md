# Instructions
Given the specified directory look for `Recipe.md` which will contain content for a methods section for a proposal. Your job is to write out methods prose in the same directory in the file `Methods.md`. 

The content in `Recipe.md` is split into three levels. Level 1 are the sections - each of these is a pinnacle, that is a major outcome of the work. The methods should be split into sections, one for each of these pinnacles. Level 2 are the topic sentences. Each paragraph in the methods section should capture a step whose outcome is largely useful on its own. These topic sentences will capture those steps. Level 3 are development sentences which fill out the step introduced by the topic. These give very specific instructions on the actions to carry out the step. As such each section should be in Lead/Development form. To better understand how the format of `Recipe.md` fits into this paradigm see `Format of Content` below. 

# Format of Content
The content will be formatted as:

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

# Rules

- This is a proposal so things should be in the future tense.
- Use the active voice.
- Vary sentence structure to maintain reader engagement. Avoid starting consecutive sentences or paragraphs with the same phrase (e.g., "We will"). Mix sentence openings by:
  - Leading with the object or method: "Tag data from collaborative research efforts will be compiled..."
  - Using passive constructions strategically when appropriate: "All movement data from the first seven days will be excluded..."
  - Starting with temporal or conditional phrases: "To obtain location estimates, the data will be..."
  - Using descriptive phrases: "Constant depth readings will indicate mortality..."
  - Embedding the subject within the sentence: "Each observation will include..."
- Treat the entire methods section as a single narrative arc, not a collection of independent method descriptions. The reader should feel pulled forward through a logical sequence of dependent steps.
- When writing each paragraph, consider: "What did the previous paragraph produce, and how does this paragraph use or build on that?"

# Narrative Flow

The methods section must read as a continuous narrative, not a collection of standalone blocks. Each paragraph should connect logically to those around it.

## Transitions Between Paragraphs
- Each paragraph (after the first in a section) should begin with or include early transitional language that:
  - References the output or outcome of the previous paragraph
  - Explains why this step follows from what was just described
  - Uses connective phrases like "With [previous output] established...", "This [result] then requires...", "Once [previous step] is complete...", "From these [outputs]..."

## The "Why This, Why Now" Principle
- Don't just state what will be done—briefly indicate why this step is necessary given what came before
- The reader should always understand how each step's output becomes the next step's input
- The methods should tell a story of transformation: raw data → validated data → filtered data → enriched data → model → application