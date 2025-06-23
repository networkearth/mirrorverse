I am going to give you a document in the following structured YAML-like form:

```yaml
results:
  - section: >
      If the dataset supports multiple distinct questions or storylines, divide the results 
      into subsections. Each should have a descriptive header that aligns with 
      the narrative of the paper but does not imply interpretation.
    header: The header for this section.
    - lead: Topic sentence in L/D structure.
      development: Sentences supporting the lead in an L/D structure.
```

Please treat the input as structured YAML text.

## Structure

- Each paragraph should begin with the lead sentence, followed by its development. The full paragraph should flow as a single block of text, not broken into lead + development.
- Each section should tell a self-contained story, marked with a clear and descriptive header.

## Style

- Write for non-technical readers. If your grandparents wouldn’t understand it, rewrite.
- Be concise. Writing is a game of removal — you can usually cut more than you think.
- Aim for clarity, rhythm, and compression — dense with meaning, light on fluff. (feel free to add transitional phrasing)
- Avoid bullet-list structure unless inherently necessary. Let each paragraph build logically, with implied connections rather than mechanical listing.
- Still avoid speculation, interpretation, or rhetorical flourishes. Stay factual and tied to the data.

