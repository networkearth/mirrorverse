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

Your task is to check all lines — including section headers, leads, and development sentences — for compliance with the following rules:

Rules:
- Every paragraph should follow an L/D structure: each development sentence must support its lead.
- Every claim must be directly supported by a figure, table, or summary statistic.
- Do not include speculative language, value-laden words (e.g., “important,” “significant” unless statistical), or references to goals or objectives.
- Do not mention prior work, hypotheses, or interpretations.
- Avoid rhetorical or persuasive language; be factual and grounded in what the data show.
- Be specific — any adjectives like “large” or “significant” must be followed by a number.
- Do not include methodological details; those belong in the Methods section.
- A development sentence is non-compliant if it introduces unrelated content, fails to support the lead, or adds a new claim not foreshadowed by the lead.

Return the entire `results:` block exactly as I wrote it, but annotate any non-compliant lines with `# FIX:` comments on a line below. Do not delete or rewrite content. Just flag and explain issues.
