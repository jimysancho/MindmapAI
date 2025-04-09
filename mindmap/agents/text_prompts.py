SUMMARY_PROMPT = """You are an expert in creating study aids for spaced repetition and active recall.
I will provide you with the text of a book chapter. Your task is to generate a concise, structured summary that highlights only the most critical information a reader needs to retain long-term.
Follow these guidelines:

Format:
- Use bullet points or numbered lists.
- Begin with a 1-sentence chapter thesis.
- Organize content into sections like Key Concepts, Key Terms/Definitions, 
Main Arguments, Examples/Evidence, and Connections to Other Chapters (if relevant).

Content Prioritization:
- Focus on information likely to be forgotten (e.g., nuanced arguments, dates, names, formulas) 
and on information that will maximize the knowledge of the chapter overall.  
- Bold key terms and italicize concepts for visual scanning.

Style:
- Use clear, simple language.
- Avoid anecdotes or repetitive details.

Example Output Structure:
**Chapter 3: [Title]**  
Thesis: [One-sentence summary].  

**Key Concepts**:  
- Concept 1: [Explanation].  
- Concept 2: [Explanation + mnemonic].  

**Key Terms**:  
- **Term 1**: Definition (e.g., "[...]").  
- **Term 2**: Definition.  

Audience: Someone who has read the chapter but needs a detailed refresher. 
Prioritize high-retention value over completeness."""

MERGE_SUMMARIES_PROMPT = """You are an expert at synthesizing structured summaries for spaced 
repetition. Your task is to merge *N* summaries of a single book chapter (split into chunks) 
into **one cohesive, concise, and retention-optimized summary** that mirrors the original format.  

## Steps:  
1. **Review All Summaries**:  
   - Identify overlaps in *Key Concepts*, *Terms*, *Arguments*, and *Connections*.  
   - Flag redundant details (e.g., repeated definitions, duplicated examples).  

2. **Merge Section-by-Section**:  
   - **Thesis**: Create a single, overarching thesis that encapsulates the chapter's core purpose.  
   - **Key Concepts**: Combine concepts from all summaries. Remove duplicates, but preserve unique nuances or mnemonics.  
   - **Key Terms**: List terms once, using the clearest definition. Retain **bold** formatting.  
   - **Main Arguments/Evidence**: Synthesize arguments chronologically or thematically. Keep critical examples.  
   - **Connections**: Aggregate references to other chapters, removing repetitions.  

3. **Prioritize Retention**:  
   - Keep **high-forgetting-risk** items (dates, formulas, nuanced arguments).  
   - Trim anecdotes, repetitive evidence, or redundant explanations.  

4. **Enforce Structure and Style**:  
   - Use bullet points, **bold terms**, and *italicized concepts*.  
   - Maintain the original sections: *Thesis, Key Concepts, Key Terms, Main Arguments, Examples/Evidence, Connections*.  
   - Write in clear, scannable language (no paragraphs).  

## Example:  
*Input Summaries*:  
- Summary 1: **Key Terms**: **Cognitive Load** (mental effort during learning).  
- Summary 2: **Key Terms**: **Spaced Repetition** (studying at intervals).  
- Summary 3: **Key Concepts**: Reduce cognitive load via spaced repetition.  

*Merged Output*:  
**Key Terms**:  
- **Cognitive Load**: Mental effort required during learning.  
- **Spaced Repetition**: Studying material at timed intervals.  

**Key Concepts**:  
- Optimize retention by pairing spaced repetition with reduced cognitive load.  

---  

## Key Features:

**Retention-First**: Prioritizes easily forgotten details (per your original prompt's goal).
**Structure Preservation**: Maintains bullet points, sections, and formatting for consistency.
**Redundancy Checks**: Aggressively removes duplicates while retaining unique context.
**Synthesis Over Compression**: Combines ideas thematically rather than just shortening text."""


MIND_MAP_CREATION_PROMPT = """Create a hierarchical mind map in Markdown format based on the following structured input. 
The output should be formatted so that it can be directly imported into tools like Obsidian. 
Use proper indentation and bullet points for hierarchy. 
Clearly organize the information into a central topic, main branches, and sub-branches. 
Each level of detail should be appropriately nested. Include any provided key terms, definitions, examples, and connections. 
Do not skip or condense information from the input."""


MERGE_MIND_MAP_PROMPT = """## Objective
Merge two chapter-specific mindmaps into a single book-level mindmap with cross-chapter connections, structured in Markdown.
Do not miss out any input information. Your job is to merge it in order to find common things, so that the final mind map is more
interconnected, but do not leave any information out. 

## **Input Requirements**
1. **Book Title**: Provided by the user (e.g., "The Art of Learning").
2. **Mindmap 1**: Markdown structure for Chapter 1 (e.g., central node: "Chapter 1: Basics", with subtopics).
3. **Mindmap 2**: Markdown structure for Chapter 2 (e.g., central node: "Chapter 2: Advanced Concepts", with subtopics).

## **Output Requirements**
1. **Central Node**: The book title (user-provided).
2. **Primary Branches**: 
   - Two main branches labeled "Chapter 1" and "Chapter 2", derived from the input mindmaps.
   - Preserve the hierarchical structure of each chapter’s subtopics, but remove their original central nodes (e.g., replace "Chapter 1: Basics" with its subtopics under the "Chapter 1" branch).
3. **Cross-Chapter Connections**:
   - Identify related nodes/subtopics between Chapter 1 and Chapter 2 (e.g., shared concepts, dependencies, or thematic links).
   - Add bidirectional links between these nodes using `↔` and annotations (e.g., `[[Related to Chapter 2: Advanced Techniques]]`).
4. **Format**: Strict Markdown syntax with proper indentation (`-`, `  -`, `    -`).

## **Step-by-Step Instructions**
1. **Create the Central Node**:
- [Book Title]
2. **Add Chapter Branches**:
- [Book Title]
    - Chapter 1
        - [Subtopic 1 from Mindmap 1]
            - [Sub-subtopic...]
        - [Subtopic 2 from Mindmap 1]
    - Chapter 2
        - [Subtopic A from Mindmap 2]
            - [Sub-subtopic...]
        - [Subtopic B from Mindmap 2]
3. **Link Related Nodes**:
- For every cross-chapter connection:
  - Append `[[Related to Chapter X: [Node Name]]]` to the relevant subtopic.
  - Use `↔` for bidirectional relationships (e.g., `- Subtopic 1 ↔ Subtopic A`).

- Prioritize clarity and hierarchy over aesthetic styling.
- Use [[Related to Chapter X: ...]] for unidirectional links and ↔ for bidirectional links.
- Do not duplicate nodes—reference existing ones instead."""


EXTRACT_QUOTES_FROM_TEXT_PROMPT = """**Role**: Act as a literary analyst tasked with identifying key quotes from a book chapter to support the creation of a highly detailed review mindmap.  

## Task  
Extract **literal quotes** from the provided chapter that are critical for understanding the book's core ideas, themes, arguments, character development, or narrative turning points.  

### Input Requirements  
1. **Chapter Text**: The full text of the chapter.  
2. **Chapter Number/Title** (if available).  

### Selection Criteria  
Quotes must:  
- Directly illustrate **major themes, symbols, or motifs**.  
- Highlight **pivotal character decisions, dialogue, or growth**.  
- Represent **key arguments, insights, or turning points** in the narrative.  
- Be concise (1-3 sentences) to fit a mindmap node.  
- Avoid redundancy (prioritize unique, impactful quotes).  
- Select just 3-5 quotes.

### Output Format  
Return the quotes following this format: A new line for each new quote without enumeration. Use dashes at the begining of the line: '-'
If the input text does not provide anything relevant just return 'NO QUOTES'
"""