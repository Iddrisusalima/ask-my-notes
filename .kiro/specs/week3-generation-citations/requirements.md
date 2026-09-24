# Requirements Document

## Introduction

Week 3 (Sep 28 – Oct 4) of the "Ask My Docs" learning project turns retrieval into answers. Week 1 delivered Configuration, the Embedder, chunking, loading, and the store interface. Week 2 delivered the persistent Chroma_Store, incremental ingest, and the Retriever, which returns a Retrieval_Result whose outcome is exactly one of RELEVANT_CONTEXT, NO_RELEVANT_CONTEXT, or EMPTY_COLLECTION, together with an ordered sequence of SearchHit values each carrying a Chunk (source path, ordinal index, start and end character offsets) and a Similarity_Score.

Week 3 consumes that Retrieval_Result and adds nothing to the storage or retrieval layer. It adds prompt construction with a bounded context budget, answer generation through the chat completions endpoint, citation markers that map to a printed source list, citation validation, an explicit refusal path, a single end-to-end command-line script, an append-only Answer_Log, an evaluation run over at least ten questions with hand-entered quality ratings, the README with a "What I Learned" section, a RAG versus fine-tuning comparison, a demo shot list, and the submission and self-check checklists. Week 1 and Week 2 modules are not modified.

Out of scope for Week 3: changing the chunking algorithm, changing the vector store, streaming responses, a web or graphical interface, multi-turn conversation memory, and re-ranking of retrieved Chunks.

## Glossary

The following terms are carried over unchanged from the Week 1 and Week 2 requirements documents and keep their existing meanings here: **Ask_My_Docs**, **Configuration**, **Embedding_Provider**, **Embedder**, **Max_Input_Length**, **Max_Retry_Attempts**, **Request_Timeout**, **Sample_Notes_Folder**, **Chunk**, **Chunk_Size**, **Chunk_Overlap**, **Similarity_Score**, **Chroma_Store**, **Retriever**, **Top_K**, **Relevance_Threshold**, **Retrieval_Result**, **SearchHit**, **No_Relevant_Context**, **Query_Script**, **Retrieval_Log**, **Question_Set**, **Learning_Notes**, and **Reporter-style redaction**. Week 3 adds the following terms.

- **Retrieval_Outcome**: The single value the Retriever reports for one question, being exactly one of `RELEVANT_CONTEXT`, `NO_RELEVANT_CONTEXT`, or `EMPTY_COLLECTION`.
- **Supplied_Chunk**: One Chunk of the Retrieval_Result that the Prompt_Builder placed into the Context_Block of one Assembled_Prompt.
- **Context_Block**: The numbered text section of the Assembled_Prompt holding one entry per Supplied_Chunk, each entry carrying a Citation_Number and the Chunk text.
- **Citation_Number**: The one-based position of one Supplied_Chunk in the Context_Block, assigned in Retrieval_Result order.
- **System_Prompt**: The fixed instruction text of the Assembled_Prompt that directs the Chat_Model to answer only from the Context_Block, to state that it does not know when the Context_Block lacks the answer, and to mark each sentence it draws from the Context_Block with the Citation_Marker of the Supplied_Chunk it used.
- **User_Prompt**: The part of the Assembled_Prompt holding the Context_Block and the learner's question string.
- **Assembled_Prompt**: The complete System_Prompt and User_Prompt pair the Prompt_Builder produces for one Retrieval_Result.
- **Prompt_Builder**: The Ask_My_Docs component that converts one Retrieval_Result and the Configuration into one Assembled_Prompt and the accompanying Citation_Table.
- **Prompt_Length**: The number of Unicode code points of the System_Prompt and the User_Prompt taken together.
- **Context_Budget**: The configured maximum permitted Prompt_Length. Setting name `Context_Budget`, default `12000`, permitted range 1000 to 200000.
- **Over_Budget**: The condition holding when the Prompt_Length of the Assembled_Prompt built from every SearchHit of the Retrieval_Result exceeds the Context_Budget.
- **Dropped_Chunk_Count**: The number of SearchHit values of the Retrieval_Result that the Prompt_Builder excluded from the Context_Block in order to satisfy the Context_Budget.
- **Chat_Provider**: The configured source of chat completions, independent of the Embedding_Provider. Week 3 supports the single value `openai`. Setting name `Chat_Provider`, default `openai`.
- **Chat_Model**: The configured chat completions model name used to generate answers. Setting name `Chat_Model`, default `gpt-4o-mini`.
- **Answer_Generator**: The Ask_My_Docs component that sends one Assembled_Prompt to the chat completions endpoint of the Chat_Provider and returns one Generated_Answer.
- **Generated_Answer**: The answer text the Chat_Model returned for one Assembled_Prompt, together with the Token_Usage of that call.
- **Token_Usage**: The record of the prompt token count, the completion token count, and the total token count the chat completions endpoint reported for one call.
- **Citation_Marker**: A bracketed one-based integer, written as `[n]`, appearing in the Generated_Answer text and referring to the Supplied_Chunk whose Citation_Number is `n`.
- **Citation_Table**: The mapping the Prompt_Builder produces from each Citation_Number to the source path, ordinal index, start character offset, end character offset, and Similarity_Score of the corresponding Supplied_Chunk.
- **Source_List**: The numbered list Ask_My_Docs prints after the answer text, holding one line per Citation_Table entry that the presented answer cites.
- **Dangling_Citation**: A Citation_Marker of the Generated_Answer whose integer has no matching Citation_Number in the Citation_Table.
- **Citation_Validator**: The Ask_My_Docs component that extracts every Citation_Marker of a Generated_Answer, matches each against the Citation_Table, and reports every Dangling_Citation.
- **Verified_Answer**: A presented answer whose Generated_Answer holds at least one Citation_Marker and no Dangling_Citation.
- **Unverified_Answer**: A presented answer whose Generated_Answer holds at least one Dangling_Citation or holds no Citation_Marker.
- **Refusal**: The response Ask_My_Docs presents in place of a Generated_Answer when the Retrieval_Outcome is `NO_RELEVANT_CONTEXT` or `EMPTY_COLLECTION`, carrying a Refusal_Reason and no Source_List.
- **Refusal_Reason**: The named cause of one Refusal, being `no-relevant-context` for the Retrieval_Outcome `NO_RELEVANT_CONTEXT` and `empty-collection` for the Retrieval_Outcome `EMPTY_COLLECTION`.
- **Ask_Script**: The runnable command-line script that accepts one question string, invokes the Retriever, the Prompt_Builder, the Answer_Generator, and the Citation_Validator, and prints either a presented answer with its Source_List or a Refusal.
- **Answer_Log**: The append-only machine-readable file to which one Answer_Log_Record is written for every Ask_Script and Evaluation_Script answer attempt. Setting name `Answer_Log`, default path `logs/answers.jsonl`.
- **Answer_Log_Record**: One JSON object, serialized on a single line, recording one answer attempt.
- **Answer_Log_Schema_Version**: The version value held in every Answer_Log_Record that identifies the field set of that Answer_Log_Record. Week 3 writes the single value `1`.
- **Prompt_Hash**: The lower-case hexadecimal SHA-256 digest of the UTF-8 encoding of the Assembled_Prompt, recorded in place of the prompt text.
- **Retrieval_Summary**: The part of one Answer_Log_Record holding the Retrieval_Outcome, the Top_K used, the Relevance_Threshold used, and, for every SearchHit of the Retrieval_Result, the source path, the ordinal index, the character offset range, and the Similarity_Score.
- **Grounding_Check**: One of the two Week 3 checks that the answer text depends on retrieved context: the Positive_Grounding_Question and the Negative_Grounding_Question.
- **Positive_Grounding_Question**: A question whose answer appears in the Sample_Notes_Folder and nowhere in the System_Prompt, used to confirm that the presented answer draws on the Context_Block.
- **Negative_Grounding_Question**: A question whose answer appears nowhere in the Sample_Notes_Folder, used to confirm that Ask_My_Docs refuses rather than inventing an answer.
- **Evaluation_Question_Set**: The UTF-8 text file holding the Week 3 evaluation questions, one question per non-empty line. Setting name `Evaluation_Question_Set`, default path `question-sets/week3-questions.txt`. Required question count at least 10, of which at least one is a Negative_Grounding_Question.
- **Evaluation_Script**: The runnable script that runs every question of the Evaluation_Question_Set, records the results, accepts the Quality_Ratings, and writes the Evaluation_Report.
- **Quality_Rating**: The hand-entered integer from 1 to 5 recording the learner's judgement of one answer, or empty for not yet rated.
- **Mean_Quality_Rating**: The arithmetic mean of the non-empty Quality_Ratings of one evaluation run, rounded to two decimal places.
- **Evaluation_Report**: The markdown file holding, for every question of the Evaluation_Question_Set, the answer, the citations, the retrieval Similarity_Scores, and the Quality_Rating, plus the Mean_Quality_Rating and the Configuration_Stamp. Setting name `Evaluation_Report`, default path `reports/week3-evaluation.md`.
- **Configuration_Stamp**: The record of every Configuration setting name and resolved value in force for one evaluation run, with the API key value replaced by Reporter-style redaction.
- **Malformed_Model_Output**: A chat completions response carrying no message content, or carrying message content that is empty after removal of surrounding whitespace.
- **Readme**: The repository file `README.md`.
- **What_I_Learned_Section**: The Readme section in which the learner explains embeddings, chunking, and the retrieval-augmented generation pipeline in the learner's own words.
- **Example_Screenshot**: The committed image file showing one example question, its presented answer, and its Source_List, referenced from the Readme.
- **RAG_Versus_Fine_Tuning_Note**: The written markdown artifact comparing retrieval-augmented generation against fine-tuning. Default path `learning-notes/rag-vs-fine-tuning.md`.
- **Demo_Script**: The written markdown shot list for the 3 to 4 minute demo video. Default path `learning-notes/demo-script.md`.
- **Submission_Checklist**: The written markdown artifact listing every submission deliverable as a checkable item. Default path `learning-notes/submission-checklist.md`.
- **Self_Check_Checklist**: The written markdown artifact listing the six self-check questions the learner must be able to answer. Default path `learning-notes/self-check.md`.

## Requirements

### Requirement 1: Week 3 Configuration Extension

**User Story:** As the learner, I want every Week 3 setting read from the environment with a documented default, so that I can change generation behaviour without editing code.

#### Acceptance Criteria

1. THE Configuration SHALL expose the settings Chat_Provider, Context_Budget, Chat_Model, Answer_Log, Evaluation_Question_Set, and Evaluation_Report, each read from an environment variable and each resolving to the default recorded in the Glossary when the environment variable is absent or empty.
2. THE Configuration SHALL retain the Week 1 and Week 2 setting names, resolved values, and validation behaviour unchanged.
3. IF the resolved Chat_Provider, after removal of surrounding whitespace and conversion to lower case, does not equal `openai`, THEN THE Configuration SHALL raise a configuration error naming the setting name Chat_Provider, the rejected value, and the supported value `openai`.
4. IF the resolved Context_Budget is not an integer in the range 1000 to 200000 inclusive, THEN THE Configuration SHALL raise a configuration error naming the setting name Context_Budget, the rejected value, and the permitted range.
5. IF the resolved Chat_Model is empty after removal of surrounding whitespace, THEN THE Configuration SHALL raise a configuration error naming the setting name Chat_Model.
6. WHEN the Configuration resolves the Answer_Log, the Evaluation_Question_Set, and the Evaluation_Report, THE Configuration SHALL resolve each relative path against the repository root and return an absolute path.
7. THE Configuration SHALL permit every combination of Embedding_Provider and Chat_Provider, so that a local Embedding_Provider resolves together with a remote Chat_Provider.
8. THE Configuration SHALL record every Week 3 setting name, its environment variable name, and its default in both `.env.example` and the Readme.

### Requirement 2: Prompt Construction

**User Story:** As the learner, I want retrieved Chunks assembled into a prompt that forbids outside knowledge, so that the model answers only from my notes.

#### Acceptance Criteria

1. WHEN the Prompt_Builder receives a Retrieval_Result whose Retrieval_Outcome is `RELEVANT_CONTEXT`, THE Prompt_Builder SHALL return one Assembled_Prompt holding a System_Prompt, a User_Prompt, and a Citation_Table.
2. THE Prompt_Builder SHALL place one Context_Block entry per Supplied_Chunk, in Retrieval_Result order, and SHALL assign each entry the Citation_Number equal to the one-based position of that entry in the Context_Block.
3. THE Context_Block SHALL hold, for each entry, the Citation_Number and the Chunk text of the corresponding Supplied_Chunk, and SHALL hold no text taken from any source other than the Supplied_Chunks.
4. THE System_Prompt SHALL instruct the Chat_Model to answer using only the Context_Block, to state that it does not know when the Context_Block lacks the answer, and to append the Citation_Marker of each Supplied_Chunk it used to the sentence drawn from that Supplied_Chunk.
5. THE User_Prompt SHALL hold the Context_Block and the learner's question string.
6. WHEN the Prompt_Builder receives the same Retrieval_Result and the same Configuration on two separate invocations, THE Prompt_Builder SHALL return two Assembled_Prompts with identical System_Prompt text, identical User_Prompt text, and identical Citation_Tables.
7. THE Prompt_Builder SHALL build the Citation_Table so that each Citation_Number maps to exactly one Supplied_Chunk and carries that Chunk's source path, ordinal index, start character offset, end character offset, and Similarity_Score.
8. THE Prompt_Builder SHALL read the Retrieval_Result without modifying the Retriever, the Chroma_Store, or any Week 1 or Week 2 module.

### Requirement 3: Context Budget

**User Story:** As the learner, I want the prompt bounded by a configured size, so that a large retrieval cannot produce an oversized or rejected request.

#### Acceptance Criteria

1. THE Prompt_Builder SHALL return an Assembled_Prompt whose Prompt_Length is less than or equal to the Context_Budget.
2. WHILE the Over_Budget condition holds, THE Prompt_Builder SHALL exclude SearchHit values from the Context_Block starting at the lowest-scoring end of the Retrieval_Result and continuing toward the highest-scoring end until the Prompt_Length is less than or equal to the Context_Budget.
3. THE Prompt_Builder SHALL produce a Supplied_Chunk sequence that is a prefix of the Retrieval_Result in Retrieval_Result order.
4. WHEN the Prompt_Builder excludes at least one SearchHit, THE Prompt_Builder SHALL report the Dropped_Chunk_Count together with the Assembled_Prompt.
5. WHEN the Prompt_Builder excludes at least one SearchHit, THE Ask_Script SHALL print a notice naming the Dropped_Chunk_Count and the Context_Budget.
6. IF the Assembled_Prompt built from the single highest-scoring SearchHit alone has a Prompt_Length greater than the Context_Budget, THEN THE Prompt_Builder SHALL raise a context budget error naming the Context_Budget, the Prompt_Length, and the Chunk_Size.
7. WHEN the Prompt_Builder excludes no SearchHit, THE Prompt_Builder SHALL report a Dropped_Chunk_Count of zero.

### Requirement 4: Answer Generation

**User Story:** As the learner, I want the assembled prompt sent to a chat model and the answer returned with its token usage, so that I can see both the answer and its cost.

#### Acceptance Criteria

1. WHEN the Answer_Generator receives an Assembled_Prompt, THE Answer_Generator SHALL send the System_Prompt and the User_Prompt to the chat completions endpoint of the Chat_Provider using the Chat_Model and SHALL return one Generated_Answer.
2. THE Answer_Generator SHALL apply the Request_Timeout, the Max_Retry_Attempts, and the retry classification of transient against permanent failures defined for the Week 1 Embedder.
3. WHERE the Chat_Provider requires an API key, THE Answer_Generator SHALL read that API key from the environment only, and SHALL apply Reporter-style redaction to every console string, error message, and log record it produces.
4. IF the Chat_Provider requires an API key and the API key environment variable of that Chat_Provider is absent or empty, THEN THE Answer_Generator SHALL raise an error naming the required environment variable and SHALL issue no chat completions request.
5. WHEN the chat completions endpoint returns a successful response, THE Answer_Generator SHALL return the Token_Usage holding the prompt token count, the completion token count, and the total token count of that response.
6. IF the chat completions endpoint reports no token counts, THEN THE Answer_Generator SHALL return a Token_Usage whose prompt token count, completion token count, and total token count are each zero.
7. THE Answer_Generator SHALL return a Token_Usage whose three counts are non-negative integers and whose total token count equals the sum of the prompt token count and the completion token count.
8. THE Answer_Generator SHALL leave the Week 1 Embedder byte-identical and SHALL live in a module added for Week 3.

### Requirement 5: Grounding Checks

**User Story:** As the learner, I want proof that answers come from my notes, so that I can show the pipeline is retrieval-augmented rather than model recall.

#### Acceptance Criteria

1. WHEN Ask_My_Docs answers the Positive_Grounding_Question, THE Ask_My_Docs SHALL present a Verified_Answer whose Source_List holds at least one entry drawn from the Sample_Notes_Folder.
2. WHEN Ask_My_Docs answers the Negative_Grounding_Question, THE Ask_My_Docs SHALL present a Refusal naming the Refusal_Reason.
3. THE Evaluation_Question_Set SHALL hold at least one Positive_Grounding_Question and at least one Negative_Grounding_Question.
4. THE Evaluation_Report SHALL identify which question served as the Positive_Grounding_Question and which served as the Negative_Grounding_Question.

### Requirement 6: Citations and Source List

**User Story:** As a reader of an answer, I want every claim traceable to a numbered source, so that I can check the answer against the original note.

#### Acceptance Criteria

1. WHEN Ask_My_Docs presents an answer, THE Ask_My_Docs SHALL print the answer text followed by the Source_List.
2. THE Source_List SHALL hold one line per cited Citation_Number, each line naming the Citation_Number, the source path, the ordinal index, the start and end character offsets, and the Similarity_Score of the corresponding Supplied_Chunk.
3. THE Ask_My_Docs SHALL map each Citation_Marker of a presented answer to exactly one Supplied_Chunk of the Context_Block of that answer's Assembled_Prompt.
4. THE Source_List SHALL hold no entry whose Citation_Number is absent from the Citation_Table.
5. WHERE the presented answer cites a subset of the Supplied_Chunks, THE Source_List SHALL hold one line for each cited Citation_Number and SHALL omit the uncited Citation_Numbers.
6. THE Ask_My_Docs SHALL print the Source_List in ascending Citation_Number order.

### Requirement 7: Citation Validation

**User Story:** As the learner, I want fabricated citation numbers caught, so that a hallucinated reference is never presented as verified.

#### Acceptance Criteria

1. WHEN the Citation_Validator receives a Generated_Answer and a Citation_Table, THE Citation_Validator SHALL extract every Citation_Marker of the Generated_Answer text and SHALL return the set of Dangling_Citations.
2. IF the Generated_Answer holds at least one Dangling_Citation, THEN THE Ask_My_Docs SHALL present the answer as an Unverified_Answer and SHALL print a warning naming every Dangling_Citation.
3. IF the Generated_Answer holds no Citation_Marker, THEN THE Ask_My_Docs SHALL present the answer as an Unverified_Answer and SHALL print a warning naming the absence of citations.
4. WHEN the Generated_Answer holds at least one Citation_Marker and no Dangling_Citation, THE Ask_My_Docs SHALL present the answer as a Verified_Answer.
5. THE Answer_Log_Record SHALL record whether the presented answer was a Verified_Answer or an Unverified_Answer and SHALL record every Dangling_Citation found.

### Requirement 8: Refusal Path

**User Story:** As a reader of an answer, I want the tool to say it does not know, so that I am never handed an invented answer.

#### Acceptance Criteria

1. WHEN the Retrieval_Outcome is `NO_RELEVANT_CONTEXT`, THE Ask_My_Docs SHALL present a Refusal whose Refusal_Reason is `no-relevant-context` and SHALL send no request to the chat completions endpoint.
2. WHEN the Retrieval_Outcome is `EMPTY_COLLECTION`, THE Ask_My_Docs SHALL present a Refusal whose Refusal_Reason is `empty-collection` and SHALL send no request to the chat completions endpoint.
3. WHEN the Retrieval_Outcome is `RELEVANT_CONTEXT`, THE Ask_My_Docs SHALL build an Assembled_Prompt and invoke the Answer_Generator.
4. WHEN Ask_My_Docs presents a Refusal whose Refusal_Reason is `no-relevant-context`, THE Ask_My_Docs SHALL print the highest Similarity_Score of the Retrieval_Result and the Relevance_Threshold.
5. WHEN Ask_My_Docs presents a Refusal whose Refusal_Reason is `empty-collection`, THE Ask_My_Docs SHALL print the command that ingests the Sample_Notes_Folder.
6. WHEN Ask_My_Docs presents a Refusal, THE Ask_My_Docs SHALL print no Source_List.

### Requirement 9: End-to-End Ask Script

**User Story:** As the learner, I want one command that answers a question, so that I can demonstrate the whole pipeline in a single step.

#### Acceptance Criteria

1. WHEN the Ask_Script is invoked with one question string, THE Ask_Script SHALL invoke the Retriever, and on the Retrieval_Outcome `RELEVANT_CONTEXT` SHALL invoke the Prompt_Builder, the Answer_Generator, and the Citation_Validator, and SHALL print the presented answer and the Source_List.
2. WHERE the Ask_Script is invoked with an optional Top_K value, THE Ask_Script SHALL pass that value to the Retriever in place of the configured Top_K.
3. IF the Ask_Script is invoked with no question string, THEN THE Ask_Script SHALL print a usage message naming the required question argument and SHALL terminate with a non-zero exit status.
4. IF the Ask_Script is invoked with a question string that is empty after removal of surrounding whitespace, THEN THE Ask_Script SHALL print an error naming the empty question and SHALL terminate with a non-zero exit status.
5. WHEN the Ask_Script completes an answer attempt, THE Ask_Script SHALL write one Answer_Log_Record and SHALL leave the Week 2 Retrieval_Log behaviour unchanged.
6. THE Ask_Script SHALL implement `main(argv) -> int` over library code, SHALL route every console string through the Reporter, and SHALL raise typed exceptions from library code rather than calling process exit from library code.
7. WHEN the Ask_Script completes without error, THE Ask_Script SHALL return the exit status 0.

### Requirement 10: Answer Log

**User Story:** As the learner, I want every answer recorded, so that I can compare runs across refinement rounds.

#### Acceptance Criteria

1. WHEN Ask_My_Docs completes one answer attempt, THE Answer_Log SHALL receive exactly one appended Answer_Log_Record serialized as one JSON object on one line.
2. THE Answer_Log_Record SHALL hold the Answer_Log_Schema_Version, a timestamp, the question string, the Retrieval_Summary, the Prompt_Hash, the Chat_Model, the Token_Usage, the answer text, the cited Citation_Table entries, the Dropped_Chunk_Count, and the Retrieval_Outcome.
3. THE Answer_Log_Record SHALL hold the Prompt_Hash in place of the Assembled_Prompt text.
4. THE Answer_Log_Record SHALL hold no API key value and no substring of the API key of length 8 or greater.
5. WHEN Ask_My_Docs presents a Refusal, THE Answer_Log SHALL receive one Answer_Log_Record holding the Refusal_Reason, an empty answer text, an empty citation list, and a Token_Usage whose three counts are each zero.
6. IF the parent directory of the Answer_Log is absent, THEN THE Ask_My_Docs SHALL create that directory before appending the Answer_Log_Record.
7. THE repository `.gitignore` SHALL exclude the Answer_Log because the Answer_Log holds verbatim note text.

### Requirement 11: Evaluation Run

**User Story:** As the learner, I want a repeatable evaluation over at least ten questions, so that I can measure answer quality and show improvement.

#### Acceptance Criteria

1. WHEN the Evaluation_Script is invoked in generate mode, THE Evaluation_Script SHALL run every distinct non-empty question of the Evaluation_Question_Set through the same path the Ask_Script uses and SHALL record, per question, the answer text, the cited Citation_Table entries, every retrieved Similarity_Score, and the Retrieval_Outcome.
2. THE Evaluation_Script SHALL write one Quality_Rating field per question, left empty in generate mode for the learner to enter by hand.
3. WHEN the Evaluation_Script is invoked in score mode, THE Evaluation_Script SHALL read the hand-entered Quality_Ratings, compute the Mean_Quality_Rating over the non-empty Quality_Ratings, and write the Evaluation_Report.
4. IF a hand-entered Quality_Rating is present and is not an integer in the range 1 to 5 inclusive, THEN THE Evaluation_Script SHALL raise an evaluation error naming the question and the rejected value.
5. IF every Quality_Rating is empty in score mode, THEN THE Evaluation_Script SHALL raise an evaluation error naming the absence of ratings.
6. IF the Evaluation_Question_Set holds fewer than 10 distinct non-empty question lines, THEN THE Evaluation_Script SHALL raise an evaluation error naming the question count and the required minimum of 10.
7. THE Evaluation_Report SHALL hold the Configuration_Stamp of the run that produced the recorded answers, with the API key value replaced by Reporter-style redaction.
8. THE Evaluation_Report SHALL identify at least one question whose Retrieval_Outcome was `NO_RELEVANT_CONTEXT` or `EMPTY_COLLECTION`, showing that the refusal path was exercised.
9. THE Evaluation_Report SHALL hold the Mean_Quality_Rating rounded to two decimal places.

### Requirement 12: Error Handling

**User Story:** As the learner, I want each failure reported with its cause and remedy, so that I can fix the run rather than read a traceback.

#### Acceptance Criteria

1. IF the Sample_Notes_Folder holds no supported source file, THEN THE Ask_Script SHALL print an error naming the Sample_Notes_Folder path and the supported file extensions and SHALL terminate with a non-zero exit status.
2. IF the Retrieval_Outcome is `NO_RELEVANT_CONTEXT`, THEN THE Ask_Script SHALL present a Refusal and SHALL return the exit status 0.
3. IF the chat completions endpoint fails on every attempt up to the Max_Retry_Attempts, THEN THE Ask_Script SHALL print an error naming the Chat_Model, the attempt count, and the final failure category, and SHALL terminate with a non-zero exit status.
4. IF the Chat_Provider requires an API key and the API key environment variable of that Chat_Provider is absent or empty, THEN THE Ask_Script SHALL print an error naming the required environment variable and the setting name Chat_Provider and SHALL terminate with a non-zero exit status.
5. IF the Prompt_Builder raises a context budget error, THEN THE Ask_Script SHALL print that error naming the Context_Budget and the Chunk_Size and SHALL terminate with a non-zero exit status.
6. IF the chat completions endpoint returns Malformed_Model_Output, THEN THE Ask_Script SHALL print an error naming the Chat_Model and the malformed response condition, SHALL write one Answer_Log_Record recording the condition, and SHALL terminate with a non-zero exit status.
7. THE Ask_My_Docs SHALL apply Reporter-style redaction to every error message, so that no error message holds the API key or any substring of the API key of length 8 or greater.
8. THE Ask_My_Docs library code SHALL raise typed exceptions for each condition of this requirement and SHALL leave exit status selection to the scripts.
9. THE Ask_Script and THE Evaluation_Script SHALL select a distinct non-zero exit status per error condition, and THE Readme SHALL list every exit status with its condition.

### Requirement 13: Readme

**User Story:** As a mentor cloning the repository, I want a README that explains the tool and lets me run it, so that I can reproduce the learner's results.

#### Acceptance Criteria

1. THE Readme SHALL hold a one-paragraph description of Ask_My_Docs and of the committed Sample_Notes_Folder dataset.
2. THE Readme SHALL hold setup instructions covering the editable install, the environment variables with their defaults, the ingest command, and the Ask_Script command.
3. THE Readme SHALL hold instructions for adding the reader's own documents to the Sample_Notes_Folder and re-running the ingest command.
4. THE Readme SHALL hold a What_I_Learned_Section explaining, in the learner's own words, what an embedding is, why Chunk_Size and Chunk_Overlap matter, and how the retrieval-augmented generation pipeline works end to end.
5. THE Readme SHALL name the chosen Chunk_Size and the chosen Top_K and SHALL give the reasoning for each, referencing the Week 2 Top_K_Report.
6. THE Readme SHALL hold at least one example question with its presented answer and its cited Source_List, shown as the committed Example_Screenshot.
7. THE Readme SHALL hold no API key value and no substring of an API key of length 8 or greater.
8. THE Readme SHALL list every exit status the Ask_Script and the Evaluation_Script return, each paired with the condition that produces it.

### Requirement 14: Written Learning Artifacts

**User Story:** As the learner, I want my understanding written down, so that I can explain the design choices without notes.

#### Acceptance Criteria

1. THE RAG_Versus_Fine_Tuning_Note SHALL compare retrieval-augmented generation against fine-tuning on at least data freshness, cost, traceability of answers, and the kind of task each suits, and SHALL state which approach this project uses and why.
2. THE Demo_Script SHALL hold an ordered shot list for a demo of 3 to 4 minutes covering the ingest step and 2 to 3 live questions, with an explanation of each pipeline stage.
3. THE Demo_Script SHALL name, for each shot, the command shown and the point the learner makes in that shot.
4. THE Learning_Notes SHALL hold the RAG_Versus_Fine_Tuning_Note and the Demo_Script as committed markdown files.

### Requirement 15: Submission and Self-Check Checklists

**User Story:** As the learner, I want the submission and self-check items written down, so that nothing is missed before the deadline.

#### Acceptance Criteria

1. THE Submission_Checklist SHALL hold one checkable item for each of: a public GitHub repository named `ask-my-docs`, the committed Sample_Notes_Folder, the recorded 3 to 4 minute demo video, and the repository link plus video link sent to the mentor before Sunday 4 October 2026 23:59.
2. THE Self_Check_Checklist SHALL hold one checkable item for each of: explaining an embedding without using the word "vector", explaining why Chunk_Size affects the quality of retrieved context, explaining how a vector database finds similar Chunks, explaining the full pipeline in under one minute, showing answers that cite their source, and showing graceful handling of a question with no good answer in the notes.
3. THE Submission_Checklist SHALL record the submission deadline as Sunday 4 October 2026 23:59.
4. THE Submission_Checklist and the Self_Check_Checklist SHALL be committed markdown files in the Learning_Notes.

### Requirement 16: Repository Hygiene

**User Story:** As a mentor cloning the repository, I want a clean, safe, reproducible repository, so that I can run the tool without receiving anyone's secrets or personal notes.

#### Acceptance Criteria

1. THE committed Sample_Notes_Folder SHALL hold small non-sensitive placeholder notes sufficient to answer the Readme example question.
2. THE repository `.gitignore` SHALL exclude `.env`, the personal note files of the Sample_Notes_Folder other than its README, the Chroma Persist_Directory, the Retrieval_Log and Answer_Log directory, and the relevance review comma-separated files.
3. THE `pyproject.toml` SHALL pin every dependency, including the chat completions client dependency, to an exact version.
4. THE Week 3 modules SHALL hold comments explaining the prompt construction rules, the context budget reduction rule, and the citation validation rule.
5. THE Week 3 work SHALL leave the Week 1 and Week 2 modules listed as frozen byte-identical, and SHALL change only `config.py` and `stores/factory.py` among pre-existing modules, and only by addition.
