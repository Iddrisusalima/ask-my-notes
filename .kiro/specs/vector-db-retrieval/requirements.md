# Requirements Document

## Introduction

Week 2 (Sep 21–27) of the "Ask My Docs" learning project replaces the Week 1 in-process list with a real, local, free vector database and builds retrieval on top of it. Week 1 delivered Configuration, Embedder (OpenAI or Sentence Transformers) with batching and retry, Similarity_Calculator, Document_Loader with PDF_Loader and Markdown_Loader, Chunker, Vector_Store_Interface, In_Memory_Store, and four scripts. Week 2 adds a second implementation of that same Vector_Store_Interface, Chroma_Store, backed by a persistent on-disk Chroma collection.

Chroma is the required database for Week 2: it installs as a Python package, needs no API key, no server process, and no account, and its persistent client writes a collection to a local directory. LanceDB is the documented alternative considered, and Pinecone is the documented hosted comparison; neither is installed.

Every Embedding_Vector stored in or queried against Chroma is produced by the Week 1 Embedder. Chroma is never asked to embed text, so the Embedder remains the single embedding path and the Week 1 loader, chunker, and embedder modules are not modified.

The Week 2 deliverables are: a persistent Chroma collection holding every Chunk of the Sample_Notes_Folder; an ingest run that is idempotent and incremental, driven by a per-file content hash, so unchanged files are skipped, changed files have their old Chunks replaced, and deleted files have their Chunks removed; a Retriever that turns a question string into the top-K most similar Chunks with scores and source metadata; a no-relevant-context path that reports the absence of a match instead of returning noise when the best Similarity_Score falls below a configured threshold; a command-line query script; an append-only machine-readable retrieval log; a Top-K tuning experiment at K values 3, 5, and 10; a manual relevance review that yields precision@K over hand-labelled results; a written comparison of Chroma against at least one alternative plus an explanation of how the database performs similarity search; and a conformance test suite that runs the same behavioural tests against both In_Memory_Store and Chroma_Store.

Out of scope for Week 2: prompt construction, LLM answer generation, citations inside generated answers, and any user interface. Those belong to Week 3, which consumes the Retriever output and uses the no-relevant-context signal to refuse to answer.

## Glossary

Terms defined in the Week 1 requirements document — Ask_My_Docs, Configuration, Embedding_Provider, Embedder, Embedding_Vector, Embedding_Dimensionality, Similarity_Calculator, Cosine_Similarity, Sample_Notes_Folder, Document_Loader, PDF_Loader, Markdown_Loader, Document, Chunker, Chunk, Chunk_Size, Chunk_Overlap, Max_Input_Length, Max_Batch_Size, Max_Chunks_Per_Run, Vector_Store_Interface, In_Memory_Store, Learning_Notes — keep their Week 1 meanings and are used unchanged here. Week 2 adds the following terms.

- **Vector_Database**: A storage system that holds Embedding_Vectors together with identifiers, text, and metadata, and that answers nearest-neighbour queries over those Embedding_Vectors.
- **Chroma**: The selected local Vector_Database, used through its Python package and its persistent client, requiring no API key, no account, and no separate server process.
- **LanceDB**: The alternative local Vector_Database considered and documented in Week 2 but not installed.
- **Pinecone**: The hosted Vector_Database considered and documented in Week 2 but not installed.
- **HNSW**: Hierarchical Navigable Small World, the multi-layer proximity-graph index structure Chroma uses to answer approximate nearest-neighbour queries.
- **Exact_Search_Limit**: The stored item count of 1000 at or below which a store query is required to return exactly the K Stored_Items with the highest Similarity_Score, and above which the Recall_Floor applies instead because the HNSW index answers the query approximately.
- **Recall_Floor**: The minimum proportion, 95 percent rounded down, of the SearchHit values a store query returns that are required to be among the K Stored_Items with the highest Similarity_Score computed by exhaustive comparison, applying while the stored item count exceeds the Exact_Search_Limit.
- **Chroma_Store**: The Ask_My_Docs component that implements the Vector_Store_Interface against a persistent Chroma collection.
- **Chroma_Collection**: The single named Chroma collection that holds every stored Chunk of the Sample_Notes_Folder.
- **Collection_Name**: The configured name of the Chroma_Collection. Default `ask_my_docs`.
- **Persist_Directory**: The configured local filesystem directory in which Chroma writes the Chroma_Collection so that the collection survives process exit. Default `.chroma`.
- **Distance_Metric**: The configured Chroma distance function recorded on the Chroma_Collection at creation time. Week 2 supports the single value `cosine`.
- **Chroma_Distance**: The non-negative number Chroma returns for a query result under the configured Distance_Metric.
- **Similarity_Score**: The number in the closed interval [-1, 1] that Ask_My_Docs derives from a Chroma_Distance, defined so that a larger Similarity_Score means greater semantic closeness and so that the value equals the Cosine_Similarity of the query Embedding_Vector and the stored Embedding_Vector.
- **Collection_Fingerprint**: The record written into the Chroma_Collection metadata at creation time, holding the Distance_Metric, the Embedding_Provider identifier, the model name, and the Embedding_Dimensionality.
- **Store_Admin_Interface**: The additive Week 2 contract, implemented by both In_Memory_Store and Chroma_Store, defining the operations needed for incremental ingest: upsert a batch, fetch stored items by chunk id, delete by chunk id, delete by source file path, and reset the store to empty.
- **SearchHit**: One result of a store query, carrying the returned Chunk, the Similarity_Score of that Chunk against the query Embedding_Vector, and the insertion index of that Chunk in the store.
- **Stored_Item**: One chunk id together with the Chunk text, the Chunk metadata, and the Embedding_Vector held under that chunk id in a store.
- **Collection_State**: The set of Stored_Items in a store, compared by chunk id, Chunk text, Chunk metadata excluding insertion index, and Embedding_Vector. Two stores have equal Collection_State when the two sets are equal under those fields.
- **Content_Hash**: The lower-case hexadecimal SHA-256 digest of the complete byte content of one source file.
- **Source_Manifest**: The local file that records, for every source file already ingested, the source file path, the Content_Hash, the number of Chunks stored for that source file, the Chunk_Size, the Chunk_Overlap, the Embedding_Provider identifier, and the model name. Default path `.chroma/ingest-manifest.json`.
- **Ingest_Script**: The runnable script that discovers, loads, chunks, embeds, and upserts Chunks into the Chroma_Collection, and that updates the Source_Manifest.
- **Unchanged_Source**: A discovered source file whose Content_Hash, Chunk_Size, Chunk_Overlap, Embedding_Provider identifier, and model name all equal the values recorded for that source file path in the Source_Manifest.
- **Changed_Source**: A discovered source file that has a Source_Manifest entry and is not an Unchanged_Source.
- **New_Source**: A discovered source file that has no Source_Manifest entry.
- **Deleted_Source**: A source file path that has a Source_Manifest entry and that is not among the discovered supported files of the Sample_Notes_Folder.
- **Retriever**: The Ask_My_Docs component that accepts a question string and a value of Top_K, embeds the question string with the Embedder, queries the Chroma_Store, and returns an ordered Retrieval_Result.
- **Top_K**: The configured number of Chunks the Retriever requests for one question. Default 5. Permitted range 1 to 100.
- **Retrieval_Result**: The ordered sequence of SearchHit values the Retriever returns for one question, each carrying the Chunk, the Similarity_Score, and the Chunk's source metadata, ordered by descending Similarity_Score with ties broken by ascending insertion index.
- **Relevance_Threshold**: The configured minimum Similarity_Score at which the highest-scoring SearchHit is treated as relevant context. Default 0.30. Permitted range -1.0 to 1.0.
- **No_Relevant_Context**: The Retriever outcome reported when the Chroma_Collection holds at least one Stored_Item and the highest Similarity_Score of the Retrieval_Result is less than the Relevance_Threshold.
- **Query_Script**: The runnable command-line script that accepts a question string and an optional value of Top_K and prints the Retrieval_Result.
- **Retrieval_Log**: The append-only machine-readable file to which one Retrieval_Log_Record is written for every Retriever invocation. Default path `logs/retrievals.jsonl`.
- **Retrieval_Log_Record**: One JSON object, serialized on a single line, recording one Retriever invocation.
- **Log_Schema_Version**: The version value held in every Retrieval_Log_Record that identifies the field set of that Retrieval_Log_Record, so that a reader can tell which fields to expect. Week 2 writes the single value `1`.
- **Question_Set**: The UTF-8 text file holding the Week 2 evaluation questions, one question per non-empty line. Default path `question-sets/retrieval-questions.txt`. Expected question count 5 to 10.
- **Question_Identifier**: The one-based position of a distinct non-empty question line of the Question_Set, counted in Question_Set file order over the distinct non-empty lines only, used to identify that question in the Top_K_Report, the Relevance_Review_File, and the Precision_Report.
- **Top_K_Experiment_Script**: The runnable script that runs the whole Question_Set at Top_K values 3, 5, and 10 and writes the Top_K_Report.
- **Top_K_Report**: The markdown file holding the per-question and aggregate Top-K statistics and the written tradeoff observation. Default path `reports/topk-experiment.md`.
- **Relevance_Review_File**: The comma-separated file, one row per retrieved Chunk, that the learner edits by hand to mark each retrieved Chunk relevant or not relevant. Default path `reports/relevance-review.csv`.
- **Relevance_Label**: The hand-entered value in one Relevance_Review_File row, being `y` for relevant, `n` for not relevant, or empty for not yet labelled.
- **Precision_At_K**: For one question and one value of Top_K, the count of rows labelled `y` divided by the count of rows present for that question at that value of Top_K.
- **Relevance_Review_Script**: The runnable script that generates the Relevance_Review_File from the Question_Set and that computes Precision_At_K from the labelled Relevance_Review_File.
- **Precision_Report**: The markdown file holding the per-question and mean Precision_At_K values. Default path `reports/precision-at-k.md`.
- **Comparison_Note**: The written markdown artifact comparing Chroma against at least one of LanceDB and Pinecone and explaining how a Vector_Database performs similarity search. Default path `learning-notes/vector-db-comparison.md`.
- **Store_Conformance_Suite**: The single set of behavioural tests that is executed once against In_Memory_Store and once against Chroma_Store.

## Requirements

### Requirement 1: Week 2 Configuration Extension

**User Story:** As a learner, I want the vector database, retrieval, and logging settings supplied through environment variables like every Week 1 setting, so that I can retune retrieval without editing code.

#### Acceptance Criteria

1. THE Configuration SHALL read the vector store selection, the Persist_Directory path, the Collection_Name, the Distance_Metric, the default Top_K, the Relevance_Threshold, the Retrieval_Log path, the Source_Manifest path, and the Question_Set path from environment variables, in addition to every setting named in Week 1 Requirement 1.
2. IF an environment variable named in criterion 1 is absent or contains only whitespace, THEN THE Configuration SHALL apply that setting's documented default value: vector store selection `chroma`, Persist_Directory `.chroma`, Collection_Name `ask_my_docs`, Distance_Metric `cosine`, Top_K 5, Relevance_Threshold 0.30, Retrieval_Log path `logs/retrievals.jsonl`, Source_Manifest path `.chroma/ingest-manifest.json`, and Question_Set path `question-sets/retrieval-questions.txt`.
3. THE Configuration SHALL match the vector store selection value and the Distance_Metric value against the supported values case-insensitively and after trimming leading and trailing whitespace.
4. IF the vector store selection value, after trimming and lower-casing, equals neither `memory` nor `chroma`, THEN THE Configuration SHALL terminate with a non-zero exit status and an error message listing the two supported values and showing the rejected value.
5. IF the Distance_Metric value, after trimming and lower-casing, does not equal `cosine`, THEN THE Configuration SHALL terminate with a non-zero exit status and an error message stating that Week 2 supports the single Distance_Metric value `cosine` and showing the rejected value.
6. IF the configured Top_K value cannot be parsed as an integer, or parses as an integer less than 1 or greater than 100, THEN THE Configuration SHALL terminate with a non-zero exit status and an error message naming the environment variable, showing the rejected value, and stating the permitted range of 1 to 100.
7. IF the configured Relevance_Threshold value cannot be parsed as a decimal number, or parses as a number less than -1.0 or greater than 1.0, THEN THE Configuration SHALL terminate with a non-zero exit status and an error message naming the environment variable, showing the rejected value, and stating the permitted range of -1.0 to 1.0.
8. IF the Collection_Name, after trimming, is shorter than 3 characters, is longer than 63 characters, has a first or last character that is not an ASCII letter or ASCII digit, or contains any character other than an ASCII letter, an ASCII digit, a hyphen, or an underscore, THEN THE Configuration SHALL terminate with a non-zero exit status and an error message naming the environment variable, showing the rejected value, and stating the permitted character set, the first-and-last-character rule, and the length range of 3 to 63 characters.
9. THE Configuration SHALL resolve the Persist_Directory path, the Retrieval_Log path, the Source_Manifest path, and the Question_Set path relative to the repository root when the configured value is not an absolute path.
10. THE Ask_My_Docs repository SHALL list every environment variable named in criterion 1, with its default value and permitted range, in the example environment file and in the README.
11. THE Ask_My_Docs repository SHALL exclude the Persist_Directory, the Source_Manifest, the Retrieval_Log, and the Relevance_Review_File from version control.
12. WHERE the selected Embedding_Provider is a Sentence Transformers model, THE Ask_My_Docs SHALL complete every Week 2 operation without reading the API key environment variable and without issuing any network request.

### Requirement 2: Local Persistent Vector Database Selection

**User Story:** As a learner, I want a vector database that runs entirely on my machine for free, so that I can index my notes without an account, a hosted service, or a second API key.

#### Acceptance Criteria

1. THE Ask_My_Docs dependency manifest SHALL pin an exact version of the Chroma Python package and SHALL name no hosted Vector_Database client package.
2. WHEN the Chroma_Store is constructed, THE Chroma_Store SHALL open a Chroma persistent client rooted at the Persist_Directory and SHALL issue no network request and read no API key.
3. WHEN the Chroma_Store is constructed and the Persist_Directory does not exist, THE Chroma_Store SHALL create the Persist_Directory together with any missing parent directory.
4. WHEN a process that used the Chroma_Store exits, no other process holds the Persist_Directory, and a later process constructs a Chroma_Store with the same Persist_Directory, Collection_Name, and Distance_Metric, THE Chroma_Store SHALL report the stored item count written by the earlier process and SHALL report a Collection_State equal to the Collection_State written by the earlier process.
5. THE README SHALL state the single command that installs the Chroma package, the command that runs the Ingest_Script, and the command that runs the Query_Script.
6. THE README SHALL state that Chroma requires no API key and that the Persist_Directory is safe to delete, and SHALL name the command sequence that rebuilds the Chroma_Collection from the Sample_Notes_Folder after that deletion.
7. IF the Chroma package is absent from the active Python environment, THEN THE Ask_My_Docs SHALL terminate with a non-zero exit status and an error message naming the missing package and the documented install command.
8. IF the resolved Persist_Directory path exists and is not a directory, or cannot be created, read, or written by the running process, THEN THE Chroma_Store SHALL terminate with a non-zero exit status and an error message naming the resolved absolute path and the failure reason, before performing any add, upsert, or query operation.
9. IF the Persist_Directory holds content that the Chroma client cannot open as a Chroma database, or holds a Chroma database written by a Chroma package version the pinned version cannot read, THEN THE Chroma_Store SHALL terminate with a non-zero exit status and an error message naming the resolved absolute path, the failure reason, and the documented reset command, SHALL leave the Persist_Directory content unmodified, and SHALL perform no add, upsert, or query operation.
10. IF a Chroma_Store construction, add, upsert, delete, or reset operation fails because another process holds the Persist_Directory, THEN THE Chroma_Store SHALL terminate with a non-zero exit status and an error message naming the resolved absolute path and stating that one Persist_Directory supports one Ask_My_Docs process at a time, and SHALL leave the Collection_State unchanged.

### Requirement 3: Chroma_Store as a Second Vector_Store_Interface Implementation

**User Story:** As a learner, I want Chroma behind the interface In_Memory_Store already implements, so that swapping the storage layer changes one module and leaves my loader, chunker, and embedder untouched.

#### Acceptance Criteria

1. THE Chroma_Store SHALL implement the Vector_Store_Interface operations defined in Week 1 Requirement 10.1: add a batch of Chunks with their Embedding_Vectors, report the stored item count, and return the K stored Chunks with the highest Similarity_Score to a supplied query Embedding_Vector.
2. WHEN the Chroma_Store adds a batch of Chunks, THE Chroma_Store SHALL supply to Chroma, for every Chunk in the batch, the chunk id produced by the Chunk, the Chunk text as the stored document, the flat metadata mapping produced by the Chunk, and the Embedding_Vector computed by the Embedder.
3. WHEN the Chroma_Store creates or opens the Chroma_Collection, THE Chroma_Store SHALL configure the Chroma_Collection with no Chroma embedding function and SHALL supply an Embedding_Vector for every add, upsert, and query operation.
4. WHEN the Chroma_Store adds a batch of Chunks, THE Chroma_Store SHALL store, in the metadata of every added Stored_Item, an insertion index held as an integer metadata value that is greater than the insertion index of every Stored_Item already held in the Chroma_Collection.
5. WHILE the stored item count is at most 1000, WHEN the Chroma_Store is queried with a query Embedding_Vector and an integer value K of at least 1, THE Chroma_Store SHALL return at most K SearchHit values that are exactly the Stored_Items with the highest Similarity_Score, ordered by descending Similarity_Score with ties broken by ascending insertion index, and SHALL return the same SearchHit values in the same order for every repetition of that query while the Collection_State is unchanged.
6. WHEN the Chroma_Store is queried with a value K greater than the stored item count, THE Chroma_Store SHALL return exactly the stored item count of SearchHit values and SHALL raise no error.
7. IF a query supplies a value of K that is not an integer or is less than 1, THEN THE Chroma_Store SHALL raise an error stating the permitted range of K and SHALL return no result.
8. IF a query supplies an Embedding_Vector whose length differs from the Embedding_Dimensionality recorded in the Collection_Fingerprint, or whose Euclidean norm is at most 1e-12, THEN THE Chroma_Store SHALL raise an error naming the violated property and reporting both lengths when the lengths differ, and SHALL issue no Chroma query.
9. IF an add operation supplies a number of Embedding_Vectors different from the number of Chunks, or supplies Embedding_Vectors that are not all of the same length, or supplies an Embedding_Vector whose length differs from the Embedding_Dimensionality recorded in the Collection_Fingerprint, THEN THE Chroma_Store SHALL raise an error reporting both counts or both lengths and SHALL leave the Collection_State and the stored item count unchanged.
10. WHEN an add operation supplies an empty batch, THE Chroma_Store SHALL leave the stored item count unchanged and SHALL raise no error.
11. IF an add operation supplies a chunk id that is already stored in the Chroma_Collection, THEN THE Chroma_Store SHALL raise an error naming every duplicate chunk id and SHALL leave the Collection_State and the stored item count unchanged.
12. IF an add operation supplies a chunk id that appears more than once within the same batch, THEN THE Chroma_Store SHALL raise an error naming every repeated chunk id and SHALL leave the Collection_State and the stored item count unchanged.
13. THE Store_Admin_Interface SHALL define an operation to upsert a batch of Chunks with their Embedding_Vectors, an operation to fetch Stored_Items by chunk id, an operation to delete Stored_Items by chunk id, an operation to delete every Stored_Item whose source file path equals a supplied path, and an operation to reset the store to a stored item count of 0.
14. THE Chroma_Store and THE In_Memory_Store SHALL each implement every Store_Admin_Interface operation with the behaviour stated in this document for both implementations.
15. WHEN the upsert operation receives a Chunk whose chunk id is already stored, THE store SHALL replace the stored Chunk text, the stored Chunk metadata, and the stored Embedding_Vector held under that chunk id, SHALL retain the insertion index already stored under that chunk id, and SHALL leave the stored item count unchanged.
16. WHEN the delete-by-source-file-path operation receives a source file path, THE store SHALL remove every Stored_Item whose stored source file path metadata equals that path, SHALL leave every other Stored_Item unchanged, and SHALL reduce the stored item count by the number of removed Stored_Items.
17. WHEN the delete-by-source-file-path operation or the delete-by-chunk-id operation receives a value that matches no Stored_Item, THE store SHALL leave the Collection_State unchanged and SHALL raise no error.
18. WHEN Chroma writes more Stored_Items in one request than the Chroma client accepts in a single request, THE Chroma_Store SHALL split the Stored_Items into consecutive segments that the Chroma client accepts and SHALL preserve the insertion index ordering of criterion 4 across those segments.
19. IF an add or upsert operation supplies a Chunk metadata mapping holding a value that is not a string, an integer, a floating-point number, or a boolean, THEN THE Chroma_Store SHALL raise an error naming every offending metadata key and stating the permitted value types, and SHALL leave the Collection_State and the stored item count unchanged.
20. WHILE the stored item count exceeds 1000, WHEN the Chroma_Store is queried with a query Embedding_Vector and an integer value K of at least 1, THE Chroma_Store SHALL return SearchHit values of which at least 95 percent, rounded down, are among the K Stored_Items with the highest Similarity_Score computed by exhaustive comparison, and SHALL order the returned SearchHit values by descending Similarity_Score with ties broken by ascending insertion index.

### Requirement 4: Explicit Distance Metric and Similarity Score Conversion

**User Story:** As a learner, I want the distance metric named explicitly and the returned distance converted into the same cosine similarity number I computed by hand in Week 1, so that scores from Chroma and scores from my own code mean the same thing.

#### Acceptance Criteria

1. WHEN the Chroma_Store creates the Chroma_Collection, THE Chroma_Store SHALL set the Chroma collection distance function to the configured Distance_Metric value `cosine` explicitly rather than relying on a Chroma default.
2. WHEN the Chroma_Store creates the Chroma_Collection, THE Chroma_Store SHALL write the Collection_Fingerprint, holding the Distance_Metric, the Embedding_Provider identifier, the model name, and the Embedding_Dimensionality, into the Chroma_Collection metadata.
3. WHERE the Distance_Metric is `cosine`, THE Chroma_Store SHALL treat the Chroma_Distance as a value in the closed interval [0.0, 2.0] that is 1 minus the Cosine_Similarity of the unit-normalized query and stored Embedding_Vectors, SHALL compute the Similarity_Score as 1 minus that Chroma_Distance independently of the Euclidean norms of the two Embedding_Vectors, and SHALL clamp the computed value into the closed interval [-1.0, 1.0].
4. FOR ALL pairs of Embedding_Vectors A and B whose lengths equal the Embedding_Dimensionality, whose Euclidean norms both lie in the closed interval [1e-3, 1e4], and whose every element has an absolute value of at most 1e3, where A is stored in the Chroma_Collection and B is supplied as the query Embedding_Vector, the Similarity_Score the Chroma_Store returns for A SHALL equal the value the Similarity_Calculator returns for the pair (A, B) within an absolute tolerance of 1e-5, that tolerance accounting for the 32-bit floating-point precision at which the Chroma_Collection stores an Embedding_Vector.
5. THE Learning_Notes SHALL state the Distance_Metric used, the formula that converts a Chroma_Distance into a Similarity_Score, and the numeric tolerance at which that conversion is verified against the Similarity_Calculator.
6. IF the Chroma_Collection opened at the Persist_Directory records a Distance_Metric, an Embedding_Provider identifier, a model name, or an Embedding_Dimensionality in the Collection_Fingerprint that differs from the corresponding Configuration value, THEN THE Chroma_Store SHALL terminate with a non-zero exit status and an error message reporting both values for every differing field and naming the documented reset command, before performing any add, upsert, or query operation.
7. FOR ALL Retrieval_Results, every Similarity_Score in the Retrieval_Result SHALL lie within the closed interval [-1.0, 1.0].
8. IF the Chroma_Collection opened at the Persist_Directory holds no Collection_Fingerprint in its metadata, or holds a Collection_Fingerprint from which the Distance_Metric, the Embedding_Provider identifier, the model name, or the Embedding_Dimensionality cannot be read, THEN THE Chroma_Store SHALL terminate with a non-zero exit status and an error message naming the resolved absolute Persist_Directory path, the Collection_Name, the missing or unreadable fields, and the documented reset command, before performing any add, upsert, or query operation.

### Requirement 5: Persistence and Round-Trip Fidelity

**User Story:** As a learner, I want the collection written to disk and read back exactly as stored, so that re-running my scripts does not re-embed unchanged documents and so that I can trust what the database gives back.

#### Acceptance Criteria

1. WHEN the Ingest_Script completes and the process exits, THE Chroma_Collection SHALL remain readable at the Persist_Directory by a later process without any further Embedder call.
2. FOR ALL batches of Chunks with Embedding_Vectors stored in the Chroma_Collection, the Chunk text fetched by chunk id SHALL equal the stored Chunk text character for character.
3. FOR ALL batches of Chunks with Embedding_Vectors stored in the Chroma_Collection, the source file path, ordinal index, start offset, and end offset fetched by chunk id SHALL equal the corresponding values of the stored Chunk.
4. FOR ALL batches of Chunks with Embedding_Vectors stored in the Chroma_Collection, every element of the Embedding_Vector fetched by chunk id SHALL equal the corresponding element of the stored Embedding_Vector within an absolute tolerance of 1e-5 where that element has an absolute value of at most 1.0, and within a relative tolerance of 1e-5 otherwise, those tolerances accounting for the 32-bit floating-point precision at which the Chroma_Collection stores an Embedding_Vector.
5. FOR ALL non-empty sets of at most 1000 Chunks stored in the Chroma_Collection, querying the Chroma_Store with the Embedding_Vector of a stored Chunk and a value K of 1 SHALL return that stored Chunk, unless another Stored_Item has a Similarity_Score equal to that of the stored Chunk within 1e-9, in which case the returned Stored_Item SHALL be whichever of the tied Stored_Items has the lowest insertion index.
6. WHEN the reset operation completes, THE Chroma_Store SHALL report a stored item count of 0 and SHALL fetch no Stored_Item for any previously stored chunk id.
7. FOR ALL sequences of upsert and delete operations, the stored item count SHALL equal the number of distinct chunk ids fetchable from the store.

### Requirement 6: Store Conformance and Differential Equivalence

**User Story:** As a learner, I want one set of behavioural tests run against both stores, so that I can prove the Chroma implementation and the in-memory implementation behave the same and that the interface is a real seam.

#### Acceptance Criteria

1. THE Store_Conformance_Suite SHALL execute every test case once against In_Memory_Store and once against Chroma_Store, using the same test code for both stores.
2. THE Store_Conformance_Suite SHALL contain at least one test case for each acceptance criterion of Week 1 Requirement 10 and for each acceptance criterion of Requirement 3 of this document that states store behaviour.
3. FOR ALL sequences of at least 1 and at most 50 Chunks with Embedding_Vectors of equal length, Euclidean norm in the closed interval [1e-3, 1e4], and every element of absolute value at most 1e3, added in the same order to an empty In_Memory_Store and an empty Chroma_Store, and FOR ALL query Embedding_Vectors satisfying those same bounds and all integer values of K from 1 to 50, the sequence of chunk ids the Chroma_Store returns SHALL equal the sequence of chunk ids the In_Memory_Store returns, except that a position where the two stores return different chunk ids is permitted when the Similarity_Scores of the two returned Stored_Items differ by at most 1e-5.
4. FOR ALL sequences described in criterion 3, the Similarity_Score the Chroma_Store returns at each position SHALL equal the Cosine_Similarity the In_Memory_Store returns at that position within an absolute tolerance of 1e-5.
5. FOR ALL Retrieval_Results and all query Embedding_Vectors, the sequence of Similarity_Scores returned by either store SHALL be monotonically non-increasing from the first position to the last position.
6. FOR ALL sequences described in criterion 3 and all pairs of integer values K1 and K2 where 1 is at most K1 and K1 is less than K2 and K2 is at most 50, the sequence of chunk ids the Chroma_Store returns for K1 SHALL equal the first K1 chunk ids of the sequence that same store returns for K2, except at a position where the Similarity_Scores of the two compared Stored_Items differ by at most 1e-5.
7. FOR ALL sequences described in criterion 3, the stored item count reported by the Chroma_Store SHALL equal the stored item count reported by the In_Memory_Store.
8. WHERE a test case in the Store_Conformance_Suite exercises the Chroma_Store, THE test case SHALL use a Persist_Directory created for that test case and removed after that test case, SHALL be the only test case accessing that Persist_Directory for the duration of that test case, and SHALL issue no request to a remote Embedding_Provider.
9. WHERE the Store_Conformance_Suite exercises the Chroma_Store with a stored item count greater than 1000, THE Store_Conformance_Suite SHALL assert the recall floor of Requirement 3 criterion 20 instead of chunk-id sequence equality, and SHALL report the measured count of returned chunk ids that appear among the K chunk ids the In_Memory_Store returns.

### Requirement 7: Content-Hash Source Manifest

**User Story:** As a learner, I want a record of what I already ingested and the hash of each file, so that a second ingest run can tell unchanged files from changed ones without re-embedding everything.

#### Acceptance Criteria

1. WHEN the Ingest_Script processes a discovered source file, THE Ingest_Script SHALL read the complete byte content of that source file in one sequential read pass, SHALL compute the Content_Hash of that source file from the bytes obtained in that read pass, and SHALL supply those same bytes to the Document_Loader, so that no further read of that source file occurs between the Content_Hash computation and the loading of that source file.
2. WHEN the Ingest_Script commits the Chunks of one source file to the Chroma_Collection, THE Ingest_Script SHALL write a Source_Manifest entry for that source file holding the source file path relative to the Sample_Notes_Folder, the Content_Hash, the number of Chunks stored for that source file, the Chunk_Size, the Chunk_Overlap, the Embedding_Provider identifier, and the model name.
3. WHEN the Ingest_Script writes the Source_Manifest, THE Ingest_Script SHALL serialize the Source_Manifest as UTF-8 JSON with source file path keys ordered by ascending code-point comparison, so that two runs over identical inputs produce byte-identical Source_Manifest content.
4. WHEN the Ingest_Script starts and the Source_Manifest file is absent, THE Ingest_Script SHALL treat every discovered source file as a New_Source.
5. IF the Source_Manifest file exists and cannot be parsed as JSON, or holds an entry missing a field named in criterion 2, THEN THE Ingest_Script SHALL terminate with a non-zero exit status and an error message naming the Source_Manifest path, the offending source file path key when one applies, and the documented reset command.
6. WHEN the Ingest_Script writes the Source_Manifest, THE Ingest_Script SHALL write the serialized content to a temporary file in the Source_Manifest's directory and then replace the Source_Manifest file with that temporary file, so that an interrupted write leaves the previous Source_Manifest content intact.
7. FOR ALL Source_Manifests written by a completed Ingest_Script run, the sum of the stored Chunk counts of all Source_Manifest entries SHALL equal the stored item count reported by the Chroma_Store.
8. WHEN the Ingest_Script computes a Content_Hash twice for a source file whose bytes are unchanged, THE Ingest_Script SHALL produce identical Content_Hash values.
9. WHEN the Ingest_Script computes the Content_Hash of two source files whose byte contents differ, THE Ingest_Script SHALL produce different Content_Hash values.
10. WHEN the upsert of one source file's Chunks completes, THE Ingest_Script SHALL recompute the Content_Hash of that source file from a further sequential read pass over that source file's bytes and SHALL compare that recomputed Content_Hash against the Content_Hash computed in criterion 1.
11. IF the Content_Hash recomputed in criterion 10 differs from the Content_Hash computed in criterion 1 for the same source file, THEN THE Ingest_Script SHALL delete every Stored_Item whose source file path metadata equals that source file path, SHALL remove any existing Source_Manifest entry for that source file path and write no new Source_Manifest entry for that source file path, SHALL emit a warning naming that source file path and stating that the source file bytes changed during the run, and SHALL continue processing the remaining discovered source files.

### Requirement 8: Idempotent Incremental Ingest

**User Story:** As a learner, I want re-running ingest to skip unchanged files, re-index changed files, and drop files I deleted, so that my collection matches my notes folder without paying to re-embed everything.

#### Acceptance Criteria

1. WHEN the Ingest_Script is run, THE Ingest_Script SHALL classify every discovered supported file of the Sample_Notes_Folder as a New_Source, a Changed_Source, or an Unchanged_Source, and SHALL classify every Source_Manifest source file path that is not among the discovered supported files as a Deleted_Source.
2. WHEN the Ingest_Script processes an Unchanged_Source, THE Ingest_Script SHALL leave every Stored_Item of that source file unchanged, SHALL leave that source file's Source_Manifest entry unchanged, and SHALL issue no Embedder call for that source file.
3. WHEN the Ingest_Script processes a New_Source, THE Ingest_Script SHALL load, chunk, and embed that source file and SHALL upsert every produced Chunk with its Embedding_Vector into the Chroma_Collection.
4. WHEN the Ingest_Script processes a Changed_Source, THE Ingest_Script SHALL delete every Stored_Item whose source file path metadata equals that source file path before writing any Stored_Item for that source file path, SHALL then load, chunk, and embed that source file, and SHALL then upsert every produced Chunk with its Embedding_Vector into the Chroma_Collection, so that when the count of produced Chunks is smaller than the Chunk count recorded for that source file path in the Source_Manifest, no Stored_Item of the previous Chunk sequence of that source file path remains in the Chroma_Collection.
5. WHEN the Ingest_Script processes a Deleted_Source, THE Ingest_Script SHALL delete every Stored_Item whose source file path equals that source file path and SHALL remove that source file path entry from the Source_Manifest.
6. FOR ALL Sample_Notes_Folder contents and all valid Configurations, running the Ingest_Script a second time with unchanged source file bytes and unchanged Configuration SHALL leave the Collection_State unchanged, SHALL leave the stored item count unchanged, SHALL leave the Source_Manifest content byte-identical, and SHALL issue zero Embedder calls (ingest idempotence).
7. FOR ALL Sample_Notes_Folder contents and all valid Configurations, the Collection_State produced by any sequence of Ingest_Script runs ending with the current Sample_Notes_Folder contents SHALL equal the Collection_State produced by resetting the Chroma_Store and running the Ingest_Script once over those same contents (incremental ingest equals full rebuild).
8. FOR ALL Sample_Notes_Folder contents, after a completed Ingest_Script run the set of source file paths appearing in Stored_Item metadata SHALL equal the set of source file paths of the discovered supported files whose loaded Document text length is greater than 0.
9. WHEN the Chunk_Size, the Chunk_Overlap, the Embedding_Provider identifier, or the model name in the Configuration differs from the value recorded in a source file's Source_Manifest entry, THE Ingest_Script SHALL classify that source file as a Changed_Source.
10. WHEN the Ingest_Script upserts the Chunks of one source file, THE Ingest_Script SHALL issue Embedder calls each containing up to the configured Max_Batch_Size of Chunk texts.
11. IF the total number of Chunks produced from all New_Sources and all Changed_Sources exceeds the configured Max_Chunks_Per_Run, THEN THE Ingest_Script SHALL terminate with a non-zero exit status and an error message stating that total and the configured maximum, before issuing any Embedder call.
12. WHEN the Ingest_Script commits one source file, THE Ingest_Script SHALL write that source file's Source_Manifest entry only after the Chroma_Collection holds every Chunk of that source file, so that a source file with an incomplete set of Stored_Items has no Source_Manifest entry claiming the source file is ingested.
13. IF an Embedder call or a Chroma write fails while the Ingest_Script processes one source file, THEN THE Ingest_Script SHALL terminate with a non-zero exit status and an error message naming that source file path, the failure reason, and the number of source files committed before the failure, and SHALL leave the Source_Manifest entries of the already committed source files in place.
14. IF an Embedder call or a Chroma write fails while the Ingest_Script processes one source file, THEN THE Ingest_Script SHALL delete every Stored_Item whose source file path metadata equals that source file path, SHALL remove any existing Source_Manifest entry for that source file path and write no new Source_Manifest entry for that source file path, so that the next Ingest_Script run classifies that source file path as a New_Source.
15. WHEN the Ingest_Script is invoked with the documented reset option, THE Ingest_Script SHALL reset the Chroma_Store to a stored item count of 0, SHALL remove every Source_Manifest entry, and SHALL then ingest every discovered supported file as a New_Source.
16. WHEN the Ingest_Script starts, THE Ingest_Script SHALL compare, for every Source_Manifest entry, the Chunk count recorded in that entry against the count of Stored_Items whose source file path metadata equals that entry's source file path, and SHALL classify that entry's source file path as a Changed_Source in place of the Unchanged_Source classification of criterion 1 when the two counts differ and that source file path is among the discovered supported files of the Sample_Notes_Folder.
17. WHEN the Ingest_Script starts, THE Ingest_Script SHALL delete every Stored_Item whose source file path metadata equals no Source_Manifest entry source file path, before issuing any Embedder call, so that no Stored_Item left behind by an interrupted run remains in the Chroma_Collection.
18. WHEN the upsert of one source file's Chunks completes, THE Ingest_Script SHALL compare the count of Stored_Items whose source file path metadata equals that source file path against the count of Chunks produced for that source file.
19. IF the two counts compared in criterion 18 differ, THEN THE Ingest_Script SHALL delete every Stored_Item whose source file path metadata equals that source file path, SHALL remove any existing Source_Manifest entry for that source file path and write no new Source_Manifest entry for that source file path, and SHALL terminate with a non-zero exit status and an error message naming that source file path, the count of produced Chunks, and the count of Stored_Items found.

### Requirement 9: Ingest Run Reporting

**User Story:** As a learner, I want the ingest run to tell me what it did, so that I can see that the second run really skipped everything and that the counts add up.

#### Acceptance Criteria

1. WHEN the Ingest_Script completes, THE Ingest_Script SHALL print the count of New_Sources, the count of Changed_Sources, the count of Unchanged_Sources, the count of Deleted_Sources, the count of source file paths reclassified as Changed_Sources by the startup reconciliation of Requirement 8 criterion 16, the count of Stored_Items deleted by the startup orphan deletion of Requirement 8 criterion 17, the count of Chunks upserted, the count of Stored_Items deleted, the count of batch Embedder calls issued, the stored item count of the Chroma_Collection, and the elapsed wall-clock seconds rounded to 1 decimal place.
2. WHEN the Ingest_Script completes, THE Ingest_Script SHALL print the resolved absolute Persist_Directory path and the Collection_Name.
3. WHILE the Ingest_Script embeds Chunks, THE Ingest_Script SHALL print, after each batch Embedder call returns, the source file path being processed, the cumulative number of Chunks embedded, and the total number of Chunks scheduled for embedding in that run.
4. WHEN the Ingest_Script is run and the Sample_Notes_Folder contains no supported files and the Source_Manifest holds no entry, THE Ingest_Script SHALL print counts of 0 for every value named in criterion 1, SHALL issue no Embedder call, and SHALL exit with a success status.
5. WHEN the Ingest_Script is run a second time with unchanged source file bytes and unchanged Configuration, THE Ingest_Script SHALL print an Unchanged_Sources count equal to the count of discovered supported files whose loaded Document text length is greater than 0, a New_Sources count of 0, a Changed_Sources count of 0, a Chunks upserted count of 0, and a batch Embedder call count of 0.
6. WHEN any Ingest_Script output, warning, or error message is produced, THE Ingest_Script SHALL exclude the API key value in whole and in part and SHALL substitute the fixed redaction marker wherever the API key would otherwise appear.
7. WHEN the Ingest_Script loads source files, THE Ingest_Script SHALL apply the Week 1 per-file isolation behaviour for unopenable, unparsable, encrypted, oversize, and whitespace-only source files, SHALL exclude every such source file from the Chroma_Collection and from the Source_Manifest, and SHALL continue processing the remaining source files.

### Requirement 10: Retriever Top-K Query

**User Story:** As a learner, I want one function that turns a question into the most similar chunks with their scores and sources, so that Week 3 has something to build an answer on.

#### Acceptance Criteria

1. WHEN the Retriever receives a question string containing at least one non-whitespace character and an integer value of Top_K of at least 1, THE Retriever SHALL embed the question string with the Embedder using the Embedding_Provider and model name recorded in the Collection_Fingerprint, query the Chroma_Store, and return a Retrieval_Result.
2. WHEN the Retriever returns a Retrieval_Result, THE Retrieval_Result SHALL carry, for every returned SearchHit, the Chunk text, the chunk id, the source file path, the Chunk ordinal index, the Chunk start and end offsets, and the Similarity_Score.
3. WHEN the Retriever is invoked without an explicit value of Top_K, THE Retriever SHALL use the configured default Top_K.
4. FOR ALL question strings containing at least one non-whitespace character and all integer values of Top_K of at least 1, the number of SearchHit values in the Retrieval_Result SHALL equal the smaller of Top_K and the stored item count of the Chroma_Collection (K clamping).
5. FOR ALL question strings containing at least one non-whitespace character, the Similarity_Scores of the Retrieval_Result SHALL be monotonically non-increasing from the first position to the last position.
6. WHEN the Retriever embeds a question string, THE Retriever SHALL issue exactly one Embedder call for that question string.
7. IF the question string holds 0 Unicode code points, or holds only Unicode whitespace characters, THEN THE Retriever SHALL raise an error identifying the question string as empty, SHALL issue no Embedder call, SHALL issue no Chroma query, and SHALL append no Retrieval_Log_Record.
8. IF the supplied value of Top_K is not an integer, is less than 1, or is greater than 100, THEN THE Retriever SHALL raise an error stating the permitted range of 1 to 100, and SHALL issue no Embedder call.
9. IF the question string holds more Unicode code points than the configured Max_Input_Length, THEN THE Retriever SHALL raise an error stating the question string length in Unicode code points and the configured Max_Input_Length, SHALL issue no Embedder call, and SHALL issue no Chroma query.
10. IF the Embedder raises an error while the Retriever embeds a question string, THEN THE Retriever SHALL raise an error naming the Embedding_Provider and the failure reason, and SHALL return no Retrieval_Result.
11. WHEN the Retriever returns a Retrieval_Result, THE Retriever SHALL report the highest Similarity_Score of the Retrieval_Result and whether that Similarity_Score is at least the Relevance_Threshold.
12. FOR ALL question strings holding at least one non-whitespace character and all integer values of Top_K from 1 to 100, two Retriever invocations that supply the identical question string and the identical value of Top_K while the Collection_State is unchanged SHALL return the same sequence of chunk ids in the same order, and the Similarity_Scores the two invocations return SHALL be equal at every position within an absolute tolerance of 1e-9.

### Requirement 11: No-Relevant-Context Path

**User Story:** As a learner, I want retrieval to say plainly that nothing relevant was found when every score is low, so that Week 3 can refuse to answer instead of quoting noise.

#### Acceptance Criteria

1. WHEN the Retriever completes a query, the Chroma_Collection holds at least one Stored_Item, and the highest Similarity_Score of the Retrieval_Result is less than the Relevance_Threshold, THE Retriever SHALL report the No_Relevant_Context outcome together with the highest Similarity_Score and the Relevance_Threshold.
2. WHEN the Retriever reports the No_Relevant_Context outcome, THE Retriever SHALL return a Retrieval_Result holding a count of SearchHit values equal to the smaller of Top_K and the stored item count of the Chroma_Collection, SHALL omit no SearchHit on the ground of that SearchHit's Similarity_Score, and SHALL mark as below threshold every returned SearchHit whose Similarity_Score is less than the Relevance_Threshold.
3. FOR ALL question strings containing at least one non-whitespace character and all Relevance_Threshold values in the closed interval [-1.0, 1.0], the Retriever SHALL report the No_Relevant_Context outcome exactly when the Chroma_Collection holds at least one Stored_Item and the highest Similarity_Score of the Retrieval_Result is less than the Relevance_Threshold.
4. WHEN the Query_Script receives a Retrieval_Result for which the No_Relevant_Context outcome is reported, THE Query_Script SHALL print a line stating that no relevant context was found, the highest Similarity_Score rounded to 4 decimal places, and the Relevance_Threshold, and SHALL exit with a success status.
5. WHEN the Query_Script prints the No_Relevant_Context outcome, THE Query_Script SHALL print the below-threshold SearchHit values under a heading stating that the listed Chunks fall below the Relevance_Threshold.
6. THE Learning_Notes SHALL record the selected Relevance_Threshold value, at least one question that produced the No_Relevant_Context outcome, at least one question that produced a Similarity_Score at or above the Relevance_Threshold, and the reason for the selected value.
7. WHEN the Retriever reports the No_Relevant_Context outcome, THE Retriever SHALL write a Retrieval_Log_Record for that query recording the No_Relevant_Context outcome.
8. WHEN the Retriever completes a query, the Chroma_Collection holds at least one Stored_Item, and the highest Similarity_Score of the Retrieval_Result equals the Relevance_Threshold, THE Retriever SHALL treat the Retrieval_Result as relevant context and SHALL report no No_Relevant_Context outcome.

### Requirement 12: Empty or Missing Collection Handling

**User Story:** As a learner, I want a clear message instead of a crash when I query before ingesting, so that I know to run the ingest script.

#### Acceptance Criteria

1. WHEN the Chroma_Store is constructed and the Persist_Directory holds no Chroma_Collection with the configured Collection_Name, THE Chroma_Store SHALL create an empty Chroma_Collection with the configured Distance_Metric and Collection_Fingerprint and SHALL report a stored item count of 0.
2. WHEN the Chroma_Store is queried while the stored item count is 0, THE Chroma_Store SHALL return an empty Retrieval_Result and SHALL raise no error.
3. WHEN the Retriever queries a Chroma_Collection whose stored item count is 0, THE Retriever SHALL return a Retrieval_Result holding 0 SearchHit values, SHALL report that the Chroma_Collection is empty, SHALL report no highest Similarity_Score, and SHALL report no No_Relevant_Context outcome.
4. WHEN the Query_Script receives an empty Retrieval_Result because the stored item count is 0, THE Query_Script SHALL print a line stating that the Chroma_Collection holds no Chunks, SHALL print the documented Ingest_Script command, and SHALL exit with a success status.
5. WHEN the Query_Script runs and the Persist_Directory does not exist, THE Query_Script SHALL print a line stating that no Persist_Directory was found at the resolved absolute path, SHALL print the documented Ingest_Script command, and SHALL exit with a success status.
6. IF the Persist_Directory exists but the Chroma client cannot open it, THEN THE Ask_My_Docs SHALL terminate with a non-zero exit status and an error message naming the resolved absolute path, the failure reason, and the documented reset command.
7. WHEN the Retriever returns an empty Retrieval_Result because the stored item count is 0, THE Retriever SHALL write a Retrieval_Log_Record for that query recording a returned Chunk count of 0.

### Requirement 13: Command-Line Query Script

**User Story:** As a learner, I want to ask a question from the terminal and see which chunks come back, so that I can feel out how retrieval behaves on my own notes.

#### Acceptance Criteria

1. WHEN the Query_Script is run with a question string argument, THE Query_Script SHALL invoke the Retriever with that question string and the configured default Top_K, and SHALL print the Retrieval_Result.
2. WHEN the Query_Script is run with a question string argument and an explicit Top_K option, THE Query_Script SHALL invoke the Retriever with that value of Top_K.
3. WHEN the Query_Script prints a Retrieval_Result, THE Query_Script SHALL print, for every SearchHit in Retrieval_Result order, the one-based rank, the source file path, the chunk id, the Similarity_Score rounded to 4 decimal places, and the Chunk text truncated to the first 200 characters with a truncation marker appended when the Chunk text is longer than 200 characters.
4. WHEN the Query_Script prints a Retrieval_Result, THE Query_Script SHALL print the question string, the value of Top_K used, the Embedding_Provider identifier, the model name, and the count of returned SearchHit values.
5. IF the Query_Script is run with no question string argument, THEN THE Query_Script SHALL print a usage line naming the question string argument and the Top_K option, and SHALL exit with a non-zero status.
6. IF the Query_Script is run with a Top_K option value that is not an integer or lies outside 1 to 100, THEN THE Query_Script SHALL print an error message stating the permitted range of 1 to 100 and the rejected value, and SHALL exit with a non-zero status.
7. WHEN the Query_Script completes a query, THE Query_Script SHALL exit with a success status and SHALL print exactly one outcome line whose wording differs across the three cases that the Chroma_Collection holds no Chunks, that the No_Relevant_Context outcome was reported, and that at least one returned SearchHit holds a Similarity_Score of at least the Relevance_Threshold.
8. WHEN any Query_Script output or error message is produced, THE Query_Script SHALL exclude the API key value in whole and in part and SHALL substitute the fixed redaction marker wherever the API key would otherwise appear.
9. IF the Retriever raises an error, THEN THE Query_Script SHALL print the error message, SHALL print no Retrieval_Result, and SHALL exit with a non-zero status.

### Requirement 14: Retrieval Logging

**User Story:** As a learner, I want every retrieval appended to a log I can read later, so that I can review retrieval quality without re-running every question.

#### Acceptance Criteria

1. WHEN the Retriever completes a query, THE Retriever SHALL append exactly one Retrieval_Log_Record to the Retrieval_Log.
2. WHEN the Retriever appends a Retrieval_Log_Record, THE Retrieval_Log_Record SHALL hold a log schema version value, the query timestamp as an ISO 8601 date and time in UTC, the question string, the value of Top_K used, the Embedding_Provider identifier, the model name, the Embedding_Dimensionality, the Distance_Metric, the Relevance_Threshold, the stored item count of the Chroma_Collection at query time, the count of returned SearchHit values, the highest Similarity_Score when at least one SearchHit is returned, and whether the No_Relevant_Context outcome was reported.
3. WHEN the Retriever appends a Retrieval_Log_Record, THE Retrieval_Log_Record SHALL hold, for every returned SearchHit in Retrieval_Result order, the one-based rank, the chunk id, the source file path, the Chunk ordinal index, the Chunk start and end offsets, the Similarity_Score, the full character length of the Chunk text, and the Chunk text truncated to the first 500 characters with a truncation marker appended when the Chunk text is longer than 500 characters.
4. WHEN the Retriever appends a Retrieval_Log_Record, THE Retriever SHALL serialize the Retrieval_Log_Record as a single-line UTF-8 JSON object terminated by one line-feed character, so that the Retrieval_Log is parsable one record per line.
5. WHEN the Retriever appends a Retrieval_Log_Record and the Retrieval_Log file already exists, THE Retriever SHALL preserve every existing byte of the Retrieval_Log, SHALL add the serialized Retrieval_Log_Record and its terminating line-feed character after the existing content in a single append-mode write, and SHALL rewrite no existing line.
6. WHEN the Retriever appends a Retrieval_Log_Record and the Retrieval_Log file or its parent directory is absent, THE Retriever SHALL create the missing parent directories and the Retrieval_Log file and SHALL then append the Retrieval_Log_Record.
7. THE Retrieval_Log SHALL exclude the API key value in whole and in part, every field of every Retrieval_Log_Record SHALL substitute the fixed redaction marker wherever the API key would otherwise appear, and THE Ask_My_Docs repository SHALL exclude the Retrieval_Log from version control because every Retrieval_Log_Record holds verbatim text of the learner's source files.
8. FOR ALL sequences of N Retriever invocations that each complete, the Retrieval_Log SHALL hold N more lines after the sequence than before the sequence, and each added line SHALL parse as one JSON object holding every field named in criteria 2 and 3.
9. IF the Retrieval_Log cannot be opened for appending or cannot be written, THEN THE Retriever SHALL raise an error naming the resolved absolute Retrieval_Log path and the failure reason, and SHALL return no Retrieval_Result.
10. THE README SHALL state the resolved Retrieval_Log path, the record format as one JSON object per line, the log schema version value, every field name of a Retrieval_Log_Record, the 500-character Chunk text truncation limit, and that the Retrieval_Log holds verbatim source file text and is excluded from version control.
11. IF the Retriever raises an error before a Retrieval_Result is produced, THEN THE Retriever SHALL append no Retrieval_Log_Record and SHALL leave the Retrieval_Log byte-identical to its content before the invocation.
12. FOR ALL sets of at least 2 and at most 16 Retriever invocations that run concurrently against the same Retrieval_Log and that each complete, the Retrieval_Log SHALL gain exactly one line per invocation, and every added line SHALL parse as one complete JSON object holding every field named in criteria 2 and 3.

### Requirement 15: Top-K Tuning Experiment

**User Story:** As a learner, I want the same questions run at several values of K, so that I can see for myself where extra context turns into extra noise.

#### Acceptance Criteria

1. WHEN the Top_K_Experiment_Script is run, THE Top_K_Experiment_Script SHALL embed each question in the Question_Set with the Embedder exactly once, SHALL reuse that question's Embedding_Vector for the store query at every value of Top_K in the ordered list 3, 5, 10, and SHALL record one result set for every combination of a question and a value of Top_K in that list.
2. WHEN the Top_K_Experiment_Script is run, THE Top_K_Experiment_Script SHALL print and write to the Top_K_Report, for every combination named in criterion 1, the question identifier, the value of Top_K, the count of returned SearchHit values, the highest Similarity_Score, the lowest Similarity_Score, the mean Similarity_Score rounded to 4 decimal places, and the count of returned SearchHit values whose Similarity_Score is less than the Relevance_Threshold.
3. WHEN the Top_K_Experiment_Script is run, THE Top_K_Experiment_Script SHALL write to the Top_K_Report, for every value of Top_K in the ordered list 3, 5, 10, the mean across the Question_Set of the highest Similarity_Score, the mean across the Question_Set of the mean Similarity_Score, and the mean across the Question_Set of the count of returned SearchHit values below the Relevance_Threshold.
4. WHEN the Top_K_Experiment_Script is run, THE Top_K_Experiment_Script SHALL write to the Top_K_Report, for every question in the Question_Set, the count of distinct source file paths appearing in the Retrieval_Result at each value of Top_K.
5. WHEN the Top_K_Experiment_Script completes, THE Top_K_Experiment_Script SHALL write the new Top_K_Report content to a temporary file in the Top_K_Report's directory and SHALL then replace the Top_K_Report file with that temporary file, so that an interrupted run leaves the previous Top_K_Report content intact.
6. THE Top_K_Report SHALL contain a written section of at least 100 words stating the observed tradeoff between the additional context obtained at a larger value of Top_K and the additional below-threshold Chunks obtained at that same value, and naming the value of Top_K the learner selected for Week 3 with the reason.
7. FOR ALL questions in the Question_Set, WHILE the stored item count of the Chroma_Collection is at most 1000, the Retrieval_Result obtained at a value of Top_K of 3 SHALL equal the first 3 SearchHit values of the Retrieval_Result obtained at a value of Top_K of 10, except at a position where the Similarity_Scores of the two compared SearchHit values differ by at most 1e-5.
8. IF the Question_Set file is absent, cannot be opened, cannot be decoded as UTF-8, or holds fewer than 1 non-empty line, THEN THE Top_K_Experiment_Script SHALL terminate with a non-zero exit status and an error message naming the resolved absolute Question_Set path, the failure reason, and the expected format of one question per non-empty line, and SHALL leave any existing Top_K_Report unchanged.
9. IF the Question_Set holds fewer than 5 or more than 10 non-empty lines, THEN THE Top_K_Experiment_Script SHALL emit a warning stating the discovered question count and the recommended range of 5 to 10 questions, and SHALL continue processing every question.
10. WHEN the Top_K_Experiment_Script runs and the stored item count of the Chroma_Collection is 0, THE Top_K_Experiment_Script SHALL print a line stating that the Chroma_Collection holds no Chunks, SHALL print the documented Ingest_Script command, SHALL leave any existing Top_K_Report unchanged, and SHALL exit with a success status.
11. IF two or more non-empty lines of the Question_Set hold the same text after trimming leading and trailing whitespace, THEN THE Top_K_Experiment_Script SHALL emit a warning naming the duplicated text and the one-based line numbers of every occurrence, SHALL process only the first occurrence, and SHALL continue processing the remaining distinct questions.
12. FOR ALL Question_Sets, the count of question texts the Top_K_Experiment_Script submits to the Embedder in one run SHALL equal the count of distinct non-empty question lines processed in that run.

### Requirement 16: Manual Relevance Review and Precision at K

**User Story:** As a learner, I want to mark retrieved chunks relevant or not by hand and get a precision number, so that I have a real measurement of retrieval quality rather than a feeling.

#### Acceptance Criteria

1. WHEN the Relevance_Review_Script is run in generate mode, THE Relevance_Review_Script SHALL invoke the Retriever once for every question in the Question_Set at the configured default Top_K and SHALL write one Relevance_Review_File row for every returned SearchHit.
2. WHEN the Relevance_Review_Script writes a Relevance_Review_File row, THE row SHALL hold the question identifier, the question string, the value of Top_K used, the one-based rank, the chunk id, the source file path, the Similarity_Score rounded to 4 decimal places, the Chunk text with every line-feed and carriage-return character replaced by one space character and then truncated to the first 300 characters with a truncation marker appended when the replaced text is longer than 300 characters, and an empty Relevance_Label field.
3. WHEN the Relevance_Review_Script writes the Relevance_Review_File, THE Relevance_Review_File SHALL be a comma-separated UTF-8 file whose first line holds the column names, whose every field value is enclosed in double-quote characters with every double-quote character inside a field value doubled, whose every row occupies exactly one line, and whose rows are ordered by ascending question identifier and then ascending rank.
4. WHEN the Relevance_Review_Script is run in generate mode and the Relevance_Review_File already exists, THE Relevance_Review_Script SHALL make no modification to the existing Relevance_Review_File, SHALL write the newly generated rows to a separate file, SHALL copy into each newly generated row the Relevance_Label of the existing row holding the same question identifier and the same chunk id when such a row exists, and SHALL print both resolved absolute paths together with the count of Relevance_Labels carried over.
5. WHEN the Relevance_Review_Script is run in score mode, THE Relevance_Review_Script SHALL read the Relevance_Review_File and compute, for every question identifier holding at least 1 row, the Precision_At_K as the count of rows whose Relevance_Label equals `y` divided by the total count of rows present for that question identifier at the value of Top_K recorded in those rows.
6. WHEN the Relevance_Review_Script is run in score mode, THE Relevance_Review_Script SHALL print and write to the Precision_Report the per-question Precision_At_K rounded to 4 decimal places, the value of Top_K used, the count of labelled rows per question, and the mean Precision_At_K across all question identifiers rounded to 4 decimal places.
7. IF any row of the Relevance_Review_File holds an empty Relevance_Label when the Relevance_Review_Script is run in score mode, THEN THE Relevance_Review_Script SHALL terminate with a non-zero exit status and an error message naming the question identifier and rank of every unlabelled row and stating that an unlabelled row is not counted as not relevant, SHALL compute no Precision_At_K, and SHALL leave any existing Precision_Report unchanged.
8. IF any row of the Relevance_Review_File holds a Relevance_Label other than `y`, `n`, or empty, THEN THE Relevance_Review_Script SHALL terminate with a non-zero exit status and an error message naming the question identifier, the rank, the rejected value, and the two permitted values.
9. FOR ALL labelled Relevance_Review_Files, every computed Precision_At_K value and the mean Precision_At_K value SHALL lie within the closed interval [0.0, 1.0].
10. WHEN every row of one question identifier holds the Relevance_Label `y`, THE Relevance_Review_Script SHALL report a Precision_At_K of 1.0 for that question identifier, and WHEN every row of one question identifier holds the Relevance_Label `n`, THE Relevance_Review_Script SHALL report a Precision_At_K of 0.0 for that question identifier.
11. THE Learning_Notes SHALL record the mean Precision_At_K value, the value of Top_K at which the review was performed, the count of labelled rows, and at least one observation about a retrieved Chunk the learner labelled `n` and the reason for that label.
12. IF the Relevance_Review_File is absent when the Relevance_Review_Script is run in score mode, THEN THE Relevance_Review_Script SHALL terminate with a non-zero exit status and an error message naming the resolved absolute Relevance_Review_File path and the documented generate-mode command.
13. IF the Relevance_Review_File holds no row for a question identifier present in the Question_Set, THEN THE Relevance_Review_Script SHALL report that question identifier's Precision_At_K as not defined, SHALL exclude that question identifier from the mean Precision_At_K, and SHALL print the count of question identifiers excluded on that ground.

### Requirement 17: Vector Database Comparison Note

**User Story:** As a learner, I want my database choice written down against the alternatives and an explanation of how similarity search actually works, so that I can justify the choice and understand the machinery underneath.

#### Acceptance Criteria

1. THE Comparison_Note SHALL compare Chroma against at least one of LanceDB and Pinecone across the dimensions local versus hosted deployment, monetary cost, persistence model, index type, metadata filtering support, and operational effort.
2. THE Comparison_Note SHALL state, for every compared Vector_Database, whether that Vector_Database runs on the learner's machine without an account and whether that Vector_Database requires an API key.
3. THE Comparison_Note SHALL state, for every compared Vector_Database, the index type used for approximate nearest-neighbour search, and SHALL name HNSW where HNSW applies.
4. THE Comparison_Note SHALL state, for every compared Vector_Database, how metadata filtering is expressed and which of the Chunk metadata fields produced in Week 1 can be filtered on.
5. THE Comparison_Note SHALL contain a section of at least 150 words explaining how a Vector_Database performs similarity search, addressing the embedding space in which Chunks and questions are placed, the role of the Distance_Metric, the behaviour of an HNSW proximity graph as an approximate nearest-neighbour index, and the tradeoff between query latency and recall that approximate search introduces.
6. THE Comparison_Note SHALL state the selected Vector_Database for Week 2, the reason for the selection, and the condition under which the learner would move to a hosted Vector_Database.
7. THE Comparison_Note SHALL state the Distance_Metric configured on the Chroma_Collection and the formula that converts a Chroma_Distance into a Similarity_Score.
8. THE Ask_My_Docs repository SHALL hold the Comparison_Note as a markdown file under version control at a path stated in the README.

### Requirement 18: Week 1 Component Isolation

**User Story:** As a learner, I want the database swap to touch only the storage layer, so that the interface I designed in Week 1 proves it was a real seam.

#### Acceptance Criteria

1. THE Ask_My_Docs SHALL implement the Chroma_Store without changing the parameters or return types of the Week 1 Chunker, Embedder, Document_Loader, PDF_Loader, Markdown_Loader, or Similarity_Calculator interfaces.
2. THE Chunker, Embedder, Document_Loader, and Similarity_Calculator modules SHALL import no Chroma symbol and no Chroma_Store symbol.
3. THE Chroma_Store module SHALL be the only Ask_My_Docs module that imports the Chroma client package.
4. THE Vector_Store_Interface SHALL retain the three operations defined in Week 1 Requirement 10.1 with unchanged names and unchanged parameter and return types, and the Store_Admin_Interface SHALL be defined as a separate additive contract.
5. WHEN the store factory receives a Configuration whose vector store selection is `chroma`, THE store factory SHALL return a Chroma_Store, and WHEN the store factory receives a Configuration whose vector store selection is `memory`, THE store factory SHALL return an In_Memory_Store.
6. WHEN the Week 1 Pipeline_Script is run under any Week 2 Configuration, THE Pipeline_Script SHALL populate an In_Memory_Store and SHALL produce the output defined by Week 1 Requirement 11.
7. IF the import graph of the Ask_My_Docs modules shows a Chroma symbol or a Chroma_Store symbol imported by the Chunker, Embedder, Document_Loader, or Similarity_Calculator module, or shows the Chroma client package imported by any module other than the Chroma_Store module, THEN THE Ask_My_Docs test suite SHALL fail with a message naming every offending module and every offending imported symbol.
8. WHEN the Ingest_Script produces Chunks, THE Ingest_Script SHALL use the chunk id and the flat metadata mapping already defined by the Week 1 Chunk without adding any field to the Chunk model other than fields derived from existing Chunk fields.
9. THE Ask_My_Docs test suite SHALL compare the current content of every Week 1 module named in criterion 1 against that module's content at the recorded Week 1 completion revision of version control, and SHALL fail with a message naming every module whose content differs.
10. WHEN the Ask_My_Docs test suite is run, THE test suite SHALL execute every Week 1 test without modification to that test's code and SHALL report a pass result for every one of those tests.

### Requirement 19: Week 2 Test Suite and Reproducibility

**User Story:** As a learner, I want the Week 2 properties checked automatically and offline, so that the store swap, the ingest logic, and the retrieval ordering stay correct as I keep building.

#### Acceptance Criteria

1. THE Ask_My_Docs dependency manifest SHALL pin an exact version for the Chroma package and for every other direct dependency added in Week 2, and SHALL exclude every dependency that supplies pre-built retrieval, re-ranking, or prompt-orchestration pipelines.
2. THE Ask_My_Docs test suite SHALL contain at least one test for each acceptance criterion of Requirements 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, and 18 of this document.
3. THE Ask_My_Docs test suite SHALL verify each FOR ALL property stated in this document using property-based tests that generate at least 100 distinct randomly generated inputs per property, and SHALL report, for every failing property, the seed value and the generated input that produced the failure so that the failing run can be repeated.
4. THE Ask_My_Docs test suite SHALL contain a property-based test for each of the following properties: differential agreement of Chroma_Store and In_Memory_Store on top-K chunk ids and scores; monotonically non-increasing Similarity_Score ordering; prefix consistency of results across values of Top_K; ingest idempotence; equality of incremental ingest and full rebuild Collection_State; round-trip fidelity of Chunk text, Chunk metadata, and Embedding_Vector; agreement of the Chroma_Distance to Similarity_Score conversion with the Similarity_Calculator; retrieval of a stored Chunk's own Embedding_Vector returning that Chunk first; K clamping when Top_K exceeds the stored item count; equality of the stored item count and the sum of the Source_Manifest Chunk counts; and the No_Relevant_Context outcome holding exactly when the highest Similarity_Score is below the Relevance_Threshold.
5. WHERE a test requires Embedding_Vectors, THE test suite SHALL use the Week 1 substitute Embedder that returns vectors of a fixed Embedding_Dimensionality, returns identical vectors for identical input text on every invocation, and issues no request to a remote Embedding_Provider.
6. WHERE a test requires a Chroma_Collection, THE test SHALL use a Persist_Directory created for that test and removed after that test, and SHALL leave the repository Persist_Directory unchanged.
7. WHEN the test suite is run with no API key present in the environment, THE test suite SHALL complete with a success status and SHALL issue no network request.
8. WHEN the test suite is run on a supported Python version, THE test suite SHALL complete within 300 seconds.
9. THE README SHALL state the single command that runs the whole test suite and the command that runs only the Store_Conformance_Suite.
10. THE test suite SHALL verify the persistence behaviour of Requirement 5 criterion 1 by constructing a second Chroma_Store against the same Persist_Directory after the first Chroma_Store has been released, and SHALL assert the stored item count and the Collection_State of the second Chroma_Store.
11. THE Ask_My_Docs test suite SHALL contain a test that issues at least 8 concurrent Retriever invocations against one Retrieval_Log and asserts that the Retrieval_Log gained exactly one parsable line per completed invocation.
