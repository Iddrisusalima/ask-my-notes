# Requirements Document

## Introduction

Week 1 (Sep 14–20) of the "Ask My Docs" learning project builds the foundation layers of a from-scratch RAG application: embedding generation, document loading, and chunking. No high-level RAG framework is used; every step is implemented manually in Python so the learner understands the mechanics.

The deliverable at the end of Week 1 is a runnable pipeline that loads 5–10 personal PDF and markdown notes, splits each document into overlapping fixed-size chunks, embeds every chunk, and holds the chunk text plus vectors in an in-memory store. The in-memory store is deliberately simple, but its interface must be defined so that Week 2 can substitute a persistent vector database (Chroma) without modifying the loader, chunker, or embedder.

Week 1 also produces two written learning artifacts: an explanation of why documents must be split, and a plain-language explanation of semantic search.

Out of scope for Week 1: persistent vector storage, retrieval ranking beyond direct similarity comparison, prompt construction, answer generation, and any user interface.

## Glossary

- **Ask_My_Docs**: The overall RAG application being built across three weekly specs. In Week 1 it refers to the collection of scripts and modules produced by this spec.
- **Embedding_Provider**: The configured source of embedding vectors. Exactly one of two implementations: the OpenAI Embeddings API (remote) or a Sentence Transformers model (local).
- **Embedder**: The Ask_My_Docs component that accepts text and returns an embedding vector by delegating to the selected Embedding_Provider.
- **Embedding_Vector**: A fixed-length list of floating-point numbers representing the semantic content of a text input.
- **Embedding_Dimensionality**: The number of elements in an Embedding_Vector produced by the active Embedding_Provider.
- **Similarity_Calculator**: The Ask_My_Docs component that computes cosine similarity between two Embedding_Vectors.
- **Cosine_Similarity**: The dot product of two Embedding_Vectors divided by the product of their Euclidean norms, bounded in the closed interval [-1, 1].
- **Sample_Notes_Folder**: A filesystem directory, default name `sample-notes`, holding the learner's personal PDF and markdown files used as the Week 1 test dataset.
- **Document_Loader**: The Ask_My_Docs component that reads a supported file from the Sample_Notes_Folder and returns its text content plus source metadata.
- **PDF_Loader**: The Document_Loader implementation for files with a `.pdf` extension.
- **Markdown_Loader**: The Document_Loader implementation for files with a `.md` or `.markdown` extension.
- **Document**: The text content of one loaded source file together with source metadata (file path, file type).
- **Chunker**: The Ask_My_Docs component that splits Document text into a sequence of Chunks.
- **Chunk**: A contiguous substring of a Document's text, together with metadata identifying its source file, ordinal index, and start and end character offsets in the source text.
- **Chunk_Size**: The configured maximum character length of a Chunk, counted in Unicode code points. Default 500. Permitted range 1 to 10000.
- **Chunk_Overlap**: The configured number of characters of the preceding Chunk repeated at the start of the following Chunk, counted in Unicode code points. Default 50. Permitted range 0 to Chunk_Size minus 1.
- **Request_Timeout**: The configured maximum number of seconds the Embedder waits for a single Embedding_Provider request to return. Default 30. Permitted range 1 to 300.
- **Max_Retry_Attempts**: The configured maximum number of retry attempts the Embedder makes after a transient Embedding_Provider failure. Default 3. Permitted range 0 to 10.
- **Max_Input_Length**: The configured maximum input character length accepted by the Embedder for a single text, counted in Unicode code points. Default 8000. Permitted range 1 to 100000.
- **Max_Batch_Size**: The configured maximum number of texts the Embedder sends in one Embedding_Provider request. Default 64. Permitted range 1 to 2048.
- **Max_Chunks_Per_Run**: The configured maximum total number of Chunks the Pipeline_Script embeds in a single run. Default 2000.
- **In_Memory_Store**: The Week 1 storage component that holds Chunk text, Chunk metadata, and the corresponding Embedding_Vectors in process memory.
- **Vector_Store_Interface**: The abstract contract that In_Memory_Store implements, defining the operations Week 2 will reimplement against Chroma.
- **Configuration**: The set of runtime settings (provider selection, model name, API key, Sample_Notes_Folder path, Chunk_Size, Chunk_Overlap, Request_Timeout, Max_Retry_Attempts, Max_Input_Length, Max_Batch_Size, Max_Chunks_Per_Run) read from environment variables with documented defaults.
- **Comparison_Script**: The runnable script that embeds three mutually similar sentences and three unrelated sentences and reports their pairwise Cosine_Similarity values.
- **Pipeline_Script**: The runnable script that performs the Week 1 wrap-up: load, chunk, embed, and populate the In_Memory_Store.
- **Learning_Notes**: The written markdown artifacts produced in Week 1, covering why splitting matters and what semantic search is.

## Requirements

### Requirement 1: Configuration via Environment Variables

**User Story:** As a learner, I want all provider and pipeline settings supplied through environment variables, so that I can switch providers and tune chunking without editing code and without committing secrets.

#### Acceptance Criteria

1. THE Configuration SHALL read the Embedding_Provider selection, model name, API key, Sample_Notes_Folder path, Chunk_Size, Chunk_Overlap, Embedding_Provider request timeout in seconds, Embedding_Provider maximum retry attempts, maximum input character length per text, and maximum batch size per Embedding_Provider request from environment variables.
2. IF an environment variable other than the API key is absent or empty, THEN THE Configuration SHALL apply that setting's documented default value: Sample_Notes_Folder path `sample-notes`, Chunk_Size 500 characters, Chunk_Overlap 50 characters, request timeout 30 seconds, maximum retry attempts 3, maximum input character length 8000 characters, maximum batch size 64 texts, and the model name documented in the README for the selected Embedding_Provider.
3. IF the selected Embedding_Provider is the OpenAI Embeddings API AND the API key environment variable is absent or contains only whitespace, THEN THE Ask_My_Docs SHALL terminate with a non-zero exit status and an error message naming the missing environment variable, before issuing any Embedding_Provider request.
4. WHERE the selected Embedding_Provider is a Sentence Transformers model, THE Ask_My_Docs SHALL complete every operation without reading the API key environment variable and without issuing any network request to the OpenAI Embeddings API.
5. IF the Embedding_Provider environment variable value, after trimming surrounding whitespace and lowercasing, does not equal one of the two supported provider identifiers, THEN THE Configuration SHALL terminate with a non-zero exit status and an error message listing the two supported identifiers and showing the rejected value.
6. THE Ask_My_Docs repository SHALL contain a `.gitignore` entry for the local environment file and an example environment file that contains placeholder values in place of secrets.
7. WHEN any error message, log record, or console output is produced, THE Ask_My_Docs SHALL exclude the API key value in whole and in part and SHALL substitute a fixed redaction marker wherever the API key would otherwise appear.
8. IF Chunk_Size is configured as a value less than 1 or greater than 10000, THEN THE Configuration SHALL terminate with a non-zero exit status and an error message naming the environment variable and stating the permitted range of 1 to 10000 characters.
9. IF Chunk_Overlap is configured as a value less than 0 or greater than or equal to Chunk_Size, THEN THE Configuration SHALL terminate with a non-zero exit status and an error message naming both environment variables, both configured values, and the permitted range of 0 to Chunk_Size minus 1.
10. THE Configuration SHALL match the Embedding_Provider environment variable value against the supported provider identifiers case-insensitively and after trimming leading and trailing whitespace.
11. IF a numeric Configuration setting holds a value that cannot be parsed as a number, THEN THE Configuration SHALL terminate with a non-zero exit status and an error message naming that environment variable, its rejected value, and the expected value type.
12. IF the request timeout is outside 1 to 300 seconds inclusive, OR the maximum retry attempts is outside 0 to 10 inclusive, OR the maximum input character length is outside 1 to 100000 inclusive, OR the maximum batch size is outside 1 to 2048 inclusive, THEN THE Configuration SHALL terminate with a non-zero exit status and an error message naming the offending environment variable and stating its permitted range.

### Requirement 2: Generate and Inspect a Single Embedding

**User Story:** As a learner, I want a script that turns one sentence into an embedding vector and prints its dimensionality, so that I can see concretely what an embedding is.

#### Acceptance Criteria

1. WHEN the Embedder receives a non-empty text input of at most the configured maximum input character length, THE Embedder SHALL return an Embedding_Vector whose element count equals the Embedding_Dimensionality of the active Embedding_Provider, whose elements are all finite floating-point numbers, and whose Euclidean norm is greater than 0.
2. WHEN the embedding script is run with a sentence, THE Ask_My_Docs SHALL print the Embedding_Dimensionality of the returned Embedding_Vector.
3. WHEN the embedding script is run with a sentence, THE Ask_My_Docs SHALL print the first 5 elements of the returned Embedding_Vector, or all elements if the Embedding_Dimensionality is less than 5.
4. FOR ALL non-empty text inputs, THE Embedder SHALL return Embedding_Vectors of identical Embedding_Dimensionality.
5. WHEN the Embedder is invoked twice with byte-identical text and unchanged Configuration, THE Embedder SHALL return Embedding_Vectors whose corresponding elements differ by no more than 1e-6 in absolute value.
6. IF the text input is empty or contains only whitespace, THEN THE Embedder SHALL raise an error identifying the input as empty and SHALL issue no Embedding_Provider request.
7. IF an Embedding_Provider request fails and the configured maximum retry attempts have been exhausted, THEN THE Embedder SHALL raise an error that names the provider, the number of attempts made, and the provider's failure reason, and SHALL return no Embedding_Vector.
8. IF a text input exceeds the configured maximum input character length, THEN THE Embedder SHALL raise an error stating the input length and the configured maximum, and SHALL issue no Embedding_Provider request.
9. IF an Embedding_Provider request does not return within the configured request timeout, OR the Embedding_Provider reports a rate limit or other transient failure, THEN THE Embedder SHALL retry the request up to the configured maximum retry attempts, waiting at least 1 second and at most 30 seconds before each retry.
10. IF the Embedding_Provider reports a failure caused by rejected credentials or rejected input, THEN THE Embedder SHALL raise an error naming the provider and the failure reason without issuing any retry attempt.

### Requirement 3: Batch Embedding

**User Story:** As a learner, I want to embed a list of texts in one call, so that embedding all chunks of my notes completes in a reasonable number of provider requests.

#### Acceptance Criteria

1. WHEN the Embedder receives a list of N non-empty texts, where N is at least 1, THE Embedder SHALL return a list of N Embedding_Vectors all having the Embedding_Dimensionality of the active Embedding_Provider.
2. WHEN the Embedder returns a list of Embedding_Vectors, THE Embedder SHALL order the returned Embedding_Vectors to match the order of the input texts.
3. FOR ALL lists of non-empty texts, the Embedding_Vector returned at position i of a batch call SHALL differ from the Embedding_Vector returned by a single-text call on the text at position i by no more than 1e-6 per element, for every position i.
4. WHEN the Embedder receives an empty list, THE Embedder SHALL return an empty list.
5. WHEN the Embedder receives a list whose length exceeds the configured maximum batch size, THE Embedder SHALL split the list into consecutive segments of at most the configured maximum batch size, issue one Embedding_Provider request per segment, and return a single list of Embedding_Vectors in input order.
6. IF any element of the supplied list is empty, contains only whitespace, or exceeds the configured maximum input character length, THEN THE Embedder SHALL raise an error naming the zero-based position of the first offending element and the reason for rejection, and SHALL issue no Embedding_Provider request.
7. IF an Embedding_Provider request for any segment fails after the configured maximum retry attempts, THEN THE Embedder SHALL raise an error naming the provider, the zero-based input positions covered by the failed segment, and the provider's failure reason, and SHALL return no partial list of Embedding_Vectors.

### Requirement 4: Cosine Similarity Calculation

**User Story:** As a learner, I want a cosine similarity function I wrote myself, so that I understand how semantic closeness is measured numerically.

#### Acceptance Criteria

1. WHEN the Similarity_Calculator receives two Embedding_Vectors of the same length, that length being at least 1, whose elements are all finite and whose Euclidean norms both exceed 1e-12, THE Similarity_Calculator SHALL return their Cosine_Similarity as a floating-point number.
2. FOR ALL pairs of Embedding_Vectors of equal length and non-zero norm, THE Similarity_Calculator SHALL return a value within the closed interval [-1, 1], allowing a floating-point tolerance of 1e-9.
3. FOR ALL pairs of Embedding_Vectors A and B of equal length and non-zero norm, the value returned for (A, B) SHALL equal the value returned for (B, A) within a tolerance of 1e-9.
4. FOR ALL Embedding_Vectors A of non-zero norm, the value returned for (A, A) SHALL equal 1.0 within a tolerance of 1e-9.
5. FOR ALL Embedding_Vectors A whose Euclidean norm exceeds 1e-12 and all scalars k in the closed interval [1e-6, 1e6], the value returned for (A, k·A) SHALL equal 1.0 within a tolerance of 1e-9.
6. IF the two supplied Embedding_Vectors have different lengths, THEN THE Similarity_Calculator SHALL raise an error reporting both lengths, and SHALL perform this length check before any element or norm check.
7. IF the computed Euclidean norm of either supplied Embedding_Vector is at most 1e-12, THEN THE Similarity_Calculator SHALL raise an error identifying which of the two inputs has that norm.
8. IF either supplied Embedding_Vector contains no elements, THEN THE Similarity_Calculator SHALL raise an error identifying the empty input.
9. IF either supplied Embedding_Vector contains an element that is not a finite number, THEN THE Similarity_Calculator SHALL raise an error identifying which input contains that element.

### Requirement 5: Similar vs Unrelated Sentence Comparison

**User Story:** As a learner, I want to compare three similar sentences against three unrelated ones, so that I can observe that semantically close text produces numerically close vectors.

#### Acceptance Criteria

1. THE Comparison_Script SHALL define a similar group of exactly 3 sentences of 40 to 200 characters each that all describe one shared topic, and an unrelated group of exactly 3 sentences of 40 to 200 characters each that share no content word with any sentence in the similar group.
2. WHEN the Comparison_Script is run, THE Comparison_Script SHALL print exactly 3 within-group lines, one for each unordered pair of the similar group, each line containing both sentence identifiers and the pair's Cosine_Similarity rounded to 4 decimal places.
3. WHEN the Comparison_Script is run, THE Comparison_Script SHALL print exactly 9 cross-group lines, one for each pair formed by one sentence of the similar group and one sentence of the unrelated group, each line containing both sentence identifiers and the pair's Cosine_Similarity rounded to 4 decimal places.
4. WHEN the Comparison_Script is run, THE Comparison_Script SHALL print the labelled mean Cosine_Similarity of the 3 within-group pairs and the labelled mean Cosine_Similarity of the 9 cross-group pairs, each rounded to 4 decimal places.
5. WHEN the Comparison_Script is run, THE Comparison_Script SHALL print as its final line both mean values and an explicit statement of whether the within-group mean exceeds the cross-group mean.
6. WHEN the Comparison_Script embeds its sentences, THE Comparison_Script SHALL embed all 6 sentences in a single batch Embedder call.
7. IF the Embedder raises an error while the Comparison_Script is running, THEN THE Comparison_Script SHALL terminate with an error message naming the Embedding_Provider and the failure reason, SHALL print no similarity or mean values, and SHALL exit with a failure status.

### Requirement 6: Sample Notes Dataset

**User Story:** As a learner, I want a folder of 5–10 of my own PDF and markdown notes, so that the pipeline is exercised on real documents rather than toy strings.

#### Acceptance Criteria

1. THE Ask_My_Docs repository SHALL contain a Sample_Notes_Folder and a README file inside that folder stating the expected file count of 5 to 10 note files, the supported file extensions `.pdf`, `.md`, and `.markdown`, and that the README file itself is not counted as a note file.
2. THE Ask_My_Docs repository SHALL exclude the learner's personal note files in the Sample_Notes_Folder from version control while retaining the folder and its README file under version control.
3. WHEN the Sample_Notes_Folder is scanned, THE Ask_My_Docs SHALL report the count of discovered supported files grouped by lower-cased file extension and the total count of skipped entries.
4. IF the Sample_Notes_Folder contains fewer than 5 or more than 10 supported files, THEN THE Ask_My_Docs SHALL emit a warning stating the discovered count and the recommended range of 5 to 10 files, and SHALL continue processing all discovered supported files.
5. IF the configured Sample_Notes_Folder path does not exist or does not resolve to a directory, THEN THE Ask_My_Docs SHALL terminate with an error message containing the resolved absolute path and stating which of the two conditions applied.
6. IF the Sample_Notes_Folder contains no supported files, THEN THE Ask_My_Docs SHALL report that no documents were found, produce an empty In_Memory_Store, and exit with a success status.
7. WHEN the Sample_Notes_Folder is scanned, THE Ask_My_Docs SHALL include files in the folder and in all of its subdirectories, SHALL match supported extensions without regard to letter case, and SHALL exclude the folder's README file and every entry whose name begins with a period.
8. WHEN the Sample_Notes_Folder is scanned, THE Ask_My_Docs SHALL order the discovered supported files by ascending code-point comparison of their forward-slash-separated paths relative to the Sample_Notes_Folder, yielding the same order for the same folder contents on every operating system.
9. IF a discovered entry is a symbolic link, THEN THE Ask_My_Docs SHALL exclude it from the discovered supported files and SHALL list its path among the skipped entries with the reason.

### Requirement 7: Document Loading

**User Story:** As a learner, I want separate loaders for PDF and markdown files, so that both note formats reach the chunker as plain text with source metadata.

#### Acceptance Criteria

1. WHEN the Document_Loader receives a file path whose lower-cased extension is `.pdf`, THE PDF_Loader SHALL extract the text content of every page and return a single Document whose text is the per-page extracted texts joined in ascending page order with a single line-feed character between consecutive pages and no other inserted characters.
2. WHEN the Document_Loader receives a file path whose lower-cased extension is `.md` or `.markdown`, THE Markdown_Loader SHALL return a Document whose text is the file's full text content with all markup characters, indentation, and blank lines preserved.
3. WHEN a Document_Loader returns a Document, THE Document SHALL carry as metadata the source file path expressed as a forward-slash-separated path relative to the Sample_Notes_Folder, which is unique among the loaded Documents, and the source file type expressed as the lower-cased extension without the leading period.
4. WHEN the Markdown_Loader reads a file, THE Markdown_Loader SHALL decode the file bytes as UTF-8 and SHALL discard a leading byte-order mark if present.
5. IF a file's bytes cannot be decoded as UTF-8, THEN THE Document_Loader SHALL decode the file text as UTF-8 with each invalid byte sequence replaced by the Unicode replacement character and SHALL emit a warning naming the file path and the number of replacements performed.
6. WHEN the Sample_Notes_Folder is scanned, THE Ask_My_Docs SHALL skip files whose lower-cased extension is not `.pdf`, `.md`, or `.markdown` and SHALL list each skipped file path with the unsupported extension.
7. IF a supported file cannot be opened or parsed, THEN THE Ask_My_Docs SHALL emit an error naming the file path and the failure reason, and SHALL continue loading the remaining files.
8. IF a loaded Document's text contains no non-whitespace characters, THEN THE Ask_My_Docs SHALL emit a warning naming the file path and stating that no extractable text was found, SHALL exclude that Document from chunking, and SHALL continue loading the remaining files.
9. WHEN the Document_Loader loads the same file twice with unchanged file bytes and unchanged Configuration, THE Document_Loader SHALL return Documents with character-identical text and identical source metadata.
10. THE Document_Loader SHALL convert carriage-return and carriage-return-line-feed sequences in Document text to a single line-feed character and SHALL apply no other whitespace trimming, whitespace collapsing, or Unicode normalization to Document text, so that Chunk start and end offsets index the returned Document text directly.
11. IF a `.pdf` file is encrypted or requires a password to open, THEN THE Ask_My_Docs SHALL emit an error naming the file path and stating that the file is encrypted, SHALL exclude that file from the loaded Documents, and SHALL continue loading the remaining files.
12. IF a supported file's size exceeds 25 megabytes, THEN THE Ask_My_Docs SHALL emit a warning naming the file path, its size, and the 25 megabyte limit, SHALL exclude that file from the loaded Documents, and SHALL continue loading the remaining files.

### Requirement 8: Fixed-Size Chunking with Overlap

**User Story:** As a learner, I want to split documents into fixed-size chunks with configurable overlap, so that each chunk fits a context window while preserving continuity across boundaries.

#### Acceptance Criteria

1. WHEN the Chunker receives a Document whose text length exceeds Chunk_Size, THE Chunker SHALL return a sequence of 2 or more Chunks ordered by strictly ascending start offset.
2. FOR ALL Documents and all valid Chunk_Size and Chunk_Overlap configurations, every returned Chunk SHALL have a character length of at most Chunk_Size.
3. FOR ALL Documents and all valid configurations that produce more than one Chunk, every returned Chunk except the last SHALL have a character length of exactly Chunk_Size, and the last Chunk SHALL have a character length greater than Chunk_Overlap and at most Chunk_Size.
4. FOR ALL Documents and all valid configurations, consecutive Chunks SHALL advance the start offset by exactly Chunk_Size minus Chunk_Overlap characters.
5. FOR ALL Documents and all valid configurations, the concatenation of the first returned Chunk's text with the text of every subsequent Chunk after removing that Chunk's leading Chunk_Overlap characters SHALL equal the Document text exactly (coverage and reconstruction property).
6. FOR ALL Documents and all valid configurations where more than one Chunk is produced, the trailing Chunk_Overlap characters of each Chunk except the last SHALL equal the leading Chunk_Overlap characters of the following Chunk (overlap invariant).
7. FOR ALL Documents, THE Chunker SHALL produce the same number of Chunks with the same text and the same start and end offsets on every invocation with the same Document text and the same Configuration (deterministic boundaries).
8. WHEN a Document's text length is greater than 0 and at most Chunk_Size, THE Chunker SHALL return exactly one Chunk containing the full Document text, including when that text length is less than or equal to Chunk_Overlap.
9. WHEN a Document's text length is 0, THE Chunker SHALL return an empty sequence of Chunks.
10. WHEN the Chunker returns a Chunk, THE Chunk SHALL carry the source file path, a zero-based ordinal index within the source Document, and the start and end character offsets of the Chunk in the Document text.
11. FOR ALL Documents, a Chunk's start offset SHALL be inclusive and its end offset exclusive, the substring of the Document text from the start offset to the end offset SHALL equal that Chunk's text, and the end offset minus the start offset SHALL equal that Chunk's character length.
12. FOR ALL Documents, the ordinal indices of the Chunks of one Document SHALL form the consecutive integer sequence starting at 0.
13. WHEN the Chunker emits a Chunk whose end offset equals the Document text length, THE Chunker SHALL emit no further Chunks for that Document.
14. THE Chunker SHALL count Chunk_Size, Chunk_Overlap, Chunk character lengths, and Chunk start and end offsets in Unicode code points, not in bytes and not in grapheme clusters.

### Requirement 9: Chunking Experiment

**User Story:** As a learner, I want to run the chunker over my notes at several chunk sizes and overlaps, so that I can see how the settings change what text groups together.

#### Acceptance Criteria

1. WHEN the chunking experiment script is run, THE Ask_My_Docs SHALL chunk the loaded Documents once for each of the 6 combinations formed by the Chunk_Size values 200, 500, and 1000 and the Chunk_Overlap values 0 and 50.
2. WHEN the chunking experiment script is run, THE Ask_My_Docs SHALL print, for each of the 6 combinations, the Chunk_Size, the Chunk_Overlap, the total Chunk count, the mean Chunk character length rounded to 1 decimal place, and the minimum and maximum Chunk character length.
3. WHEN the chunking experiment script is run, THE Ask_My_Docs SHALL print, for each of the 6 combinations, the text of the first 2 Chunks truncated to the first 200 characters with a truncation marker appended, or the text of all produced Chunks when fewer than 2 Chunks were produced.
4. WHEN the chunking experiment script completes, THE Ask_My_Docs SHALL write the statistics named in criterion 2 for all 6 combinations to a single markdown file whose location is stated in the README, replacing the entire previous contents of that file when the file already exists.
5. THE Learning_Notes SHALL record, for at least 2 of the 6 combinations, the learner's observation of how Chunk_Size and Chunk_Overlap changed which sentences appeared together in one Chunk.
6. IF the Sample_Notes_Folder contains no supported files when the chunking experiment script is run, THEN THE Ask_My_Docs SHALL print a Document count of 0, SHALL leave any existing statistics file unchanged, and SHALL exit with a success status.

### Requirement 10: In-Memory Chunk and Vector Store

**User Story:** As a learner, I want chunk text and its embedding held together in memory behind a small interface, so that Week 2 can replace the storage with Chroma without touching my loaders, chunker, or embedder.

#### Acceptance Criteria

1. THE Vector_Store_Interface SHALL define an operation to add a batch of Chunks with their Embedding_Vectors, an operation to return the stored item count, and an operation to return the K stored Chunks with the highest Cosine_Similarity to a supplied query Embedding_Vector.
2. THE In_Memory_Store SHALL implement the Vector_Store_Interface and SHALL hold Chunk text, Chunk metadata, and Embedding_Vectors in process memory.
3. THE Chunker and THE Embedder SHALL expose interfaces whose parameters and return values reference no Vector_Store_Interface type and no In_Memory_Store type, and the Chunker and Embedder modules SHALL import neither the Vector_Store_Interface nor the In_Memory_Store.
4. WHEN Chunks and Embedding_Vectors are added to the In_Memory_Store, THE In_Memory_Store SHALL preserve the association between each Chunk and its Embedding_Vector.
5. WHEN N Chunks are added to an empty In_Memory_Store, THE In_Memory_Store SHALL report a stored item count of N.
6. FOR ALL sequences of add operations, the stored item count SHALL equal the sum of the Chunk counts of those operations, and an add operation supplying an empty batch SHALL leave the stored item count unchanged and SHALL NOT raise an error.
7. IF an add operation supplies a number of Embedding_Vectors different from the number of Chunks, THEN THE In_Memory_Store SHALL raise an error reporting both counts and SHALL leave the stored items and the stored item count unchanged.
8. IF an add operation supplies Embedding_Vectors that are not all of the same length, or supplies an Embedding_Vector whose length differs from the length established by the first completed non-empty add operation, THEN THE In_Memory_Store SHALL raise an error reporting both lengths and SHALL leave the stored items and the stored item count unchanged.
9. WHEN the In_Memory_Store is queried with a query Embedding_Vector and an integer value K of at least 1, THE In_Memory_Store SHALL return at most K stored Chunks ordered by descending Cosine_Similarity to the query Embedding_Vector, and SHALL return the same Chunks in the same order for every repetition of that query while its stored contents are unchanged.
10. WHEN the In_Memory_Store holds fewer than K items and is queried with a value K, THE In_Memory_Store SHALL return all stored Chunks ordered by descending Cosine_Similarity.
11. WHEN an empty In_Memory_Store is queried with any query Embedding_Vector whose Euclidean norm exceeds 1e-12 and any integer K of at least 1, THE In_Memory_Store SHALL return an empty result and SHALL NOT raise an error.
12. WHEN the In_Memory_Store is queried with the Embedding_Vector of a stored Chunk and a value K of 1, THE In_Memory_Store SHALL return that stored Chunk, unless another stored Chunk has an identical Cosine_Similarity value to that query Embedding_Vector, in which case THE In_Memory_Store SHALL return whichever of the tied Chunks has the lowest insertion index.
13. IF a query supplies a value of K that is not an integer or is less than 1, THEN THE In_Memory_Store SHALL raise an error stating the permitted range of K and SHALL return no result.
14. IF a query supplies an Embedding_Vector whose length differs from the length established by the first completed non-empty add operation, or whose Euclidean norm is at most 1e-12, THEN THE In_Memory_Store SHALL raise an error naming the violated property and reporting both lengths when the lengths differ.
15. WHEN two or more stored Chunks have identical Cosine_Similarity values to the query Embedding_Vector, THE In_Memory_Store SHALL order those Chunks among themselves by ascending insertion index.

### Requirement 11: Week 1 End-to-End Pipeline

**User Story:** As a learner, I want one script that loads, chunks, and embeds all my notes into the in-memory store, so that I have a working foundation to build retrieval on in Week 2.

#### Acceptance Criteria

1. WHEN the Pipeline_Script is run, THE Ask_My_Docs SHALL load every supported file in the Sample_Notes_Folder, chunk each loaded Document, embed every Chunk, and add every Chunk with its Embedding_Vector to the In_Memory_Store.
2. WHEN the Pipeline_Script completes, THE Ask_My_Docs SHALL print the Document count, the total Chunk count, the Embedding_Dimensionality, the In_Memory_Store item count, the number of batch Embedder calls issued, and the elapsed wall-clock seconds rounded to 1 decimal place.
3. WHEN the Pipeline_Script completes, THE In_Memory_Store item count SHALL equal the total number of Chunks produced from the loaded Documents.
4. WHEN the Pipeline_Script is run and the Sample_Notes_Folder contains no supported files, THE Ask_My_Docs SHALL print a Document count of 0 and a Chunk count of 0, SHALL issue no Embedder call, SHALL print a line stating that no Chunks were available for the demonstration query, and SHALL exit with a success status.
5. WHEN the Pipeline_Script is run twice with unchanged files and unchanged Configuration, THE Ask_My_Docs SHALL produce the same total Chunk count and the same ordered Chunk texts.
6. WHEN the Pipeline_Script embeds Chunks, THE Ask_My_Docs SHALL issue Embedder calls each containing up to the configured maximum batch size of Chunks, defaulting to 64 Chunks per call, and SHALL issue no more Embedder calls than the total Chunk count divided by the maximum batch size rounded up.
7. WHILE the Pipeline_Script is embedding Chunks, THE Ask_My_Docs SHALL print, after each batch Embedder call returns, the cumulative number of Chunks embedded and the total number of Chunks.
8. WHEN the Pipeline_Script finishes populating a non-empty In_Memory_Store, THE Ask_My_Docs SHALL print the demonstration query text and then, for each of the 3 highest-similarity stored Chunks or for all stored Chunks when the In_Memory_Store item count is less than 3, in descending Cosine_Similarity order, that Chunk's source file path, its Cosine_Similarity rounded to 4 decimal places, and its text truncated to the first 200 characters.
9. IF the total Chunk count produced from the loaded Documents exceeds the configured maximum Chunks per run, defaulting to 2000, THEN THE Ask_My_Docs SHALL terminate with an error message stating the total Chunk count and the configured maximum before issuing any Embedder call.
10. IF an Embedder call fails while the Pipeline_Script is embedding Chunks, THEN THE Ask_My_Docs SHALL terminate with an error message naming the Embedding_Provider, the failure reason, and the number of Chunks embedded before the failure, SHALL discard the partially populated In_Memory_Store, and SHALL exit with a failure status.

### Requirement 12: Written Learning Artifacts

**User Story:** As a learner, I want my understanding written down, so that I can confirm I grasped the concepts rather than only running code.

#### Acceptance Criteria

1. THE Learning_Notes SHALL contain a section, under its own markdown heading, of at least 100 words explaining why documents are split into Chunks, addressing both model context length limits and retrieval relevance precision.
2. THE Learning_Notes SHALL contain a section, under its own markdown heading, explaining semantic search in 3 to 4 sentences that contain no formulas, no mathematical operators, and no variable symbols.
3. THE Learning_Notes SHALL record the selected Embedding_Provider, the selected model name, the Embedding_Dimensionality, and the reason for the selection.
4. THE Learning_Notes SHALL record the within-group mean and cross-group mean Cosine_Similarity values printed by the Comparison_Script, each to at least 4 decimal places, together with the 6 sentences used.
5. THE Learning_Notes SHALL record the chosen Chunk_Size and Chunk_Overlap for the Week 1 wrap-up and the reason for the choice.
6. THE Ask_My_Docs repository SHALL contain the Learning_Notes as one or more markdown files under version control at a location stated in the README, and the README SHALL name the file containing each section required by this requirement.

### Requirement 13: Project Setup and Reproducibility

**User Story:** As a learner, I want a clean Python project with pinned dependencies and a test suite, so that the Week 1 code runs the same way tomorrow and the properties above stay verified.

#### Acceptance Criteria

1. THE Ask_My_Docs repository SHALL contain a dependency manifest that pins an exact version for every direct dependency and that states the supported Python version range of 3.10 through 3.12.
2. THE Ask_My_Docs repository SHALL exclude from the dependency manifest every dependency that supplies pre-built document loading, chunking, retrieval, or prompt-orchestration pipelines.
3. THE Ask_My_Docs repository SHALL contain a README file stating the setup commands, the environment variables, and the command to run each Week 1 script.
4. THE Ask_My_Docs repository SHALL contain a test suite that is runnable with a single command stated in the README and that contains at least one test for each acceptance criterion of Requirements 4, 8, and 10.
5. WHERE a test requires Embedding_Vectors, THE test suite SHALL use a substitute Embedder that returns vectors of a fixed Embedding_Dimensionality, returns identical vectors for identical input text on every invocation, and issues no request to a remote Embedding_Provider.
6. WHEN the test suite is run with no API key present in the environment, THE test suite SHALL complete with a success status and SHALL issue no network request to any Embedding_Provider.
7. THE test suite SHALL verify each FOR ALL property stated in Requirements 4, 8, and 10 using property-based tests that generate at least 100 distinct randomly generated inputs per property.
8. WHEN the test suite is run on a supported Python version, THE test suite SHALL complete within 120 seconds.
