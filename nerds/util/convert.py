from nerds.core.model.input.annotation import Annotation
from nerds.core.model.input.document import AnnotatedDocument
from nerds.util.nlp import sentence_to_tokens, document_to_sentences


def transform_annotated_documents_to_bio_format(
        annotated_documents, tokenizer=sentence_to_tokens):
    """ Wrapper function that applies `transform_annotated_document_to_bio_format`
        for a batch of annotated documents.

        Args:
            annotated_documents (list(AnnotatedDocument)): The annotated
                document objects to be converted to BIO format.
            tokenizer (function, optional): A function that accepts string
                as input and returns a list of strings - used in tokenization.
                Defaults to `sentence_to_tokens`.

        Returns:
            2-tuple: Both the first and the second elements of this tuple
                contain a list of lists of string, the first representing the
                tokens in each document and the second the BIO tags.
    """
    X = []
    y = []
    for annotated_document in annotated_documents:
        tokens, bio_tags = transform_annotated_document_to_bio_format(
            annotated_document, tokenizer)
        X.append(tokens)
        y.append(bio_tags)
    return (X, y)


def transform_annotated_document_to_bio_format(
        annotated_document, tokenizer=sentence_to_tokens):
    """ Transforms an annotated set of documents to the format that
        the model requires as input for training. That is two vectors of
        strings per document containing tokens and tags (i.e. BIO) in
        consecutive order.

        Args:
            annotated_document (AnnotatedDocument): The annotated document
                object to be converted to BIO format.
            tokenizer (function, optional): A function that accepts string
                as input and returns a list of strings - used in tokenization.
                Defaults to `sentence_to_tokens`.

        Returns:
            2-tuple: (list(str), list(str)): The tokenized document
                and the BIO tags corresponding to each of the tokens.

        Example:
            ['Barack', 'Obama', 'lives', 'in', 'the', 'White', 'House']
            ['B_Person', 'I_Person', '0', '0', 'B_Institution',
            'I_Institution','I_Institution']

    """
    content = annotated_document.plain_text_
    # If they're not annotated documents or no annotations are available.
    if not (isinstance(annotated_document, AnnotatedDocument) and
            annotated_document.annotations):
        tokens = tokenizer(content)
        labels = ["O" for _ in tokens]
        return tokens, labels

    tokens = []
    labels = []
    annotations = annotated_document.annotations
    substring_index = 0
    for ann in annotations:
        # Tokens from the end of the previous annotation or the start of the
        # sentence, to the beginning of this one.
        non_tagged_tokens = tokenizer(content[substring_index:ann.offset[0]])

        # Tokens corresponding to the annotation itself.
        tagged_tokens = tokenizer(content[ann.offset[0]:ann.offset[1] + 1])

        if isinstance(ann.label, bytes):
            label = ann.label.decode(annotated_document.encoding)
        else:
            label = ann.label

        # Adjust the index to reflect the next starting point.
        substring_index = ann.offset[1] + 1

        # Fill in the labels.
        non_tagged_labels = ["O" for token in non_tagged_tokens]
        # B_tag for the first token then I_tag for residual tokens.
        tagged_labels = ["B_" + label]\
            + ["I_" + label for i in range(len(tagged_tokens) - 1)]

        tokens += non_tagged_tokens + tagged_tokens
        labels += non_tagged_labels + tagged_labels

    # Also take into account the substring from the last token to
    # the end of the sentence.
    if substring_index < len(content):
        non_tagged_tokens = tokenizer(
            content[substring_index:len(content)])
        non_tagged_labels = ["O" for token in non_tagged_tokens]

        tokens += non_tagged_tokens
        labels += non_tagged_labels

    return tokens, labels


def transform_bio_tags_to_annotated_documents(tokens, bio_tags, documents):
    """ Wrapper function that applies `transform_bio_tags_to_annotated_document`
        for a batch of BIO tag - token lists. It's the inverse transformation
        of `transform_annotated_documents_to_bio_format`.

        Args:
            tokens (list(list(str))): The tokens for each document in
                the input.
            bio_tags (list(list(str))): The BIO tags for each list
                of tokens.
            documents (list(Document)): The original input documents.

        Returns:
            list(AnnotatedDocument)
    """
    annotated_documents = []
    for doc_idx, document in enumerate(documents):
        annotated_documents.append(
            transform_bio_tags_to_annotated_document(
                tokens[doc_idx], bio_tags[doc_idx], document))
    return annotated_documents


def transform_bio_tags_to_annotated_document(tokens, bio_tags, document):
    """ Given a list of tokens, a list of BIO tags, and a document object,
        this function returns annotated documents formed from this information.

        Example:
            doc -> "Barack Obama lives in the White House"
            tokens ->
            [['Barack', 'Obama', 'lives', 'in', 'the', 'White', 'House']]
            bio ->
            [['B_Person', 'I_Person', '0', '0', 'B_Institution',
            'I_Institution','I_Institution']]

        It returns:
        AnnotatedDocument(
        content = "Barack Obama lives in the White House"
        annotations = (
            (Barack Obama, Person, (0, 11))
            (White House, Person, (26, 36))
            )
        )
    """
    content = document.plain_text_

    cur_token_idx = 0
    cur_substring_idx = 0

    annotations = []
    while cur_token_idx < len(bio_tags):
        cur_token = tokens[cur_token_idx]
        cur_tag = bio_tags[cur_token_idx]

        if not cur_tag.startswith("B"):
            cur_substring_idx += len(cur_token)
            cur_token_idx += 1
            continue

        cur_label = cur_tag.split("_")[1]

        # Get the absolute start of the entity, given the index
        # which stores information about the previously detected
        # entity offset.
        start_idx = content.find(cur_token, cur_substring_idx)
        end_idx = start_idx + len(cur_token)

        if cur_token_idx + 1 < len(bio_tags):
            next_tag = bio_tags[cur_token_idx + 1]
            # If last word skip the following
            if next_tag.startswith("I"):
                while next_tag.startswith("I"):
                    cur_token_idx += 1
                    cur_token = tokens[cur_token_idx]
                    try:
                        next_tag = bio_tags[cur_token_idx + 1]
                    except IndexError:
                        break

                tmp_idx = content.find(cur_token, cur_substring_idx)
                # This line overwrites end_idx, in case there is a
                # multi-term annotation.
                end_idx = tmp_idx + len(cur_token)

        # Ends at the last character, and not after!
        idx_tuple = (start_idx, end_idx - 1)
        cur_substring_idx = end_idx

        annotations.append(Annotation(
            content[start_idx:end_idx],
            cur_label,
            idx_tuple))

        cur_token_idx += 1

    return AnnotatedDocument(
        document.content, annotations=annotations, encoding=document.encoding)


def transform_annotated_documents_to_multiclass_dictionary(
        annotated_documents, dict_filename, 
        stopwords=None, write_entity_type=True):
    """ Convert a collection of AnnotatedDocument objects to (phrase, 
        entity_type) tuples and writes them out to dict_filename. 
        
        Args:
            annotated_documents -- collection of AnnotatedDocument objects
            dict_filename -- path to dictionary file to create
            stopwords -- specify set of phrases (usually english stopwords)
                that should not be marked up as entities. Default = None
                implies no stopword filtering
            write_entity_type -- if True, writes out entities as TSV (phrase,
                entity_type), else writes out just the phrase, one per line.
                Former format suitable for ExactMatchMultiClassDictionaryNER,
                latter format suitable for ExactMatchDictionaryNER.

        Returns:
            None
    """
    
    fdict = open(dict_filename, "w")
    for annotated_document in annotated_documents:
        tokens, tags = transform_annotated_document_to_bio_format(annotated_document)
        phrase_tokens, prev_tag, already_seen_phrases = [], None, set()
        for token, tag in zip(tokens, tags):
            # print("token:", token, "tag:", tag)
            if tag == "O":
                if len(phrase_tokens) > 0:
                    phrase = " ".join(phrase_tokens)
                    prev_tag = prev_tag[2:]  # remove B_ and I_ prefix
                    # print("... phrase:", phrase, "tag:", prev_tag)
                    if phrase not in already_seen_phrases:
                        if stopwords is not None and phrase not in stopwords:
                            if write_entity_type:
                                fdict.write("{:s}\t{:s}\n".format(phrase, prev_tag))
                            else:
                                fdict.write("{:s}\n".format(phrase))
                            already_seen_phrases.add(phrase)
                    phrase_tokens, prev_tag = [], None
                continue
            else:
                phrase_tokens.append(token)
                prev_tag = tag

        if len(phrase_tokens) > 0:
            phrase = " ".join(phrase_tokens)
            prev_tag = prev_tag[2:]  # remove B_ and I_ prefix
            # print("... (last) phrase:", phrase, "tag:", prev_tag)
            if phrase not in already_seen_phrases:
                if stopwords is not None and phrase not in stopwords:
                    if write_entity_type:
                        fdict.write("{:s}\t{:s}\n".format(phrase, prev_tag))
                    else:
                        fdict.write("{:s}\n".format(phrase))

    fdict.close()


def split_annotated_documents(
        annotated_documents, splitter=document_to_sentences):
    """ Wrapper function that applies `split_annotated_document` to a
        batch of documents.
    """
    result_ann = []
    for annotated_document in annotated_documents:
        result_ann.extend(split_annotated_document(
            annotated_document, splitter))
    return result_ann


def split_annotated_document(
        annotated_document, splitter=document_to_sentences):
    """ Splits an annotated document and maintains the annotation offsets.

        This function accepts an AnnotatedDocument object as parameter along
        with an optional tokenization method. It splits the document according
        to the tokenization method, and returns a list of AnnotatedDocument
        objects, where the annotation offsets have been adjusted.

        Args:
            annotated_document (AnnotatedDocument): The document that will be
                split into more documents.
            splitter: (func, optional): A function that accepts a string as
                input and returns a list of strings. Defaults to
                `document_to_sentences`, which is the default sentence splitter
                for this library.

        Returns:
            list(AnnotatedDocument): A list of annotated documents.
    """

    snippets = [
        snippet.strip() for snippet in
        splitter(annotated_document.plain_text_)]
    annotations = annotated_document.annotations

    cur_snippet_idx = 0
    cur_ann_idx = 0

    result_ann = []

    # Iterate every snippet of text and isolate its annotations.
    # Then construct a single AnnotatedDocument object with them.
    while cur_snippet_idx < len(snippets) and cur_ann_idx < len(annotations):
        cur_substring_idx = 0

        token_ann = []

        cur_snippet = snippets[cur_snippet_idx]

        cur_annotation_text = annotations[cur_ann_idx].text
        cur_annotation_label = annotations[cur_ann_idx].label
        idx_found = cur_snippet.find(cur_annotation_text, cur_substring_idx)
        # Iterate the annotations for as long as we keep finding them in
        # the current snippet of text.
        while idx_found != -1:
            cur_annotation_offsets = (
                idx_found, idx_found + len(cur_annotation_text) - 1)
            token_ann.append(Annotation(
                cur_annotation_text,
                cur_annotation_label,
                cur_annotation_offsets))

            cur_substring_idx = idx_found + len(cur_annotation_text)
            cur_ann_idx += 1

            if cur_ann_idx < len(annotations):
                cur_annotation_text = annotations[cur_ann_idx].text
                cur_annotation_label = annotations[cur_ann_idx].label
                idx_found = cur_snippet.find(
                    cur_annotation_text, cur_substring_idx)
            else:
                break

        result_ann.append(AnnotatedDocument(
            cur_snippet.encode(annotated_document.encoding),
            token_ann))
        cur_snippet_idx += 1

    return result_ann
