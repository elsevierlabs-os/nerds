def calculate_precision_recall_f1score(y_pred, y_true, entity_label=None):
    """ Calculates precision recall and F1-score metrics.

        Args:
            y_pred (list(AnnotatedDocument)): The predictions of an NER
                model in the form of a list of annotated documents.
            y_true (list(AnnotatedDocument)): The ground truth set of
                annotated documents.
            entity_label (str, optional): The label of the entity for which
                the scores are calculated. It defaults to None, which means
                all annotated entities.

        Returns:
            (3-tuple(float)): (Precision, Recall, F1-score)
    """

    # Flatten all annotations
    all_y_pred_ann = []
    all_y_true_ann = []
    if entity_label is None:
        for annotated_document in y_pred:
            all_y_pred_ann.extend(annotated_document.annotations)
        for annotated_document in y_true:
            all_y_true_ann.extend(annotated_document.annotations)
    else:
        for annotated_document in y_pred:
            all_y_pred_ann.extend([
                annotation for annotation in annotated_document.annotations
                if annotation.label == entity_label
            ])
        for annotated_document in y_true:
            all_y_true_ann.extend([
                annotation for annotation in annotated_document.annotations
                if annotation.label == entity_label
            ])

    tp = 0.0
    fp = 0.0
    fn = 0.0

    # Convert true annotations to a set in O(n) for quick lookup
    all_y_true_ann_lookup = set(all_y_true_ann)
    # True positives are predicted annotations that are confirmed by
    # their existence in the ground truth dataset. False positives are
    # predicted annotations that are not in the ground truth dataset.
    for annotation in all_y_pred_ann:
        if annotation in all_y_true_ann_lookup:
            tp += 1.0
        else:
            fp += 1.0

    # Convert predictions to a set in O(n) for quick lookup
    all_y_pred_ann_lookup = set(all_y_pred_ann)
    # False negatives are annotations in the ground truth dataset that
    # were never predicted by the system.
    for annotation in all_y_true_ann:
        if annotation not in all_y_pred_ann_lookup:
            fn += 1.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.
    f1_score = (2 * precision * recall) / (precision + recall) if\
        (precision + recall) > 0 else 0.

    return (precision, recall, f1_score)


def classification_report(y_pred, y_true, entity_labels):
    """ Pretty prints a classification report based on precision, 
        recall, and f1-scores from `calculate_precision_recall_f1score`
        for each entity label supplied and the aggregate.

        Args:
            y_pred (list(AnnotatedDocument)): The predictions of an NER
                model in the form of a list of annotated documents.
            y_true (list(AnnotatedDocument)): The ground truth set of
                annotated documents.
            entity_labels (list(str)): The entity labels for which
                the scores are calculated.

        Returns:
            None
    """
    
    print("                    precision    recall  f1-score")
    for l in sorted(entity_labels):
        p, r, f = calculate_precision_recall_f1score(y_pred, y_true, entity_label=l)
        print("{:20s}    {:.3f}     {:.3f}     {:.3f}".format(l, p, r, f))
    p, r, f = calculate_precision_recall_f1score(y_pred, y_true)
    print("")
    print("{:20s}    {:.3f}     {:.3f}     {:.3f}".format("--all--", p, r, f))
