import webdataset as wds
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider


def load_refs_from_wds(eval_shards):
    """Load all reference captions from the eval WDS shards.

    Returns:
        dict mapping image_id (int) -> list of reference caption strings,
        in the format expected by pycocoevalcap scorers.
    """
    refs = {}
    dataset = wds.WebDataset(
        eval_shards, shardshuffle=False, nodesplitter=None
    ).decode()
    for sample in dataset:
        image_id = int(sample["__key__"])
        raw = sample["json"]
        candidates = raw["captions"]
        captions = [str(c).strip() for c in candidates]
        refs[image_id] = captions
    return refs


def compute_metrics(preds, refs):
    metrics = {}

    scorer = Bleu(4)
    score, _ = scorer.compute_score(refs, preds)
    metrics["BLEU-4"] = score[3]

    scorer = Meteor()
    score, _ = scorer.compute_score(refs, preds)
    metrics["METEOR"] = score

    scorer = Cider()
    score, _ = scorer.compute_score(refs, preds)
    metrics["CIDEr"] = score * 100

    return metrics
