import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str)
    parser.add_argument("--gts", type=str)
    args = parser.parse_args()
    return args

def main(args):
    ref_dict = json.load(open(args.ref, "r"))
    gts_dict = json.load(open(args.gts, "r"))

    scorer_bleu = Bleu(n = 4)
    score_bleu, scores_bleu = scorer_bleu.compute_score(ref_dict, gts_dict)

    scorer_rouge = Rouge()
    score_rouge, scores_rouge = scorer_rouge.compute_score(ref_dict, gts_dict)

    scorer_cider = Cider()
    score_cider, scores_cider = scorer_cider.compute_score(ref_dict, gts_dict)

    scorer_meteor = Meteor()
    score_meteor, scores_meteor = scorer_meteor.compute_score(ref_dict, gts_dict)

    scorer_spice = Spice()
    score_spice, scores_spice = scorer_spice.compute_score(ref_dict, gts_dict)
    print(score_bleu[0], score_bleu[3], score_meteor, score_rouge, score_cider, score_spice)
    
if __name__ == "__main__":
    args = get_args()
    main(args)