import os
import json
import numpy as np
from collections import defaultdict
import argparse


def evaluate_counting(fname, ref_file_path = "datasets/NSR-1K/spatial/spatial.val.json"):
    # load prediction results
    # pred_filename = f'{args.gpt_type}.{args.setting}.{args.icl_type}.k_{args.K}.px_{args.canvas_size}.json'
    # prediction_list = json.load(open(os.path.join(args.prediction_dir, pred_filename)))
    pred_filename = os.path.basename(fname)
    prediction_list = json.load(open(fname))

    # load gt val examples
    val_example_list = json.load(open(ref_file_path))
    id2subtype = {d['id']:d['sub-type'] for d in val_example_list}
    ref_file = {x['id']: x for x in val_example_list}


    precision_list = []
    recall_list = []
    iou_list = []
    mae_list = []
    acc_list = []
    for pred_eg in prediction_list:
        val_eg = ref_file[int(pred_eg['query_id'])]
        pred_object_count = defaultdict(lambda: 0)
        for category, _ in pred_eg['object_list']:
            if category is None: continue
            for x in val_eg['num_object']:
                if category.lstrip("a ").lstrip("an ").lstrip("the ") in x[0] or x[0] in category.lstrip("a ").lstrip("an ").lstrip("the "):
                    category = x[0]
            pred_object_count[category] += 1
        
        if id2subtype[pred_eg['query_id']] == 'comparison':
            (obj1, gt_num1), (obj2, gt_num2) = val_eg['num_object']
            pred_num1 = pred_object_count[obj1]
            pred_num2 = pred_object_count[obj2]
            
            # equal cases
            if gt_num1 == gt_num2 == pred_num1 == pred_num2:
                acc_list.append(1)
            # < or >
            elif gt_num1 == pred_num1 and (gt_num1-gt_num2)*(pred_num1-pred_num2) > 0:
                acc_list.append(1)
            else:
                acc_list.append(0)

        else:
            cnt_gt_total = 0
            cnt_pred_total = sum(pred_object_count.values())
            cnt_intersection_total = 0
            cnt_union_total = 0
            absolute_error = 0
            appeared_category_list = []
            all_matched = True

            for category, gt_cnt in val_eg['num_object']:
                cnt_gt_total += gt_cnt
                pred_cnt = pred_object_count[category]
                cnt_intersection_total += min(pred_cnt, gt_cnt)
                cnt_union_total += max(pred_cnt, gt_cnt)
                absolute_error += abs(pred_cnt - gt_cnt)
                appeared_category_list.append(category)
                if pred_cnt != gt_cnt:  # check if all the mentioned objects are predicted correctly
                    all_matched = False

            # accuracy
            acc_list.append(1 if all_matched else 0)

            # MAE
            if not len(appeared_category_list):
                mae_list.append(0)
            else:
                mae_list.append(float(absolute_error) / len(appeared_category_list))

            # precision, recall, IoU
            if not cnt_intersection_total:
                precision_list.append(0)
                recall_list.append(0)
                iou_list.append(0)
            else:
                precision_list.append(float(cnt_intersection_total) / cnt_pred_total)
                recall_list.append(float(cnt_intersection_total) / cnt_gt_total)
                iou_list.append(float(cnt_intersection_total) / cnt_union_total)

    # print results
    # print(f'Setting: {args.setting} (#eg: {len(prediction_list)})\tGPT-3: {args.gpt_type} - {args.icl_type}\tk = {args.K}')
    print(f'{pred_filename}, #eg: {len(prediction_list)}')
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_acc = np.mean(acc_list)
    avg_mae = np.mean(mae_list)
    avg_iou = np.mean(iou_list)
    score_info = {
        'precision': avg_precision,
        'recall': avg_recall,
        'precision_list': precision_list,
        'recall_list': recall_list,
        'acc': avg_acc,
        'acc_list': acc_list,
        'mae': avg_mae,
        'mae_list': mae_list,
        'iou': avg_iou,
        'iou_list': iou_list,
    }
    print(f'\tPrecision = {avg_precision*100:.2f} %\n\tRecall = {avg_recall*100:.2f} %' \
            f'\n\tIoU = {avg_iou*100:.2f} %\n\tMAE = {avg_mae:.2f}\n\tacc = {avg_acc*100:.2f} %')
    return score_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str)
    args = parser.parse_args()

    evaluate_counting(args.file, "datasets/NSR-1K/counting/counting.val.json")