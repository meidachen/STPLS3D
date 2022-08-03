import logging
import os,json
import numpy as np
from sys import argv

##############
##### log ####
##############


def create_logger(log_file):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    log_format = '[%(asctime)s  %(levelname)s  %(filename)s  line %(lineno)d  %(process)d]  %(message)s'
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)

    logging.basicConfig(level=logging.DEBUG, format=log_format, filename=log_file)    # filename: build a FileHandler
    return logger



################
##### eval #####
################


# ---------- Label info ---------- #
CLASS_LABELS = ['Build', 'LowVeg', 'MediumVeg', 'HighVeg', 'Vehicle', 'Truck', 'Aircraft', 'MilitaryVeh', 'Bike', 'Motorcycle', 'LightPole', 'StreetSign', 'Clutter', 'Fence']
VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
# ---------- Evaluation params ---------- #
# overlaps for evaluation
OVERLAPS             = np.append(np.arange(0.5,0.95,0.05), 0.25)
# minimum region size for evaluation [verts]
MIN_REGION_SIZES     = np.array( [ 10 ] )
# distance thresholds [m]
DISTANCE_THRESHES    = np.array( [  float('inf') ] )
# distance confidences
DISTANCE_CONFS       = np.array( [ -float('inf') ] )


def evaluate_matches(matches):
    overlaps = OVERLAPS
    min_region_sizes = [MIN_REGION_SIZES[0]]
    dist_threshes = [DISTANCE_THRESHES[0]]
    dist_confs = [DISTANCE_CONFS[0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), np.float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    # filter groups in ground truth
                    gt_instances = [gt for gt in gt_instances if
                                    gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=np.bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (
                            gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (
                            gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt['dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    if(len(y_true_sorted_cumsum) == 0):
                        num_true_examples = 0
                    else:
                        num_true_examples = y_true_sorted_cumsum[-1]
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di, li, oi] = ap_current
    return ap


def compute_averages(aps):
    d_inf = 0
    o50   = np.where(np.isclose(OVERLAPS,0.5))
    o25   = np.where(np.isclose(OVERLAPS,0.25))
    oAllBut25  = np.where(np.logical_not(np.isclose(OVERLAPS,0.25)))
    avg_dict = {}
    #avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
    avg_dict["classes"]  = {}
    for (li,label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name]             = {}
        #avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,oAllBut25])
        avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ d_inf,li,o50])
        avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ d_inf,li,o25])
    return avg_dict


def assign_instances_for_scan(scene_name, pred_info, gt_ids):

    # get gt instances
    ### CLASS_LABELS is name of semantic label
    ### VALID_CLASS_IDS are id of semantic labels
    ### ID_TO_LABEL id to label name
    gt_instances = get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)

    # associate
    gt2pred = gt_instances.copy()
    for label in gt2pred:
        ### go over all instances id in this semantic
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids//1000, VALID_CLASS_IDS))
    # go thru all prediction masks
    nMask = pred_info['label_id'].shape[0]

    for i in range(nMask):
        label_id = int(pred_info['label_id'][i])
        conf = pred_info['conf'][i]
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info['mask'][i]   # (N), long
        if len(pred_mask) != len(gt_ids):
            print('wrong number of lines in mask#%d: ' % (i)  + '(%d) vs #mesh vertices (%d)' % (len(pred_mask), len(gt_ids)))
        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < MIN_REGION_SIZES[0]:
            continue  # skip if empty

        pred_instance = {}
        pred_instance['filename'] = '{}_{:03d}'.format(scene_name, num_pred_instances)
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection']   = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt


def print_results(avgs,log_file):

    logger = create_logger(log_file)
    sep     = ""
    col1    = ":"
    lineLen = 64

    logger.info("")
    logger.info("#" * lineLen)
    line  = ""
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("AP"        ) + sep
    line += "{:>15}".format("AP_50%"    ) + sep
    line += "{:>15}".format("AP_25%"    ) + sep
    logger.info(line)
    logger.info("#" * lineLen)

    for (li,label_name) in enumerate(CLASS_LABELS):
        ap_avg  = avgs["classes"][label_name]["ap"]
        ap_50o  = avgs["classes"][label_name]["ap50%"]
        ap_25o  = avgs["classes"][label_name]["ap25%"]
        line  = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg ) + sep
        line += sep + "{:>15.3f}".format(ap_50o ) + sep
        line += sep + "{:>15.3f}".format(ap_25o ) + sep
        logger.info(line)

    all_ap_avg  = avgs["all_ap"]
    all_ap_50o  = avgs["all_ap_50%"]
    all_ap_25o  = avgs["all_ap_25%"]

    logger.info("-"*lineLen)
    line  = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg)  + sep
    line += "{:>15.3f}".format(all_ap_50o)  + sep
    line += "{:>15.3f}".format(all_ap_25o)  + sep
    logger.info(line)
    logger.info("")


####################
##### utils_3d #####
####################

# ------------ Instance Utils ------------ #

class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    ### instance_id is unique instance: int
    ### mesh_vert_instances is all instances: 1d array
    def __init__(self, mesh_vert_instances, instance_id):
        if (instance_id == -1):
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        ### the number of point labels as instance_id
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))
    ### semantic and instance labels are stored in single number by semantic_label * 1000 + inst_id + 1
    ### label_id means semantic id
    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"]    = self.label_id
        dict["vert_count"]  = self.vert_count
        dict["med_dist"]    = self.med_dist
        dict["dist_conf"]   = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id     = int(data["instance_id"])
        self.label_id        = int(data["label_id"])
        self.vert_count      = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist    = float(data["med_dist"])
            self.dist_conf   = float(data["dist_conf"])

    def __str__(self):
        return "("+str(self.instance_id)+")"

### ids is semanitc id
### class_labels is name of labels
### class_ids are number of labels
### id2label number to name
def get_instances(ids, class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    ### get unique instance label
    instance_ids = np.unique(ids)

    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        ### label_id is semantic, check if semantic is in valid semantic list
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances

if __name__ == "__main__":

    ### prepare for evaluation

    # Default I/O directories:
    default_input_dir = r'E:\ECCV_workshop\evaluation_test\input'
    default_output_dir = r'E:\ECCV_workshop\evaluation_test\output'

    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = argv[2]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    score_file = open(os.path.join(output_dir, 'scores.txt'), 'w')

    import glob
    data_path = os.path.join(input_dir, 'ref')
    results_path = os.path.join(input_dir, 'res')
    instance_paths = sorted(glob.glob(os.path.join(results_path, '*.txt')))

    matches = {}
    for instance_path in instance_paths:
        img_id = os.path.basename(instance_path)[:-4]

        try:
            gt = os.path.join(data_path, img_id + '.npy')
            assert os.path.isfile(gt)
            data = np.load(gt)
            coords, rgb, semantic_label, instance_label = data[:,:3], data[:,3:6], np.squeeze(data[:,6]), np.squeeze(data[:,7])
            gt_ids = semantic_label*1000 + instance_label

            pred_infos = open(instance_path, 'r').readlines()
            pred_infos = [x.rstrip().split() for x in pred_infos]
            mask_path, labels, scores = list(zip(*pred_infos))

            preSem = []
            preIns = []
            preConf = []
            for mask_path, label, score in pred_infos:
                mask_full_path = os.path.join(results_path, mask_path)
                mask = np.array(open(mask_full_path).read().splitlines(), dtype=int)
                preIns.append(mask)
                preSem.append(label)
                preConf.append(score)
            preConf = np.array(preConf, dtype=float)
            preSem = np.array(preSem, dtype=int)
            preIns = np.array(preIns)

            pred_info = {}
            pred_info['conf'] = preConf
            pred_info['label_id'] = preSem
            pred_info['mask'] = preIns

            gt2pred, pred2gt = assign_instances_for_scan(str(img_id), pred_info, gt_ids)

            matches[str(img_id)] = {}
            matches[str(img_id)]['gt'] = gt2pred
            matches[str(img_id)]['pred'] = pred2gt

            matches[str(img_id)]['seg_gt'] = semantic_label
            matches[str(img_id)]['seg_pred'] = preSem

        except Exception as inst:
            print("======= ERROR evaluating for" + img_id.capitalize() + " =======")

    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)
    score_file.write('AP' + ": %0.4f\n" % avgs['all_ap'])
    score_file.write('AP50' + ": %0.4f\n" % avgs['all_ap_50%'])
    score_file.write('AP25' + ": %0.4f\n" % avgs['all_ap_25%'])

    for CLASS_LABEL in CLASS_LABELS:
        score_file.write('%s'% (CLASS_LABEL) + ": %0.4f\n" % (avgs['classes'][CLASS_LABEL]['ap']))
        score_file.write('%s_AP50'% (CLASS_LABEL) + ": %0.4f\n" % (avgs['classes'][CLASS_LABEL]['ap50%']))
        score_file.write('%s_AP25'% (CLASS_LABEL) + ": %0.4f\n" % (avgs['classes'][CLASS_LABEL]['ap25%']))
    score_file.close()

    # log_file = r'E:\ECCV_workshop\test\log.txt'
    # print_results(avgs,log_file)
