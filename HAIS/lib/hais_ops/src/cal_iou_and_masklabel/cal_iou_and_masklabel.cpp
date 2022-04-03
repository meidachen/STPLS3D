/*
Get the IoU between predictions and gt masks
*/

#include "cal_iou_and_masklabel.h"

void cal_iou_and_masklabel(at::Tensor proposals_idx_tensor, at::Tensor proposals_offset_tensor, \
        at::Tensor instance_labels_tensor, at::Tensor instance_pointnum_tensor,  \
        at::Tensor proposals_iou_tensor, int nInstance, int nProposal, 
        at::Tensor mask_scores_sigmoid_tensor, at::Tensor mask_labels_tensor,
        int mode){
    int *proposals_idx = proposals_idx_tensor.data<int>();
    int *proposals_offset = proposals_offset_tensor.data<int>();
    long *instance_labels = instance_labels_tensor.data<long>();
    int *instance_pointnum = instance_pointnum_tensor.data<int>();

    float *proposals_iou = proposals_iou_tensor.data<float>();

    float *mask_scores_sigmoid = mask_scores_sigmoid_tensor.data<float>();
    float *mask_label = mask_labels_tensor.data<float>();




    //input: nInstance (1,), int
    //input: nProposal (1,), int
    //input: proposals_idx (sumNPoint), int
    //input: proposals_offset (nProposal + 1), int
    //input: instance_labels (N), long, 0~total_nInst-1, -100
    //input: instance_pointnum (total_nInst), int
    //input: mask_scores_sigmoid (sumNPoint, 1), float
    //output: proposals_iou (nProposal, total_nInst), float
    //output: mask_label (sumNPoint, 1), float
    cal_iou_and_masklabel_cuda(nInstance, nProposal, proposals_idx, proposals_offset, instance_labels, 
        instance_pointnum, proposals_iou, mask_scores_sigmoid, mask_label, mode);
}