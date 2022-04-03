/*
Get the IoU between predictions and gt masks
*/

#ifndef CAL_IOU_AND_MASKLABEL_H
#define CAL_IOU_AND_MASKLABEL_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include "../datatype/datatype.h"

//
void cal_iou_and_masklabel_cuda(int nInstance, int nProposal, int *proposals_idx, int *proposals_offset, \
    long *instance_labels, int *instance_pointnum, float *proposals_iou, float *mask_scores_sigmoid, float *mask_label, int mode);

void cal_iou_and_masklabel(at::Tensor proposals_idx_tensor, at::Tensor proposals_offset_tensor, \
    at::Tensor instance_labels_tensor, at::Tensor instance_pointnum_tensor, \
    at::Tensor proposals_iou_tensor, int nInstance, int nProposal, at::Tensor mask_scores_sigmoid_tensor, at::Tensor mask_labels_tensor, int mode);

#endif //CAL_IOU_AND_MASKLABEL_H