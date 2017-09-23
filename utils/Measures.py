import numpy
import tensorflow as tf
from sklearn.metrics import confusion_matrix, average_precision_score

from utils import Constants
# from Log import log


def create_confusion_matrix(pred, targets, n_classes):
  result = None
  targets = targets.reshape((targets.shape[0], -1))
  pred = pred.reshape((pred.shape[0], -1))
  for i in range(pred.shape[0]):
    conf_matrix = confusion_matrix(targets[i],
                                   pred[i],
                                   range(0, n_classes))
    conf_matrix = conf_matrix[numpy.newaxis, :, :]

    if result is None:
      result = conf_matrix
    else:
      result = numpy.append(result, conf_matrix, axis=0)
  return result


def get_average_precision(targets, outputs, conf_matrix):
  targets = targets.reshape(targets.shape[0], -1)
  outputs = outputs[:, :, :, :, 1]
  outputs = outputs.reshape(outputs.shape[1], -1)

  ap = numpy.empty(outputs.shape[0], numpy.float64)
  # ap_interpolated = numpy.empty(outputs.shape[0], numpy.float64)
  for i in range(outputs.shape[0]):
    # precision, recall, thresholds = precision_recall_curve(targets[i], outputs[i])
    ap[i] = average_precision_score(targets[i].flatten(), outputs[i].flatten())
    # result = eng.get_ap(matlab.double(outputs[i].tolist()), matlab.double(targets[i].tolist()))
    # ap_interpolated[i] = result

  ap = numpy.nan_to_num(ap)
  # ap_interpolated = numpy.nan_to_num(ap_interpolated)
  return ap


def compute_binary_ious_tf(targets, outputs):
  binary_ious = [compute_iou_for_binary_segmentation(target, output) for target, output in
                 zip(targets, outputs)]
  return numpy.sum(binary_ious, dtype="float32")


def compute_iou_for_binary_segmentation(y_argmax, target):
  I = numpy.logical_and(y_argmax == 1, target == 1).sum()
  U = numpy.logical_or(y_argmax == 1, target == 1).sum()
  if U == 0:
    IOU = 1.0
  else:
    IOU = float(I) / U
  return IOU


def compute_measures_for_binary_segmentation(prediction, target):
  T = target.sum()
  P = prediction.sum()
  I = numpy.logical_and(prediction == 1, target == 1).sum()
  U = numpy.logical_or(prediction == 1, target == 1).sum()

  if U == 0:
    recall = 1.0
    precision = 1.0
    iou = 1.0
  else:
    if T == 0:
      recall = 1.0
    else:
      recall = float(I) / T

    if P == 0:
      precision = 1.0
    else:
      precision = float(I) / P

    iou = float(I) / U

  measures = {"recall": recall, "precision": precision, "iou": iou}
  return measures


def average_measures(measures_dicts):
  keys = measures_dicts[0].keys()
  averaged_measures = {}
  for k in keys:
    vals = [m[k] for m in measures_dicts]
    val = numpy.mean(vals)
    averaged_measures[k] = val
  return averaged_measures


def compute_iou_from_logits(preds, labels, num_labels):
  """
  Computes the intersection over union (IoU) score for given logit tensor and target labels
  :param logits: 4D tensor of shape [batch_size, height, width, num_classes]
  :param labels: 3D tensor of shape [batch_size, height, width] and type int32 or int64
  :param num_labels: tensor with the number of labels
  :return: 1D tensor of shape [num_classes] with intersection over union for each class, averaged over batch
  """
  with tf.variable_scope("IoU"):
    # compute predictions
    # probs = softmax(logits, axis=-1)
    # preds = tf.arg_max(probs, dimension=3)
    # num_labels = preds.get_shape().as_list()[-1];
    IoUs = []
    for label in range(num_labels):
      # find pixels with given label
      P = tf.equal(preds, label)
      L = tf.equal(labels, label)
      # Union
      U = tf.logical_or(P, L)
      U = tf.reduce_sum(tf.cast(U, tf.float32))
      # intersection
      I = tf.logical_and(P, L)
      I = tf.reduce_sum(tf.cast(I, tf.float32))

      IOU = tf.cast(I, tf.float32) / tf.cast(U, tf.float32)
      # U might be 0!
      IOU = tf.where(tf.equal(U, 0), 1, IOU)
      IOU = tf.Print(IOU, [IOU], "iou" + label)
      IoUs.append(IOU)
    return tf.reshape(tf.stack(IoUs), (num_labels,))


def calc_measures_avg(measures, n_imgs, ignore_classes):
  measures_result = {}
  if Constants.ERRORS in measures:
    measures_result[Constants.ERRORS] = measures[Constants.ERRORS] / n_imgs

  if Constants.IOU in measures:
    measures_result[Constants.IOU] = measures[Constants.IOU] / n_imgs

  if Constants.AP in measures:
    measures_result[Constants.AP] = numpy.sum(measures[Constants.AP]) / n_imgs

  if Constants.AP_INTERPOLATED in measures:
    measures_result[Constants.AP_INTERPOLATED] = numpy.sum(measures[Constants.AP_INTERPOLATED]) / n_imgs

  if Constants.BINARY_IOU in measures:
    measures_result[Constants.BINARY_IOU] = numpy.sum(measures[Constants.BINARY_IOU]) / n_imgs

  # TODO: This has to be added as IOU instead of conf matrix.
  if Constants.CONFUSION_MATRIX in measures:
    measures_result[Constants.IOU] = calc_iou(measures, n_imgs, ignore_classes)

  if Constants.CLICKS in measures:
    clicks = [int(x.rsplit(':', 1)[-1]) for x in measures[Constants.CLICKS]]
    measures_result[Constants.CLICKS] = float(numpy.sum(clicks)) / n_imgs

  return measures_result


def calc_iou(measures, n_imgs, ignore_classes):
  assert Constants.CONFUSION_MATRIX in measures
  conf_matrix = measures[Constants.CONFUSION_MATRIX]
  assert conf_matrix.shape[0] == n_imgs # not sure, if/why we need these n_imgs

  I = (numpy.diagonal(conf_matrix, axis1=1, axis2=2)).astype("float32")
  sum_predictions = numpy.sum(conf_matrix, axis=1)
  sum_labels = numpy.sum(conf_matrix, axis=2)
  U = sum_predictions + sum_labels - I
  n_classes = conf_matrix.shape[-1]
  class_mask = numpy.ones((n_classes,))
  # Temporary fix to avoid index out of bounds when there is a void label in the list of classes to be ignored.
  ignore_classes = numpy.array(ignore_classes)
  ignore_classes = ignore_classes[numpy.where(ignore_classes<=n_classes)]
  class_mask[ignore_classes] = 0

  ious = []
  for i, u in zip(I, U):
    mask = numpy.logical_and(class_mask, u != 0)
    if mask.any():
      iou = (i[mask] / u[mask]).mean()
    else:
      print >> log.v5, "warning, mask only consists of ignore_classes"
      iou = 1.0
    ious.append(iou)
  IOU_avg = numpy.mean(ious)
  return IOU_avg


def calc_measures_sum(measures1, measures2):
  measures_result = {}

  if not measures1:
    return measures2

  if not measures2:
    return measures1

  if Constants.ERRORS in measures1 and Constants.ERRORS in measures2:
    measures_result[Constants.ERRORS] = measures1[Constants.ERRORS] + measures2[Constants.ERRORS]

  if Constants.IOU in measures1 and Constants.IOU in measures2:
    measures_result[Constants.IOU] = measures1[Constants.IOU] + measures2[Constants.IOU]

  if Constants.BINARY_IOU in measures1 and Constants.BINARY_IOU in measures2:
    measures_result[Constants.BINARY_IOU] = measures1[Constants.BINARY_IOU] + measures2[Constants.BINARY_IOU]

  if Constants.CONFUSION_MATRIX in measures1 and Constants.CONFUSION_MATRIX in measures2:
    conf_matrix1 = measures1[Constants.CONFUSION_MATRIX]
    conf_matrix2 = measures2[Constants.CONFUSION_MATRIX]

    measures_result[Constants.CONFUSION_MATRIX] = numpy.append(conf_matrix2, conf_matrix1, axis=0)

  if Constants.AP in measures1 and Constants.AP in measures2:
    measures_result[Constants.AP] = measures1[Constants.AP] + measures2[Constants.AP]

  if Constants.AP_INTERPOLATED in measures1 and Constants.AP_INTERPOLATED in measures2:
    measures_result[Constants.AP_INTERPOLATED] = measures1[Constants.AP_INTERPOLATED] + measures2[
      Constants.AP_INTERPOLATED]

  if Constants.CLICKS in measures1 and Constants.CLICKS in measures2:
    measures_result[Constants.CLICKS] = numpy.append(measures1[Constants.CLICKS], measures2[Constants.CLICKS])

  return measures_result


def get_error_string(measures, task):
  result_string = ""

  if task == "train":
    result_string += "train_err:"
  else:
    result_string += "valid_err:"
  if Constants.ERRORS in measures:
    result_string += "  %4f" % measures[Constants.ERRORS]

  if Constants.IOU in measures:
    result_string += "(IOU) %4f" % measures[Constants.IOU]

  if Constants.BINARY_IOU in measures:
    result_string += "(binary IOU) %4f" % measures[Constants.BINARY_IOU]

  if Constants.CONFUSION_MATRIX in measures:
    result_string += "(IOU) %4f" % measures[Constants.CONFUSION_MATRIX]

  if Constants.AP in measures:
    result_string += " (mAP) %4f" % measures[Constants.AP]

  if Constants.AP_INTERPOLATED in measures:
    result_string += " (mAP-intepolated) %4f" % measures[Constants.AP_INTERPOLATED]

  if Constants.RANKS in measures:
    result_string += " " + measures[Constants.RANKS]

  if Constants.CLICKS in measures:
    # clicks = [x.rsplit(':', 1)[-1] for x in measures[Constants.TAGS]]
    result_string += " (Avg Clicks): %4f" %measures[Constants.CLICKS]

  return result_string


def calc_ap_from_cm(conf_matrix):
  tp = conf_matrix[1][1]
  fp = conf_matrix[0][1]
  fn = conf_matrix[1][0]

  precision = tp.astype(float) / (tp + fp).astype(float)
  recall = tp.astype(float) / (tp + fn).astype(float)

  return precision, recall
