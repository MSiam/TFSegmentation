from sklearn.metrics import confusion_matrix
import numpy
#import pdb
import math

class Metrics(object):
    """
    Contains evaluation metrics

    """
    def __init__(self, nclasses):
        self.nclasses= nclasses
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.cost = 0
        self.error = 0
        self.prec = 0
        self.rec = 0
        self.fmes = 0
        self.conf_mat= numpy.zeros([nclasses, nclasses], dtype= numpy.float32)

    def reset(self):
        self.fp = 0
        self.fn = 0
        self.tp = 0
        self.cost = 0
        self.error = 0
        self.prec = 0
        self.rec = 0
        self.fmes = 0
        self.conf_mat= numpy.zeros([self.nclasses, self.nclasses], dtype= numpy.float32)
        self.mean_iou_index= 0

    def update_metrics_batch(self, preds, y):
        error = 0
        cost = 0
        for i in range(preds.shape[0]):
            self.update_metrics(preds[i],y[i],error,cost)

    def update_metrics(self, preds, y, error, cost):
        self.error += error
        self.cost += cost
        cf_m= confusion_matrix(y.flatten(), preds.flatten(), range(0, self.nclasses))
        self.conf_mat+=cf_m

    def compute_rates(self, nonignore=None):
        perclass_tp = numpy.diagonal(self.conf_mat).astype(numpy.float32)
        if nonignore is not None:
            perclass_fp = self.conf_mat[nonignore,:].sum(axis=0) - perclass_tp #SUM(nji) - nii
        else:
            perclass_fp = self.conf_mat.sum(axis=0) - perclass_tp #SUM(nji) - nii

        perclass_fn = self.conf_mat.sum(axis=1) - perclass_tp

        if self.nclasses == 2:
            self.tp= perclass_tp[1]
            self.fp= perclass_fp[1]
            self.fn= perclass_fn[1]
        self.iou= perclass_tp/(perclass_fp+ perclass_tp+ perclass_fn)
        if nonignore is not None:
            self.iou= self.iou[nonignore]
        else:
            self.iou= self.iou[:-1]

        self.mean_iou_index= self.getScoreAverage(self.iou)

    def getScoreAverage(self, scoreList):
        validScores = 0
        scoreSum    = 0.0
        for score in scoreList:
            if not math.isnan(score):
                validScores += 1
                scoreSum += score
        if validScores == 0:
            return float('nan')
        return scoreSum / validScores

    def compute_final_metrics(self, nmini_batches, nonignore=None):
        self.compute_rates(nonignore)
        if not (self.tp==0 and (self.fp==0 or self.fn==0)):
            self.prec= numpy.true_divide(self.tp,self.tp+self.fp)
            self.rec= numpy.true_divide(self.tp, self.tp+self.fn)
            self.fmes= (2*self.rec*self.prec)/(self.rec+self.prec)
        self.error /= nmini_batches
        self.cost /= nmini_batches
        return self.mean_iou_index
