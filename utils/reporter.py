"""
This File will contain a reporter class which will allow us to report the whole experiment.
"""

import json
import numpy as np


class Reporter:
    """
    This class will contain APIs to facilitate the process of reporting the experiment
    """

    def __init__(self, file_json_to_save, args):
        self.report_dict = dict()
        self.json_file = file_json_to_save
        # init main keys in the report
        self.report_dict['train-acc'] = {}
        self.report_dict['train-loss'] = {}
        self.report_dict['validation-acc'] = {}
        self.report_dict['validation-loss'] = {}
        self.report_dict['avg_inference_time_on_validation'] = {}
        self.report_dict['validation-mean-iou'] = {}
        self.report_dict['validation-total-mean-iou'] = {}
        self.report_dict['validation-mean-iou']['road'] = {}
        self.report_dict['validation-mean-iou']['sidewalk'] = {}
        self.report_dict['validation-mean-iou']['building'] = {}
        self.report_dict['validation-mean-iou']['wall'] = {}
        self.report_dict['validation-mean-iou']['fence'] = {}
        self.report_dict['validation-mean-iou']['pole'] = {}
        self.report_dict['validation-mean-iou']['traffic light'] = {}
        self.report_dict['validation-mean-iou']['traffic sign'] = {}
        self.report_dict['validation-mean-iou']['vegetation'] = {}
        self.report_dict['validation-mean-iou']['terrain'] = {}
        self.report_dict['validation-mean-iou']['sky'] = {}
        self.report_dict['validation-mean-iou']['person'] = {}
        self.report_dict['validation-mean-iou']['rider'] = {}
        self.report_dict['validation-mean-iou']['car'] = {}
        self.report_dict['validation-mean-iou']['truck'] = {}
        self.report_dict['validation-mean-iou']['bus'] = {}
        self.report_dict['validation-mean-iou']['train'] = {}
        self.report_dict['validation-mean-iou']['motorcycle'] = {}
        self.report_dict['validation-mean-iou']['bicycle'] = {}
        self.report_dict['validation-mean-iou']['ignore'] = {}
        # put the arguments of the report
        self.report_dict['arguments-of-the-experiment'] = {}
        for key, value in sorted(vars(args).items()):
            self.report_dict['arguments-of-the-experiment'][key] = value

    def finalize(self):
        with open(self.json_file, 'w') as file:
            json.dump(self.report_dict, file, sort_keys=True, indent=4)

    def report(self, key, value):
        self.report_dict[key] = value

    def report_experiment_statistics(self, statistics, epoch, value):
        self.report_dict[statistics][epoch] = value

    def report_experiment_validation_iou(self, epoch, mean_iou, per_class_mean_iou):
        self.report_dict['validation-total-mean-iou'][epoch] = mean_iou
        self.report_dict['validation-mean-iou']['road'][epoch] = str(per_class_mean_iou[0])
        self.report_dict['validation-mean-iou']['sidewalk'][epoch] = str(per_class_mean_iou[1])
        self.report_dict['validation-mean-iou']['building'][epoch] = str(per_class_mean_iou[2])
        self.report_dict['validation-mean-iou']['wall'][epoch] = str(per_class_mean_iou[3])
        self.report_dict['validation-mean-iou']['fence'][epoch] = str(per_class_mean_iou[4])
        self.report_dict['validation-mean-iou']['pole'][epoch] = str(per_class_mean_iou[5])
        self.report_dict['validation-mean-iou']['traffic light'][epoch] = str(per_class_mean_iou[6])
        self.report_dict['validation-mean-iou']['traffic sign'][epoch] = str(per_class_mean_iou[7])
        self.report_dict['validation-mean-iou']['vegetation'][epoch] = str(per_class_mean_iou[8])
        self.report_dict['validation-mean-iou']['terrain'][epoch] = str(per_class_mean_iou[9])
        self.report_dict['validation-mean-iou']['sky'][epoch] = str(per_class_mean_iou[10])
        self.report_dict['validation-mean-iou']['person'][epoch] = str(per_class_mean_iou[11])
        self.report_dict['validation-mean-iou']['rider'][epoch] = str(per_class_mean_iou[12])
        self.report_dict['validation-mean-iou']['car'][epoch] = str(per_class_mean_iou[13])
        self.report_dict['validation-mean-iou']['truck'][epoch] = str(per_class_mean_iou[14])
        self.report_dict['validation-mean-iou']['bus'][epoch] = str(per_class_mean_iou[15])
        self.report_dict['validation-mean-iou']['train'][epoch] = str(per_class_mean_iou[16])
        self.report_dict['validation-mean-iou']['motorcycle'][epoch] = str(per_class_mean_iou[17])
        self.report_dict['validation-mean-iou']['bicycle'][epoch] = str(per_class_mean_iou[18])


if __name__ == '__main__':
    class Empty():
        pass


    empty = Empty()
    empty.lololololololo = 1.23562662436
    empty.tatatatatatattt = 2.2346346346346
    reporter = Reporter('../logs/x.json', empty)
    reporter.report_experiment_validation_iou('epoch-0', 0.0, np.zeros((20,)))
    reporter.report_experiment_validation_iou('epoch-1', 0.0, np.zeros((20,)))
    reporter.report_experiment_validation_iou('epoch-2', 0.0, np.zeros((20,)))
    reporter.finalize()
