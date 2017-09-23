"""
This File will contain a reporter class which will allow us to report the whole experiment.
"""

"""
# TODO mainly how mean iou 
 a nd loss are chaning thats what I mainly care about isA mean iou on validation 
 of course if you can also log the inference time
 avg inference time a2so
on validation
the problem with tensorboard
is that sometimes
I am running on a server with no gui
then I have to reroute the tensorboard
its a headache
so its much easier to have a log file that I can open in vim
to observe these
one last thing can you also
log the perclass iou
so if you have the metrics object metrics lets say
just do metrics.iou

"""

import json


class Reporter:
    """
    This class will contain APIs to facilitate the process of reporting the experiment
    """

    def __init__(self, file_json_to_save):
        self.report_dict = dict()
        self.json_file = file_json_to_save
        # init main keys in the report

    def finalize(self):
        with open(self.json_file, 'w') as file:
            json.dump(self.report_dict, file, sort_keys=True, indent=4, separators=(',', ': '))

    def report(self, key, value):
        self.report_dict[key] = value


if __name__ == '__main__':
    reporter = Reporter('../logs/x.json')
    reporter.report('mean-iou', [[1, 1], 2, 3, 4, 5, 6, 7])
    reporter.report('mean-iou_2', [1, 2, 3, 4, 5, 6, 7])
    reporter.report('mean-iou_3', [1, 2, 3, 4, 5, 6, 7])
    reporter.report('mean-iou_4', {'x': 'y'})
    reporter.finalize()
