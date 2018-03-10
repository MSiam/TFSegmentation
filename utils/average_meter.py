class AverageMeter:
    def __init__(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        raise NotImplementedError()

    def update_n(self, val, n):
        raise NotImplementedError()

    @property
    def val(self):
        return self.avg


class FPSMeter(AverageMeter):
    """
    Class to measure frame per second in our networks
    """

    def __init__(self):
        super().__init__()
        self.frame_per_second = 0.0
        self.f_in_milliseconds = 0.0

        self.frame_count = 0
        self.milliseconds = 0.0

    def reset(self):
        self.frame_per_second = 0.0
        self.f_in_milliseconds = 0.0

        self.frame_count = 0

    def update(self, seconds):
        self.milliseconds += seconds * 1000
        self.frame_count += 1

        self.frame_per_second = self.frame_count / (self.milliseconds / 1000.0)
        self.f_in_milliseconds = self.milliseconds / self.frame_count

    @property
    def mspf(self):
        return self.f_in_milliseconds

    @property
    def fps(self):
        return self.frame_per_second

    def print_statistics(self):
        print("""
Statistics of the FPSMeter
Frame per second: {:.2f} fps
Milliseconds per frame: {:.2f} ms in one frame
These statistics are calculated based on
{:d} Frames and the whole taken time is {:.4f} Seconds
        """.format(self.frame_per_second, self.f_in_milliseconds, self.frame_count, self.milliseconds / 1000.0))


def main_test_fps():
    fpsfps = FPSMeter()
    fpsfps.update(0.025)
    fpsfps.update(0.025)
    fpsfps.update(0.025)
    fpsfps.update(0.025)
    fpsfps.print_statistics()


if __name__ == '__main__':
    main_test_fps()
