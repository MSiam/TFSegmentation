from Forwarding.OneshotForwarder import OneshotForwarder
from datasets.Util.Timer import Timer
from utils.Measures import average_measures
# from Log import log
import tensorflow as tf
import numpy
from scipy.ndimage.morphology import distance_transform_edt, grey_erosion
import numpy as np
import os
import cv2
from utils.one_shot_utils import process_forward_result
VOID_LABEL = 255


class OnlineAdaptingForwarder(OneshotForwarder):
    def __init__(self,  sess,config,model,trainer):
        super(OnlineAdaptingForwarder, self).__init__(sess,config,model,trainer)
        self.config = config
        self.trainer=trainer
        self.n_adaptation_steps = self.config.n_adaptation_steps
        self.adaptation_interval = self.config.adaptation_interval
        self.adaptation_learning_rate = self.config.adaptation_learning_rate
        self.posterior_positive_threshold = self.config.posterior_positive_threshold
        self.distance_negative_threshold = self.config.distance_negative_threshold
        self.adaptation_loss_scale = self.config.adaptation_loss_scale
        self.debug = self.config.adapt_debug
        self.erosion_size = self.config.adaptation_erosion_size
        self.use_positives = self.config.use_positives
        self.use_negatives = self.config.use_negatives

        self.mot_dir = '/home/gemy/work/datasets/Motion/Motion'
        self.short_dir = '/home/gemy/work/datasets/Motion_4/'
        self.long_dir = '/home/gemy/work/datasets/ARP/'
        self.correct_th = 0.3
        self.neg_th = 0.8

    def process_forward(self ,data, network, save_logits, save_results, targets, ys, start_frame_idx):
        main_folder = "forwarded/" + "/" + data.subset + "/"
        tf.gfile.MakeDirs(main_folder)

        ys_argmax = tf.arg_max(ys, 3)
        feed_dict = data.feed_dict_for_video_frame(frame_idx=start_frame_idx, with_annotations=True)
        feed_dict[self.model.is_training] = False

        if self.config.n_test_samples > 1:

          ys_argmax_val, logits_val, targets_val, tags_val, n = self._run_minibatch_multi_sample(
            feed_dict, ys, targets, network["tags"], network.index_imgs)
          extraction_vals = []
        else:
          ys_argmax_val, logits_val, targets_val, tags_val, n, extraction_vals = self._run_minibatch_single_sample(
            feed_dict, ys, ys_argmax, [], targets, self.model.valid_tags, 1, save_logits)

        measures = []
        for y_argmax, logit, target, tag in zip(ys_argmax_val, logits_val, targets_val, tags_val):
          measure = process_forward_result(y_argmax, logit, target, tag)
          measures.append(measure)
        return n, measures, ys_argmax_val, logits_val, targets_val

    def _oneshot_forward_video(self, video_idx, save_logits):
        # with Timer():
        # finetune on first frame
        self._finetune(video_idx, n_finetune_steps=self.n_finetune_steps)

        network = self.model.test_net
        targets = self.model.valid_raw_labels
        ys = self.model.test_net['out']
        ys = self._adjust_results_to_targets(ys, targets)
        data = self.model.valid_data
        #
        # feed_dict = self.valid_data.feed_dict_for_video_frame(frame_idx=i, with_annotations=True)
        # feed_dict[self.is_training] = False
        # if self.config.n_test_samples <= 1:
        #     ys_argmax_val, targets_val, tags_val, logits_val = self.single_run(
        #         [ys_argmax, self.valid_raw_labels, self.valid_tags, self.test_net['out']], feed_dict)
        # else:
        #     ys_argmax_val, logits_val, targets_val, tags_val = self.multiple_run(
        #         [results, self.valid_index_imgs, self.valid_raw_labels, self.valid_tags], feed_dict)
        #
        #

        n, measures, ys_argmax_val, logits_val, targets_val = self.process_forward(
            data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=0)
        last_mask = targets_val[0]

        adapt_flag = True
        assert n == 1
        n_frames = data.num_examples_per_epoch()

        measures_video = []

        dirs = sorted(os.listdir(self.mot_dir))
        motype = np.load(self.mot_dir + '/' + dirs[video_idx] + '/motype.npy')
        print('Motion Type of this Sequence is ', motype)

        #      masks= np.load(self.mot_dir+dirs[video_idx]+'/mask_'+dirs[video_idx]+'.npy')
        #      indices= np.load(self.mot_dir+dirs[video_idx]+'/indices.npy')

        for t in range(0, n_frames):
            def get_posteriors():
                n_, _, _, logits_val_, _ = self.process_forward(
                    data, network, save_logits=False, save_results=False, targets=targets, ys=ys, start_frame_idx=t)
                assert n_ == 1
                return logits_val_[0]

            if motype == 'static':
                temp = cv2.imread(self.long_dir + dirs[video_idx] + ('/%05d.png' % t), 0)
                temp = (temp - temp.min()) * 1.0 / (temp.max() - temp.min())
                last_mask = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.uint8)
                last_mask[temp > 0.5] = 1
                last_mask = np.expand_dims(last_mask, axis=2)
                self.distance_negative_threshold = 10
                if adapt_flag:
                    negatives = self._adapt(video_idx, t, last_mask, get_posteriors, adapt_flag=1)
                    #             adapt_flag= False
                n, measures, ys_argmax_val, posteriors_val, targets_val = self.process_forward(
                    data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)
                assert n == 1
                assert len(measures) == 1
                measure = measures[0]
                print ( "Motion Adapted frame", t, ":", measure, " factor ", float(ys_argmax_val.sum()) / (
                854 * 480))
            else:
                if t < n_frames - 1:
                    temp = cv2.imread(self.short_dir + dirs[video_idx] + ('/%05d.png' % t), 0)
                    temp = (temp - temp.min()) * 1.0 / (temp.max() - temp.min())
                    last_mask = np.zeros((temp.shape[0], temp.shape[1]), dtype=np.uint8)
                    last_mask[temp > self.neg_th] = 1
                    last_mask = np.expand_dims(last_mask, axis=2)
                    if adapt_flag:
                        negatives = self._adapt(video_idx, t, last_mask, get_posteriors, adapt_flag=1)
                        adapt_flag = False

                n, measures, ys_argmax_val, posteriors_val, targets_val = self.process_forward(
                    data, network, save_logits, self.save_oneshot, targets, ys, start_frame_idx=t)
                assert n == 1
                assert len(measures) == 1
                measure = measures[0]
                print("Motion Adapted frame", t, ":", measure, " factor ", float(ys_argmax_val.sum()) / (
                854 * 480))

            measures_video.append(measure)
            # last_mask = ys_argmax_val[0]

            # prune negatives from last mask
            # negatives are None if we think that the target is lost
            #      if negatives is not None and self.use_negatives:
            #        last_mask[negatives] = 0
        #########

        measures_video[:-1] = measures_video[:-1]
        measures_video = average_measures(measures_video)
        print ( "sequence", video_idx + 1, data.video_tag(video_idx), measures_video)

    def _adapt(self, video_idx, frame_idx, last_mask, get_posteriors_fn, adapt_flag=0):
        eroded_mask = grey_erosion(last_mask, size=(self.erosion_size, self.erosion_size, 1))
        dt = distance_transform_edt(numpy.logical_not(eroded_mask))

        adaptation_target = numpy.zeros_like(last_mask)
        adaptation_target[:] = VOID_LABEL

        current_posteriors = get_posteriors_fn()
        if adapt_flag == 0:
            positives = current_posteriors[:, :, 1] > self.posterior_positive_threshold
        else:
            positives = last_mask == 1

        if self.use_positives:
            adaptation_target[positives] = 1

        threshold = self.distance_negative_threshold
        negatives = dt > threshold
        if self.use_negatives:
            adaptation_target[negatives] = 0

        do_adaptation = eroded_mask.sum() > 0

        if self.debug:
            adaptation_target_visualization = adaptation_target.copy()
            adaptation_target_visualization[adaptation_target == 1] = 128
            if not do_adaptation:
                adaptation_target_visualization[:] = VOID_LABEL
            from scipy.misc import imsave
            folder = self.val_data.video_tag().replace("__", "/")
            if not os.path.exists("forwarded" + "/valid/" + folder):
                os.makedirs("forwarded" + "/valid/" + folder)
            imsave("forwarded" + "/valid/" + folder + "/adaptation_%05d.png" % frame_idx,
                   numpy.squeeze(adaptation_target_visualization))

        self.train_data.set_video_idx(video_idx)

        for idx in range(self.n_adaptation_steps):
            do_step = True
            # if idx % self.adaptation_interval == 0:
            if do_adaptation:
                feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx, with_annotations=True)
                feed_dict[self.model.is_training] = True

                feed_dict[self.train_data.get_label_placeholder()] = adaptation_target
                loss_scale = self.adaptation_loss_scale
                adaption_frame_idx = frame_idx
            else:
                print ("skipping current frame adaptation, since the target seems to be lost")
                do_step = False
                # else:
                # mix in first frame to avoid drift
                # (do this even if we think the target is lost, since then this can help to find back the target)
            #  feed_dict = self.train_data.feed_dict_for_video_frame(frame_idx=0, with_annotations=True)
            #  loss_scale = 1.0
            #  adaption_frame_idx = 0

            if do_step:
                loss,  measures = self.trainer.train_step(epoch=idx, feed_dict=feed_dict, loss_scale=loss_scale,
                                                          learning_rate=self.adaptation_learning_rate)
                # assert n_imgs == 1
                print ( "adapting on frame", adaption_frame_idx, "of sequence", video_idx + 1, \
                self.train_data.video_tag(video_idx), "loss:", loss)
        if do_adaptation:
            return negatives
        else:
            return None
