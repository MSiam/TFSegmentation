from Forwarding.Forwarder import ImageForwarder
from Forwarding.OneshotForwarder import OneshotForwarder
from Forwarding.OnlineAdaptingForwarder import OnlineAdaptingForwarder


def forward(engine, network, data, dataset_name, save_results, save_logits):
  forwarder = ImageForwarder(engine)
  forwarder.forward(network, data, save_results, save_logits)


def oneshot_forward(engine, save_results, save_logits):
  if engine.dataset in ("davis", "davis17", "davis2017", "davis_video", "oxford", "youtube", "youtubeobjects",
                        "youtubefull", "youtubeobjectsfull", "segtrackv2"):
    forwarder = OneshotForwarder(engine)
  else:
    assert False, "unknown dataset for oneshot: " + engine.dataset
  forwarder.forward(None, None, save_results, save_logits)


def online_forward(engine, save_results, save_logits):
  forwarder = OnlineAdaptingForwarder(engine)
  forwarder.forward(None, None, save_results, save_logits)
