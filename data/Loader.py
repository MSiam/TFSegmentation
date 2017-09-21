from DAVIS.DAVIS import DAVISDataset, DAVIS2017Dataset
from datasets.Custom.Custom import CustomDataset
from datasets.DAVIS.DAVIS2017_oneshot import Davis2017OneshotDataset
from datasets.DAVIS.DAVIS_oneshot import DavisOneshotDataset
from datasets.PascalVOC.PascalVOC_instance import PascalVOCInstanceDataset
from datasets.PascalVOC.PascalVOC_objectness import PascalVOCObjectnessDataset
from datasets.PascalVOC.PascalVOC_semantic import PascalVOCSemanticDataset
from datasets.Segtrackv2.Segtrackv2_oneshot import Segtrackv2OneshotDataset
from datasets.YoutubeObjects.YoutubeObjectsFull_oneshot import YoutubeObjectsFullOneshotDataset
from datasets.YoutubeObjects.YoutubeObjects_oneshot import YoutubeObjectsOneshotDataset


def load_dataset(config, subset, session, coordinator):
  name = config.unicode("dataset").lower()
  task = config.unicode("task", "")
  if task in ("oneshot", "oneshot_forward", "online", "offline"):
    if name == "davis":
      return DavisOneshotDataset(config, subset, use_old_label=False)
    elif name in ("davis17", "davis2017"):
      return Davis2017OneshotDataset(config, subset)
    elif name in ("youtube", "youtubeobjects"):
      return YoutubeObjectsOneshotDataset(config, subset)
    elif name in ("youtubefull", "youtubeobjectsfull"):
      return YoutubeObjectsFullOneshotDataset(config, subset)
    elif name == "segtrackv2":
      return Segtrackv2OneshotDataset(config, subset)
    else:
      assert False, "Unknown dataset for oneshot: " + name

  if task == "forward" and name == "davis_mask":
    return DavisOneshotDataset(config, subset, use_old_label=True)

  if name == "davis":
    return DAVISDataset(config, subset, coordinator)
  elif name in ("davis17", "davis2017"):
    return DAVIS2017Dataset(config, subset, coordinator)
  elif name in ("davis17_test", "davis2017_test"):
    return DAVIS2017Dataset(config, subset, coordinator, fraction=0.002)
  elif name == "davis_test":
    return DAVISDataset(config, subset, coordinator, fraction=0.05)
  elif name == "pascalvoc_instance":
    return PascalVOCInstanceDataset(config, subset, coordinator)
  elif name == "pascalvoc_instance_test":
    return PascalVOCInstanceDataset(config, subset, coordinator, fraction=0.03)
  elif name == "pascalvoc_objectness":
    return PascalVOCObjectnessDataset(config, subset, coordinator)
  elif name == "pascalvoc_objectness_test":
    return PascalVOCObjectnessDataset(config, subset, coordinator, fraction=0.03)
  elif name == "pascalvoc":
    return PascalVOCSemanticDataset(config, subset, coordinator)
  elif name == "pascalvoc_test":
    return PascalVOCSemanticDataset(config, subset, coordinator, fraction=0.03)
  elif name == "custom":
    return CustomDataset(config, subset, coordinator)
  assert False, "Unknown dataset " + name
