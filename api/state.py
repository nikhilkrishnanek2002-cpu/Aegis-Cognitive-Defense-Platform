"""
Shared radar processing state (tracker, cognitive controller, EW defense)
kept alive across multiple API requests using module-level singletons.
"""
from src.config import get_config
from src.tracker import MultiTargetTracker
from src.cognitive_controller import CognitiveRadarController
from src.ew_defense import EWDefenseController

_cfg = get_config()

tracker: MultiTargetTracker = MultiTargetTracker(_cfg.get("tracker", {}))
cognitive_controller: CognitiveRadarController = CognitiveRadarController(_cfg.get("cognitive_controller", {}))
ew_defense: EWDefenseController = EWDefenseController(_cfg.get("ew_defense", {}))
