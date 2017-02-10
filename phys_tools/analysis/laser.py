from .base import Session, Unit

class LaserUnit(Session):
    pass

class LaserSession(Session):
    unit_type = LaserUnit

    def _make_stimuli(self, meta_fn):
        pass

    def get_laser_times(self):
        pass

    def get_laser_epochs(self, stim_id, prepad_ms, postpad_ms, units='all'):
        pass


class LaserOdorUnit(Unit):
    pass


class LaserOdorSession(OdorSession, LaserSession):
    unit_type = LaserOdorUnit

    def _make_stimuli(self, meta_fn):
        pass

    pass