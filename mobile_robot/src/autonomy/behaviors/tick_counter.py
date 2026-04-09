import py_trees


class TickCounter(py_trees.behaviour.Behaviour):
    """Return RUNNING for N ticks, then SUCCESS.

    This is useful as a stand-in for actions like "navigate to pose" or
    "load tray" while you are wiring up a behaviour tree.

    Args:
        name: behaviour name
        ticks_to_succeed: number of ticks before returning SUCCESS
        status_after_success: status to return after first success (defaults to SUCCESS)
            - SUCCESS: stays successful forever
            - RUNNING: will keep running even after completing once
            - FAILURE: will report failure after completing once
    """

    def __init__(
        self,
        name: str,
        ticks_to_succeed: int,
        status_after_success: py_trees.common.Status = py_trees.common.Status.SUCCESS,
    ):
        super().__init__(name=name)
        if ticks_to_succeed < 1:
            raise ValueError("ticks_to_succeed must be >= 1")
        self.ticks_to_succeed = ticks_to_succeed
        self.status_after_success = status_after_success
        self._ticks = 0
        self._completed_once = False

    def initialise(self):
        # Only reset when the behaviour is entered (i.e. transitions from
        # INVALID to RUNNING). This mimics action server semantics.
        self._ticks = 0

    def update(self) -> py_trees.common.Status:
        if self._completed_once:
            return self.status_after_success

        self._ticks += 1
        if self._ticks < self.ticks_to_succeed:
            return py_trees.common.Status.RUNNING

        self._completed_once = True
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status):
        # If pre-empted (e.g. phase switch), allow it to run again when re-entered.
        if new_status == py_trees.common.Status.INVALID:
            self._completed_once = False
