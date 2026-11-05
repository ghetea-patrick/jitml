from jitdef import Ctx
from netdef import Tnn
from mtpdef import MTP
from tgpdef import TGP


def is_match(tnn: Tnn, ctx: Ctx) -> bool:
    ...


def is_alive(tnn: Tnn) -> bool:
    ...


class Step:
    @staticmethod
    def doneplay(ctx: Ctx) -> None:
        ...

    @staticmethod
    def uirender(ctx: Ctx) -> None:
        ...

    @staticmethod
    def keepbusy(ctx: Ctx) -> None:
        ...

    @staticmethod
    def killdead(ctx: Ctx) -> None:
        ...

    @staticmethod
    def givelife(ctx: Ctx) -> None:
        ...

    @staticmethod
    def makemove(ctx: Ctx) -> None:
        ...

    @staticmethod
    def needstgp(ctx: Ctx) -> None:
        ...

    @staticmethod
    def needsmtp(ctx: Ctx) -> None:
        ...