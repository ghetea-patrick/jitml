from dataclasses import dataclass

from envdef import Env
from netdef import Ann, Bnn, Snn, Wnn, Que, Tnn
from stepdef import Step


@dataclass
class Ctx:
    runme: bool = True

    every: int = 32
    ticks: int = 64
    clock: int = 0

    myenv: Env = Env()

    myque: Que = []
    okque: Que = []
    dfque: Que = [Tnn()]

    mysnn: Snn = Snn()
    myann: Ann = Ann()
    mywnn: Wnn = Wnn()
    mybnn: Bnn = Bnn()


def jitml(ctx: Ctx) -> None:
    while ctx.runme:
        Step.doneplay(ctx)
        Step.uirender(ctx)
        Step.keepbusy(ctx)
        Step.killdead(ctx)
        Step.givelife(ctx)
        Step.needstgp(ctx)
        Step.makemove(ctx)
        Step.needsmtp(ctx)