from dataclasses import dataclass
from envdef      import Env
from netdef      import Ann, Bnn, Snn, Wnn, Que, Siamese, Tnn
from stepdef     import Step


@dataclass
class Ctx:
    runme   : bool   = True

    every   : int    = 3
    clock   : int    = 0

    tnnum   : int    = 7
    mnnum   : int    = 7

    myque   : Que    = []
    okque   : Que    = []
    dfque   : Que    = [Tnn()]

    myenv   : Env    = Env()
    mysnn   : Snn    = Snn()
    myann   : Ann    = Ann()
    mywnn   : Wnn    = Wnn()
    mybnn   : Bnn    = Bnn()
    siamese: Siamese = Siamese()


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