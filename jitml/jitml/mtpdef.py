from canddef import of_MTP
from jitdef  import Ctx
from netdef  import Ann, Bnn, Snn, Wnn


class Vote:
    @staticmethod
    def snn_from(cands: list[Snn]) -> Snn:
        ...

    @staticmethod
    def ann_from(cands: list[Ann]) -> Ann:
        ...

    @staticmethod
    def wnn_from(cands: list[Wnn]) -> Wnn:
        ...

    @staticmethod
    def bnn_from(cands: list[Bnn]) -> Bnn:
        ...


def MTP(ctx: Ctx) -> None:
    ctx.clock += 1

    if ctx.clock % ctx.every:
        return

    snn_needs = ctx.mysnn.lifescore >= ctx.mysnn.lifelimit
    ann_needs = ctx.myann.lifescore >= ctx.myann.lifelimit
    wnn_needs = ctx.mywnn.lifescore >= ctx.mywnn.lifelimit
    bnn_needs = ctx.mybnn.lifescore >= ctx.mybnn.lifelimit

    if snn_needs:
        ctx.mysnn.trainme(ctx.myque)
    if ann_needs:
        ctx.myann.trainme(ctx.myque)
    if wnn_needs:
        ctx.mywnn.trainme(ctx.myque)
    if bnn_needs:
        ctx.mybnn.trainme(ctx.myque)

    if snn_needs:
        ctx.mysnn = Vote.snn_from([ctx.mysnn] + of_MTP(ctx.mysnn, ctx.mnnum))
    if ann_needs:
        ctx.myann = Vote.ann_from([ctx.myann] + of_MTP(ctx.myann, ctx.mnnum))
    if wnn_needs:
        ctx.mywnn = Vote.wnn_from([ctx.mywnn] + of_MTP(ctx.mywnn, ctx.mnnum))
    if bnn_needs:
        ctx.mybnn = Vote.bnn_from([ctx.mybnn] + of_MTP(ctx.mybnn, ctx.mnnum))

    if snn_needs:
        ctx.mysnn.lifescore = 0
    if ann_needs:
        ctx.myann.lifescore = 0
    if wnn_needs:
        ctx.mywnn.lifescore = 0
    if bnn_needs:
        ctx.mybnn.lifescore = 0