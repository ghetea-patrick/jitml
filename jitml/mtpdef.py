from jitdef import Ctx
from netdef import Ann, Bnn, Snn, Wnn


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


def candidates_of[T](L: T, num: int) -> list[T]:
    ...


def MTP(ctx: Ctx) -> None:
    ctx.clock += 1

    if ctx.clock == ctx.ticks:
        ctx.clock = 0
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
        ctx.mysnn = Vote.snn_from([ctx.mysnn] + candidates_of(ctx.mysnn, ctx.mnnum))
    if ann_needs:
        ctx.myann = Vote.ann_from([ctx.myann] + candidates_of(ctx.myann, ctx.mnnum))
    if wnn_needs:
        ctx.mywnn = Vote.wnn_from([ctx.mywnn] + candidates_of(ctx.mywnn, ctx.mnnum))
    if bnn_needs:
        ctx.mybnn = Vote.bnn_from([ctx.mybnn] + candidates_of(ctx.mybnn, ctx.mnnum))

    if snn_needs:
        ctx.mysnn.lifescore += 1
    if ann_needs:
        ctx.myann.lifescore += 1
    if wnn_needs:
        ctx.mywnn.lifescore += 1
    if bnn_needs:
        ctx.mybnn.lifescore += 1