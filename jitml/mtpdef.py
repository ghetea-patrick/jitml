from jitdef import Ctx


def MTP(ctx: Ctx) -> None:
    ctx.mysnn.trainme(ctx.myque)
    ctx.myann.trainme(ctx.myque)
    ctx.mywnn.trainme(ctx.myque)
    ctx.mybnn.trainme(ctx.myque)