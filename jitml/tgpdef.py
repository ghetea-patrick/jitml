from jitdef import Ctx
from netdef import Cube, Grid, Tess, Tnn


class Vote:
    @staticmethod
    def layersize_from(cands: Grid[int]) -> list[int]:
        ...

    @staticmethod
    def activfunc_from(cands: Grid) -> list:
        ...

    @staticmethod
    def weigvalue_from(cands: Tess[float]) -> Cube[float]:
        ...

    @staticmethod
    def biasvalue_from(cands: Cube[float]) -> Grid[float]:
        ...


class Oops:
    @staticmethod
    def layersize_from(cands: Grid[int]) -> list[int]:
        ...

    @staticmethod
    def activfunc_from(cands: Grid) -> list:
        ...

    @staticmethod
    def weigvalue_from(cands: Tess[float]) -> Cube[float]:
        ...

    @staticmethod
    def biasvalue_from(cands: Cube[float]) -> Grid[float]:
        ...


def candidates_of[T](L: list[T], num: int) -> Grid[T]:
    ...


def TGP(ctx: Ctx) -> Tnn:
    laysinit  = ctx.mysnn.gen_layersize()
    actiinit  = ctx.myann.gen_activfunc()
    weiginit  = ctx.mywnn.gen_weigvalue()
    biasinit  = ctx.mybnn.gen_biasvalue()

    layscands = [laysinit] + candidates_of(laysinit, ctx.tnnum)
    acticands = [actiinit] + candidates_of(actiinit, ctx.tnnum)
    weigcands = [weiginit] + candidates_of(weiginit, ctx.tnnum)
    biascands = [biasinit] + candidates_of(biasinit, ctx.tnnum)

    layersize = Vote.layersize_from(layscands) or Oops.layersize_from(layscands)
    activfunc = Vote.activfunc_from(acticands) or Oops.activfunc_from(layscands)
    weigvalue = Vote.weigvalue_from(weigcands) or Oops.weigvalue_from(weigcands)
    biasvalue = Vote.biasvalue_from(biascands) or Oops.biasvalue_from(biascands)
 
    ctx.mysnn.lifescore += 1
    ctx.myann.lifescore += 1
    ctx.mywnn.lifescore += 1
    ctx.mybnn.lifescore += 1

    return Tnn(
        layersize=layersize,
        activfunc=activfunc,
        weigvalue=weigvalue,
        biasvalue=biasvalue,
    )