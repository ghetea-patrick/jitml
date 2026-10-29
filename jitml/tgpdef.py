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


def candidates_of[T](L: list[T]) -> Grid[T]:
    ...


def TGP(ctx: Ctx) -> Tnn:
    grid = ctx.myenv.near()

    laysinit = ctx.mysnn.gen_layersize()
    actiinit = ctx.myann.gen_activfunc()
    weiginit = ctx.mywnn.gen_weigvalue()
    biasinit = ctx.mybnn.gen_biasvalue()

    layscand = candidates_of(laysinit)
    acticand = candidates_of(actiinit)
    weigcand = candidates_of(weiginit)
    biascand = candidates_of(biasinit)

    layersize = Vote.layersize_from(layscand)
    activfunc = Vote.activfunc_from(acticand)
    weigvalue = Vote.weigvalue_from(weigcand)
    biasvalue = Vote.biasvalue_from(biascand)

    return Tnn(
        layersize=layersize,
        activfunc=activfunc,
        weigvalue=weigvalue,
        biasvalue=biasvalue)