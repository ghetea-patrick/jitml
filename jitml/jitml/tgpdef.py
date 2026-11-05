from dataclasses import dataclass, field
from canddef     import of_TGP
from jitdef      import Ctx
from netdef      import Cube, Grid, Tess, Tnn


@dataclass
class Elec:
    @dataclass
    class Lays:
        init: list[int]   = field(default_factory=list)
        cand: Grid[int]   = field(default_factory=list)

    @dataclass
    class Acti:
        init: list        = field(default_factory=list)
        cand: Grid        = field(default_factory=list)

    @dataclass
    class Weig:
        init: Cube[float] = field(default_factory=list)
        cand: Tess[float] = field(default_factory=list)

    @dataclass
    class Bias:
        init: Grid[float] = field(default_factory=list)
        cand: Cube[float] = field(default_factory=list)

    lays: Lays = field(default_factory=Lays)
    acti: Acti = field(default_factory=Acti)
    weig: Weig = field(default_factory=Weig)
    bias: Bias = field(default_factory=Bias)


class Vote:
    @staticmethod
    def layersize_from(elec: Elec) -> None:
        ...

    @staticmethod
    def activfunc_from(elec: Elec) -> None:
        ...

    @staticmethod
    def weigvalue_from(elec: Elec) -> None:
        ...

    @staticmethod
    def biasvalue_from(elec: Elec) -> None:
        ...


class Oops:
    @staticmethod
    def layersize_from(elec: Elec) -> None:
        ...

    @staticmethod
    def activfunc_from(elec: Elec) -> None:
        ...

    @staticmethod
    def weigvalue_from(elec: Elec) -> None:
        ...

    @staticmethod
    def biasvalue_from(elec: Elec) -> None:
        ...


def TGP(ctx: Ctx) -> Tnn:
    elec = Elec()

    ctx.mysnn.gen_layersize(elec)
    ctx.myann.gen_activfunc(elec)
    ctx.mywnn.gen_weigvalue(elec)
    ctx.mybnn.gen_biasvalue(elec)

    elec.lays.cand = [elec.lays.init] + of_TGP(elec.lays.init, ctx.tnnum)
    elec.acti.cand = [elec.acti.init] + of_TGP(elec.acti.init, ctx.tnnum)
    elec.weig.cand = [elec.weig.init] + of_TGP(elec.weig.init, ctx.tnnum)
    elec.bias.cand = [elec.bias.init] + of_TGP(elec.bias.init, ctx.tnnum)

    layersize = Vote.layersize_from(elec) or Oops.layersize_from(elec)
    activfunc = Vote.activfunc_from(elec) or Oops.activfunc_from(elec)
    weigvalue = Vote.weigvalue_from(elec) or Oops.weigvalue_from(elec)
    biasvalue = Vote.biasvalue_from(elec) or Oops.biasvalue_from(elec)
 
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